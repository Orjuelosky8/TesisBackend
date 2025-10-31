# /app/scripts/import_xls.py
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -------------------------------------------------------------------
# DDL: staging alineado a columnas reales y PK(codigo)
# -------------------------------------------------------------------
DDL_STAGING = """
CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.licitaciones_xlsx_raw (
  codigo            TEXT PRIMARY KEY,      -- clave natural del Excel
  entidad           TEXT NOT NULL,
  objeto            TEXT,
  cuantia_raw       TEXT,
  modalidad         TEXT,
  numero            TEXT,
  estado            TEXT,
  fecha_public_raw  TEXT,
  ubicacion         TEXT,
  act_econ          TEXT,
  enlace            TEXT,
  portal_origen     TEXT,
  contratistas      TEXT,
  cant_docs         TEXT
);

-- staging.chunks (lic_id = codigo del Excel)
CREATE TABLE IF NOT EXISTS staging.chunks (
  chunk_id         TEXT PRIMARY KEY,
  lic_id           TEXT,
  doc_id           BIGINT,
  doc_chunk_index  BIGINT,
  lic_chunk_index  BIGINT,
  text             TEXT,
  created_at       TEXT
);
"""

# -------------------------------------------------------------------
# SQL: Sincronización hacia public.licitacion (UPDATE + INSERT)
# -------------------------------------------------------------------
SQL_SYNC_PUBLIC_LICITACION = """
WITH src AS (
  SELECT
    trim(entidad) AS entidad,
    objeto,
    NULLIF(REGEXP_REPLACE(COALESCE(cuantia_raw,''), '[^0-9.-]+', '', 'g'), '') AS cuantia_clean,
    trim(modalidad) AS modalidad,
    trim(numero)    AS numero,
    trim(estado)    AS estado,
    fecha_public_raw,
    ubicacion,
    act_econ,
    enlace,
    portal_origen,
    contratistas
  FROM staging.licitaciones_xlsx_raw
),
norm AS (
  SELECT
    entidad,
    objeto,
    CASE
      WHEN cuantia_clean IS NULL THEN NULL
      ELSE CAST(REPLACE(cuantia_clean, ',', '.') AS numeric(18,2))
    END AS cuantia,
    modalidad,
    numero,
    estado,
    CASE
      WHEN fecha_public_raw ~ '^\d{4}-\d{2}-\d{2}'
           THEN to_timestamp(substr(fecha_public_raw,1,19), 'YYYY-MM-DD HH24:MI:SS')::date
      WHEN fecha_public_raw ~ '^\d{2}/\d{2}/\d{4}'
           THEN to_date(fecha_public_raw, 'DD/MM/YYYY')
      ELSE NULL
    END AS fecha_public,
    ubicacion,
    act_econ,
    enlace,
    portal_origen,
    CONCAT_WS(' ',
      entidad, modalidad, numero, estado,
      to_char(
        COALESCE(
          CASE
            WHEN fecha_public_raw ~ '^\d{4}-\d{2}-\d{2}'
              THEN to_timestamp(substr(fecha_public_raw,1,19), 'YYYY-MM-DD HH24:MI:SS')::date
            WHEN fecha_public_raw ~ '^\d{2}/\d{2}/\d{4}'
              THEN to_date(fecha_public_raw, 'DD/MM/YYYY')
            ELSE NULL
          END,
          current_date
        ), 'YYYY-MM-DD'
      ),
      ubicacion, act_econ, COALESCE(objeto,'')
    ) AS texto_idx
  FROM src
),

-- 1) UPDATE filas existentes (match por entidad+numero)
upd AS (
  UPDATE public.licitacion l
  SET objeto         = n.objeto,
      cuantia        = n.cuantia,
      modalidad      = n.modalidad,
      estado         = n.estado,
      fecha_public   = n.fecha_public,
      ubicacion      = n.ubicacion,
      act_econ       = n.act_econ,
      enlace         = n.enlace,
      portal_origen  = n.portal_origen,
      texto_indexado = n.texto_idx
  FROM norm n
  WHERE COALESCE(l.entidad,'') = COALESCE(n.entidad,'')
    AND COALESCE(l.numero,'')  = COALESCE(n.numero,'')
  RETURNING 1
)

-- 2) INSERT faltantes
INSERT INTO public.licitacion (
  entidad, objeto, cuantia, modalidad, numero, estado,
  fecha_public, ubicacion, act_econ, enlace, portal_origen,
  objeto_vec, texto_indexado
)
SELECT
  n.entidad, n.objeto, n.cuantia, n.modalidad, n.numero, n.estado,
  n.fecha_public, n.ubicacion, n.act_econ, n.enlace, n.portal_origen,
  NULL, n.texto_idx
FROM norm n
WHERE NOT EXISTS (
  SELECT 1 FROM public.licitacion l
  WHERE COALESCE(l.entidad,'') = COALESCE(n.entidad,'')
    AND COALESCE(l.numero,'')  = COALESCE(n.numero,'')
);
"""

# -------------------------------------------------------------------
# SQL: Keymap y chunks → public
# -------------------------------------------------------------------
SQL_UPSERT_KEYMAP = """
WITH chosen AS (
  -- todas las licitaciones reales
  SELECT
    id AS licitacion_id,
    COALESCE(NULLIF(numero,''), id::text) AS numero_norm,
    ROW_NUMBER() OVER (
      PARTITION BY COALESCE(NULLIF(numero,''), id::text)
      ORDER BY fecha_public DESC NULLS LAST, id DESC
    ) rn
  FROM public.licitacion
),
src AS (
  -- todo lo que vino del Excel
  SELECT
    r.codigo AS lic_ext_id,
    r.numero AS numero_xls
  FROM staging.licitaciones_xlsx_raw r
  WHERE r.codigo IS NOT NULL AND r.codigo <> ''
),
pick AS (
  -- joineamos por numero (o por id texto) y nos quedamos solo con la
  -- mejor licitación real (rn = 1)
  SELECT
    c.licitacion_id,
    s.lic_ext_id
  FROM chosen c
  JOIN public.licitacion l ON l.id = c.licitacion_id
  JOIN src s
    ON COALESCE(NULLIF(l.numero,''), l.id::text)
     = COALESCE(NULLIF(s.numero_xls,''), l.id::text)
  WHERE c.rn = 1
),
filtered AS (
  -- aquí evitamos meter un lic_ext_id que ya esté usado por OTRA licitación
  SELECT p.*
  FROM pick p
  LEFT JOIN public.licitacion_keymap k
    ON k.lic_ext_id = p.lic_ext_id
  WHERE k.lic_ext_id IS NULL
     OR k.licitacion_id = p.licitacion_id
),
dedup AS (
  -- 👈 AQUI está la magia: si en filtered quedaron varias filas con el mismo
  -- licitacion_id (porque el Excel tenía repetidos), nos quedamos con UNA sola
  -- usando DISTINCT ON
  SELECT DISTINCT ON (licitacion_id)
         licitacion_id,
         lic_ext_id
  FROM filtered
  ORDER BY licitacion_id, lic_ext_id
)
INSERT INTO public.licitacion_keymap (licitacion_id, lic_ext_id)
SELECT licitacion_id, lic_ext_id
FROM dedup
ON CONFLICT (licitacion_id)
DO UPDATE SET lic_ext_id = EXCLUDED.lic_ext_id;
"""

SQL_UPSERT_CHUNKS_TO_PUBLIC = """
INSERT INTO public.licitacion_chunk (licitacion_id, chunk_idx, chunk_text)
SELECT
  k.licitacion_id,
  c.lic_chunk_index::int,
  c.text
FROM staging.chunks c
JOIN public.licitacion_keymap k
  ON k.lic_ext_id = c.lic_id
ON CONFLICT (licitacion_id, chunk_idx) DO UPDATE
  SET chunk_text = EXCLUDED.chunk_text;
"""

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def chunk_text(txt: str, chunk_len: int = 1200, overlap: int = 100) -> List[str]:
    """Chunker simple por caracteres (con solape)."""
    if not txt:
        return []
    txt = str(txt).strip()
    if not txt:
        return []
    chunks: List[str] = []
    i = 0
    n = len(txt)
    while i < n:
        j = min(i + chunk_len, n)
        chunks.append(txt[i:j])
        if j == n:
            break
        i = j - overlap if j - overlap > i else j
    return chunks


def ensure_staging(engine: Engine):
    with engine.begin() as cx:
        cx.execute(text(DDL_STAGING))


def load_excel_to_staging_raw(engine: Engine, xls_path: str, sheet: Optional[str] = None) -> int:
    """
    Carga Excel (.xls/.xlsx) → staging.licitaciones_xlsx_raw con UPSERT por codigo.
    """
    ext = os.path.splitext(xls_path)[1].lower()
    engine_name = "xlrd" if ext == ".xls" else "openpyxl"

    sheet_name = sheet if sheet is not None else 0
    df = pd.read_excel(xls_path, sheet_name=sheet_name, engine=engine_name)

    colmap = {
        "Codigo": "codigo",
        "Entidad": "entidad",
        "Objeto": "objeto",
        "Cuantía": "cuantia_raw",
        "Modalidad": "modalidad",
        "Número": "numero",
        "Estado": "estado",
        "F. Publicación": "fecha_public_raw",
        "Ubicación": "ubicacion",
        "Actividad Económica": "act_econ",
        "Enlace": "enlace",
        "Portal de origen": "portal_origen",
        "Contratista(s)": "contratistas",
        "Cantidad Documentos": "cant_docs",
    }

    df = df.rename(columns=colmap)

    for c in colmap.values():
        if c not in df.columns:
            df[c] = None

    if "codigo" in df.columns:
        df["codigo"] = df["codigo"].astype(str).str.strip()
    if "cuantia_raw" in df.columns:
        df["cuantia_raw"] = df["cuantia_raw"].astype(str).str.strip()

    records = df[list(colmap.values())].where(pd.notnull(df), None).to_dict(orient="records")
    if not records:
        return 0

    cols = list(colmap.values())
    placeholders = ", ".join(cols)
    values_clause = ", ".join([f":{k}" for k in cols])
    update_clause = ", ".join([f"{k} = EXCLUDED.{k}" for k in cols if k != "codigo"])

    sql = text(f"""
        INSERT INTO staging.licitaciones_xlsx_raw ({placeholders})
        VALUES ({values_clause})
        ON CONFLICT (codigo) DO UPDATE
        SET {update_clause}
    """)

    with engine.begin() as cx:
        cx.execute(sql, records)
    return len(records)


def build_staging_chunks(engine: Engine, from_objeto_only: bool = True, chunk_len: int = 1200, overlap: int = 100) -> int:
    """
    Construye staging.chunks a partir de staging.licitaciones_xlsx_raw.
    - lic_id = codigo (externo)
    - contenido por defecto: Objeto (o campos alternos si from_objeto_only=False)
    """
    q = text("""
        SELECT codigo, entidad, objeto, modalidad, numero, estado, ubicacion, act_econ
        FROM staging.licitaciones_xlsx_raw
        WHERE codigo IS NOT NULL AND codigo <> ''
    """)
    with engine.begin() as cx:
        rows = cx.execute(q).fetchall()

    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    inserts: List[dict] = []
    for (codigo, entidad, objeto, modalidad, numero, estado, ubicacion, act_econ) in rows:
        lic_id = str(codigo).strip()
        base_text = (objeto or "").strip()
        if not base_text and not from_objeto_only:
            base_text = " ".join([
                str(entidad or ""), str(modalidad or ""), str(numero or ""),
                str(estado or ""), str(ubicacion or ""), str(act_econ or "")
            ]).strip()
        if not base_text:
            continue

        parts = chunk_text(base_text, chunk_len=chunk_len, overlap=overlap)
        for i, ch in enumerate(parts):
            chunk_id = f"{lic_id}:{i}"
            inserts.append({
                "chunk_id": chunk_id,
                "lic_id": lic_id,
                "doc_id": 0,
                "doc_chunk_index": i,
                "lic_chunk_index": i,
                "text": ch,
                "created_at": now_iso
            })

    if not inserts:
        return 0

    cols = ["chunk_id","lic_id","doc_id","doc_chunk_index","lic_chunk_index","text","created_at"]
    ph  = ", ".join([f":{c}" for c in cols])
    sql = text(f"""
        INSERT INTO staging.chunks ({", ".join(cols)})
        VALUES ({ph})
        ON CONFLICT (chunk_id) DO UPDATE
        SET text = EXCLUDED.text,
            created_at = EXCLUDED.created_at
    """)
    with engine.begin() as cx:
        cx.execute(sql, inserts)
    return len(inserts)


def upsert_public(engine: Engine):
    with engine.begin() as cx:
        cx.execute(text(SQL_SYNC_PUBLIC_LICITACION))
        cx.execute(text(SQL_UPSERT_KEYMAP))
        cx.execute(text(SQL_UPSERT_CHUNKS_TO_PUBLIC))


def main():
    ap = argparse.ArgumentParser(description="Importa Excel → staging y public; crea chunks y keymap.")
    ap.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@db:5432/licita_db"),
        help="SQLAlchemy DSN (DATABASE_URL)"
    )
    ap.add_argument("--excel", required=True, help="Ruta del Excel (.xls/.xlsx), p.ej. Secop_2.xlsx")
    ap.add_argument("--sheet", default=None, help="Nombre o índice de la hoja (por defecto 0)")
    ap.add_argument("--chunk-len", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--raw-only", action="store_true", help="Solo cargar staging.licitaciones_xlsx_raw")
    ap.add_argument("--no-public", action="store_true", help="No tocar public.* (solo staging.* y chunks)")
    ap.add_argument("--from-objeto-only", action="store_true", help="Chunks solo del campo Objeto")
    args = ap.parse_args()

    engine = create_engine(args.dsn)

    print(">> Asegurando staging.* ...")
    ensure_staging(engine)

    print(">> Cargando Excel a staging.licitaciones_xlsx_raw (UPSERT por codigo) ...")
    n = load_excel_to_staging_raw(engine, args.excel, args.sheet)
    print(f"   Filas crudas upserted: {n}")

    if args.raw_only:
        print(">> Finalizado (raw-only).")
        return

    print(">> Construyendo staging.chunks ...")
    m = build_staging_chunks(
        engine,
        from_objeto_only=args.from_objeto_only,
        chunk_len=args.chunk_len,
        overlap=args.overlap
    )
    print(f"   Chunks upserted: {m}")

    if not args.no_public:
        print(">> Sincronizando public.licitacion + keymap + licitacion_chunk ...")
        upsert_public(engine)
        print("   Listo.")


if __name__ == "__main__":
    main()
