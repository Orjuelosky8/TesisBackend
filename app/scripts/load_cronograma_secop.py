# /app/scripts/load_cronograma_secop.py
from __future__ import annotations
import argparse, os, re
from typing import Optional, Dict, List
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from datetime import datetime

DDL = """
CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.secop_calendario_raw (
  archivo                                  TEXT NOT NULL,
  aceptacion_ofertas_raw                   TEXT,
  apertura_ofertas_raw                     TEXT,
  fecha_publicacion_raw                    TEXT,
  presentacion_ofertas_raw                 TEXT,
  apertura_sobres_raw                      TEXT,
  apertura_sobre_economico_raw             TEXT,
  apertura_req_hab_tec_raw                 TEXT,
  fecha_limite_presentacion_ofertas_raw    TEXT,
  fecha_limite_apertura_sobres_raw         TEXT,
  fecha_publicacion_proceso_raw            TEXT
);

CREATE TABLE IF NOT EXISTS staging.secop_calendario_norm (
  archivo                                  TEXT PRIMARY KEY,
  aceptacion_ofertas_ts                    TIMESTAMP NULL,
  apertura_ofertas_ts                      TIMESTAMP NULL,
  fecha_publicacion_ts                     TIMESTAMP NULL,
  presentacion_ofertas_ts                  TIMESTAMP NULL,
  apertura_sobres_ts                       TIMESTAMP NULL,
  apertura_sobre_economico_ts              TIMESTAMP NULL,
  apertura_req_hab_tec_ts                  TIMESTAMP NULL,
  fecha_limite_presentacion_ofertas_ts     TIMESTAMP NULL,
  fecha_limite_apertura_sobres_ts          TIMESTAMP NULL,
  fecha_publicacion_proceso_ts             TIMESTAMP NULL
);
-- por si ya existía, añadimos columnas nuevas
ALTER TABLE staging.secop_calendario_raw
  ADD COLUMN IF NOT EXISTS apertura_sobres_raw TEXT,
  ADD COLUMN IF NOT EXISTS apertura_sobre_economico_raw TEXT,
  ADD COLUMN IF NOT EXISTS apertura_req_hab_tec_raw TEXT,
  ADD COLUMN IF NOT EXISTS fecha_limite_presentacion_ofertas_raw TEXT,
  ADD COLUMN IF NOT EXISTS fecha_limite_apertura_sobres_raw TEXT,
  ADD COLUMN IF NOT EXISTS fecha_publicacion_proceso_raw TEXT;

ALTER TABLE staging.secop_calendario_norm
  ADD COLUMN IF NOT EXISTS apertura_sobres_ts TIMESTAMP NULL,
  ADD COLUMN IF NOT EXISTS apertura_sobre_economico_ts TIMESTAMP NULL,
  ADD COLUMN IF NOT EXISTS apertura_req_hab_tec_ts TIMESTAMP NULL,
  ADD COLUMN IF NOT EXISTS fecha_limite_presentacion_ofertas_ts TIMESTAMP NULL,
  ADD COLUMN IF NOT EXISTS fecha_limite_apertura_sobres_ts TIMESTAMP NULL,
  ADD COLUMN IF NOT EXISTS fecha_publicacion_proceso_ts TIMESTAMP NULL;
"""

# mapeo ampliado
COLMAP: Dict[str, str] = {
    "Archivo": "archivo",
    "Aceptación de ofertas": "aceptacion_ofertas_raw",
    "Apertura de Ofertas": "apertura_ofertas_raw",
    "Fecha de publicación": "fecha_publicacion_raw",
    "Presentación de Ofertas": "presentacion_ofertas_raw",

    # nuevas que sí aparecen en tu Excel
    "Apertura de sobres": "apertura_sobres_raw",
    "Apertura del sobre económico": "apertura_sobre_economico_raw",
    "Apertura de sobre de requisitos habilitantes y técnicos": "apertura_req_hab_tec_raw",
    "Fecha límite de presentación de ofertas": "fecha_limite_presentacion_ofertas_raw",
    "Fecha límite para la apertura de sobres": "fecha_limite_apertura_sobres_raw",
    "Fecha de publicación del proceso": "fecha_publicacion_proceso_raw",
}

ES2EN = {
    "Ene": "Jan", "Feb": "Feb", "Mar": "Mar", "Abr": "Apr", "May": "May", "Jun": "Jun",
    "Jul": "Jul", "Ago": "Aug", "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dic": "Dec"
}

DATE_RE = re.compile(r"(\d{1,2}/[A-Za-zÁÉÍÓÚáéíóú]{3}/\d{4})\s*-\s*(\d{1,2}:\d{2})\s*(am|pm)", re.IGNORECASE)
DATE_ONLY_RE = re.compile(r"(\d{1,2}/[A-Za-zÁÉÍÓÚáéíóú]{3}/\d{4})", re.IGNORECASE)

def normalize_es_datetime(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    # normalizar meses
    for es, en in ES2EN.items():
        s = re.sub(rf"(?i)\b{es}\b", en, s)

    # 1) date+time (toma la PRIMERA que encuentre, sirve para "Cronograma: ...")
    m = DATE_RE.search(s)
    if m:
        dt_str = f"{m.group(1)} {m.group(2)} {m.group(3).upper()}"
        try:
            return datetime.strptime(dt_str, "%d/%b/%Y %I:%M %p")
        except Exception:
            pass

    # 2) solo fecha
    m = DATE_ONLY_RE.search(s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d/%b/%Y")
        except Exception:
            pass

    # 3) fallback pandas
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return None if pd.isna(dt) else dt.to_pydatetime()
    except Exception:
        return None

def ensure_schema(engine: Engine):
    with engine.begin() as cx:
        cx.execute(text(DDL))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/licita_db"))
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--truncate", action="store_true")
    args = ap.parse_args()

    engine = create_engine(args.dsn)
    ensure_schema(engine)

    df = pd.read_excel(args.excel, sheet_name=args.sheet, engine="openpyxl")

    # renombra las que tenemos mapeadas; las que no, las deja
    df = df.rename(columns={k: v for k, v in COLMAP.items() if k in df.columns})

    # asegurar columnas que pusimos en RAW
    for v in COLMAP.values():
        if v not in df.columns:
            df[v] = None

    raw = df[list(COLMAP.values())].copy()

    with engine.begin() as cx:
        if args.truncate:
            cx.execute(text("TRUNCATE staging.secop_calendario_raw;"))
            cx.execute(text("TRUNCATE staging.secop_calendario_norm;"))

        cols = list(raw.columns)
        ph = ", ".join([f":{c}" for c in cols])
        sql = text(f"INSERT INTO staging.secop_calendario_raw ({', '.join(cols)}) VALUES ({ph})")
        cx.execute(sql, raw.where(pd.notnull(raw), None).to_dict(orient="records"))

    # normalizar
    norm_rows: List[dict] = []
    for _, r in raw.iterrows():
        archivo = (str(r["archivo"]).strip() if pd.notna(r["archivo"]) else None)
        if not archivo:
            continue
        norm_rows.append({
            "archivo": archivo,
            "aceptacion_ofertas_ts":                normalize_es_datetime(r.get("aceptacion_ofertas_raw")),
            "apertura_ofertas_ts":                  normalize_es_datetime(r.get("apertura_ofertas_raw")),
            "fecha_publicacion_ts":                 normalize_es_datetime(r.get("fecha_publicacion_raw")),
            "presentacion_ofertas_ts":              normalize_es_datetime(r.get("presentacion_ofertas_raw")),
            "apertura_sobres_ts":                   normalize_es_datetime(r.get("apertura_sobres_raw")),
            "apertura_sobre_economico_ts":          normalize_es_datetime(r.get("apertura_sobre_economico_raw")),
            "apertura_req_hab_tec_ts":              normalize_es_datetime(r.get("apertura_req_hab_tec_raw")),
            "fecha_limite_presentacion_ofertas_ts": normalize_es_datetime(r.get("fecha_limite_presentacion_ofertas_raw")),
            "fecha_limite_apertura_sobres_ts":      normalize_es_datetime(r.get("fecha_limite_apertura_sobres_raw")),
            "fecha_publicacion_proceso_ts":         normalize_es_datetime(r.get("fecha_publicacion_proceso_raw")),
        })

    if norm_rows:
        with engine.begin() as cx:
            upsert = text("""
                INSERT INTO staging.secop_calendario_norm (
                  archivo,
                  aceptacion_ofertas_ts,
                  apertura_ofertas_ts,
                  fecha_publicacion_ts,
                  presentacion_ofertas_ts,
                  apertura_sobres_ts,
                  apertura_sobre_economico_ts,
                  apertura_req_hab_tec_ts,
                  fecha_limite_presentacion_ofertas_ts,
                  fecha_limite_apertura_sobres_ts,
                  fecha_publicacion_proceso_ts
                )
                VALUES (
                  :archivo,
                  :aceptacion_ofertas_ts,
                  :apertura_ofertas_ts,
                  :fecha_publicacion_ts,
                  :presentacion_ofertas_ts,
                  :apertura_sobres_ts,
                  :apertura_sobre_economico_ts,
                  :apertura_req_hab_tec_ts,
                  :fecha_limite_presentacion_ofertas_ts,
                  :fecha_limite_apertura_sobres_ts,
                  :fecha_publicacion_proceso_ts
                )
                ON CONFLICT (archivo) DO UPDATE SET
                  aceptacion_ofertas_ts                = EXCLUDED.aceptacion_ofertas_ts,
                  apertura_ofertas_ts                  = EXCLUDED.apertura_ofertas_ts,
                  fecha_publicacion_ts                 = EXCLUDED.fecha_publicacion_ts,
                  presentacion_ofertas_ts              = EXCLUDED.presentacion_ofertas_ts,
                  apertura_sobres_ts                   = EXCLUDED.apertura_sobres_ts,
                  apertura_sobre_economico_ts          = EXCLUDED.apertura_sobre_economico_ts,
                  apertura_req_hab_tec_ts              = EXCLUDED.apertura_req_hab_tec_ts,
                  fecha_limite_presentacion_ofertas_ts = EXCLUDED.fecha_limite_presentacion_ofertas_ts,
                  fecha_limite_apertura_sobres_ts      = EXCLUDED.fecha_limite_apertura_sobres_ts,
                  fecha_publicacion_proceso_ts         = EXCLUDED.fecha_publicacion_proceso_ts;
            """)
            cx.execute(upsert, norm_rows)

    print(f">> Filas RAW insertadas: {len(raw)}; normalizadas: {len(norm_rows)}")

if __name__ == "__main__":
    main()
