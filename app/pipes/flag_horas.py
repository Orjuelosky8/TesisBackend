# app/pipes/flag_hora_1159.py
from __future__ import annotations
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from db import repo  

# 
FLAG_CODE = "hora_1159"   # 8 chars
FLAG_NAME = "Hora 11:59"
FLAG_DESC = (
    "Marca si algún hito crítico del cronograma (aceptación, presentación, apertura) "
    "está configurado con hora {horas_objetivo}."
)


def _ensure_flag(db: Session, horas_objetivo: str) -> None:
    """Crea/actualiza la flag en public.flags sin usar DO $$ ... $$."""
    desc = FLAG_DESC.format(horas_objetivo=horas_objetivo)

    # 1) ¿ya existe?
    row = db.execute(
        text("SELECT id FROM public.flags WHERE codigo = :c"),
        {"c": FLAG_CODE},
    ).fetchone()

    if row is None:
        
        new_id = db.execute(
            text("SELECT COALESCE(MAX(id), 0) + 1 FROM public.flags")
        ).scalar_one()

        db.execute(
            text(
                "INSERT INTO public.flags (id, codigo, nombre, descripcion) "
                "VALUES (:id, :c, :n, :d)"
            ),
            {
                "id": new_id,
                "c": FLAG_CODE,
                "n": FLAG_NAME,
                "d": desc,
            },
        )
    else:
        # 3) sí existe → actualizo
        db.execute(
            text(
                "UPDATE public.flags "
                "SET nombre = :n, descripcion = :d "
                "WHERE codigo = :c"
            ),
            {
                "c": FLAG_CODE,
                "n": FLAG_NAME,
                "d": desc,
            },
        )


def _parse_target_times(json_override: dict) -> list[tuple[int, int]]:
    """
    Recibe algo como:
      { "target_times": ["23:59", "00:00"] }
    y lo convierte a [(23,59), (0,0)].
    Por defecto solo ["23:59"].
    """
    raw = json_override.get("target_times") or ["23:59"]
    parsed: list[tuple[int, int]] = []

    for s in raw:
        try:
            h_str, m_str = s.split(":")
            h = int(h_str)
            m = int(m_str)
            if 0 <= h <= 23 and 0 <= m <= 59:
                parsed.append((h, m))
        except Exception:
            # si viene algo raro lo ignoramos para no romper el pipe
            continue

    # fallback por si quedó vacío
    if not parsed:
        parsed = [(23, 59)]
    return parsed


def _is_target_time(
    dt: Optional[datetime],
    targets: list[tuple[int, int]],
) -> bool:
    if not isinstance(dt, datetime):
        return False
    return any((dt.hour, dt.minute) == t for t in targets)


def run_flag_hora_1159_for_one(
    db: Session,
    licitacion_id: int,
    json_override: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Revisa si los hitos del cronograma están configurados en 11:59 (por defecto)
    u otras horas objetivo.

    json_override opcional:
      {
        "target_times": ["23:59", "00:00"]
      }
    """
    json_override = json_override or {}
    target_times = _parse_target_times(json_override)
    target_times_str = ", ".join(f"{h:02d}:{m:02d}" for h, m in target_times)

    # 1) Traer fechas del staging (mismos campos que usas en gap_fechas)
    row = db.execute(
        text(
            """
            SELECT
              n.archivo,
              n.aceptacion_ofertas_ts,
              n.presentacion_ofertas_ts,
              COALESCE(n.apertura_ofertas_ts, n.presentacion_ofertas_ts) AS apertura_ts
            FROM public.licitacion_keymap k
            JOIN staging.secop_calendario_norm n
              ON n.archivo::text = k.lic_ext_id
            WHERE k.licitacion_id = :lid
            LIMIT 1;
            """
        ),
        {"lid": licitacion_id},
    ).fetchone()

    if not row:
        return {"ok": False, "flow": "hora_1159", "reason": "sin_calendario"}

    eventos: dict[str, Optional[datetime]] = {
        "aceptacion_ofertas_ts": row.aceptacion_ofertas_ts,
        "presentacion_ofertas_ts": row.presentacion_ofertas_ts,
        "apertura_ts": row.apertura_ts,
    }

    # 2) Ver si alguno está en 11:59 (23:59 por defecto)
    eventos_en_1159: dict[str, datetime] = {}
    for nombre, dt in eventos.items():
        if _is_target_time(dt, target_times):
            eventos_en_1159[nombre] = dt

    flag_val = bool(eventos_en_1159)

    # 3) asegurar registro en public.flags
    _ensure_flag(db, target_times_str)

    # 4) comentario explicativo
    partes_eventos = []
    for nombre, dt in eventos.items():
        partes_eventos.append(f"{nombre}={dt}")

    comentario = (
        f"Horas objetivo: {target_times_str}. "
        f"archivo={row.archivo}. "
        f"Eventos: " + "; ".join(partes_eventos)
    )

    if flag_val:
        detalle_hits = ", ".join(
            f"{k}={v}" for k, v in eventos_en_1159.items()
        )
        comentario += (
            f". Se encontró al menos un hito con hora exacta de cierre "
            f"({target_times_str}): {detalle_hits}."
        )
    else:
        comentario += (
            f". Ningún hito usa exactamente las horas objetivo ({target_times_str})."
        )

    # 5) registrar en flags_licitaciones
    repo.set_flag_for_licitacion(
        session=db,
        licitacion_id=licitacion_id,
        flag_codigo=FLAG_CODE,
        valor=flag_val,
        comentario=comentario,
        fuente="pipe:hora_1159",
        usuario_log="pipeline",
    )

    return {
        "ok": True,
        "flow": "hora_1159",
        "flag_applied": flag_val,
        "detail": {
            "archivo": row.archivo,
            "target_times": target_times_str,
            "eventos_en_1159": eventos_en_1159,
        },
    }
