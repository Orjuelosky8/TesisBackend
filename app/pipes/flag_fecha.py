# app/pipes/flag_fecha.py
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from db import repo  # para registrar en flags_licitaciones


FLAG_CODE = "gap_fecha"   # 9 chars, estilo de tus otras flags
FLAG_NAME = "Gap de fechas"
FLAG_DESC = (
    "Diferencia de días hábiles entre hitos del cronograma (aceptación, presentación, apertura). "
    "Son {dias} días hábiles; la regla vigente espera ≤ {threshold} días hábiles."
)


def _ensure_flag(db: Session, threshold: int) -> None:
    """Crea/actualiza la flag en public.flags sin usar DO $$ ... $$."""
    desc = FLAG_DESC.format(dias="{n}", threshold=threshold)

    # 1) ¿ya existe?
    row = db.execute(
        text("SELECT id FROM public.flags WHERE codigo = :c"),
        {"c": FLAG_CODE},
    ).fetchone()

    if row is None:
        # 2) no existe → saco siguiente id
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


def _business_days(
    d1: Optional[datetime],
    d2: Optional[datetime],
    holidays: set[str] | None,
) -> Optional[int]:
    if not d1 or not d2:
        return None
    if d2 < d1:
        d1, d2 = d2, d1

    cur = d1.date()
    end = d2.date()
    days = 0
    while cur < end:
        if cur.weekday() < 5 and (not holidays or cur.isoformat() not in holidays):
            days += 1
        cur = cur + timedelta(days=1)
    return days


def run_flag_fecha_for_one(
    db: Session,
    licitacion_id: int,
    json_override: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    json_override opcional:
      {
        "threshold": 5,
        "holidays": ["2025-01-01", ...]
      }
    """
    json_override = json_override or {}
    threshold = int(json_override.get("threshold", 5))
    holidays = set(json_override.get("holidays", []))

    # 1) Traer fechas del staging, usando apertura o, si no hay, presentación
    row = db.execute(
        text(
            """
            SELECT
              n.archivo,
              n.aceptacion_ofertas_ts,
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
        return {"ok": False, "flow": "gap_fechas", "reason": "sin_calendario"}

    aceptacion = row.aceptacion_ofertas_ts
    apertura = row.apertura_ts

    dias = _business_days(aceptacion, apertura, holidays)
    if dias is None:
        return {"ok": False, "flow": "gap_fechas", "reason": "fechas_incompletas"}

    # 2) asegurar registro en public.flags
    _ensure_flag(db, threshold)

    comentario = (
        f"Gap: {dias} días hábiles "
        f"(Aceptación: {aceptacion}, Apertura/Presentación: {apertura}; archivo={row.archivo}). "
        f"Se espera ≤ {threshold} días hábiles."
    )

    # 3) registrar en flags_licitaciones
    repo.set_flag_for_licitacion(
        session=db,
        licitacion_id=licitacion_id,
        flag_codigo=FLAG_CODE,     # ← aquí también usamos el corto
        valor=bool(dias > threshold),
        comentario=comentario,
        fuente="pipe:gap_fechas",
        usuario_log="pipeline",
    )

    return {
        "ok": True,
        "flow": "gap_fechas",
        "flag_applied": dias > threshold,
        "detail": {
            "dias_habiles": dias,
            "threshold": threshold,
            "archivo": row.archivo,
            "aceptacion_ofertas_ts": aceptacion,
            "apertura_ofertas_ts": apertura,
        },
    }
