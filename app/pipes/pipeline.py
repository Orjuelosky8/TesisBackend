# app/pipes/pipeline.py
from __future__ import annotations
from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

from .flag_redcontactos import run_red_contactos

try:
    from .flag_precio import run_flag_precio_for_one as _run_precio_one
    HAS_PRECIO = True
except Exception:
    HAS_PRECIO = False

try:
    from .flag_fecha import run_flag_fecha_for_one as _run_gap_fecha_one
    HAS_GAP_FECHA = True
except Exception:
    HAS_GAP_FECHA = False


# ---- NUEVO: separar flujos computables vs interactivos ----
def get_computable_flows() -> List[str]:
    """
    Flujos que NO requieren payload externo (se pueden correr en batch).
    """
    base: List[str] = []
    if HAS_PRECIO:
        base.append("red_precio")
    if HAS_GAP_FECHA:
        base.append("gap_fechas")
    return base

def get_interactive_flows() -> List[str]:
    """
    Flujos que requieren JSON/payload del frontend (no se incluyen en 'all').
    """
    return ["red_contactos"]

def get_available_flows() -> List[str]:
    """
    Para el frontend/UI: lista total visible.
    Nota: 'all' == solo computables.
    """
    return get_computable_flows() + get_interactive_flows() + ["all"]


def _run_one_flow(db: Session, lic_id: int, flow: str, json_override: Optional[dict]) -> dict:
    # Interactivo: requiere JSON
    if flow == "red_contactos":
        if not json_override:
            # En vez de ejecutar y dejar comentario "JSON inválido", falla explícito:
            return {"flow": "red_contactos", "ok": False, "error": "json_required"}
        return {
            "flow": "red_contactos",
            "result": run_red_contactos(db, lic_id, json_override=json_override or {}),
        }

    # Computables:
    if flow == "red_precio" and HAS_PRECIO:
        res = _run_precio_one(db, lic_id)
        return {
            "flow": "red_precio",
            "result": {
                "ok": True,
                "flag_applied": (res.target_cuantia is not None) and (
                    res.target_cuantia < res.stats.lower
                    or res.target_cuantia > res.stats.upper
                    or abs(res.stats.z_mad) >= 2.8
                ),
                "detail": {
                    "method": res.method,
                    "n_comparables": res.n_comparables,
                    "median": res.stats.median,
                    "z_mad": res.stats.z_mad,
                    "lower": res.stats.lower,
                    "upper": res.stats.upper,
                    "neighbors": res.neighbor_ids,
                },
            },
        }

    if flow == "gap_fechas" and HAS_GAP_FECHA:
        return {
            "flow": "gap_fechas",
            "result": _run_gap_fecha_one(db, lic_id, json_override=json_override or {}),
        }

    raise ValueError(f"Flow desconocido o no disponible: {flow}")



def run_flow_for_one(
    db: Session,
    licitacion_id: int,
    flow: str = "all",
    json_override: Optional[dict] = None,
) -> dict:
    if flow == "all":
        flows = get_computable_flows()
    else:
        flows = [flow]

    applied = [_run_one_flow(db, licitacion_id, f, json_override) for f in flows]

    
    db.commit()

    return {"licitacion_id": licitacion_id, "applied": applied}



def run_flow_batch(
    db: Session,
    ksflow: str = "all",
    lic_ids: Optional[List[int]] = None,
    where_clause: Optional[str] = None,
    limit: Optional[int] = None,
    json_override: Optional[dict] = None,
) -> List[dict]:
    # Validaciones según tipo de flujo
    if ksflow == "all":
        # OK (solo computables)
        pass
    elif ksflow in get_computable_flows():
        # OK
        pass
    elif ksflow == "red_contactos":
        # Requiere JSON explícito y lic_ids (para endpoint dedicado)
        if not json_override:
            raise ValueError("red_contactos requiere json_override (payload PersonasPayload).")
        if not lic_ids:
            raise ValueError("red_contactos requiere lic_ids (lista de IDs a evaluar).")
    else:
        raise ValueError(f"Flow desconocido: {ksflow}")

    # Cursor de IDs si no vienen dados (solo para computables)
    if lic_ids is None:
        sql = "SELECT id FROM public.licitacion"
        if where_clause:
            sql += f" WHERE {where_clause}"
        sql += " ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        lic_ids = [r[0] for r in db.execute(text(sql)).fetchall()]

    out = []
    for lid in lic_ids:
        out.append(run_flow_for_one(db, lid, flow=ksflow, json_override=json_override))
    return out
