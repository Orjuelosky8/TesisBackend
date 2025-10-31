# app/pipes/flag_redcontactos.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import deque, defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import text

from db.schema import Licitacion
from db import repo

# ---------- Constantes del flag ----------
FLAG_CODE = "red_contac"  # <= 10 chars
FLAG_NAME = "Red de contactos"
FLAG_DESC = "Posible conflicto por red de contactos"

# ---------- Modelo interno ----------
@dataclass
class Trabajo:
    cargo: Optional[str] = None
    entidad: Optional[str] = None
    anio_inicio: Optional[int] = None
    anio_fin: Optional[int] = None
    descripcion: Optional[str] = None

@dataclass
class Persona:
    id: str
    nombre: str
    entidad: Optional[str] = None
    ent_publica: Optional[bool] = None
    es_contratista: bool = False
    trabajos: List[Trabajo] = field(default_factory=list)
    # conexiones como lista de (id o nombre, tipo, fuente)
    conexiones: List[Dict] = field(default_factory=list)

def _safe_str(x) -> Optional[str]:
    return (str(x).strip() if x is not None and str(x).strip() != "" else None)

def _norm_name(x: Optional[str]) -> Optional[str]:
    return _safe_str(x.lower()) if x else None

# ---------- Parseo de payload v1/v2 ----------
def _from_v1_people(json_override: dict) -> List[Persona]:
    """Compat: personas[] con posibles claves antiguas y diccionario conexiones {nombre: {id:..}}"""
    people_raw = json_override.get("personas") or []
    res: List[Persona] = []
    for p in people_raw:
        nombre = _safe_str(p.get("Nombre") or p.get("nombre"))
        pid = _safe_str(p.get("id") or nombre or "")
        entidad = _safe_str(p.get("Entidad") or p.get("entidad"))
        ent_publica = p.get("Ent_publica") if "Ent_publica" in p else p.get("ent_publica")
        es_contratista = bool(p.get("es_contratista", False))

        # trabajos antiguos
        at_list = p.get("Anteriores_trabajos") or p.get("anteriores_trabajos") or []
        if isinstance(at_list, dict):
            at_list = [at_list]
        trabajos = []
        for t in (at_list or []):
            trabajos.append(Trabajo(
                cargo=_safe_str(t.get("cargo")),
                entidad=None,
                anio_inicio=None,
                anio_fin=None,
                descripcion=None
            ))

        # conexiones antiguas: dict nombre -> {id: ...}
        conns = []
        conn_map = p.get("conexion_directas") or p.get("conexiones") or {}
        if isinstance(conn_map, dict):
            for nom, meta in conn_map.items():
                conns.append({
                    "con_id": _safe_str(meta.get("id")),
                    "con_nombre": _safe_str(nom),
                    "tipo": None,
                    "fuente": None,
                })

        res.append(Persona(
            id=pid or nombre or "",
            nombre=nombre or pid or "persona_sin_nombre",
            entidad=entidad,
            ent_publica=bool(ent_publica) if ent_publica is not None else None,
            es_contratista=es_contratista,
            trabajos=trabajos,
            conexiones=conns
        ))
    return res

def _from_v2_people(json_override: dict) -> Tuple[List[Persona], List[Dict]]:
    """
    v2:
    {
      "aprobadores": [ { nombre, entidad, tipo_actor, ... } ],
      "personas": [
        { id, nombre, ent_publica, entidad, es_contratista, trabajos:[], conexiones:[] }
      ]
    }
    """
    personas_raw = json_override.get("personas") or []
    res: List[Persona] = []
    for p in personas_raw:
        res.append(Persona(
            id=_safe_str(p.get("id") or p.get("persona_id") or p.get("nombre") or ""),
            nombre=_safe_str(p.get("nombre") or p.get("id") or "persona_sin_nombre"),
            entidad=_safe_str(p.get("entidad")),
            ent_publica=bool(p.get("ent_publica")) if p.get("ent_publica") is not None else None,
            es_contratista=bool(p.get("es_contratista", False)),
            trabajos=[
                Trabajo(
                    cargo=_safe_str(t.get("cargo")),
                    entidad=_safe_str(t.get("entidad")),
                    anio_inicio=int(t.get("anio_inicio")) if t.get("anio_inicio") is not None else None,
                    anio_fin=int(t.get("anio_fin")) if t.get("anio_fin") is not None else None,
                    descripcion=_safe_str(t.get("descripcion")),
                ) for t in (p.get("trabajos") or [])
            ],
            conexiones=[
                {
                    "con_id": _safe_str(c.get("con_id")),
                    "con_nombre": _safe_str(c.get("con_nombre")),
                    "tipo": _safe_str(c.get("tipo")),
                    "fuente": _safe_str(c.get("fuente")),
                } for c in (p.get("conexiones") or [])
            ]
        ))
    aprobadores = json_override.get("aprobadores") or []
    return res, aprobadores

# ---------- Grafo ----------
def build_graph(people: List[Persona]) -> Tuple[Dict[str, Persona], Dict[str, Set[str]]]:
    by_id: Dict[str, Persona] = {p.id: p for p in people}
    # name->id (primera ocurrencia)
    name_to_id = {}
    for p in people:
        if p.nombre:
            name_to_id.setdefault(_norm_name(p.nombre), p.id)
    # adjacency
    adj: Dict[str, Set[str]] = {p.id: set() for p in people}
    for p in people:
        for c in (p.conexiones or []):
            tid = c.get("con_id")
            tnm = c.get("con_nombre")
            target = _safe_str(tid) or name_to_id.get(_norm_name(tnm))
            if target and target != p.id:
                adj.setdefault(p.id, set()).add(target)
                adj.setdefault(target, set()).add(p.id)
    return by_id, adj

def shortest_path(adj: Dict[str, Set[str]], src: str, dst: str, max_depth: int = 2) -> Optional[List[str]]:
    if src == dst or src not in adj or dst not in adj:
        return None
    q = deque([(src, [src])]); seen = {src}
    while q:
        node, path = q.popleft()
        if len(path) - 1 >= max_depth:
            pass
        for nb in adj.get(node, []):
            if nb in seen:
                continue
            newp = path + [nb]
            if nb == dst:
                return newp
            if len(newp) - 1 <= max_depth:
                seen.add(nb)
                q.append((nb, newp))
    return None

# ---------- Selección de actores ----------
def pick_official_ids(lic_entidad: Optional[str], people: List[Persona], aprobadores: List[Dict]) -> List[str]:
    ids: List[str] = []
    lic_ent = _norm_name(lic_entidad)
    # 1) prefer aprobadores v2 -> público y misma entidad
    for ap in (aprobadores or []):
        if _norm_name(ap.get("tipo_actor")) == "publico":
            if not lic_ent or _norm_name(ap.get("entidad")) == lic_ent:
                ids.append(_safe_str(ap.get("nombre")) or "")
    # map nombres a ids presentes
    by_name = { _norm_name(p.nombre): p.id for p in people if p.nombre }
    mapped = [by_name.get(_norm_name(n)) for n in ids if _norm_name(n) in by_name]
    mapped = [x for x in mapped if x]
    # 2) si no hay aprobadores mapeados, cae a funcionarios de la misma entidad
    if not mapped:
        mapped = [
            p.id for p in people
            if (p.ent_publica is True)
            and lic_ent and _norm_name(p.entidad) == lic_ent
        ]
    return list(dict.fromkeys(mapped))

def pick_contractor_ids(people: List[Persona], json_override: dict) -> List[str]:
    first = [p.id for p in people if p.es_contratista]
    by_name = { _norm_name(p.nombre): p.id for p in people if p.nombre }
    extra = []
    for n in (json_override.get("contratistas") or []):
        nid = by_name.get(_norm_name(n))
        if nid:
            extra.append(nid)
    ids = list(dict.fromkeys(first + extra))
    return ids

# ---------- Scoring ----------
REL_WEIGHTS = {
    "familiar": 8,
    "socio": 6,
    "jefe_subalterno": 5,
    "supervision": 4,
    "colegas_previos": 3,
    "academico": 2,
}

def score_path(path: List[str], people: Dict[str, Persona]) -> int:
    base = 30 if len(path) == 2 else 18
    bonus = 0
    return base + bonus

# ---------- Flag principal ----------
def ensure_flag_exists(db: Session, codigo: str, nombre: str, descripcion: str = "") -> None:
    db.execute(text("""
        INSERT INTO public.flags (codigo, nombre, descripcion)
        VALUES (:c, :n, :d)
        ON CONFLICT (codigo) DO UPDATE
        SET nombre = EXCLUDED.nombre,
            descripcion = EXCLUDED.descripcion
    """), {"c": codigo, "n": nombre, "d": descripcion})
    db.commit()

def run_red_contactos(db: Session, licitacion_id: int, json_override: dict) -> dict:
    lic: Licitacion | None = db.get(Licitacion, licitacion_id)
    if not lic:
        return {"ok": False, "detail": "Licitación no existe"}

    if not json_override:
        return {"ok": True, "flag_applied": False, "detail": "JSON vacío"}

    # --- Parseo v2 o v1 ---
    people_v2, aprobadores = _from_v2_people(json_override)
    if people_v2:
        people = people_v2
    else:
        people = _from_v1_people(json_override)
        aprobadores = json_override.get("aprobadores", [])

    if not people:
        return {"ok": True, "flag_applied": False, "detail": "Sin 'personas' válidas"}

    by_id, adj = build_graph(people)
    official_ids = pick_official_ids(lic_entidad=lic.entidad, people=people, aprobadores=aprobadores)
    contractor_ids = pick_contractor_ids(people, json_override)

    matches = []
    best = None
    best_score = -10**9

    for oid in official_ids:
        for cid in contractor_ids:
            path = shortest_path(adj, oid, cid, max_depth=2)
            if not path:
                continue
            s = score_path(path, by_id)
            if s > best_score:
                best_score = s
                best = (oid, cid, path)
            matches.append({
                "oficial_id": oid, "oficial_nombre": by_id.get(oid).nombre if by_id.get(oid) else oid,
                "contratista_id": cid, "contratista_nombre": by_id.get(cid).nombre if by_id.get(cid) else cid,
                "path_ids": path, "path_len": len(path)-1, "score": s
            })

    if best:
        oid, cid, path = best
        oficial = by_id.get(oid)
        contrat = by_id.get(cid)
        saltos = len(path) - 1
        comment = (
            f"[red_contactos] Mejor coincidencia: {oficial.nombre if oficial else oid} ↔ "
            f"{contrat.nombre if contrat else cid} en {saltos} salto(s). "
            f"Total coincidencias: {len(matches)}."
        )
        apply_flag = True
    else:
        degs = [(oid, len(adj.get(oid, []))) for oid in official_ids]
        degs.sort(key=lambda x: x[1], reverse=True)
        top = degs[0] if degs else (None, 0)
        if top[0] and top[1] >= 5:
            p = by_id.get(top[0])
            comment = (
                f"[red_contactos] Sin caminos ≤2 saltos; oficial con conectividad alta: "
                f"{p.nombre if p else top[0]} (grado={top[1]})."
            )
        else:
            comment = "[red_contactos] No hay suficientes evidencias de conexión."
        apply_flag = False

    # 👇 AQUÍ usamos el código corto
    ensure_flag_exists(db, FLAG_CODE, FLAG_NAME, FLAG_DESC)
    repo.set_flag_for_licitacion(
        session=db,
        licitacion_id=licitacion_id,
        flag_codigo=FLAG_CODE,
        valor=bool(apply_flag),
        comentario=comment,
        fuente="pipeline.redcontactos(json-v2)",
        usuario_log="pipeline",
    )

    detail = {
        "lic_entidad": lic.entidad,
        "official_ids": official_ids,
        "contractor_ids": contractor_ids,
        "matches": matches,
    }
    return {"ok": True, "flag_applied": apply_flag, "comment": comment, "detail": detail}
