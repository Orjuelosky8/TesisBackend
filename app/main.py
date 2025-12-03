from __future__ import annotations

from datetime import date
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

from db.deps import get_db
from db import repo
from db.schema import Licitacion, Flags, FlagsLicitaciones, FlagsLog, LicitacionChunk, LicitacionKeymap
from pipes.pipeline import get_available_flows, run_flow_for_one, run_flow_batch

from ai_router import router as ai_router

api = FastAPI(title="Licita API", version="1.0.0")


api.include_router(ai_router)


@api.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@api.get("/")
def index():
    return {
        "name": "Licita API",
        "endpoints": [
            "/health",
            "/licitaciones/search",
            "/licitaciones",
            "/flags/{licitacion_id}",
            "/pipelines/flows",
            "/pipelines/run/{licitacion_id}",
            "/pipelines/batch",
            "/pipes/red-contactos/run",
            "/pipes/red-contactos/run-v2",
            "/pipes/flags/{flag_code}/run/{licitacion_id}",
            "/ai/query",
            "/ai/graphs/assistant",
        ],
    }


# --------- Schemas ----------

class LicitacionIn(BaseModel):
    entidad: str
    objeto: Optional[str] = None
    cuantia: Optional[float] = None
    modalidad: Optional[str] = None
    numero: Optional[str] = None
    estado: Optional[str] = None
    fecha_public: Optional[date] = None
    ubicacion: Optional[str] = None
    act_econ: Optional[str] = None
    enlace: Optional[str] = None
    portal_origen: Optional[str] = None
    texto_indexado: Optional[str] = None


class FlagSetIn(BaseModel):
    flag_codigo: str = Field(..., examples=["red1"])
    valor: bool
    comentario: Optional[str] = None
    fuente: Optional[str] = "manual"


# --------- Rutas básicas ----------

@api.post("/licitaciones", response_model=dict)
def create(lic_in: LicitacionIn, db: Session = Depends(get_db)):
    lic = repo.create_licitacion(db, **lic_in.model_dump())
    db.commit()
    return {"id": lic.id}


@api.get("/licitaciones/search", response_model=List[dict])
def search(q: str, limit: int = 50, db: Session = Depends(get_db)):
    rows = repo.search_licitaciones(db, q=q, limit=limit)
    return [
        {
            "id": x.id,
            "entidad": x.entidad,
            "estado": x.estado,
            "fecha_public": x.fecha_public,
            "cuantia": float(x.cuantia) if x.cuantia is not None else None,
        }
        for x in rows
    ]


@api.post("/flags/{licitacion_id}", response_model=dict)
def set_flag(licitacion_id: int, body: FlagSetIn, db: Session = Depends(get_db)):
    lic: Licitacion | None = db.get(Licitacion, licitacion_id)
    if not lic:
        raise HTTPException(status_code=404, detail="Licitación no encontrada")

    fli = repo.set_flag_for_licitacion(
        session=db,
        licitacion_id=licitacion_id,
        flag_codigo=body.flag_codigo,
        valor=body.valor,
        comentario=body.comentario,
        fuente=body.fuente,
        usuario_log="api",
    )
    db.commit()
    return {"flags_licitaciones_id": fli.id, "ok": True}


# --------- Orquestador ----------

class BatchRequest(BaseModel):
    flow: str = "all"
    where: Optional[str] = None
    limit: Optional[int] = None


@api.get("/pipelines/flows", response_model=List[str])
def list_flows():
    return get_available_flows()


@api.post("/pipelines/run/{licitacion_id}", response_model=dict)
def run_pipeline_one(
    licitacion_id: int,
    flow: str = Query(default="all"),
    db: Session = Depends(get_db),
):
    try:
        return run_flow_for_one(db, licitacion_id, flow=flow)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@api.post("/pipelines/batch", response_model=List[dict])
def run_pipeline_batch_ep(
    payload: BatchRequest = Body(...),
    db: Session = Depends(get_db),
):
    try:
        return run_flow_batch(
            db,
            ksflow=payload.flow,
            where_clause=payload.where,
            limit=payload.limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------- Red de contactos (JSON in-memory) ----------

class PersonasPayload(BaseModel):
    personas: List[Dict[str, Any]]
    contratistas: Optional[List[str]] = None


class RunRedContactosRequest(BaseModel):
    licitacion_ids: List[int]
    data: PersonasPayload


@api.post("/pipes/red-contactos/run", response_model=List[dict])
def run_red_contactos_endpoint(
    payload: RunRedContactosRequest,
    db: Session = Depends(get_db),
):
    return run_flow_batch(
        db,
        ksflow="red_contactos",
        lic_ids=payload.licitacion_ids,
        json_override=payload.data.model_dump(),
    )


class OneFlagRequest(BaseModel):
    json_override: Optional[Dict[str, Any]] = None


@api.post("/pipes/flags/{flag_code}/run/{licitacion_id}", response_model=dict)
def run_one_flag_endpoint(
    flag_code: str,
    licitacion_id: int,
    payload: OneFlagRequest = Body(default=OneFlagRequest()),
    db: Session = Depends(get_db),
):
    aliases = {
        "red_precio": "red_precio",
        "gap_fechas": "gap_fechas",
        "red_contactos": "red_contactos",
        "hora_1159": "hora_1159", 
    }
    flow = aliases.get(flag_code)
    if not flow:
        raise HTTPException(status_code=400, detail=f"Flag desconocido: {flag_code}")

    try:
        result = run_flow_for_one(
            db,
            licitacion_id,
            flow=flow,
            json_override=payload.json_override or {},
        )
        return {"ok": True, "flow": flow, "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------- Red de contactos V2 (aprobadores + personas) ----------

class Trabajo(BaseModel):
    cargo: str
    entidad: str
    anio_inicio: int
    anio_fin: int
    descripcion: Optional[str] = None


class Conexion(BaseModel):
    con_id: Optional[str] = None
    con_nombre: Optional[str] = None
    tipo: Optional[str] = None
    fuente: Optional[str] = None


class PersonaV2(BaseModel):
    id: str
    nombre: str
    ent_publica: bool
    entidad: str
    es_contratista: bool = False
    trabajos: List[Trabajo] = []
    conexiones: List[Conexion] = []


class Aprobador(BaseModel):
    licitacion_id: int
    nombre: str
    rol: str
    cargo: Optional[str] = None
    entidad: Optional[str] = None
    tipo_actor: Optional[str] = "publico"  # publico/privado
    identificacion: Optional[str] = None
    correo: Optional[str] = None


class PersonasPayloadV2(BaseModel):
    aprobadores: List[Aprobador]
    personas: List[PersonaV2]


class RunRedContactosV2(BaseModel):
    licitacion_id: int
    data: PersonasPayloadV2


@api.post("/pipes/red-contactos/run-v2", response_model=dict)
def run_red_contactos_v2(
    payload: RunRedContactosV2,
    db: Session = Depends(get_db),
):
   
    return run_flow_for_one(
        db,
        payload.licitacion_id,
        flow="red_contactos",
        json_override=payload.data.model_dump(),  # <- aprobadores + personas
    )
