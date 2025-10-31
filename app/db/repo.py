
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import select, update, func
from sqlalchemy.orm import Session

from db.schema import Licitacion, Flags, FlagsLicitaciones, FlagsLog



# ==============
# LICITACIONES
# ==============
def create_licitacion(
    session: Session,
    entidad: str,
    objeto: Optional[str] = None,
    cuantia: Optional[float] = None,
    modalidad: Optional[str] = None,
    numero: Optional[str] = None,
    estado: Optional[str] = None,
    fecha_public=None,
    ubicacion: Optional[str] = None,
    act_econ: Optional[str] = None,
    enlace: Optional[str] = None,
    portal_origen: Optional[str] = None,
    embedding: Optional[str] = None,
    texto_indexado: Optional[str] = None,
) -> Licitacion:
    lic = Licitacion(
        entidad=entidad,
        objeto=objeto,
        cuantia=cuantia,
        modalidad=modalidad,
        numero=numero,
        estado=estado,
        fecha_public=fecha_public,
        ubicacion=ubicacion,
        act_econ=act_econ,
        enlace=enlace,
        portal_origen=portal_origen,
        texto_indexado=texto_indexado,
    )
    session.add(lic)
    session.flush()
    return lic


def search_licitaciones(session: Session, q: str, limit: int = 50) -> Iterable[Licitacion]:
    # Búsqueda simple por entidad/objeto; mejora con pg_trgm si lo habilitas.
    stmt = (
        select(Licitacion)
        .where(
            (Licitacion.entidad.ilike(f"%{q}%"))
            | (Licitacion.objeto.ilike(f"%{q}%"))
            | (Licitacion.texto_indexado.ilike(f"%{q}%"))
        )
        .order_by(func.coalesce(Licitacion.fecha_public, func.current_date()).desc())
        .limit(limit)
    )
    return session.execute(stmt).scalars().all()


# =========
# FLAGS
# =========
def ensure_flag_by_codigo(session: Session, codigo: str, nombre: Optional[str] = None) -> Flags:
    flag = session.execute(select(Flags).where(Flags.codigo == codigo)).scalar_one_or_none()
    if flag is None:
        flag = Flags(codigo=codigo, nombre=nombre or codigo)
        session.add(flag)
        session.flush()
    return flag


def set_flag_for_licitacion(
    session: Session,
    licitacion_id: int,
    flag_codigo: str,
    valor: bool,
    comentario: Optional[str] = None,
    fuente: Optional[str] = None,
    usuario_log: Optional[str] = None,
    fecha: Optional[datetime] = None,
) -> FlagsLicitaciones:
    """Crea/actualiza el flag de una licitación y registra un log."""
    fecha = fecha or datetime.now()
    flag = ensure_flag_by_codigo(session, flag_codigo)

    fli = session.execute(
        select(FlagsLicitaciones)
        .where(FlagsLicitaciones.licitacion_id == licitacion_id, FlagsLicitaciones.flag_id == flag.id)
        .limit(1)
    ).scalar_one_or_none()

    if fli is None:
        fli = FlagsLicitaciones(
            licitacion_id=licitacion_id,
            flag_id=flag.id,
            valor=valor,
            fecha_detectado=fecha,
            comentario=comentario,
            fuente=fuente,
        )
        session.add(fli)
        session.flush()
    else:
        fli.valor = valor
        fli.fecha_detectado = fecha
        fli.comentario = comentario
        fli.fuente = fuente

    # Log
    log = FlagsLog(
        flags_licitaciones_id=fli.id,
        cambio=f"valor={valor}; comentario={comentario or ''}; fuente={fuente or ''}",
        fecha=fecha,
        usuario=usuario_log or "sistema",
    )
    session.add(log)

    return fli


def get_flags_activos_por_licitacion(session: Session, licitacion_id: int):
    stmt = (
        select(Flags.codigo, Flags.nombre, FlagsLicitaciones.valor, FlagsLicitaciones.fecha_detectado)
        .join(FlagsLicitaciones, Flags.id == FlagsLicitaciones.flag_id)
        .where(FlagsLicitaciones.licitacion_id == licitacion_id, FlagsLicitaciones.valor.is_(True))
        .order_by(FlagsLicitaciones.fecha_detectado.desc())
    )
    return session.execute(stmt).all()


# ========================
# BANCO_FLAGUEADO (vista)
# ========================
def banco_flagueado_por_flag(session: Session, flag_codigo: str, limit: int = 100):
    stmt = (
        select(BancoFlagueado)
        .where(BancoFlagueado.flag_codigo == flag_codigo)
        .order_by(BancoFlagueado.fecha_detectado.desc())
        .limit(limit)
    )
    return session.execute(stmt).scalars().all()
