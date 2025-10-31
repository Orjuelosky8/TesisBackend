# app/db/schema.py
from __future__ import annotations

import os
from datetime import date, datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    BigInteger,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# --- Base declarativa (¡no SessionLocal!)
class Base(DeclarativeBase):
    pass

# Tamaño del vector según entorno (default 1024 para tu caso actual)
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "1024"))


# ===================== PUBLIC =====================

class Licitacion(Base):
    __tablename__ = "licitacion"
    __table_args__ = (
        Index("ix_licitacion_entidad", "entidad"),
        Index("ix_licitacion_estado", "estado"),
        Index("ix_licitacion_fecha_public", "fecha_public"),
        {"schema": "public"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True)
    entidad: Mapped[str] = mapped_column(String(255), nullable=False)
    objeto: Mapped[Optional[str]] = mapped_column(Text)
    cuantia: Mapped[Optional[float]] = mapped_column(Numeric(18, 2))
    modalidad: Mapped[Optional[str]] = mapped_column(String(255))
    numero: Mapped[Optional[str]] = mapped_column(String(255))
    estado: Mapped[Optional[str]] = mapped_column(String(100))
    fecha_public: Mapped[Optional[date]] = mapped_column(Date)
    ubicacion: Mapped[Optional[str]] = mapped_column(String(255))
    act_econ: Mapped[Optional[str]] = mapped_column(String(255))
    enlace: Mapped[Optional[str]] = mapped_column(Text)
    portal_origen: Mapped[Optional[str]] = mapped_column(String(255))
    texto_indexado: Mapped[Optional[str]] = mapped_column(Text)

    # Relaciones
    flags_detalle: Mapped[List["FlagsLicitaciones"]] = relationship(
        back_populates="licitacion",
        cascade="save-update, merge",
        passive_deletes=True,
    )


class Flags(Base):
    __tablename__ = "flags"
    __table_args__ = (
        UniqueConstraint("codigo", name="uq_flags_codigo"),
        Index("ix_flags_codigo", "codigo"),
        {"schema": "public"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True)
    codigo: Mapped[Optional[str]] = mapped_column(String(10))
    nombre: Mapped[Optional[str]] = mapped_column(String(255))
    descripcion: Mapped[Optional[str]] = mapped_column(Text)

    licitaciones: Mapped[List["FlagsLicitaciones"]] = relationship(
        back_populates="flag",
        cascade="save-update, merge",
        passive_deletes=True,
    )


class FlagsLicitaciones(Base):
    __tablename__ = "flags_licitaciones"
    __table_args__ = (
        UniqueConstraint("licitacion_id", "flag_id", name="uq_flags_licitacion_flag"),
        Index("ix_flags_licitaciones_lid_fid", "licitacion_id", "flag_id"),
        CheckConstraint("fecha_detectado IS NOT NULL", name="ck_fecha_detectado_not_null"),
        {"schema": "public"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True)
    licitacion_id: Mapped[int] = mapped_column(
        ForeignKey("public.licitacion.id", ondelete="NO ACTION", onupdate="NO ACTION"),
        nullable=False,
    )
    flag_id: Mapped[int] = mapped_column(
        ForeignKey("public.flags.id", ondelete="NO ACTION", onupdate="NO ACTION"),
        nullable=False,
    )
    valor: Mapped[bool] = mapped_column(Boolean, default=False)
    fecha_detectado: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), nullable=False
    )
    comentario: Mapped[Optional[str]] = mapped_column(Text)
    fuente: Mapped[Optional[str]] = mapped_column(Text)

    licitacion: Mapped["Licitacion"] = relationship(back_populates="flags_detalle")
    flag: Mapped["Flags"] = relationship(back_populates="licitaciones")
    logs: Mapped[List["FlagsLog"]] = relationship(
        back_populates="flag_licitacion",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class FlagsLog(Base):
    __tablename__ = "flags_log"
    __table_args__ = (
        Index("ix_flags_log_flags_licitaciones_id", "flags_licitaciones_id"),
        Index("ix_flags_log_fecha", "fecha"),
        {"schema": "public"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True)
    flags_licitaciones_id: Mapped[int] = mapped_column(
        ForeignKey("public.flags_licitaciones.id", ondelete="NO ACTION", onupdate="NO ACTION"),
        nullable=False,
    )
    cambio: Mapped[Optional[str]] = mapped_column(Text)
    fecha: Mapped[datetime] = mapped_column(DateTime(timezone=False))
    usuario: Mapped[Optional[str]] = mapped_column(Text)

    flag_licitacion: Mapped["FlagsLicitaciones"] = relationship(back_populates="logs")


class LicitacionChunk(Base):
    __tablename__ = "licitacion_chunk"
    __table_args__ = (
        UniqueConstraint("licitacion_id", "chunk_idx", name="uq_lic_chunk_lid_idx"),
        Index("ix_lic_chunk_licid_idx", "licitacion_id", "chunk_idx"),
        {"schema": "public"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    licitacion_id: Mapped[int] = mapped_column(
        ForeignKey("public.licitacion.id", ondelete="NO ACTION", onupdate="NO ACTION"),
        nullable=False,
    )
    chunk_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[Optional[str]] = mapped_column(Text)
    # En DB está como vector(1024) (ajústalo con EMBED_DIMS si cambiaste)
    embedding_vec: Mapped[Optional[List[float]]] = mapped_column(Vector(EMBED_DIMS))


class LicitacionKeymap(Base):
    __tablename__ = "licitacion_keymap"
    __table_args__ = (
        UniqueConstraint("lic_ext_id", name="uq_licitacion_keymap_lic_ext_id"),
        {"schema": "public"},
    )

    licitacion_id: Mapped[int] = mapped_column(
        ForeignKey("public.licitacion.id", ondelete="CASCADE", onupdate="CASCADE"),
        primary_key=True,
    )
    lic_ext_id: Mapped[Optional[str]] = mapped_column(Text, unique=True)
