from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@db:5432/licita_db")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db           # ← aquí ejecutan tus endpoints/flows
        db.commit()        # ← COMMIT si todo fue bien
    except Exception:
        db.rollback()      # ← ROLLBACK si algo falló
        raise
    finally:
        db.close()
