-- V20251028__staging_calendar_tables.sql
CREATE SCHEMA IF NOT EXISTS staging;

CREATE TABLE IF NOT EXISTS staging.secop_calendario_raw (
  archivo                  TEXT NOT NULL,
  aceptacion_ofertas_raw   TEXT,
  apertura_ofertas_raw     TEXT,
  fecha_publicacion_raw    TEXT,
  presentacion_ofertas_raw TEXT
);

CREATE TABLE IF NOT EXISTS staging.secop_calendario_norm (
  archivo                  TEXT PRIMARY KEY,
  aceptacion_ofertas_ts    TIMESTAMP NULL,
  apertura_ofertas_ts      TIMESTAMP NULL,
  fecha_publicacion_ts     TIMESTAMP NULL,
  presentacion_ofertas_ts  TIMESTAMP NULL,
  fuente                   TEXT,
  created_at               TIMESTAMPTZ DEFAULT now(),
  updated_at               TIMESTAMPTZ DEFAULT now()
);

-- Mantén actualizado updated_at
CREATE OR REPLACE FUNCTION staging.set_updated_at() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END$$;

DROP TRIGGER IF EXISTS trg_set_updated_at ON staging.secop_calendario_norm;
CREATE TRIGGER trg_set_updated_at
BEFORE INSERT OR UPDATE ON staging.secop_calendario_norm
FOR EACH ROW EXECUTE FUNCTION staging.set_updated_at();

-- Cola de trabajo (para que el trigger no “llame” a Python)
CREATE SCHEMA IF NOT EXISTS jobs;
CREATE TABLE IF NOT EXISTS jobs.flag_gap_fecha_queue (
  id             BIGSERIAL PRIMARY KEY,
  licitacion_id  INT NOT NULL,
  archivo        TEXT NOT NULL,
  created_at     TIMESTAMPTZ DEFAULT now(),
  UNIQUE (licitacion_id, archivo)
);
