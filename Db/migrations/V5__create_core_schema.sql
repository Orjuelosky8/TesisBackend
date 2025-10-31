CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla principal
CREATE TABLE IF NOT EXISTS public.licitacion (
  id             SERIAL PRIMARY KEY,
  entidad        VARCHAR(255) NOT NULL,
  objeto         TEXT,
  cuantia        NUMERIC(18,2),
  modalidad      VARCHAR(255),
  numero         VARCHAR(255),
  estado         VARCHAR(100),
  fecha_public   DATE,
  ubicacion      VARCHAR(255),
  act_econ       VARCHAR(255),
  enlace         TEXT,
  portal_origen  VARCHAR(255),

  -- Para import desde SQLite sin dolor:
  embedding      TEXT,             -- luego convertimos a vector en V6
  texto_indexado TEXT
);

-- Flags
CREATE TABLE IF NOT EXISTS public.flags (
  id          SERIAL PRIMARY KEY,
  codigo      VARCHAR(10) UNIQUE,
  nombre      VARCHAR(255),
  descripcion TEXT
);

-- Relación flags-licitaciones
CREATE TABLE IF NOT EXISTS public.flags_licitaciones (
  id             SERIAL PRIMARY KEY,
  licitacion_id  INT REFERENCES public.licitacion(id) ON DELETE CASCADE,
  flag_id        INT REFERENCES public.flags(id) ON DELETE CASCADE,
  valor          BOOLEAN DEFAULT FALSE,
  fecha_detectado TIMESTAMP DEFAULT NOW(),
  comentario     TEXT,
  fuente         TEXT
);

-- Auditoría
CREATE TABLE IF NOT EXISTS public.flags_log (
  id                    SERIAL PRIMARY KEY,
  flags_licitaciones_id INT REFERENCES public.flags_licitaciones(id) ON DELETE CASCADE,
  cambio                TEXT,
  fecha                 TIMESTAMP DEFAULT NOW(),
  usuario               TEXT
);
