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
  texto_indexado TEXT
);

CREATE TABLE IF NOT EXISTS public.licitacion_keymap (
  licitacion_id INT  PRIMARY KEY REFERENCES public.licitacion(id),
  lic_ext_id    TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS public.licitacion_chunk (
  id             BIGSERIAL PRIMARY KEY,
  licitacion_id  INT NOT NULL REFERENCES public.licitacion(id),
  chunk_idx      INT NOT NULL,
  chunk_text     TEXT,
  embedding      TEXT,
  embedding_vec  VECTOR(1024),
  UNIQUE (licitacion_id, chunk_idx)
);

CREATE TABLE IF NOT EXISTS public.flags (
  id          SERIAL PRIMARY KEY,
  codigo      VARCHAR(10) UNIQUE,
  nombre      VARCHAR(255),
  descripcion TEXT
);

CREATE TABLE IF NOT EXISTS public.flags_licitaciones (
  id               SERIAL PRIMARY KEY,
  licitacion_id    INT REFERENCES public.licitacion(id),
  flag_id          INT REFERENCES public.flags(id),
  valor            BOOLEAN,
  fecha_detectado  TIMESTAMP,
  comentario       TEXT,
  fuente           TEXT,
  CONSTRAINT uq_flags_licitacion_flag UNIQUE (licitacion_id, flag_id)
);

CREATE TABLE IF NOT EXISTS public.flags_log (
  id                     SERIAL PRIMARY KEY,
  flags_licitaciones_id  INT REFERENCES public.flags_licitaciones(id),
  cambio                 TEXT,
  fecha                  TIMESTAMP,
  usuario                TEXT
);
