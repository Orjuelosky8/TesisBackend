Table licitacion {
  id SERIAL PRIMARY KEY,
  entidad VARCHAR(255) NOT NULL,
  objeto TEXT,
  cuantia NUMERIC(18,2),
  modalidad VARCHAR(255),
  numero VARCHAR(255),
  estado VARCHAR(100),
  fecha_public DATE,
  ubicacion VARCHAR(255),
  act_econ VARCHAR(255),
  enlace TEXT,
  portal_origen VARCHAR(255),

  -- (LangChain/pgvector)
  embedding VECTOR(1536),   -- requiere extensión pgvector
  texto_indexado TEXT       -- texto completo usado para generar el embedding
}

Table flags {
  id SERIAL PRIMARY KEY,
  codigo VARCHAR(10) UNIQUE,    -- ejemplo: red1, red2...
  nombre VARCHAR(255),
  descripcion TEXT
}

Table flags_licitaciones {
  id SERIAL PRIMARY KEY,
  licitacion_id INT REFERENCES licitacion(id) ON DELETE CASCADE,
  flag_id INT REFERENCES flags(id) ON DELETE CASCADE,
  valor BOOLEAN DEFAULT FALSE,
  fecha_detectado TIMESTAMP DEFAULT NOW(),
  comentario TEXT,              -- opcional: razón o fuente del flag
  fuente TEXT                   -- "modelo_llm", "script_py", "analista", etc.
}

--Auditoría
Table flags_log {
  id SERIAL PRIMARY KEY,
  flags_licitaciones_id INT REFERENCES flags_licitaciones(id) ON DELETE CASCADE,
  cambio TEXT,
  fecha TIMESTAMP DEFAULT NOW(),
  usuario TEXT
}
