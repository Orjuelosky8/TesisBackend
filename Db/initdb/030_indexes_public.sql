-- Índices varios (deja todos los que ya tienes)
CREATE INDEX IF NOT EXISTS ix_licitacion_entidad       ON public.licitacion (entidad);
CREATE INDEX IF NOT EXISTS ix_licitacion_estado        ON public.licitacion (estado);
CREATE INDEX IF NOT EXISTS ix_licitacion_fecha_public  ON public.licitacion (fecha_public);

CREATE INDEX IF NOT EXISTS ix_flags_codigo             ON public.flags (codigo);
CREATE INDEX IF NOT EXISTS ix_flags_licit_lid_fid      ON public.flags_licitaciones (licitacion_id, flag_id);
CREATE INDEX IF NOT EXISTS ix_flags_licit_fecha        ON public.flags_licitaciones (fecha_detectado);
CREATE INDEX IF NOT EXISTS ix_flags_log_ref            ON public.flags_log (flags_licitaciones_id);
CREATE INDEX IF NOT EXISTS ix_flags_log_fecha          ON public.flags_log (fecha);

-- ANN por chunk (pgvector, métrica coseno). NO dejes otro CREATE INDEX duplicado.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    -- (Opcional) si la columna existe pero no tiene dimensión, la forzamos a 1536.
    BEGIN
      ALTER TABLE public.licitacion_chunk
        ALTER COLUMN embedding_vec TYPE vector(1536);
    EXCEPTION WHEN others THEN
      -- Ignorar si ya está en vector(1536) o si la columna aún no existe.
      NULL;
    END;

    IF EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema='public'
        AND table_name='licitacion_chunk'
        AND column_name='embedding_vec'
    ) THEN
      CREATE INDEX IF NOT EXISTS licitacion_chunk_embedding_vec_ivfflat
        ON public.licitacion_chunk
        USING ivfflat (embedding_vec vector_cosine_ops)
        WITH (lists = 100);
    END IF;
  END IF;
END $$;
