DO $$
BEGIN
  IF to_regclass('public.licitacion') IS NOT NULL THEN
    IF EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema = 'public'
        AND table_name   = 'licitacion'
        AND column_name  = 'embedding_vec'
    ) THEN
      ANALYZE public.licitacion;
      CREATE INDEX IF NOT EXISTS licitacion_embedding_vec_ivfflat
        ON public.licitacion
        USING ivfflat (embedding_vec vector_l2_ops)
        WITH (lists = 100);
    ELSE
      RAISE NOTICE 'No existe public.licitacion.embedding_vec; se omite índice IVFFLAT';
    END IF;
  ELSE
    RAISE NOTICE 'Tabla public.licitacion no existe; se omite V4__ivfflat_index.sql';
  END IF;
END $$;
