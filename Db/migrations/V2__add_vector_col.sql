CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
  IF to_regclass('public.licitacion') IS NOT NULL THEN
    IF NOT EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema = 'public'
        AND table_name   = 'licitacion'
        AND column_name  = 'embedding_vec'
    ) THEN
      ALTER TABLE public.licitacion
        ADD COLUMN embedding_vec vector(1536);
    END IF;
  ELSE
    RAISE NOTICE 'Tabla public.licitacion no existe; se omite V2__add_vector_col.sql';
  END IF;
END $$;
