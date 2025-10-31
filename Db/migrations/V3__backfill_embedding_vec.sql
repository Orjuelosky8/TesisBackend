DO $$
BEGIN
  IF to_regclass('public.licitacion') IS NOT NULL THEN
    IF EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema = 'public'
        AND table_name   = 'licitacion'
        AND column_name  = 'embedding_vec'
    )
    AND EXISTS (
      SELECT 1
      FROM information_schema.columns
      WHERE table_schema = 'public'
        AND table_name   = 'licitacion'
        AND column_name  = 'embedding'   -- <- columna fuente tipo text/json-ish
    ) THEN
      -- Intenta parsear embeddings estilo JSON/() a JSON y castear a vector
      UPDATE public.licitacion
      SET embedding_vec = (
        regexp_replace(embedding, '^\s*\((.*)\)\s*$', '[\1]')
      )::vector
      WHERE embedding IS NOT NULL
        AND embedding_vec IS NULL
        AND embedding ~ '^\s*[\[\(].*[\]\)]\s*$';

      ANALYZE public.licitacion;
    ELSE
      RAISE NOTICE 'Faltan columnas: embedding_vec y/o embedding en public.licitacion; se omite backfill';
    END IF;
  ELSE
    RAISE NOTICE 'Tabla public.licitacion no existe; se omite V3__backfill_embedding_vec.sql';
  END IF;
END $$;
