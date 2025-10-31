CREATE EXTENSION IF NOT EXISTS vector;

-- Asegurar columnas en public.licitacion_chunk
DO $$
BEGIN
  IF to_regclass('public.licitacion_chunk') IS NULL THEN
    CREATE TABLE public.licitacion_chunk (
      id            BIGSERIAL PRIMARY KEY,
      licitacion_id INT NOT NULL REFERENCES public.licitacion(id) ON DELETE CASCADE,
      chunk_idx     INT,
      chunk_text    TEXT,
      embedding     TEXT,
      embedding_vec vector(1536)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS licitacion_chunk_uq ON public.licitacion_chunk(licitacion_id, chunk_idx);
  ELSE
    -- añade columnas si faltan
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='licitacion_chunk' AND column_name='embedding') THEN
      ALTER TABLE public.licitacion_chunk ADD COLUMN embedding TEXT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='licitacion_chunk' AND column_name='embedding_vec') THEN
      ALTER TABLE public.licitacion_chunk ADD COLUMN embedding_vec vector(1536);
    END IF;

    -- clave natural para upserts idempotentes
    DO $do$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname='public' AND tablename='licitacion_chunk' AND indexname='licitacion_chunk_uq'
      ) THEN
        CREATE UNIQUE INDEX licitacion_chunk_uq ON public.licitacion_chunk(licitacion_id, chunk_idx);
      END IF;
    END
    $do$;
  END IF;

  -- índice KNN (IVFFLAT) para búsquedas vectoriales
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname='public' AND tablename='licitacion_chunk' AND indexname='licitacion_chunk_embedding_ivfflat'
  ) THEN
    CREATE INDEX licitacion_chunk_embedding_ivfflat
      ON public.licitacion_chunk USING ivfflat (embedding_vec vector_l2_ops)
      WITH (lists = 100);
  END IF;
END $$;
