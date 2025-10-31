-- V20251029__gap_fecha_triggers.sql
-- Función que mete trabajo a la cola al tocar calendario_norm
CREATE OR REPLACE FUNCTION staging.trg_gap_fecha_on_calnorm_fn()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE v_lic_id INT;
BEGIN
  -- Si faltan fechas, no hacemos nada
  IF NEW.aceptacion_ofertas_ts IS NULL OR NEW.apertura_ofertas_ts IS NULL THEN
    RETURN NEW;
  END IF;

  -- Resuelve licitacion_id via keymap
  SELECT k.licitacion_id INTO v_lic_id
  FROM public.licitacion_keymap k
  WHERE k.lic_ext_id = NEW.archivo
  LIMIT 1;

  -- Si hay match, encola trabajo
  IF v_lic_id IS NOT NULL THEN
    INSERT INTO jobs.flag_gap_fecha_queue (licitacion_id, archivo)
    VALUES (v_lic_id, NEW.archivo)
    ON CONFLICT DO NOTHING;
  END IF;

  RETURN NEW;
END$$;

-- Crea el trigger SOLO si la tabla existe
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema='staging' AND table_name='secop_calendario_norm'
  ) THEN
    EXECUTE $TG$
      DROP TRIGGER IF EXISTS trg_gap_fecha_on_calnorm ON staging.secop_calendario_norm;
      CREATE TRIGGER trg_gap_fecha_on_calnorm
      AFTER INSERT OR UPDATE OF aceptacion_ofertas_ts, apertura_ofertas_ts
      ON staging.secop_calendario_norm
      FOR EACH ROW
      EXECUTE FUNCTION staging.trg_gap_fecha_on_calnorm_fn();
    $TG$;
  END IF;
END$$;
