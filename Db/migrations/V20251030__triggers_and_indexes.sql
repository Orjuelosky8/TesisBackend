-- ============================================================
-- V20251029__triggers_and_indexes.sql
-- Triggers e índices para:
--   - Flag "F-GAP-APERT" (gap de días hábiles entre Aceptación y Apertura)
--   - Auditoría de cambios en flags_licitaciones -> flags_log
--   - Índices de soporte/búsqueda
--   - (Opcional) seed automático de keymap desde licitacion.numero
--
-- Requisitos previos:
--   * V20251028__staging_calendar_tables.sql (crea staging.secop_calendario_* y jobs.* si los usas)
--   * Esquemas: public, staging
-- ============================================================

-- 0) Tabla de festivos (opcional)
CREATE TABLE IF NOT EXISTS public.holidays (
  holiday date PRIMARY KEY
);
COMMENT ON TABLE public.holidays IS 'Días festivos (se excluyen del conteo de días hábiles).';

-- 1) Índices idempotentes
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE schemaname='public' AND indexname='ux_flags_licitaciones_lic_flag'
  ) THEN
    CREATE UNIQUE INDEX ux_flags_licitaciones_lic_flag
      ON public.flags_licitaciones(licitacion_id, flag_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE schemaname='public' AND indexname='idx_licitacion_fecha_public'
  ) THEN
    CREATE INDEX idx_licitacion_fecha_public ON public.licitacion(fecha_public);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE schemaname='public' AND indexname='idx_licitacion_keymap_lic_ext_id'
  ) THEN
    CREATE UNIQUE INDEX idx_licitacion_keymap_lic_ext_id
      ON public.licitacion_keymap(lic_ext_id);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE schemaname='public' AND indexname='idx_licitacion_texto_idx_tsv_es'
  ) THEN
    CREATE INDEX idx_licitacion_texto_idx_tsv_es
      ON public.licitacion USING GIN (to_tsvector('spanish', coalesce(texto_indexado,'')));
  END IF;
END $$;

-- 2) Helper para asegurar flag y devolver id
CREATE OR REPLACE FUNCTION public.ensure_flag(p_code text, p_name text, p_desc text)
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE fid integer;
BEGIN
  SELECT id INTO fid FROM public.flags WHERE codigo = p_code;
  IF fid IS NULL THEN
    SELECT COALESCE(MAX(id),0)+1 INTO fid FROM public.flags;
    INSERT INTO public.flags(id, codigo, nombre, descripcion)
    VALUES (fid, p_code, p_name, p_desc);
  END IF;
  RETURN fid;
END;
$$;

-- 3) Conteo de días hábiles (lun–vie) excluyendo public.holidays
CREATE OR REPLACE FUNCTION public.biz_days(d1 date, d2 date)
RETURNS integer
LANGUAGE plpgsql
STABLE
AS $$
DECLARE s date; e date; cnt integer;
BEGIN
  IF d1 IS NULL OR d2 IS NULL THEN
    RETURN NULL;
  END IF;

  IF d2 < d1 THEN s := d2; e := d1; ELSE s := d1; e := d2; END IF;

  SELECT COUNT(*) INTO cnt
  FROM generate_series(s, e - INTERVAL '1 day', INTERVAL '1 day') AS g(d)
  WHERE EXTRACT(DOW FROM g.d) NOT IN (0,6)  -- 0=domingo, 6=sábado
    AND NOT EXISTS (SELECT 1 FROM public.holidays h WHERE h.holiday = g.d::date);

  RETURN cnt;
END;
$$;
COMMENT ON FUNCTION public.biz_days(date, date) IS
'Cuenta días hábiles (lun–vie) excluyendo public.holidays entre [d1, d2).';

-- 4) Función principal: computar flag F-GAP-APERT
--    Regla: flag = TRUE si gap < p_threshold (por defecto 5 días hábiles)
CREATE OR REPLACE FUNCTION public.compute_gap_fecha(p_licitacion_id integer, p_threshold integer DEFAULT 5)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
  v_archivo text;
  v_acc date;
  v_ap  date;
  v_gap integer;
  v_flag_id integer;
  v_comment text;
BEGIN
  SELECT n.archivo::text,
         n.aceptacion_ofertas_ts::date,
         n.apertura_ofertas_ts::date
  INTO v_archivo, v_acc, v_ap
  FROM public.licitacion_keymap k
  JOIN staging.secop_calendario_norm n
    ON n.archivo::text = k.lic_ext_id
  WHERE k.licitacion_id = p_licitacion_id
  LIMIT 1;

  IF v_archivo IS NULL THEN
    RETURN;  -- sin calendario
  END IF;

  v_gap := public.biz_days(v_acc, v_ap);
  IF v_gap IS NULL THEN
    RETURN;  -- fechas incompletas
  END IF;

  v_flag_id := public.ensure_flag(
    'F-GAP-APERT',
    'Gap aceptación vs apertura (días hábiles)',
    'Diferencia de días hábiles entre la Aceptación de ofertas y la Apertura de Ofertas. Son n días hábiles que duró el proceso; se marca si es < umbral.'
  );

  v_comment := format(
    'Gap: %s días hábiles (Aceptación=%s, Apertura=%s; archivo=%s). Son %s días hábiles que duró el proceso; umbral <%s.',
    v_gap, v_acc, v_ap, v_archivo, v_gap, p_threshold
  );

  INSERT INTO public.flags_licitaciones(licitacion_id, flag_id, valor, fecha_detectado, comentario, fuente)
  VALUES (p_licitacion_id, v_flag_id, (v_gap < p_threshold), now(), v_comment, 'trigger:gap_fechas')
  ON CONFLICT (licitacion_id, flag_id) DO UPDATE
    SET valor = EXCLUDED.valor,
        fecha_detectado = EXCLUDED.fecha_detectado,
        comentario = EXCLUDED.comentario,
        fuente = EXCLUDED.fuente;
END;
$$;

-- 5a) Trigger en keymap → recomputa al insertar/actualizar mapeo
CREATE OR REPLACE FUNCTION public.trg_gap_fecha_on_keymap_fn()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  PERFORM public.compute_gap_fecha(NEW.licitacion_id, 5);
  RETURN NEW;
END;
$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_gap_fecha_on_keymap') THEN
    CREATE TRIGGER trg_gap_fecha_on_keymap
    AFTER INSERT OR UPDATE ON public.licitacion_keymap
    FOR EACH ROW
    EXECUTE FUNCTION public.trg_gap_fecha_on_keymap_fn();
  END IF;
END $$;

-- 5b) Trigger en staging.secop_calendario_norm (solo si existe la tabla)
CREATE OR REPLACE FUNCTION staging.trg_gap_fecha_on_calnorm_fn()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE v_lid integer;
BEGIN
  FOR v_lid IN
    SELECT licitacion_id
    FROM public.licitacion_keymap
    WHERE lic_ext_id = NEW.archivo::text
  LOOP
    PERFORM public.compute_gap_fecha(v_lid, 5);
  END LOOP;
  RETURN NEW;
END;
$$;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema='staging' AND table_name='secop_calendario_norm'
  ) AND NOT EXISTS (
    SELECT 1 FROM pg_trigger WHERE tgname = 'trg_gap_fecha_on_calnorm'
  ) THEN
    CREATE TRIGGER trg_gap_fecha_on_calnorm
    AFTER INSERT OR UPDATE OF aceptacion_ofertas_ts, apertura_ofertas_ts
    ON staging.secop_calendario_norm
    FOR EACH ROW
    EXECUTE FUNCTION staging.trg_gap_fecha_on_calnorm_fn();
  END IF;
END $$;

-- 6) Auditoría flags_licitaciones → flags_log
CREATE OR REPLACE FUNCTION public.trg_flags_audit_fn()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE v_msg text; v_id integer;
BEGIN
  IF TG_OP = 'INSERT' THEN
    v_id := NEW.id;
    v_msg := format('INSERT valor=%s, comentario=%s, fuente=%s',
                    NEW.valor, coalesce(NEW.comentario,''), coalesce(NEW.fuente,''));
    INSERT INTO public.flags_log(flags_licitaciones_id, cambio, fecha, usuario)
    VALUES (v_id, v_msg, now(), current_user);
    RETURN NEW;

  ELSIF TG_OP = 'UPDATE' THEN
    v_id := NEW.id;
    v_msg := format('UPDATE valor: %s -> %s; comentario=%s; fuente=%s',
                    coalesce(OLD.valor::text,'NULL'),
                    coalesce(NEW.valor::text,'NULL'),
                    coalesce(NEW.comentario,''), coalesce(NEW.fuente,''));
    INSERT INTO public.flags_log(flags_licitaciones_id, cambio, fecha, usuario)
    VALUES (v_id, v_msg, now(), current_user);
    RETURN NEW;

  ELSIF TG_OP = 'DELETE' THEN
    v_id := OLD.id;
    v_msg := format('DELETE valor=%s, comentario=%s, fuente=%s',
                    OLD.valor, coalesce(OLD.comentario,''), coalesce(OLD.fuente,''));
    INSERT INTO public.flags_log(flags_licitaciones_id, cambio, fecha, usuario)
    VALUES (v_id, v_msg, now(), current_user);
    RETURN OLD;
  END IF;
  RETURN NULL;
END;
$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_flags_audit') THEN
    CREATE TRIGGER trg_flags_audit
    AFTER INSERT OR UPDATE OR DELETE ON public.flags_licitaciones
    FOR EACH ROW
    EXECUTE FUNCTION public.trg_flags_audit_fn();
  END IF;
END $$;

-- 7) (Opcional) Seed automático de keymap desde licitacion.numero
CREATE OR REPLACE FUNCTION public.trg_seed_keymap_from_num_fn()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE v_num text;
BEGIN
  v_num := NULLIF(btrim(COALESCE(NEW.numero,'')), '');
  IF v_num IS NOT NULL THEN
    INSERT INTO public.licitacion_keymap(licitacion_id, lic_ext_id)
    VALUES (NEW.id, v_num) ON CONFLICT DO NOTHING;
  END IF;
  RETURN NEW;
END;
$$;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trg_seed_keymap_from_num') THEN
    CREATE TRIGGER trg_seed_keymap_from_num
    AFTER INSERT ON public.licitacion
    FOR EACH ROW
    EXECUTE FUNCTION public.trg_seed_keymap_from_num_fn();
  END IF;
END $$;
