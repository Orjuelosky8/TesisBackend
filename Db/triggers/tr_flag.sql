CREATE OR REPLACE FUNCTION actualizar_flag_estado()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.valor = TRUE THEN
    INSERT INTO flags_log(flags_licitaciones_id, cambio, usuario)
    VALUES (NEW.id, CONCAT('Flag activado: ', NEW.flag_id), current_user);
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_flag_update
AFTER INSERT OR UPDATE ON flags_licitaciones
FOR EACH ROW
EXECUTE FUNCTION actualizar_flag_estado();
