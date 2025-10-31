CREATE MATERIALIZED VIEW banco_flagueado AS
SELECT 
    l.id AS licitacion_id,
    l.entidad,
    l.objeto,
    f.codigo AS flag_codigo,
    fl.valor,
    fl.fecha_detectado,
    fl.comentario
FROM licitacion l
JOIN flags_licitaciones fl ON l.id = fl.licitacion_id
JOIN flags f ON fl.flag_id = f.id
WHERE fl.valor = TRUE;
