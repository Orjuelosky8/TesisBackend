

-Para inicializar :
*Correr flag tiempo:
docker compose exec app python /app/scripts/run_pipeline_batch.py --flow gap_fechas --holidays-file /app/scripts/holidays_co.txt
*Correr flag precio:
docker compose exec app python /app/scripts/run_pipeline_batch.py --flow red_precio --batch 200

*Correr flag redContactos:
docker compose exec app python /app/scripts/run_pipeline_batch.py `
  --flow red_contactos_data `
  --ids 10,102,1000 `
  --json-file /app/scripts/red_contactos.json



-CORRER POR BATCH:
docker compose exec app python /app/scripts/run_pipeline_batch.py --flow gap_fechas --holidays-file /app/scripts/holidays_co.txt --limit 200


