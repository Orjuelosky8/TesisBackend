#!/bin/sh
set -eu

echo "[init] Esperando Postgres…"
until pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; do
  sleep 1
done

# Por si la BD no existe
createdb -U "$POSTGRES_USER" "$POSTGRES_DB" 2>/dev/null || true

if [ -f /docker-entrypoint-initdb.d/backup.dump ]; then
  echo "[init] Encontrado backup.dump, restaurando con pg_restore…"
  pg_restore \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    --clean --if-exists \
    --no-owner --no-privileges \
    -j 4 \
    /docker-entrypoint-initdb.d/backup.dump
  echo "[init] Restauración finalizada."
else
  echo "[init][WARN] No se encontró /docker-entrypoint-initdb.d/backup.dump"
fi

echo "[init] Habilitando extensión vector…"
psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "[init] Listo ✅"
