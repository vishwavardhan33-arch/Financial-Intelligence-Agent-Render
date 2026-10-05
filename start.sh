#!/bin/sh
# Container entrypoint: wait for the database, create/seed tables, serve.
set -e

attempt=1
until python -m db.init_db; do
  if [ "$attempt" -ge 10 ]; then
    echo "Database still unreachable after $attempt attempts, giving up." >&2
    exit 1
  fi
  echo "Database not ready (attempt $attempt), retrying in 3s..."
  attempt=$((attempt + 1))
  sleep 3
done

# One worker on purpose: the embedding model lives in memory once, and
# free-tier instances only have 512 MB.
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-10000}" --workers 1 --proxy-headers --forwarded-allow-ips "*"
