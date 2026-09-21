#!/usr/bin/env bash
set -euo pipefail

docker-compose build --no-cache
docker-compose up -d

echo "Waiting for services..."
sleep 10

curl -sSf http://localhost:8000/health >/dev/null
curl -sS -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data-binary @sample.json || true

docker-compose logs -f
