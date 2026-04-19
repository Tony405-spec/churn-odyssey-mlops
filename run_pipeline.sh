#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-local}"

case "$MODE" in
  local)
    ./scripts/local_deploy.sh
    ;;
  sagemaker)
    ./scripts/sagemaker_deploy.sh
    ;;
  docker)
    docker compose up --build
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 1
    ;;
esac
