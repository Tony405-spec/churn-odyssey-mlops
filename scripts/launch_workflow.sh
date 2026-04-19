#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-local}"

if [[ "$MODE" == "local" ]]; then
  python scripts/preprocess.py
  python scripts/feature_engineering.py
  python scripts/train.py
  python scripts/evaluate.py
  python scripts/explain.py
elif [[ "$MODE" == "docker" ]]; then
  docker compose up --build
elif [[ "$MODE" == "sagemaker" ]]; then
  python -m pip install sagemaker
  python - <<'PY'
from sagemaker.sklearn.estimator import SKLearn

estimator = SKLearn(
    entry_point="scripts/train.py",
    role="SageMakerExecutionRole",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    framework_version="1.2-1",
    py_version="py3",
)
estimator.fit()
print("SageMaker training job submitted")
PY
else
  echo "Unknown mode: $MODE"
  exit 1
fi
