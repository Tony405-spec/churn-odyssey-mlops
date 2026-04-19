#!/usr/bin/env bash
set -euo pipefail

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
