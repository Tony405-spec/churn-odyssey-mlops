#!/usr/bin/env bash
set -euo pipefail

python scripts/preprocess.py
python scripts/feature_engineering.py
python scripts/train.py
python scripts/evaluate.py
python scripts/explain.py
