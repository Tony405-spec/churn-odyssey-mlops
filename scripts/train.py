import json
from pathlib import Path

import pandas as pd

from churn_odyssey.pipeline import run_pipeline


def main():
    result = run_pipeline("data/processed/features.csv", "artifacts")
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/train_summary.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
