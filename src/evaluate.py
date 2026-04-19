from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, f1_score, roc_auc_score


def evaluate(model_path: str, test_path: str) -> dict[str, float]:
    model = joblib.load(model_path)
    df = pd.read_parquet(test_path)
    y = df["churn"]
    X = pd.get_dummies(df.drop(columns=["churn"]), drop_first=False)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y, probs).figure_.savefig(reports / "roc_curve.png", dpi=200)
    PrecisionRecallDisplay.from_predictions(y, probs).figure_.savefig(reports / "pr_curve.png", dpi=200)
    ConfusionMatrixDisplay.from_predictions(y, preds).figure_.savefig(reports / "confusion_matrix.png", dpi=200)
    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.tight_layout()
    plt.savefig(reports / "calibration_curve.png", dpi=200)
    plt.close()

    return {"roc_auc": float(roc_auc_score(y, probs)), "f1": float(f1_score(y, preds, zero_division=0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--input", default="data/imputed/test_imputed.parquet")
    args = parser.parse_args()
    metrics = evaluate(args.ensemble, args.input)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    Path("metrics/final_test_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
