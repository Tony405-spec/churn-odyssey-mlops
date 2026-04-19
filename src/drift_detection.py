from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    eps = 1e-6
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(reference, quantiles))
    if len(breakpoints) < 3:
        breakpoints = np.array([reference.min() - eps, reference.mean(), reference.max() + eps])
    ref_hist, _ = np.histogram(reference, bins=breakpoints)
    cur_hist, _ = np.histogram(current, bins=breakpoints)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), eps, 1)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), eps, 1)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass
class DriftDetector:
    psi_history: dict[str, deque]
    accuracy_history: deque

    def __init__(self) -> None:
        self.psi_history = {}
        self.accuracy_history = deque(maxlen=120)

    def detect_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
    ) -> dict:
        metrics: dict[str, dict[str, float | bool | str]] = {}
        severe_features = 0
        for col in numeric_features:
            ks_stat, p_val = ks_2samp(reference_df[col], current_df[col])
            psi = _psi(reference_df[col], current_df[col])
            level = "none" if psi < 0.1 else "moderate" if psi < 0.2 else "severe"
            severe_features += int(level == "severe")
            history = self.psi_history.setdefault(col, deque(maxlen=3))
            history.append(psi)
            metrics[col] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(p_val),
                "ks_drift": bool(p_val < 0.05),
                "psi": float(psi),
                "psi_level": level,
                "adwin": bool(abs(reference_df[col].mean() - current_df[col].mean()) > 3 * (reference_df[col].std() + 1e-6)),
                "ddm": bool(abs((current_df[col] > reference_df[col].median()).mean() - (reference_df[col] > reference_df[col].median()).mean()) > 0.2),
                "page_hinkley": bool((current_df[col].mean() - reference_df[col].mean()) > (reference_df[col].std() + 1e-6)),
            }
        for col in categorical_features:
            ref_counts = reference_df[col].value_counts().sort_index()
            cur_counts = current_df[col].value_counts().reindex(ref_counts.index, fill_value=0)
            _, p_val = chisquare(f_obs=cur_counts.values + 1e-6, f_exp=ref_counts.values + 1e-6)
            metrics[col] = {"chi2_pvalue": float(p_val), "chi2_drift": bool(p_val < 0.05)}

        return {"metrics": metrics, "severe_feature_count": severe_features}

    def should_retrain(self, drift_report: dict, latest_accuracy: float | None = None) -> bool:
        if latest_accuracy is not None:
            self.accuracy_history.append(latest_accuracy)
        severe_psi_consecutive = any(
            len(history) == 3 and all(v > 0.2 for v in history) for history in self.psi_history.values()
        )
        low_accuracy = len(self.accuracy_history) >= 2 and all(v < 0.7 for v in list(self.accuracy_history)[-2:])
        multi_feature_severe = drift_report.get("severe_feature_count", 0) >= 3
        return bool(severe_psi_consecutive or low_accuracy or multi_feature_severe)


def run_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    ks_pvalue_threshold: float = 0.05,
    psi_threshold: float = 0.2,
) -> dict:
    detector = DriftDetector()
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(reference_df[c])]
    categorical_cols = [c for c in columns if c not in numeric_cols]
    report = detector.detect_drift(reference_df, current_df, numeric_cols, categorical_cols)
    report["retrain_flag"] = detector.should_retrain(report)
    report["thresholds"] = {"ks_pvalue_threshold": ks_pvalue_threshold, "psi_threshold": psi_threshold}
    report["timestamp"] = datetime.utcnow().isoformat()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--reference", type=str, default="data/imputed/train_imputed.parquet")
    parser.add_argument("--current", type=str, default="data/imputed/test_imputed.parquet")
    args = parser.parse_args()

    ref = pd.read_parquet(args.reference)
    cur = pd.read_parquet(args.current)
    cols = [c for c in ref.columns if c in cur.columns and c != "churn"]
    report = run_drift_detection(ref, cur, cols)

    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("logs/drift_metrics.db").write_text(json.dumps(report))
    Path("reports/drift_report.html").write_text(f"<html><body><pre>{json.dumps(report, indent=2)}</pre></body></html>")
    print(json.dumps({"retrain_flag": report["retrain_flag"]}))


if __name__ == "__main__":
    main()
