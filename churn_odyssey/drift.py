from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    breaks = np.linspace(0, 1, buckets + 1)
    expected_q = np.unique(np.quantile(expected, breaks))
    if expected_q.size < 2:
        return 0.0
    expected_bins = np.histogram(expected, bins=expected_q)[0] / len(expected)
    actual_bins = np.histogram(actual, bins=expected_q)[0] / len(actual)
    expected_bins = np.clip(expected_bins, 1e-6, None)
    actual_bins = np.clip(actual_bins, 1e-6, None)
    return float(np.sum((actual_bins - expected_bins) * np.log(actual_bins / expected_bins)))


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    ks_pvalue_threshold: float = 0.05,
    psi_threshold: float = 0.2,
) -> dict:
    metrics = {}
    drifted = False
    for col in columns:
        stat = ks_2samp(reference_df[col], current_df[col])
        psi_value = _psi(reference_df[col], current_df[col])
        col_drift = stat.pvalue < ks_pvalue_threshold or psi_value > psi_threshold
        drifted = drifted or col_drift
        metrics[col] = {
            "ks_stat": float(stat.statistic),
            "ks_pvalue": float(stat.pvalue),
            "psi": psi_value,
            "drift": col_drift,
        }
    return {"retrain_flag": drifted, "metrics": metrics}
