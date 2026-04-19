from __future__ import annotations

import numpy as np
import pandas as pd

from churn_odyssey.drift import detect_drift



def _adwin_proxy(reference: pd.Series, current: pd.Series, threshold: float = 0.15) -> bool:
    return abs(float(reference.mean()) - float(current.mean())) > threshold * (float(reference.std()) + 1e-6)



def _ddm_proxy(reference: pd.Series, current: pd.Series, threshold: float = 0.2) -> bool:
    ref_error = float((reference > reference.median()).mean())
    cur_error = float((current > reference.median()).mean())
    return abs(cur_error - ref_error) > threshold



def run_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str],
    ks_pvalue_threshold: float = 0.05,
    psi_threshold: float = 0.2,
) -> dict:
    base = detect_drift(
        reference_df=reference_df,
        current_df=current_df,
        columns=columns,
        ks_pvalue_threshold=ks_pvalue_threshold,
        psi_threshold=psi_threshold,
    )
    adwin_flags: dict[str, bool] = {}
    ddm_flags: dict[str, bool] = {}
    for col in columns:
        adwin_flags[col] = _adwin_proxy(reference_df[col], current_df[col])
        ddm_flags[col] = _ddm_proxy(reference_df[col], current_df[col])
        base["metrics"][col]["adwin"] = adwin_flags[col]
        base["metrics"][col]["ddm"] = ddm_flags[col]
    base["adwin"] = adwin_flags
    base["ddm"] = ddm_flags
    base["retrain_flag"] = bool(base["retrain_flag"] or any(adwin_flags.values()) or any(ddm_flags.values()))
    return base
