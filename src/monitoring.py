from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.drift_detection import run_drift_detection


@dataclass
class MonitoringSummary:
    sample_count: int
    drift_detected: bool
    monitored_columns: list[str]



def monitor_batch(reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list[str]) -> dict:
    drift_report = run_drift_detection(reference_df, current_df, columns)
    summary = MonitoringSummary(
        sample_count=len(current_df),
        drift_detected=bool(drift_report["retrain_flag"]),
        monitored_columns=columns,
    )
    return {"summary": summary.__dict__, "drift_report": drift_report}
