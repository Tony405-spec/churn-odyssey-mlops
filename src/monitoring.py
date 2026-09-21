from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from prometheus_client import Gauge, Histogram, Info

from src.drift_detection import run_drift_detection


prediction_latency_histogram = Histogram("prediction_latency_histogram", "Prediction latency in seconds")
drift_score_gauge = Gauge("drift_score_gauge", "Current drift score")
model_version_info = Info("model_version_info", "Model version metadata")
business_kpi_saved = Gauge("business_kpi_saved", "Estimated savings from churn prevention")


@dataclass
class MonitoringSummary:
    sample_count: int
    drift_detected: bool
    monitored_columns: list[str]


def monitor_batch(reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list[str], churn_prevented: int = 0, avg_clv: float = 0.0) -> dict:
    drift_report = run_drift_detection(reference_df, current_df, columns)
    max_drift = max(
        [
            metric.get("psi", 0.0)
            for metric in drift_report.get("metrics", {}).values()
            if isinstance(metric, dict)
        ]
        or [0.0]
    )
    drift_score_gauge.set(max_drift)
    business_kpi_saved.set(churn_prevented * avg_clv)
    model_version_info.info({"version": "v1"})

    summary = MonitoringSummary(
        sample_count=len(current_df),
        drift_detected=bool(drift_report.get("retrain_flag")),
        monitored_columns=columns,
    )
    return {"summary": summary.__dict__, "drift_report": drift_report}
