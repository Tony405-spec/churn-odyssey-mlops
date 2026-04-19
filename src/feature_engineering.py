from __future__ import annotations

import numpy as np
import pandas as pd

from churn_odyssey.features import generate_synthetic_features


DERIVED_FEATURE_COLUMNS = [
    "spend_per_ticket",
    "tenure_log",
    "spend_log",
    "engagement_score",
    "seasonal_churn_risk",
    "ticket_rate",
    "spend_tenure_ratio",
    "login_recency_inverse",
    "satisfaction_inverse",
    "is_long_tenure",
    "is_high_spender",
    "is_ticket_heavy",
    "spend_x_satisfaction",
    "tenure_x_login",
    "ticket_x_login",
    "contract_monthly_flag",
    "contract_yearly_flag",
    "age_tenure_ratio",
    "spend_age_ratio",
    "digital_decay",
    "service_friction",
    "stability_index",
    "engagement_x_spend",
    "ticket_burden_ratio",
    "tenure_spend_stability",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    out = generate_synthetic_features(df)
    out["engagement_x_spend"] = out["engagement_score"] * np.log1p(out["monthly_spend"])
    out["ticket_burden_ratio"] = out["support_tickets"] / (out["satisfaction_score"] + 1)
    out["tenure_spend_stability"] = out["tenure_months"] / (np.sqrt(out["monthly_spend"]) + 1)
    return out


def feature_count() -> int:
    return len(DERIVED_FEATURE_COLUMNS)
