import numpy as np
import pandas as pd


def generate_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    out["spend_per_ticket"] = out["monthly_spend"] / (out["support_tickets"] + 1)
    out["tenure_log"] = np.log1p(out["tenure_months"])
    out["spend_log"] = np.log1p(out["monthly_spend"])
    out["engagement_score"] = (30 - out["last_login_days"]).clip(lower=0) * (out["satisfaction_score"] / 10)
    out["seasonal_churn_risk"] = np.sin((out["tenure_months"] % 12) / 12 * 2 * np.pi)
    out["ticket_rate"] = out["support_tickets"] / (out["tenure_months"] + 1)
    out["spend_tenure_ratio"] = out["monthly_spend"] / (out["tenure_months"] + 1)
    out["login_recency_inverse"] = 1 / (out["last_login_days"] + 1)
    out["satisfaction_inverse"] = 1 / (out["satisfaction_score"] + eps)
    out["is_long_tenure"] = (out["tenure_months"] >= 24).astype(int)
    out["is_high_spender"] = (out["monthly_spend"] >= out["monthly_spend"].median()).astype(int)
    out["is_ticket_heavy"] = (out["support_tickets"] >= out["support_tickets"].median()).astype(int)
    out["spend_x_satisfaction"] = out["monthly_spend"] * out["satisfaction_score"]
    out["tenure_x_login"] = out["tenure_months"] * out["last_login_days"]
    out["ticket_x_login"] = out["support_tickets"] * out["last_login_days"]
    out["contract_monthly_flag"] = (out["contract_type"] == "Monthly").astype(int)
    out["contract_yearly_flag"] = (out["contract_type"] == "Yearly").astype(int)
    out["age_tenure_ratio"] = out["age"] / (out["tenure_months"] + 1)
    out["spend_age_ratio"] = out["monthly_spend"] / (out["age"] + 1)
    out["digital_decay"] = np.exp(-out["last_login_days"] / 30)
    out["service_friction"] = out["support_tickets"] * (11 - out["satisfaction_score"])
    out["stability_index"] = out["tenure_months"] / (out["support_tickets"] + 1)

    return out
