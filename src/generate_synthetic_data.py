from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


GENDERS = ["Male", "Female", "Other"]
CONTRACT_TYPES = ["Monthly", "Quarterly", "Yearly"]


def generate_synthetic_data(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_rows + 1),
            "age": rng.integers(18, 80, size=n_rows),
            "gender": rng.choice(GENDERS, size=n_rows),
            "tenure_months": rng.integers(0, 72, size=n_rows),
            "monthly_spend": rng.uniform(10.0, 250.0, size=n_rows).round(2),
            "contract_type": rng.choice(CONTRACT_TYPES, size=n_rows, p=[0.5, 0.2, 0.3]),
            "support_tickets": rng.integers(0, 12, size=n_rows),
            "last_login_days": rng.integers(0, 60, size=n_rows),
            "satisfaction_score": rng.integers(1, 11, size=n_rows),
        }
    )
    churn_logit = (
        0.02 * df["last_login_days"]
        + 0.03 * df["support_tickets"]
        + 0.02 * (10 - df["satisfaction_score"])
        - 0.01 * df["tenure_months"]
    )
    churn_prob = 1 / (1 + np.exp(-(churn_logit - 0.5)))
    df["churn"] = (rng.random(n_rows) < churn_prob).astype(int)
    return df


if __name__ == "__main__":
    out = Path("data/raw/customers.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_data().to_csv(out, index=False)
