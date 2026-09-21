from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


GENDERS = ["Male", "Female", "Other"]
CONTRACT_TYPES = ["Monthly", "Yearly", "Quarterly"]
TARGET_AGE_MEAN = 42.3
TARGET_AGE_STD = 15.2


def generate_synthetic_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    age = np.clip(np.round(rng.normal(TARGET_AGE_MEAN, TARGET_AGE_STD, size=n_rows)), 0, 120).astype(int)
    spend = rng.gamma(shape=2.5, scale=50.0, size=n_rows)
    contracts = rng.choice(CONTRACT_TYPES, size=n_rows, p=[0.55, 0.3, 0.15])
    support_tickets = rng.poisson(lam=2.2, size=n_rows)
    satisfaction = np.clip(np.round(rng.normal(6.5, 1.8, size=n_rows)), 0, 10).astype(int)
    tenure = rng.integers(0, 120, size=n_rows)
    last_login = rng.integers(0, 365, size=n_rows)
    season = rng.choice(["Q1", "Q2", "Q3", "Q4"], size=n_rows)

    churn_linear = (
        0.7 * support_tickets
        - 0.4 * satisfaction
        + 0.3 * (contracts == "Monthly").astype(float)
        + 0.01 * last_login
        - 0.005 * tenure
    )
    churn_prob = 1.0 / (1.0 + np.exp(-(-1.2 + 0.12 * churn_linear)))
    season_factor = np.where(season == "Q4", 1.5, np.where(season == "Q1", 1.0, 1.2))
    churn_prob = np.clip(churn_prob * season_factor, 0, 1)

    frame = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_rows + 1),
            "age": age,
            "gender": rng.choice(GENDERS, size=n_rows),
            "tenure_months": tenure,
            "monthly_spend": np.round(spend, 2),
            "contract_type": contracts,
            "support_tickets": support_tickets,
            "last_login_days": last_login,
            "satisfaction_score": satisfaction,
            "churn": (rng.random(n_rows) < churn_prob).astype(int),
            "_name": [fake.name() for _ in range(n_rows)],
            "_season": season,
        }
    )
    return frame.drop(columns=["_name", "_season"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/raw/fictitious_company_data.csv")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_data(n_rows=args.rows, seed=args.seed).to_csv(out, index=False)


if __name__ == "__main__":
    main()
