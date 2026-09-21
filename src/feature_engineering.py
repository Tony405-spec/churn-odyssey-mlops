from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


DERIVED_FEATURE_COLUMNS = [
    "spend_per_ticket",
    "spend_per_login_day",
    "annual_spend",
    "spend_velocity",
    "spend_decile",
    "tenure_log",
    "tenure_squared",
    "tenure_bucket",
    "churn_risk_by_tenure",
    "engagement_score",
    "login_frequency_bin",
    "support_intensity",
    "age_spend_interaction",
    "tenure_satisfaction",
    "support_login_ratio",
    "contract_spend_interaction",
    "risk_composite",
    "days_since_last_login_capped",
    "login_decay",
    "support_decay",
    "recency_score",
    "gender_encoded",
    "contract_risk",
    "tenure_age_cluster",
    "high_value_customer",
]


def _safe_qcut(values: pd.Series, q: int) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=q, labels=False, duplicates="drop")


def _target_encode_contract_type(df: pd.DataFrame, target: pd.Series, m: float = 50.0) -> pd.Series:
    global_mean = float(target.mean())
    grouped = pd.DataFrame({"contract_type": df["contract_type"], "target": target}).groupby("contract_type")["target"]
    stats = grouped.agg(["mean", "count"]).rename(columns={"mean": "contract_mean", "count": "contract_count"})
    smoothing = (stats["contract_mean"] * stats["contract_count"] + global_mean * m) / (stats["contract_count"] + m)
    return df["contract_type"].map(smoothing).fillna(global_mean)


def _top_polynomial_features(df: pd.DataFrame, target: pd.Series, limit: int = 10) -> pd.DataFrame:
    poly_base = df[["age", "tenure_months", "monthly_spend"]]
    transformer = PolynomialFeatures(degree=2, include_bias=False)
    poly_values = transformer.fit_transform(poly_base)
    names = transformer.get_feature_names_out(["age", "tenure_months", "monthly_spend"])
    poly_df = pd.DataFrame(poly_values, columns=names, index=df.index)
    correlations = poly_df.apply(lambda c: abs(c.corr(target)) if c.nunique() > 1 else 0.0)
    selected = correlations.sort_values(ascending=False).head(limit).index
    return poly_df[selected].add_prefix("poly_")


def create_features(df: pd.DataFrame, target: pd.Series | None = None, include_polynomial: bool = True) -> pd.DataFrame:
    out = df.copy()
    eps = 0.01

    out["spend_per_ticket"] = out["monthly_spend"] / (out["support_tickets"] + eps)
    out["spend_per_login_day"] = out["monthly_spend"] / (out["last_login_days"] + eps)
    out["annual_spend"] = out["monthly_spend"] * 12
    out["spend_velocity"] = out["monthly_spend"] / (out["tenure_months"] + eps)
    out["spend_decile"] = _safe_qcut(out["monthly_spend"], 10).astype(float).fillna(0).astype(int)

    out["tenure_log"] = np.log1p(out["tenure_months"])
    out["tenure_squared"] = out["tenure_months"] ** 2
    out["tenure_bucket"] = pd.cut(
        out["tenure_months"],
        bins=[0, 3, 6, 12, 24, 60, 120],
        labels=["new", "trial", "starter", "regular", "loyal", "veteran"],
        include_lowest=True,
    ).astype(str)
    out["churn_risk_by_tenure"] = np.where(out["tenure_months"] < 6, 0.8, np.where(out["tenure_months"] < 12, 0.5, 0.2))

    out["engagement_score"] = (
        (out["satisfaction_score"] * 0.5)
        + (((30 - out["last_login_days"]) / 30) * 0.3)
        + (1 - (out["support_tickets"] / 20) * 0.2)
    )
    out["login_frequency_bin"] = pd.cut(
        out["last_login_days"],
        bins=[0, 7, 14, 30, 60, 365],
        labels=["daily", "weekly", "biweekly", "monthly", "inactive"],
        include_lowest=True,
    ).astype(str)
    out["support_intensity"] = out["support_tickets"] / (out["tenure_months"] + eps)

    out["age_spend_interaction"] = out["age"] * out["monthly_spend"] / 100
    out["tenure_satisfaction"] = out["tenure_months"] * out["satisfaction_score"]
    out["support_login_ratio"] = out["support_tickets"] / (out["last_login_days"] + eps)
    out["contract_spend_interaction"] = (out["contract_type"] == "Yearly").astype(float) * out["monthly_spend"] * 0.9
    out["risk_composite"] = (
        (out["support_tickets"] * 0.4)
        + ((10 - out["satisfaction_score"]) * 0.3)
        + ((out["contract_type"] == "Monthly").astype(float) * 0.3)
    )

    out["days_since_last_login_capped"] = np.minimum(out["last_login_days"], 365)
    out["login_decay"] = np.exp(-0.1 * out["last_login_days"])
    out["support_decay"] = np.exp(-0.2 * out["support_tickets"])
    out["recency_score"] = 1 / (1 + np.exp(-(30 - out["last_login_days"]) / 10))

    out["gender_encoded"] = (out["gender"] == "Male").astype(int)
    out["contract_risk"] = (out["contract_type"] == "Monthly").astype(float) * 2 + (out["contract_type"] == "Yearly").astype(float) * 0.5
    out["tenure_age_cluster"] = _safe_qcut(out["tenure_months"] * out["age"], 5).astype(float).fillna(0).astype(int)
    out["high_value_customer"] = ((out["monthly_spend"] > 200).astype(int) & (out["tenure_months"] > 12).astype(int)).astype(int)

    if target is not None:
        out["contract_type_target_encoded"] = _target_encode_contract_type(out, target, m=50.0)
        if include_polynomial:
            poly = _top_polynomial_features(out, target, limit=10)
            out = pd.concat([out, poly], axis=1)

    return out


def feature_count() -> int:
    return len(DERIVED_FEATURE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/features")
    parser.add_argument("--target_encode_smoothing", type=float, default=50.0)
    parser.add_argument("--include_polynomial", action="store_true", default=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    target = df["churn"] if "churn" in df.columns else None
    out = create_features(df, target=target, include_polynomial=args.include_polynomial)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_dir / "train_features.parquet", index=False)
    out.to_parquet(output_dir / "test_features.parquet", index=False)
    (output_dir / "feature_columns.json").write_text(json.dumps(list(out.columns), indent=2))


if __name__ == "__main__":
    main()
