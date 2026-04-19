from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd

from churn_odyssey.drift import detect_drift
from churn_odyssey.features import generate_synthetic_features
from churn_odyssey.imputation import compare_imputation_strategies
from churn_odyssey.models import run_nested_cv
from churn_odyssey.optuna_search import run_bayesian_search
from churn_odyssey.validation import CustomerRecord


def run_pipeline(input_csv: str, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    _ = [CustomerRecord.model_validate(r) for r in df.to_dict(orient="records")]

    fe_df = generate_synthetic_features(df)
    fe_df = pd.get_dummies(fe_df, columns=["gender", "contract_type"], drop_first=True)

    numeric_cols = [c for c in fe_df.columns if c != "churn"]
    best_strategy, strategy_scores, imputed_df = compare_imputation_strategies(fe_df, numeric_cols)

    X = imputed_df.drop(columns=["churn"])
    y = imputed_df["churn"]

    nested_results = run_nested_cv(X, y)
    optuna_result = run_bayesian_search(X, y, trials=200)
    drift = detect_drift(X, X.copy(), list(X.columns[: min(5, len(X.columns))]))

    with mlflow.start_run():
        mlflow.log_param("best_imputation_strategy", best_strategy)
        mlflow.log_metrics({f"imputation_{k}": v for k, v in strategy_scores.items()})
        for result in nested_results:
            mlflow.log_metric(f"nested_auc_{result.model_name}", result.mean_auc)
        mlflow.log_metric("optuna_best_auc", optuna_result["best_value"])
        mlflow.log_dict(optuna_result, "optuna_best.json")
        mlflow.log_dict(drift, "drift_report.json")

    summary = {
        "best_imputation": best_strategy,
        "imputation_scores": strategy_scores,
        "nested_cv": [r.__dict__ for r in nested_results],
        "optuna": optuna_result,
        "drift": drift,
    }
    (output_path / "summary.json").write_text(json.dumps(summary))
    return summary
