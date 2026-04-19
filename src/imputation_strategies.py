from __future__ import annotations

import pandas as pd

from churn_odyssey.imputation import compare_imputation_strategies, get_imputer


SUPPORTED_STRATEGIES = ("knn", "mice", "iterativeimputer")


def run_imputation_comparison(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target_col: str = "churn",
) -> tuple[str, dict[str, float], pd.DataFrame]:
    return compare_imputation_strategies(df=df, numeric_columns=numeric_columns, target_col=target_col)


def build_imputer(strategy: str):
    return get_imputer(strategy)
