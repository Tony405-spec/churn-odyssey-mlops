from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def get_imputer(strategy: str):
    strategy = strategy.lower()
    if strategy == "knn":
        return KNNImputer(n_neighbors=5)
    if strategy == "mice":
        return IterativeImputer(random_state=42, sample_posterior=True, max_iter=15)
    if strategy == "iterativeimputer":
        return IterativeImputer(random_state=42, max_iter=20)
    raise ValueError(f"Unknown strategy: {strategy}")


def custom_imputation_score(X: pd.DataFrame, y: pd.Series) -> float:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores: list[float] = []
    for train_idx, test_idx in cv.split(X, y):
        model = LogisticRegression(max_iter=500)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[test_idx], prob))
    stability_penalty = float(np.std(scores))
    return float(np.mean(scores) - 0.1 * stability_penalty)


def compare_imputation_strategies(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target_col: str = "churn",
) -> tuple[str, dict[str, float], pd.DataFrame]:
    scores: dict[str, float] = {}
    best_frame = df.copy()
    best_name = ""
    best_score = -1.0

    for strategy in ["knn", "mice", "iterativeimputer"]:
        imputer = get_imputer(strategy)
        transformed = df.copy()
        transformed[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        score = custom_imputation_score(transformed[numeric_columns], transformed[target_col])
        scores[strategy] = score
        if score > best_score:
            best_name = strategy
            best_score = score
            best_frame = transformed

    return best_name, scores, best_frame
