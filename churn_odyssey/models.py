from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


@dataclass
class NestedCVResult:
    model_name: str
    mean_auc: float
    std_auc: float
    best_params: dict


def _xgboost_classifier():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(eval_metric="logloss", random_state=42)
    except Exception:
        return RandomForestClassifier(random_state=42)


def _lightgbm_classifier():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=42, verbose=-1)
    except Exception:
        return RandomForestClassifier(random_state=42)


def run_nested_cv(X: pd.DataFrame, y: pd.Series) -> list[NestedCVResult]:
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    model_specs = {
        "xgboost": (
            _xgboost_classifier(),
            {"max_depth": [3, 5], "n_estimators": [100, 200]},
        ),
        "lightgbm": (
            _lightgbm_classifier(),
            {"num_leaves": [31, 63], "n_estimators": [100, 200]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {"max_depth": [None, 12], "n_estimators": [200, 400]},
        ),
    }

    results: list[NestedCVResult] = []
    for model_name, (model, param_grid) in model_specs.items():
        fold_scores = []
        best_params = {}
        for train_idx, test_idx in outer.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=inner,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_params = search.best_params_
            prob = search.best_estimator_.predict_proba(X_test)[:, 1]
            fold_scores.append(roc_auc_score(y_test, prob))

        results.append(
            NestedCVResult(
                model_name=model_name,
                mean_auc=float(np.mean(fold_scores)),
                std_auc=float(np.std(fold_scores)),
                best_params=best_params,
            )
        )

    return results
