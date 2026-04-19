from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3


@dataclass
class NestedCVFoldResult:
    model_name: str
    best_params: dict
    inner_val_score: float
    train_score: float
    val_score: float
    generalization_gap: float


def _safe_xgboost():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(eval_metric="logloss", random_state=42)
    except Exception:
        return GradientBoostingClassifier(random_state=42)


def _safe_lightgbm():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=42, verbose=-1)
    except Exception:
        return GradientBoostingClassifier(random_state=42)


def _safe_catboost():
    try:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(verbose=False, random_state=42)
    except Exception:
        return GradientBoostingClassifier(random_state=42)


def _model_grids() -> dict[str, tuple[object, dict]]:
    return {
        "xgboost": (
            _safe_xgboost(),
            {
                "n_estimators": [100, 300, 500, 1000],
                "max_depth": [3, 5, 7, 9, 11],
                "learning_rate": [0.01, 0.05, 0.1, 0.3],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0, 0.1, 0.3, 0.5],
                "reg_alpha": [0, 0.01, 0.1, 1],
                "reg_lambda": [0.1, 1, 10],
                "scale_pos_weight": [1, 2],
            },
        ),
        "lightgbm": (
            _safe_lightgbm(),
            {
                "num_leaves": [15, 31, 63, 127, 255],
                "max_depth": [-1, 5, 10, 15, 20],
                "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
                "n_estimators": [100, 300, 500, 800],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0, 0.01, 0.1, 1],
                "reg_lambda": [0, 0.01, 0.1, 1],
                "min_child_samples": [5, 10, 20, 50],
            },
        ),
        "randomforest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 300, 500],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
        ),
        "catboost": (_safe_catboost(), {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1], "iterations": [200, 500]}),
        "gradientboosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 300], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
        ),
    }


def run_nested_cv_tuning(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=42)
    inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=42)
    grids = _model_grids()

    all_results: list[NestedCVFoldResult] = []
    oof_predictions = {model_name: np.zeros(len(y)) for model_name in grids}

    for model_name, (estimator, param_grid) in grids.items():
        for train_idx, val_idx in outer.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=inner, scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            train_prob = best.predict_proba(X_train)[:, 1]
            val_prob = best.predict_proba(X_val)[:, 1]
            oof_predictions[model_name][val_idx] = val_prob
            train_score = float(roc_auc_score(y_train, train_prob))
            val_score = float(roc_auc_score(y_val, val_prob))
            all_results.append(
                NestedCVFoldResult(
                    model_name=model_name,
                    best_params=search.best_params_,
                    inner_val_score=float(search.best_score_),
                    train_score=train_score,
                    val_score=val_score,
                    generalization_gap=train_score - val_score,
                )
            )

    return {
        "outer_folds": N_OUTER_FOLDS,
        "inner_folds": N_INNER_FOLDS,
        "results": [asdict(r) for r in all_results],
        "oof_predictions": {k: v.tolist() for k, v in oof_predictions.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/imputed/train_imputed.parquet")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    y = frame["churn"]
    X = pd.get_dummies(frame.drop(columns=["churn"]), drop_first=False)
    result = run_nested_cv_tuning(X, y)
    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "nested_cv_results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
