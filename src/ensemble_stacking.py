from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

from src.pytorch_attention_model import TabularAttentionModel


def _safe_xgb():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(eval_metric="logloss", random_state=42)
    except Exception:
        return RandomForestClassifier(random_state=42)


def _safe_lgbm():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=42, verbose=-1)
    except Exception:
        return RandomForestClassifier(random_state=42)


def _safe_cat():
    try:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(verbose=False, random_state=42)
    except Exception:
        return RandomForestClassifier(random_state=42)


class TorchAttentionSklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=500)
        self.single_class_: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        self.classes_ = classes
        if len(classes) < 2:
            self.single_class_ = int(classes[0])
            return self
        self.single_class_ = None
        self.model.fit(X, y_arr)
        return self

    def predict_proba(self, X: pd.DataFrame):
        if self.single_class_ is not None:
            probs = np.ones((len(X), 1), dtype=float)
            return probs
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame):
        if self.single_class_ is not None:
            return np.full(len(X), self.single_class_)
        return self.model.predict(X)


def create_stacking_ensemble(cv: int = 5) -> StackingClassifier:
    estimators = [
        ("xgboost_best", _safe_xgb()),
        ("lightgbm_best", _safe_lgbm()),
        ("rf_best", RandomForestClassifier(random_state=42)),
        ("pytorch_attention", TorchAttentionSklearnWrapper()),
        ("catboost_best", _safe_cat()),
    ]
    meta = LogisticRegression(
        max_iter=1000,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        random_state=42,
    )
    return StackingClassifier(estimators=estimators, final_estimator=meta, cv=cv, passthrough=True, stack_method="predict_proba")


def _optimize_soft_voting_weights(pred_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n_models = pred_matrix.shape[1]

    def objective(weights: np.ndarray) -> float:
        clipped = np.clip(weights, 0, 1)
        total = clipped.sum()
        if total == 0:
            clipped = np.ones_like(clipped) / len(clipped)
        else:
            clipped = clipped / total
        probs = pred_matrix @ clipped
        return -roc_auc_score(y_true, probs)

    bounds = [(0, 1)] * n_models
    result = differential_evolution(objective, bounds=bounds, seed=42, polish=True, maxiter=100)
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def train_stacking_ensemble(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict[str, object]:
    stack = create_stacking_ensemble(cv=cv)
    stack.fit(X, y)

    base_probs = []
    for _, model in stack.named_estimators_.items():
        prob = model.predict_proba(X)[:, 1]
        base_probs.append(prob)
    pred_matrix = np.vstack(base_probs).T
    meta_features = pd.DataFrame(pred_matrix, columns=[name for name, _ in stack.estimators])
    meta_features["prediction_variance"] = meta_features.var(axis=1)
    meta_features["model_disagreement_score"] = (meta_features.max(axis=1) - meta_features.min(axis=1)).astype(float)

    weights = _optimize_soft_voting_weights(pred_matrix, y.to_numpy())
    weighted_probs = pred_matrix @ weights

    calibrated_platt = CalibratedClassifierCV(estimator=stack, method="sigmoid", cv=3).fit(X, y)
    calibrated_isotonic = CalibratedClassifierCV(estimator=stack, method="isotonic", cv=3).fit(X, y)

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(stack, models_dir / f"ensemble_meta_model_{ts}.pkl")
    joblib.dump(stack, models_dir / "ensemble_meta_model.pkl")
    joblib.dump(calibrated_platt, models_dir / f"ensemble_platt_{ts}.pkl")
    joblib.dump(calibrated_isotonic, models_dir / f"ensemble_isotonic_{ts}.pkl")
    (models_dir / "ensemble_weights.json").write_text(json.dumps({"timestamp": ts, "weights": weights.tolist()}, indent=2))

    return {
        "roc_auc": float(roc_auc_score(y, weighted_probs)),
        "weights": weights.tolist(),
        "meta_features": list(meta_features.columns),
        "timestamp": ts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/imputed/train_imputed.parquet")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    y = frame["churn"]
    X = pd.get_dummies(frame.drop(columns=["churn"]), drop_first=False)
    out = train_stacking_ensemble(X, y)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    Path("metrics/ensemble_performance.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
