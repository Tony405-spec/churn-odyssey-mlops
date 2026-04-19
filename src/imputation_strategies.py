from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SUPPORTED_STRATEGIES = ("knn", "iterative", "mice")


def _build_model():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=42, verbose=-1)
    except Exception:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=42)


def _get_imputer(strategy: str):
    if strategy == "knn":
        return KNNImputer(n_neighbors=5, weights="distance")
    if strategy == "iterative":
        return IterativeImputer(max_iter=20, random_state=42, estimator=BayesianRidge())
    if strategy == "mice":
        return IterativeImputer(max_iter=20, random_state=42, estimator=ExtraTreesRegressor(max_depth=10, random_state=42))
    raise ValueError(f"Unknown strategy {strategy}")


def _ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= left) & (y_prob < right if i < bins - 1 else y_prob <= right)
        if np.any(mask):
            conf = float(np.mean(y_prob[mask]))
            acc = float(np.mean(y_true[mask]))
            ece += (np.sum(mask) / len(y_true)) * abs(acc - conf)
    return float(ece)


def _evaluate_split(X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict[str, float]:
    model = _build_model()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_eval)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_eval, probs)),
        "pr_auc": float(average_precision_score(y_eval, probs)),
        "logloss": float(log_loss(y_eval, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y_eval, probs)),
        "f1": float(f1_score(y_eval, preds, zero_division=0)),
        "recall": float(recall_score(y_eval, preds, zero_division=0)),
        "precision": float(precision_score(y_eval, preds, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_eval, preds)),
        "mcc": float(matthews_corrcoef(y_eval, preds)),
        "ece": _ece(y_eval.to_numpy(), probs),
    }


def compare_imputation_strategies(
    df: pd.DataFrame, numeric_columns: list[str] | None = None, target_col: str = "churn"
) -> tuple[str, dict[str, dict[str, float]], pd.DataFrame]:
    if numeric_columns is None:
        numeric_columns = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df[target_col], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[target_col], random_state=42)
    scores: dict[str, dict[str, float]] = {}
    best_name = ""
    best_composite = -10_000.0
    best_frame = df.copy()

    for strategy in SUPPORTED_STRATEGIES:
        imputer = _get_imputer(strategy)
        train_imp = train_df.copy()
        val_imp = val_df.copy()
        test_imp = test_df.copy()

        train_imp[numeric_columns] = imputer.fit_transform(train_df[numeric_columns])
        val_imp[numeric_columns] = imputer.transform(val_df[numeric_columns])
        test_imp[numeric_columns] = imputer.transform(test_df[numeric_columns])

        val_metrics = _evaluate_split(train_imp[numeric_columns], train_imp[target_col], val_imp[numeric_columns], val_imp[target_col])
        test_metrics = _evaluate_split(train_imp[numeric_columns], train_imp[target_col], test_imp[numeric_columns], test_imp[target_col])
        combined = {**{f"val_{k}": v for k, v in val_metrics.items()}, **{f"test_{k}": v for k, v in test_metrics.items()}}
        combined["composite_score"] = float(val_metrics["roc_auc"] - val_metrics["ece"])
        scores[strategy] = combined

        if combined["composite_score"] > best_composite:
            best_name = strategy
            best_composite = combined["composite_score"]
            best_frame = pd.concat([train_imp, val_imp, test_imp]).sort_index()

    return best_name, scores, best_frame


def save_imputation_plot(original: pd.DataFrame, imputed: pd.DataFrame, numeric_columns: list[str], path: str) -> None:
    selected = numeric_columns[:9]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, col in zip(axes.flatten(), selected):
        ax.hist(original[col].dropna(), bins=20, alpha=0.5, label="original")
        ax.hist(imputed[col].dropna(), bins=20, alpha=0.5, label="imputed")
        ax.set_title(col)
    for ax in axes.flatten()[len(selected) :]:
        ax.axis("off")
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def run_imputation_comparison(
    df: pd.DataFrame, numeric_columns: list[str] | None = None, target_col: str = "churn"
) -> tuple[str, dict[str, dict[str, float]], pd.DataFrame]:
    return compare_imputation_strategies(df, numeric_columns=numeric_columns, target_col=target_col)


def build_imputer(strategy: str):
    normalized = strategy.lower()
    if normalized == "iterativeimputer":
        normalized = "iterative"
    return _get_imputer(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--input", type=str, default="data/features/train_features.parquet")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    numeric = [c for c in frame.columns if c != "churn" and pd.api.types.is_numeric_dtype(frame[c])]
    best, scores, imputed = compare_imputation_strategies(frame, numeric_columns=numeric, target_col="churn")
    Path("data/imputed").mkdir(parents=True, exist_ok=True)
    imputed.to_parquet("data/imputed/train_imputed.parquet", index=False)
    imputed.to_parquet("data/imputed/test_imputed.parquet", index=False)
    save_imputation_plot(frame, imputed, numeric, "reports/imputation_comparison.png")
    Path("metrics").mkdir(parents=True, exist_ok=True)
    Path("metrics/imputation_scores.json").write_text(json.dumps({"best_strategy": best, "scores": scores}, indent=2))
    if args.best or args.compare:
        print(best)


if __name__ == "__main__":
    main()
