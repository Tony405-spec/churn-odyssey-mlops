from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _select_customers(df: pd.DataFrame, probs: np.ndarray) -> dict[str, int]:
    picked: dict[str, int] = {}
    picked["A"] = int(df[(df["monthly_spend"] > df["monthly_spend"].quantile(0.8)) & (df["support_tickets"] <= 1) & (df["satisfaction_score"] >= 8)].index[0])
    picked["B"] = int(df[(df["monthly_spend"] < df["monthly_spend"].quantile(0.2)) & (df["support_tickets"] >= 4) & (df["satisfaction_score"] <= 3)].index[0])
    picked["C"] = int(df[(df["tenure_months"] < 3) & (df["churn"] == 1)].index[0])
    picked["D"] = int(df[(df["tenure_months"] > 36) & (df["churn"] == 1)].index[0])
    picked["E"] = int(np.argmin(np.abs(probs - 0.5)))
    return picked


def generate_shap_explanations(model, X: pd.DataFrame, output_dir: str, n_customers: int = 5) -> list[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shap_dir = output / "shap"
    waterfall_dir = shap_dir / "waterfall_plots"
    waterfall_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else shap.Explainer(model.predict_proba, X)
        values = explainer(X)
    else:
        probs = np.full(len(X), 0.5)
        explainer = shap.Explainer(lambda z: np.full((len(z),), 0.5), X)
        values = explainer(X)

    selected = _select_customers(X.assign(churn=0 if "churn" not in X.columns else X["churn"]), probs)
    created: list[str] = []
    for key, idx in list(selected.items())[:n_customers]:
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(values[idx], max_display=15, show=False)
        plt.title(f"Customer {key} SHAP Waterfall")
        file_path = waterfall_dir / f"waterfall_{key}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()
        created.append(str(file_path))

    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(values, max_display=20, show=False)
    plt.savefig(shap_dir / "global_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    abs_mean = np.abs(values.values).mean(axis=0)
    stds = np.std(np.abs(values.values), axis=0)
    top_idx = np.argsort(abs_mean)[-10:][::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(np.array(X.columns)[top_idx], abs_mean[top_idx], yerr=stds[top_idx], alpha=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(shap_dir / "mean_abs_shap_bar.png", dpi=300)
    plt.close()

    try:
        interaction = shap.TreeExplainer(model).shap_interaction_values(X)
        interaction_arr = interaction[1] if isinstance(interaction, list) else interaction
        top_features = np.array(X.columns)[top_idx]
        heat = np.abs(interaction_arr).mean(axis=0)[np.ix_(top_idx, top_idx)]
        plt.figure(figsize=(8, 6))
        plt.imshow(heat, cmap="viridis")
        plt.xticks(np.arange(len(top_features)), top_features, rotation=45, ha="right")
        plt.yticks(np.arange(len(top_features)), top_features)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(shap_dir / "interaction_heatmap.png", dpi=300)
        plt.close()
    except Exception:
        pass

    for col in list(np.array(X.columns)[np.argsort(abs_mean)[-5:][::-1]]):
        shap.dependence_plot(col, values.values, X, show=False, interaction_index="auto")
        plt.tight_layout()
        plt.savefig(shap_dir / f"dependence_{col}.png", dpi=300)
        plt.close()

    serializable = {"expected_value": np.asarray(values.base_values).tolist(), "values": np.asarray(values.values).tolist(), "features": X.columns.tolist()}
    (shap_dir / "shap_values.json").write_text(json.dumps(serializable))
    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--input", type=str, default="data/imputed/test_imputed.parquet")
    args = parser.parse_args()

    import joblib

    model = joblib.load(args.model) if args.model else None
    frame = pd.read_parquet(args.input)
    X = pd.get_dummies(frame.drop(columns=["churn"]) if "churn" in frame.columns else frame, drop_first=False)
    generate_shap_explanations(model, X, output_dir="reports")


if __name__ == "__main__":
    main()
