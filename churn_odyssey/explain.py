from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_shap_waterfalls(model, X: pd.DataFrame, output_dir: str, n_customers: int = 5) -> list[str]:
    import matplotlib.pyplot as plt
    import shap

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X.iloc[:n_customers])

    paths: list[str] = []
    for i in range(min(n_customers, len(X))):
        shap.plots.waterfall(shap_values[i], show=False)
        file_path = out_dir / f"customer_{i + 1}_waterfall.png"
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        paths.append(str(file_path))

    return paths
