from __future__ import annotations

import pandas as pd

from churn_odyssey.explain import generate_shap_waterfalls


def generate_shap_explanations(model, X: pd.DataFrame, output_dir: str, n_customers: int = 5) -> list[str]:
    return generate_shap_waterfalls(model=model, X=X, output_dir=output_dir, n_customers=n_customers)
