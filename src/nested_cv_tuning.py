from __future__ import annotations

import pandas as pd

from churn_odyssey.models import NestedCVResult, run_nested_cv


N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3


def run_nested_cv_tuning(X: pd.DataFrame, y: pd.Series) -> list[NestedCVResult]:
    return run_nested_cv(X, y)
