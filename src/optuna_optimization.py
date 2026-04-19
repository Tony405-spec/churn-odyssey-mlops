from __future__ import annotations

import pandas as pd

from churn_odyssey.optuna_search import run_bayesian_search


DEFAULT_OPTUNA_TRIALS = 200


def run_optuna_optimization(X: pd.DataFrame, y: pd.Series, trials: int = DEFAULT_OPTUNA_TRIALS) -> dict:
    return run_bayesian_search(X, y, trials=trials)
