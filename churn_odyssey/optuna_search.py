from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def run_bayesian_search(X: pd.DataFrame, y: pd.Series, trials: int = 200) -> dict:
    import optuna
    from xgboost import XGBClassifier

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
            "random_state": 42,
        }
        scores = []
        for train_idx, valid_idx in cv.split(X, y):
            model = XGBClassifier(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[valid_idx], prob))
            trial.report(sum(scores) / len(scores), step=len(scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=trials)
    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
