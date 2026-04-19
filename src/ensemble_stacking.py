from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression



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



def create_stacking_ensemble(cv: int = 5) -> StackingClassifier:
    estimators = [
        ("rf", RandomForestClassifier(random_state=42)),
        ("xgb", _safe_xgb()),
        ("lgbm", _safe_lgbm()),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=True,
        cv=cv,
    )



def train_stacking_ensemble(X: pd.DataFrame, y: pd.Series, cv: int = 5) -> StackingClassifier:
    model = create_stacking_ensemble(cv=cv)
    model.fit(X, y)
    return model
