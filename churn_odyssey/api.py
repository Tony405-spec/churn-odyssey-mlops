from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI

from churn_odyssey.features import generate_synthetic_features
from churn_odyssey.validation import BatchPredictionRequest

app = FastAPI(title="Churn Odyssey API")
MODEL_PATH = Path("artifacts/model.joblib")


def _load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def _background_retrain() -> None:
    # Background retraining hook for scheduler/queue integration.
    return None


@app.post("/predict")
def predict(payload: BatchPredictionRequest):
    model = _load_model()
    records = [r.model_dump() for r in payload.records]
    frame = pd.DataFrame(records)
    feats = generate_synthetic_features(frame)
    model_input = pd.get_dummies(feats, columns=["gender", "contract_type"], drop_first=True)
    if model is None:
        probs = [0.5] * len(model_input)
    else:
        probs = model.predict_proba(model_input)[:, 1].tolist()
    return {"predictions": probs}


@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_background_retrain)
    return {"status": "retraining_scheduled"}
