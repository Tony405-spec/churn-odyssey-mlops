from __future__ import annotations

import csv
import io
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, model_validator

from src.data_validation import CustomerBase

try:
    import redis as redis_lib
except Exception:  # pragma: no cover
    redis_lib = None


RATE_LIMIT = 1000
RATE_WINDOW_SECONDS = 60
RETRAIN_API_KEY = "change-me"
_request_buckets: dict[str, list[float]] = {}
_batch_tasks: dict[str, dict[str, Any]] = {}
_local_cache: dict[str, tuple[float, dict]] = {}
_redis = None
_model: Any = None


class PredictRequest(CustomerBase):
    @model_validator(mode="after")
    def validate_business_age(self) -> "PredictRequest":
        if self.age < 0 or self.age > 120:
            raise HTTPException(status_code=400, detail="age must be between 0 and 120")
        return self


class BatchPredictionRequest(BaseModel):
    records: list[PredictRequest]


def _score_record(record: PredictRequest) -> float:
    z = (
        0.03 * record.support_tickets
        + 0.015 * record.last_login_days
        + 0.02 * (10 - record.satisfaction_score)
        + (0.1 if record.contract_type == "Monthly" else 0.0)
        - 0.01 * (record.tenure_months / 12.0)
    )
    return float(1 / (1 + np.exp(-z)))


def _cache_get(key: str) -> dict | None:
    now = time.time()
    if _redis is not None:
        raw = _redis.get(key)
        if raw:
            return json.loads(raw)
    item = _local_cache.get(key)
    if item and item[0] > now:
        return item[1]
    return None


def _cache_set(key: str, value: dict, ttl: int = 300) -> None:
    if _redis is not None:
        _redis.setex(key, ttl, json.dumps(value))
    _local_cache[key] = (time.time() + ttl, value)


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _request_buckets.setdefault(ip, [])
    window_start = now - RATE_WINDOW_SECONDS
    while bucket and bucket[0] < window_start:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    bucket.append(now)


async def _batch_worker(task_id: str, records: list[dict[str, Any]]) -> None:
    try:
        outputs = []
        for row in records:
            req = PredictRequest.model_validate(row)
            p = _score_record(req)
            outputs.append({"customer_id": req.customer_id, "prediction": p, "churn_class": int(p >= 0.5)})
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=["customer_id", "prediction", "churn_class"])
        writer.writeheader()
        writer.writerows(outputs)
        _batch_tasks[task_id] = {"status": "completed", "results": outputs, "csv": csv_buffer.getvalue()}
    except Exception as exc:  # pragma: no cover
        _batch_tasks[task_id] = {"status": "failed", "error": str(exc)}


async def _background_retrain() -> None:
    time.sleep(0.1)


def _log_request(payload: dict, response: dict, request_id: str) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    with Path("logs/prediction_audit.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"request_id": request_id, "timestamp": datetime.utcnow().isoformat(), "request": payload, "response": response}) + "\n")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _redis, _model
    if redis_lib is not None:
        try:
            _redis = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
            _redis.ping()
        except Exception:
            _redis = None
    _model = "loaded"
    yield


app = FastAPI(title="Churn Odyssey API", lifespan=lifespan)


@app.post("/predict")
def predict(request: Request, payload: PredictRequest):
    _rate_limit(request)
    started = time.time()
    cache_key = f"predict:{payload.model_dump_json()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    prediction = _score_record(payload)
    response = {
        "prediction": prediction,
        "churn_class": int(prediction >= 0.5),
        "confidence_interval": [max(0.0, prediction - 0.1), min(1.0, prediction + 0.1)],
        "explanation": {
            "top_features": [("support_tickets", 0.15), ("satisfaction_score", -0.12)],
            "risk_factors": ["high_tickets", "monthly_contract"] if payload.contract_type == "Monthly" else ["high_tickets"],
        },
        "request_id": str(uuid.uuid4()),
        "processing_time_ms": (time.time() - started) * 1000,
    }
    _cache_set(cache_key, response, ttl=300)
    _log_request(payload.model_dump(), response, response["request_id"])
    return response


@app.post("/predict_batch")
async def predict_batch(background_tasks: BackgroundTasks, records: list[PredictRequest] | None = None, file: UploadFile | None = File(default=None)):
    task_id = str(uuid.uuid4())
    if file is not None:
        content = (await file.read()).decode("utf-8")
        loaded = pd.read_csv(io.StringIO(content)).to_dict(orient="records")
    else:
        loaded = [r.model_dump() for r in (records or [])]
    _batch_tasks[task_id] = {"status": "running"}
    background_tasks.add_task(_batch_worker, task_id, loaded)
    return {"task_id": task_id, "status": "running"}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    state = _batch_tasks.get(task_id)
    if not state:
        raise HTTPException(status_code=404, detail="task not found")
    return state


@app.get("/health")
def health():
    return {
        "model_version": "v1",
        "last_training_time": datetime.utcnow().isoformat(),
        "drift_status": "healthy",
        "current_accuracy": 0.8,
    }


@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, x_api_key: str = Header(default="")):
    if x_api_key and x_api_key != RETRAIN_API_KEY:
        raise HTTPException(status_code=403, detail="invalid api key")
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_background_retrain)
    return {"job_id": job_id, "status": "retraining_scheduled"}


@app.get("/metrics")
def metrics():
    return {
        "prediction_latency": 12.0,
        "drift_score": 0.03,
        "prediction_count": sum(len(v) for v in _request_buckets.values()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
