from __future__ import annotations

import pandas as pd

from churn_odyssey.validation import BatchPredictionRequest, CustomerRecord


def validate_records(df: pd.DataFrame) -> list[CustomerRecord]:
    return [CustomerRecord.model_validate(record) for record in df.to_dict(orient="records")]


def validate_payload(payload: dict) -> BatchPredictionRequest:
    return BatchPredictionRequest.model_validate(payload)
