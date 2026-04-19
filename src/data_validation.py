from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, conint, confloat, field_validator, model_validator


ALLOWED_CONTRACT_TYPES = {"Monthly", "Yearly", "Quarterly"}


class CustomerBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: int = Field(gt=0)
    age: conint(ge=0, le=120)
    gender: str
    tenure_months: conint(ge=0)
    monthly_spend: confloat(ge=0)
    contract_type: str
    support_tickets: conint(ge=0)
    last_login_days: conint(ge=0)
    satisfaction_score: float

    @field_validator("contract_type", mode="before")
    @classmethod
    def validate_contract_type(cls, value: Any) -> str:
        if value in (None, "", "quarterly", "QUARTERLY"):
            return "Monthly"
        if isinstance(value, str):
            normalized = value.strip().title()
            if normalized == "Quarterly":
                return "Quarterly"
            if normalized in {"Monthly", "Yearly"}:
                return normalized
        raise ValueError("contract_type must be one of Monthly, Yearly, Quarterly")

    @field_validator("satisfaction_score")
    @classmethod
    def validate_satisfaction_range(cls, value: float) -> float:
        if value < 0 or value > 10:
            raise ValueError("satisfaction_score must be between 0 and 10")
        return value

    @model_validator(mode="after")
    def validate_tenure_spend_consistency(self) -> "CustomerBase":
        if self.tenure_months < 1 and self.monthly_spend != 0:
            raise ValueError("monthly_spend must be 0 when tenure_months < 1")
        return self


class ChurnPredictionDataset(CustomerBase):
    churn: conint(ge=0, le=1)

    @field_validator("churn")
    @classmethod
    def validate_churn_binary(cls, value: int) -> int:
        if value not in {0, 1}:
            raise ValueError("churn must be in {0,1}")
        return value


CustomerRecord = ChurnPredictionDataset


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    records: list[CustomerRecord]


def validate_records(df: pd.DataFrame) -> list[CustomerRecord]:
    return [CustomerRecord.model_validate(record) for record in df.to_dict(orient="records")]


def validate_payload(payload: dict) -> BatchPredictionRequest:
    return BatchPredictionRequest.model_validate(payload)


def validate_dataset(input_path: str | Path, output_path: str | Path = "data/processed/validated_data.parquet") -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    records = validate_records(frame)
    validated = pd.DataFrame([r.model_dump() for r in records])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(output_path, index=False)
    return validated


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python src/data_validation.py <csv_path>")

    in_path = Path(sys.argv[1])
    out_path = Path("data/processed/validated_data.parquet")
    validated_df = validate_dataset(in_path, out_path)
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = {"rows": int(len(validated_df)), "columns": list(validated_df.columns), "status": "ok"}
    (metrics_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
