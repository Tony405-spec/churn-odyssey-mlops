from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: int = Field(gt=0)
    age: int = Field(ge=18, le=120)
    gender: str
    tenure_months: int = Field(ge=0)
    monthly_spend: float = Field(ge=0)
    contract_type: str
    support_tickets: int = Field(ge=0)
    last_login_days: int = Field(ge=0)
    satisfaction_score: int = Field(ge=1, le=10)
    churn: int = Field(ge=0, le=1)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        if value not in {"Male", "Female", "Other"}:
            raise ValueError("gender must be Male, Female, or Other")
        return value

    @field_validator("contract_type")
    @classmethod
    def validate_contract_type(cls, value: str) -> str:
        if value not in {"Monthly", "Yearly", "Quarterly"}:
            raise ValueError("contract_type must be Monthly, Quarterly, or Yearly")
        return value


class BatchPredictionRequest(BaseModel):
    records: list[CustomerRecord]
