import pandas as pd
import pytest
from pydantic import ValidationError

from src.data_validation import ChurnPredictionDataset, CustomerBase, validate_payload, validate_records


def _valid(**overrides):
    payload = {
        "customer_id": 1,
        "age": 42,
        "gender": "Male",
        "tenure_months": 12,
        "monthly_spend": 100.0,
        "contract_type": "Monthly",
        "support_tickets": 2,
        "last_login_days": 7,
        "satisfaction_score": 8,
        "churn": 1,
    }
    payload.update(overrides)
    return payload


def test_valid_customer_passes_validation():
    model = ChurnPredictionDataset.model_validate(_valid())
    assert model.customer_id == 1


def test_invalid_age_raises_error():
    with pytest.raises(ValidationError):
        ChurnPredictionDataset.model_validate(_valid(age=-1))


def test_negative_monthly_spend_raises_error():
    with pytest.raises(ValidationError):
        ChurnPredictionDataset.model_validate(_valid(monthly_spend=-0.1))


def test_churn_outside_0_1_raises_error():
    with pytest.raises(ValidationError):
        ChurnPredictionDataset.model_validate(_valid(churn=2))


def test_missing_required_field_raises_error():
    bad = _valid()
    bad.pop("age")
    with pytest.raises(ValidationError):
        ChurnPredictionDataset.model_validate(bad)


def test_contract_type_mapping_works():
    model = ChurnPredictionDataset.model_validate(_valid(contract_type=None))
    assert model.contract_type == "Monthly"


def test_tenure_spend_consistency_validation():
    with pytest.raises(ValidationError):
        ChurnPredictionDataset.model_validate(_valid(tenure_months=0, monthly_spend=10.0))


def test_satisfaction_score_boundary():
    assert ChurnPredictionDataset.model_validate(_valid(satisfaction_score=0)).satisfaction_score == 0
    assert ChurnPredictionDataset.model_validate(_valid(satisfaction_score=10)).satisfaction_score == 10
    with pytest.raises(ValidationError):
        ChurnPredictionDataset.model_validate(_valid(satisfaction_score=11))


def test_negative_age_edge_case():
    with pytest.raises(ValidationError):
        CustomerBase.model_validate(_valid(age=-99, churn=0))


def test_validate_records_batch():
    records = validate_records(pd.DataFrame([_valid(customer_id=1), _valid(customer_id=2)]))
    assert len(records) == 2


def test_validate_payload_batch_model():
    model = validate_payload({"records": [_valid(customer_id=10), _valid(customer_id=11)]})
    assert len(model.records) == 2


def test_impossible_churn_pattern_tenure_zero_spend_non_zero():
    with pytest.raises(ValidationError):
        ChurnPredictionDataset.model_validate(_valid(tenure_months=0, monthly_spend=99.0, churn=0))
