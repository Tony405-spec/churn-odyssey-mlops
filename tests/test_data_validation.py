import pytest
from pydantic import ValidationError

from src.data_validation import CustomerRecord, validate_payload, validate_records


def valid_record(**overrides):
    base = {
        "customer_id": 1,
        "age": 56,
        "gender": "Male",
        "tenure_months": 58,
        "monthly_spend": 77.18,
        "contract_type": "Yearly",
        "support_tickets": 1,
        "last_login_days": 11,
        "satisfaction_score": 8,
        "churn": 1,
    }
    base.update(overrides)
    return base


def test_valid_record_passes():
    model = CustomerRecord.model_validate(valid_record())
    assert model.customer_id == 1


@pytest.mark.parametrize("field,value", [("customer_id", 0), ("age", 17), ("age", 121)])
def test_invalid_ranges_raise(field, value):
    payload = valid_record(**{field: value})
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(payload)


@pytest.mark.parametrize("gender", ["Unknown", ""])
def test_invalid_gender_raises(gender):
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(valid_record(gender=gender))


@pytest.mark.parametrize("contract", ["Biennial", ""])
def test_invalid_contract_type_raises(contract):
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(valid_record(contract_type=contract))


def test_extra_field_is_forbidden():
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate({**valid_record(), "extra": 1})


def test_validate_records_batch_length_matches():
    records = validate_records(
        __import__("pandas").DataFrame([valid_record(customer_id=1), valid_record(customer_id=2)])
    )
    assert len(records) == 2


def test_validate_payload_accepts_batch_request():
    payload = {"records": [valid_record(), valid_record(customer_id=2)]}
    model = validate_payload(payload)
    assert len(model.records) == 2


def test_invalid_churn_value_raises():
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(valid_record(churn=3))
