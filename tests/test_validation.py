import pytest
from pydantic import ValidationError

from churn_odyssey.validation import CustomerRecord


def test_customer_record_valid_payload():
    payload = {
        "customer_id": 1,
        "age": 56,
        "gender": "Male",
        "tenure_months": 58,
        "monthly_spend": 77.18,
        "contract_type": "Yearly",
        "support_tickets": 1,
        "last_login_days": 11,
        "satisfaction_score": 1,
        "churn": 1,
    }
    model = CustomerRecord.model_validate(payload)
    assert model.customer_id == 1


@pytest.mark.parametrize("field,value", [("age", 10), ("satisfaction_score", 11), ("churn", 3)])
def test_customer_record_rejects_invalid_ranges(field, value):
    payload = {
        "customer_id": 1,
        "age": 56,
        "gender": "Male",
        "tenure_months": 58,
        "monthly_spend": 77.18,
        "contract_type": "Yearly",
        "support_tickets": 1,
        "last_login_days": 11,
        "satisfaction_score": 1,
        "churn": 1,
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(payload)


def test_customer_record_rejects_invalid_categories():
    payload = {
        "customer_id": 1,
        "age": 56,
        "gender": "Unknown",
        "tenure_months": 58,
        "monthly_spend": 77.18,
        "contract_type": "Biennial",
        "support_tickets": 1,
        "last_login_days": 11,
        "satisfaction_score": 1,
        "churn": 1,
    }
    with pytest.raises(ValidationError):
        CustomerRecord.model_validate(payload)
