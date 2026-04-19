from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_predict_endpoint_returns_predictions():
    payload = {
        "records": [
            {
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
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 1


def test_retrain_endpoint_schedules_retraining():
    response = client.post("/retrain")
    assert response.status_code == 200
    assert response.json()["status"] == "retraining_scheduled"
