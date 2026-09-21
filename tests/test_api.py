from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def _payload():
    return {
        "customer_id": 1,
        "age": 56,
        "gender": "Male",
        "tenure_months": 58,
        "monthly_spend": 77.18,
        "contract_type": "Yearly",
        "support_tickets": 1,
        "last_login_days": 11,
        "satisfaction_score": 8,
    }


def test_predict_endpoint_returns_200():
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_invalid_input_returns_422():
    bad = _payload()
    bad["age"] = 130
    response = client.post("/predict", json=bad)
    assert response.status_code in (400, 422)


def test_batch_prediction_async():
    response = client.post("/predict_batch", json=[_payload(), {**_payload(), "customer_id": 2}])
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    status = client.get(f"/status/{task_id}")
    assert status.status_code == 200


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_version" in response.json()


def test_rate_limiting():
    for _ in range(5):
        response = client.post("/predict", json=_payload())
        assert response.status_code == 200
