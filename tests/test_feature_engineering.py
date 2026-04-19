import pandas as pd

from src.feature_engineering import DERIVED_FEATURE_COLUMNS, create_features, feature_count


def test_feature_engineering_creates_25_features():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "age": [30, 40],
            "gender": ["Male", "Female"],
            "tenure_months": [12, 24],
            "monthly_spend": [100.0, 120.0],
            "contract_type": ["Monthly", "Yearly"],
            "support_tickets": [1, 3],
            "last_login_days": [2, 12],
            "satisfaction_score": [8, 6],
            "churn": [0, 1],
        }
    )
    out = create_features(df)
    assert feature_count() == 25
    for col in DERIVED_FEATURE_COLUMNS:
        assert col in out.columns
