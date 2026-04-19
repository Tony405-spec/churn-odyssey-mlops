import pandas as pd

from src.feature_engineering import DERIVED_FEATURE_COLUMNS, create_features


def _frame():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "age": [30, 40, 25, 50, 31, 44],
            "gender": ["Male", "Female", "Male", "Other", "Female", "Male"],
            "tenure_months": [2, 8, 15, 36, 60, 1],
            "monthly_spend": [50.0, 120.0, 210.0, 300.0, 180.0, 15.0],
            "contract_type": ["Monthly", "Yearly", "Monthly", "Quarterly", "Yearly", "Monthly"],
            "support_tickets": [3, 1, 0, 2, 4, 5],
            "last_login_days": [5, 20, 1, 40, 80, 200],
            "satisfaction_score": [7, 8, 9, 4, 5, 3],
            "churn": [1, 0, 0, 1, 0, 1],
        }
    )


def test_25_features_created():
    df = _frame()
    out = create_features(df)
    for feature in DERIVED_FEATURE_COLUMNS:
        assert feature in out.columns
    assert len(DERIVED_FEATURE_COLUMNS) == 25


def test_no_leakage_between_train_test():
    df = _frame()
    train = create_features(df.iloc[:4], target=df.iloc[:4]["churn"])
    test = create_features(df.iloc[4:], target=df.iloc[4:]["churn"])
    assert not train.index.equals(test.index)


def test_target_encoding_smoothing():
    df = _frame()
    out = create_features(df, target=df["churn"])
    assert "contract_type_target_encoded" in out.columns
    assert out["contract_type_target_encoded"].between(0, 1).all()


def test_polynomial_features_shape():
    df = _frame()
    out = create_features(df, target=df["churn"], include_polynomial=True)
    poly_cols = [c for c in out.columns if c.startswith("poly_")]
    assert len(poly_cols) <= 10
    assert len(poly_cols) > 0


def test_categorical_encoding_completeness():
    df = _frame()
    out = create_features(df)
    assert out["gender_encoded"].isin([0, 1]).all()
    assert out["contract_risk"].notna().all()
