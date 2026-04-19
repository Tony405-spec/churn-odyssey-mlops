import pandas as pd

from churn_odyssey.explain import generate_shap_waterfalls
from sklearn.ensemble import RandomForestClassifier


def main():
    df = pd.read_csv("data/processed/features.csv")
    X = pd.get_dummies(df.drop(columns=["churn"]), columns=["gender", "contract_type"], drop_first=True)
    y = df["churn"]
    model = RandomForestClassifier(random_state=42).fit(X, y)
    generate_shap_waterfalls(model, X, "artifacts/explanations", n_customers=5)


if __name__ == "__main__":
    main()
