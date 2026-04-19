import pandas as pd

from churn_odyssey.validation import CustomerRecord


def main():
    df = pd.read_csv("data/raw/customers.csv")
    _ = [CustomerRecord.model_validate(r) for r in df.to_dict(orient="records")]
    df.to_csv("data/processed/validated.csv", index=False)


if __name__ == "__main__":
    main()
