import pandas as pd

from churn_odyssey.features import generate_synthetic_features


def main():
    df = pd.read_csv("data/processed/validated.csv")
    out = generate_synthetic_features(df)
    out.to_csv("data/processed/features.csv", index=False)


if __name__ == "__main__":
    main()
