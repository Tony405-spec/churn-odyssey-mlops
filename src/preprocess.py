from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(input_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(input_path)
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["churn"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["churn"])
    return train, test, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    args = parser.parse_args()
    train, test, val = preprocess(args.input_path)
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)


if __name__ == "__main__":
    main()
