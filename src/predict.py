"""Assign new customers to the trained segments.

Usage:  python -m src.predict path/to/new_customers.csv [output.csv]
The input can use the original Kaggle column names or the cleaned project names.
"""
import json
import sys

import joblib
import pandas as pd

from src import config
from src.data_prep import COLUMN_MAP, clean
from src.features import add_features, build_matrix, scale


def load_artifacts():
    scaler = joblib.load(config.MODELS_DIR / "scaler.joblib")
    kmeans = joblib.load(config.MODELS_DIR / "kmeans.joblib")
    names = {int(k): v for k, v in json.loads((config.MODELS_DIR / "segment_names.json").read_text()).items()}
    return scaler, kmeans, names


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    scaler, kmeans, names = load_artifacts()
    if "Monthly Premium Auto" in df.columns:  # raw Kaggle-style columns
        df = clean(df)
    df = add_features(df)
    Xs, _ = scale(build_matrix(df), scaler)
    out = df.copy()
    out["cluster"] = kmeans.predict(Xs)
    out["segment"] = out["cluster"].map(names)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    result = assign_segments(pd.read_csv(sys.argv[1]))
    target = sys.argv[2] if len(sys.argv) > 2 else "segmented_output.csv"
    result.to_csv(target, index=False)
    print(result["segment"].value_counts().to_string())
    print(f"\nSaved to {target}")
