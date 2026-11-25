"""Feature engineering and scaling for clustering."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add insurance-specific engineered features."""
    df = df.copy()
    annual_premium = df["monthly_premium"] * 12
    df["annual_premium"] = annual_premium
    # How much has the customer claimed relative to one year of premium?
    df["claim_to_premium_ratio"] = df["total_claim_amount"] / annual_premium
    df["is_new_customer"] = (df["tenure_months"] <= 12).astype(int)
    return df


def build_matrix(df: pd.DataFrame, features=None, log_features=None) -> pd.DataFrame:
    """Return a numeric frame with skewed columns log-transformed (not yet scaled)."""
    features = features or config.CLUSTER_FEATURES
    log_features = log_features or config.LOG_FEATURES
    X = df[features].astype(float).copy()
    for c in log_features:
        if c in X.columns:
            X[c] = np.log1p(X[c].clip(lower=0))
    return X


def scale(X: pd.DataFrame, scaler: StandardScaler | None = None):
    """Standardise. Fits a new scaler when none is supplied."""
    if scaler is None:
        scaler = StandardScaler().fit(X)
    Xs = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return Xs, scaler
