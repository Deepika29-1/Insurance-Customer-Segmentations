"""Load, rename and clean the raw customer data."""
import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

# Kaggle column name -> project column name
COLUMN_MAP = {
    "Customer": "customer_id",
    "State": "state",
    "Customer Lifetime Value": "clv",
    "Response": "response",
    "Coverage": "coverage",
    "Education": "education",
    "Effective To Date": "effective_date",
    "EmploymentStatus": "employment_status",
    "Gender": "gender",
    "Income": "income",
    "Location Code": "location_code",
    "Marital Status": "marital_status",
    "Monthly Premium Auto": "monthly_premium",
    "Months Since Last Claim": "months_since_last_claim",
    "Months Since Policy Inception": "tenure_months",
    "Number of Open Complaints": "open_complaints",
    "Number of Policies": "num_policies",
    "Policy Type": "policy_type",
    "Policy": "policy",
    "Renew Offer Type": "renew_offer_type",
    "Sales Channel": "sales_channel",
    "Total Claim Amount": "total_claim_amount",
    "Vehicle Class": "vehicle_class",
    "Vehicle Size": "vehicle_size",
}

NUMERIC_COLS = ["clv", "income", "monthly_premium", "months_since_last_claim", "tenure_months",
                "open_complaints", "num_policies", "total_claim_amount"]


def resolve_data_path():
    """Prefer the real Kaggle file, fall back to synthetic data."""
    if config.REAL_DATA_FILE.exists():
        return config.REAL_DATA_FILE, "real"
    if not config.SYNTHETIC_DATA_FILE.exists():
        from src.generate_data import generate
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        generate().to_csv(config.SYNTHETIC_DATA_FILE, index=False)
    return config.SYNTHETIC_DATA_FILE, "synthetic"


def load_raw(path=None) -> pd.DataFrame:
    if path is None:
        path, source = resolve_data_path()
        if source == "real":
            log.info("Using REAL data: %s", path.name)
        else:
            log.warning("Real data not found. Using SYNTHETIC data - do not quote these numbers on a resume.")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, fix types, remove duplicates, impute missing values."""
    df = df.rename(columns=COLUMN_MAP).copy()

    missing_cols = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    before = len(df)
    df = df.drop_duplicates(subset="customer_id" if "customer_id" in df.columns else None)
    log.info("Dropped %d duplicate rows", before - len(df))

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Median imputation for the (few) missing numeric values.
    # Note: income == 0 is meaningful (unemployed / retired) so it is NOT treated as missing.
    for c in NUMERIC_COLS:
        n_missing = int(df[c].isna().sum())
        if n_missing:
            df[c] = df[c].fillna(df[c].median())
            log.info("Imputed %d missing values in %s with the median", n_missing, c)

    # Basic sanity filters
    df = df[(df["monthly_premium"] > 0) & (df["tenure_months"] >= 0)].reset_index(drop=True)
    return df


def load_clean(path=None) -> pd.DataFrame:
    return clean(load_raw(path))


def data_source() -> str:
    return resolve_data_path()[1]
