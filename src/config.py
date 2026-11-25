"""Central configuration for the segmentation project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

# Drop the Kaggle file here to run on real data. If it is missing, the pipeline
# falls back to the synthetic file produced by src/generate_data.py.
REAL_DATA_FILE = RAW_DIR / "auto_insurance_clv.csv"
SYNTHETIC_DATA_FILE = RAW_DIR / "synthetic_auto_insurance.csv"

RANDOM_STATE = 42

# Features used for clustering (numeric, behavioural and value based)
CLUSTER_FEATURES = [
    "monthly_premium",
    "total_claim_amount",
    "claim_to_premium_ratio",
    "income",
    "tenure_months",
    "months_since_last_claim",
    "num_policies",
    "open_complaints",
    "clv",
]

# Heavily right-skewed columns get a log1p transform before scaling
LOG_FEATURES = ["total_claim_amount", "claim_to_premium_ratio", "income", "clv", "monthly_premium"]

# Categorical columns used only to PROFILE segments (not to cluster)
PROFILE_CATEGORICALS = ["coverage", "vehicle_class", "sales_channel", "employment_status", "policy_type"]

# Range of k to evaluate
K_RANGE = range(2, 9)

# Set to an int to force a specific k; None lets the pipeline choose by silhouette
FORCE_K = None
# A segment smaller than this share of customers is considered too small to act on
MIN_SEGMENT_SHARE = 0.05
