"""Generate a SYNTHETIC auto-insurance customer dataset.

The columns mirror the public Kaggle "Auto Insurance Customer Lifetime Value"
dataset so the same pipeline runs on either file. The data contains four hidden
customer groups so clustering has real structure to find.

IMPORTANT: results from synthetic data are for demonstration only. Run the
pipeline on the real Kaggle file before quoting any numbers on a resume.

Usage:  python -m src.generate_data
"""
import numpy as np
import pandas as pd

from src import config


def generate(n: int = 6000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # name: (weight, premium mean/sd, claim scale, tenure range, policies range,
    #        income mean, P(employed), complaint rate)
    groups = {
        "loyal":   dict(w=0.28, prem=(135, 22), claim=250,  tenure=(48, 99), pol=(3, 9), inc=(78000, 18000), emp=0.97, comp=0.03),
        "risky":   dict(w=0.20, prem=(125, 25), claim=1150, tenure=(18, 80), pol=(1, 4), inc=(48000, 20000), emp=0.85, comp=0.35),
        "new":     dict(w=0.27, prem=(82, 14),  claim=380,  tenure=(0, 14),  pol=(1, 3), inc=(52000, 22000), emp=0.88, comp=0.08),
        "steady":  dict(w=0.25, prem=(72, 12),  claim=480,  tenure=(20, 70), pol=(1, 3), inc=(28000, 15000), emp=0.55, comp=0.10),
    }
    names = list(groups)
    probs = [groups[g]["w"] for g in names]
    labels = rng.choice(names, size=n, p=probs)

    rows = []
    for g in labels:
        p = groups[g]
        premium = max(61, rng.normal(*p["prem"]))
        tenure = int(rng.integers(*p["tenure"]))
        policies = int(rng.integers(*p["pol"]))
        employed = rng.random() < p["emp"]
        income = max(10000, rng.normal(*p["inc"])) if employed else 0
        claims = rng.gamma(shape=2.0, scale=p["claim"] / 2.0) + rng.normal(0, 40)
        claims = max(0.5, claims)
        months_last_claim = int(min(35, rng.exponential(10 if g != "risky" else 5)))
        complaints = int(rng.poisson(p["comp"] * 3))
        complaints = min(complaints, 5)
        clv = premium * 12 * (1 + tenure / 36) * (0.7 + 0.15 * policies) * rng.lognormal(0, 0.18)
        rows.append((premium, claims, tenure, policies, income, months_last_claim, complaints, clv, employed))

    df = pd.DataFrame(rows, columns=[
        "Monthly Premium Auto", "Total Claim Amount", "Months Since Policy Inception",
        "Number of Policies", "Income", "Months Since Last Claim",
        "Number of Open Complaints", "Customer Lifetime Value", "_employed"])

    df["Customer"] = [f"CU{100000 + i}" for i in range(n)]
    df["State"] = rng.choice(["California", "Oregon", "Arizona", "Nevada", "Washington"], n, p=[.34, .26, .19, .11, .10])
    df["Response"] = rng.choice(["No", "Yes"], n, p=[.86, .14])
    df["Coverage"] = rng.choice(["Basic", "Extended", "Premium"], n, p=[.6, .3, .1])
    df["Education"] = rng.choice(["High School or Below", "College", "Bachelor", "Master", "Doctor"], n, p=[.28, .27, .3, .11, .04])
    df["Effective To Date"] = pd.to_datetime("2011-01-01") + pd.to_timedelta(rng.integers(0, 59, n), unit="D")
    df["EmploymentStatus"] = np.where(df["_employed"], "Employed", rng.choice(["Unemployed", "Retired", "Medical Leave"], n, p=[.6, .3, .1]))
    df["Gender"] = rng.choice(["F", "M"], n)
    df["Location Code"] = rng.choice(["Suburban", "Rural", "Urban"], n, p=[.6, .2, .2])
    df["Marital Status"] = rng.choice(["Married", "Single", "Divorced"], n, p=[.58, .27, .15])
    df["Policy Type"] = rng.choice(["Personal Auto", "Corporate Auto", "Special Auto"], n, p=[.74, .21, .05])
    df["Policy"] = df["Policy Type"].str.split().str[0] + " L" + rng.integers(1, 4, n).astype(str)
    df["Renew Offer Type"] = rng.choice(["Offer1", "Offer2", "Offer3", "Offer4"], n)
    df["Sales Channel"] = rng.choice(["Agent", "Branch", "Call Center", "Web"], n, p=[.37, .28, .20, .15])
    df["Vehicle Class"] = rng.choice(["Four-Door Car", "Two-Door Car", "SUV", "Sports Car", "Luxury SUV", "Luxury Car"], n, p=[.5, .19, .2, .05, .03, .03])
    df["Vehicle Size"] = rng.choice(["Medsize", "Small", "Large"], n, p=[.7, .2, .1])

    # Inject a little realistic mess so the cleaning step has something to do
    miss = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[miss, "Income"] = np.nan
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)  # a few duplicates

    df = df.drop(columns="_employed")
    cols = ["Customer", "State", "Customer Lifetime Value", "Response", "Coverage", "Education",
            "Effective To Date", "EmploymentStatus", "Gender", "Income", "Location Code", "Marital Status",
            "Monthly Premium Auto", "Months Since Last Claim", "Months Since Policy Inception",
            "Number of Open Complaints", "Number of Policies", "Policy Type", "Policy", "Renew Offer Type",
            "Sales Channel", "Total Claim Amount", "Vehicle Class", "Vehicle Size"]
    return df[cols]


if __name__ == "__main__":
    out = generate()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.SYNTHETIC_DATA_FILE, index=False)
    print(f"Wrote {len(out):,} rows to {config.SYNTHETIC_DATA_FILE}")
