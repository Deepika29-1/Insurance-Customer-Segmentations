"""Segment profiling, business naming and recommended actions."""
import numpy as np
import pandas as pd

from src import config

# (base name) -> recommended action
ACTIONS = {
    "Loyal High-Value": "Protect and reward: loyalty pricing, multi-policy bundles, proactive renewal outreach.",
    "High-Claim Risk Watch": "Manage risk: underwriting review, repricing at renewal, claims-prevention programmes.",
    "New Customers": "Onboard and grow: welcome journey, early cross-sell, first-year engagement to improve retention.",
    "Low-Value Steady": "Serve efficiently: low-cost digital engagement, selective upsell, avoid heavy discounting.",
    "Core Mid-Value": "Nurture: targeted upgrade offers (coverage tiers, extra policies) to move them up in value.",
}

# Order matters: the first matching rule wins.
def _label(idx: dict) -> str:
    if idx["claim_to_premium_ratio"] > 1.4:
        return "High-Claim Risk Watch"
    if idx["tenure_months"] < 0.5:
        return "New Customers"
    if idx["clv"] > 1.15 and idx["claim_to_premium_ratio"] < 1.0:
        return "Loyal High-Value"
    if idx["clv"] < 0.9:
        return "Low-Value Steady"
    return "Core Mid-Value"


def profile_segments(df: pd.DataFrame, labels) -> pd.DataFrame:
    """One row per cluster: size, value and risk metrics, indexed vs the overall average."""
    d = df.copy()
    d["cluster"] = np.asarray(labels)

    g = d.groupby("cluster")
    prof = pd.DataFrame({
        "customers": g.size(),
        "avg_monthly_premium": g["monthly_premium"].mean(),
        "avg_claim_amount": g["total_claim_amount"].mean(),
        "claim_to_premium_ratio": g["claim_to_premium_ratio"].mean(),
        "avg_income": g["income"].mean(),
        "tenure_months": g["tenure_months"].mean(),
        "avg_policies": g["num_policies"].mean(),
        "avg_open_complaints": g["open_complaints"].mean(),
        "clv": g["clv"].mean(),
    })
    # Portfolio-level loss ratio proxy: total claims / total annual premium
    prof["loss_ratio_proxy"] = g["total_claim_amount"].sum() / g["annual_premium"].sum()
    prof["share_of_customers"] = prof["customers"] / prof["customers"].sum()
    prof["share_of_premium"] = g["annual_premium"].sum() / d["annual_premium"].sum()
    prof["share_of_clv"] = g["clv"].sum() / d["clv"].sum()

    overall = {
        "claim_to_premium_ratio": d["claim_to_premium_ratio"].mean(),
        "tenure_months": d["tenure_months"].mean(),
        "clv": d["clv"].mean(),
    }
    names, used = {}, {}
    for c, row in prof.iterrows():
        idx = {k: row[k] / overall[k] for k in overall}
        base = _label(idx)
        used[base] = used.get(base, 0) + 1
        names[c] = base if used[base] == 1 else f"{base} ({used[base]})"
    prof["segment"] = pd.Series(names)
    prof["recommended_action"] = [ACTIONS[n.split(" (")[0]] for n in prof["segment"]]
    return prof.sort_values("share_of_premium", ascending=False)


def categorical_profile(df: pd.DataFrame, labels, columns=None) -> pd.DataFrame:
    """Most common category (and its share) per segment, for descriptive profiling."""
    columns = [c for c in (columns or config.PROFILE_CATEGORICALS) if c in df.columns]
    d = df.assign(cluster=np.asarray(labels))
    out = {}
    for c in columns:
        top = d.groupby("cluster")[c].agg(lambda s: s.value_counts().index[0])
        share = d.groupby("cluster")[c].agg(lambda s: s.value_counts(normalize=True).iloc[0])
        out[f"top_{c}"] = top + " (" + (share * 100).round(0).astype(int).astype(str) + "%)"
    return pd.DataFrame(out)
