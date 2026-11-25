"""Interactive segment dashboard.  Run:  streamlit run dashboard/app.py
(Run `python run_pipeline.py` first so data/processed/customers_segmented.csv exists.)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "customers_segmented.csv"

st.set_page_config(page_title="Insurance Customer Segments", layout="wide")
st.title("Insurance Customer Segmentation Dashboard")

if not DATA.exists():
    st.error("Segmented data not found. Run `python run_pipeline.py` first.")
    st.stop()


@st.cache_data
def load():
    return pd.read_csv(DATA)


df = load()

# ---- Sidebar filters
st.sidebar.header("Filters")
segs = st.sidebar.multiselect("Segment", sorted(df["segment"].unique()), default=sorted(df["segment"].unique()))
filters = {}
for col, label in [("coverage", "Coverage"), ("sales_channel", "Sales channel"), ("state", "State")]:
    if col in df.columns:
        picked = st.sidebar.multiselect(label, sorted(df[col].dropna().unique()))
        if picked:
            filters[col] = picked
view = df[df["segment"].isin(segs)]
for col, picked in filters.items():
    view = view[view[col].isin(picked)]

if view.empty:
    st.warning("No customers match the current filters.")
    st.stop()

# ---- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(view):,}")
c2.metric("Annual premium", f"${view['annual_premium'].sum():,.0f}")
c3.metric("Avg CLV", f"${view['clv'].mean():,.0f}")
c4.metric("Claims / premium", f"{view['total_claim_amount'].sum() / view['annual_premium'].sum():.2f}")

# ---- Charts
summary = view.groupby("segment").agg(
    customers=("customer_id", "count"),
    premium=("annual_premium", "sum"),
    avg_clv=("clv", "mean"),
    claim_ratio=("claim_to_premium_ratio", "mean"),
    tenure=("tenure_months", "mean"),
).sort_values("premium", ascending=False)

left, right = st.columns(2)
with left:
    st.subheader("Customers vs premium share")
    share = pd.DataFrame({
        "% of customers": summary["customers"] / summary["customers"].sum() * 100,
        "% of premium": summary["premium"] / summary["premium"].sum() * 100,
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    share.plot(kind="bar", ax=ax, color=["#1f4e79", "#e07b39"])
    ax.set_xlabel("")
    ax.set_ylabel("%")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    st.pyplot(fig)
with right:
    st.subheader("Average CLV by segment")
    fig, ax = plt.subplots(figsize=(6, 4))
    summary["avg_clv"].plot(kind="bar", ax=ax, color="#3a9d6b")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    st.pyplot(fig)

st.subheader("Segment summary and recommended action")
actions = df.drop_duplicates("segment").set_index("segment")["recommended_action"]
table = summary.join(actions).round(2)
st.dataframe(table, width="stretch")

st.subheader("Customer list")
st.dataframe(view[["customer_id", "segment", "monthly_premium", "total_claim_amount", "tenure_months", "clv"]].head(500),
             width="stretch")
st.download_button("Download filtered customers (CSV)", view.to_csv(index=False), "segmented_customers.csv")
