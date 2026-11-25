"""All project charts. Every function saves a PNG to reports/figures and returns the path."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from src import config

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = ["#1f4e79", "#e07b39", "#3a9d6b", "#b0413e", "#7b5ea7", "#8c8c8c", "#c2a23a", "#2e9bb3"]


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _colors(names):
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def plot_distributions(df: pd.DataFrame):
    cols = ["monthly_premium", "total_claim_amount", "income", "clv", "tenure_months", "num_policies"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(df[c], bins=40, ax=ax, color=PALETTE[0])
        ax.set_title(c.replace("_", " ").title())
        ax.set_xlabel("")
    fig.suptitle("Distributions of key customer features", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "01_feature_distributions.png")


def plot_correlation(X: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title("Correlation of clustering features (after log transform)")
    return _save(fig, "02_correlation_heatmap.png")


def plot_k_selection(metrics: pd.DataFrame, chosen_k: int):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    spec = [("inertia", "Elbow: inertia (lower is tighter)"),
            ("silhouette", "Silhouette (higher is better)"),
            ("davies_bouldin", "Davies-Bouldin (lower is better)")]
    for ax, (col, title) in zip(axes, spec):
        ax.plot(metrics["k"], metrics[col], marker="o", color=PALETTE[0])
        ax.axvline(chosen_k, color=PALETTE[1], ls="--", label=f"chosen k = {chosen_k}")
        ax.set_title(title)
        ax.set_xlabel("Number of clusters (k)")
        ax.legend()
    fig.tight_layout()
    return _save(fig, "03_k_selection.png")


def plot_pca(Xs: pd.DataFrame, seg_labels):
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    pts = pca.fit_transform(Xs)
    seg_labels = np.asarray(seg_labels)
    col = _colors(sorted(set(seg_labels)))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for s in sorted(set(seg_labels)):
        m = seg_labels == s
        ax.scatter(pts[m, 0], pts[m, 1], s=10, alpha=.55, color=col[s], label=s)
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0f}% of variance)")
    ax.set_title("Customer segments projected to 2D with PCA")
    ax.legend(markerscale=3, frameon=True)
    return _save(fig, "04_pca_segments.png")


def plot_profile_heatmap(Xs: pd.DataFrame, seg_labels):
    z = Xs.assign(segment=np.asarray(seg_labels)).groupby("segment").mean()
    fig, ax = plt.subplots(figsize=(11, 0.9 * len(z) + 2.5))
    sns.heatmap(z, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Mean z-score"})
    ax.set_ylabel("")
    ax.set_title("Segment profile: how each segment differs from the average customer (z-scores)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, "05_segment_profile_heatmap.png")


def plot_size_vs_value(prof: pd.DataFrame):
    p = prof.set_index("segment")[["share_of_customers", "share_of_premium", "share_of_clv"]] * 100
    p.columns = ["% of customers", "% of premium", "% of lifetime value"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    p.plot(kind="bar", ax=ax, color=PALETTE[:3], width=.8)
    ax.set_ylabel("Share (%)")
    ax.set_xlabel("")
    ax.set_title("Segment size versus revenue contribution")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, "06_segment_size_vs_value.png")


def plot_boxplots(df: pd.DataFrame, seg_labels):
    d = df.assign(segment=np.asarray(seg_labels))
    order = d.groupby("segment")["clv"].median().sort_values(ascending=False).index
    cols = [("monthly_premium", "Monthly premium"), ("claim_to_premium_ratio", "Claims / annual premium"),
            ("tenure_months", "Tenure (months)"), ("clv", "Customer lifetime value")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (c, t) in zip(axes.ravel(), cols):
        sns.boxplot(data=d, x="segment", y=c, order=order, ax=ax, palette=PALETTE[:len(order)], hue="segment",
                    legend=False, showfliers=False)
        ax.set_title(t)
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return _save(fig, "07_segment_boxplots.png")


def plot_radar(Xs: pd.DataFrame, seg_labels):
    z = Xs.assign(segment=np.asarray(seg_labels)).groupby("segment").mean()
    cats = [c.replace("_", "\n") for c in z.columns]
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    ang += ang[:1]
    col = _colors(list(z.index))
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"polar": True})
    for s, row in z.iterrows():
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(ang, vals, color=col[s], lw=2, label=s)
        ax.fill(ang, vals, color=col[s], alpha=.10)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_title("Segment fingerprints (z-scores)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    return _save(fig, "08_segment_radar.png")


def plot_categorical_mix(df: pd.DataFrame, seg_labels, column="coverage"):
    d = df.assign(segment=np.asarray(seg_labels))
    ct = pd.crosstab(d["segment"], d[column], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=PALETTE[:ct.shape[1]])
    ax.set_ylabel("% of segment")
    ax.set_xlabel("")
    ax.set_title(f"{column.replace('_', ' ').title()} mix by segment")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title=column.replace("_", " "), bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, "09_coverage_mix_by_segment.png")


def plot_feature_importance(importance: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.sort_values().plot(kind="barh", ax=ax, color=PALETTE[0])
    ax.set_title("Which features drive the segments? (Random Forest importance)")
    ax.set_xlabel("Importance")
    return _save(fig, "10_feature_importance.png")
