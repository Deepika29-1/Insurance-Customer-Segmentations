"""Model selection, fitting, stability checks and segment explainability."""
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from src import config

log = logging.getLogger(__name__)


def evaluate_k(Xs: pd.DataFrame, k_range=None, random_state=config.RANDOM_STATE) -> pd.DataFrame:
    """Fit K-Means for each k and collect internal validity metrics."""
    k_range = k_range or config.K_RANGE
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Xs)
        shares = np.bincount(km.labels_) / len(Xs)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(Xs, km.labels_),
            "davies_bouldin": davies_bouldin_score(Xs, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(Xs, km.labels_),
            "min_segment_share": shares.min(),
        })
    return pd.DataFrame(rows)


def choose_k(metrics: pd.DataFrame, min_share=config.MIN_SEGMENT_SHARE, forced_k=config.FORCE_K) -> int:
    """Highest silhouette among k values with no tiny (non-actionable) segment.

    Business note: k=2 often wins on silhouette but is too coarse to act on, so
    we require k >= 3 unless nothing else qualifies.
    """
    if forced_k is not None:
        return int(forced_k)
    valid = metrics[(metrics["min_segment_share"] >= min_share) & (metrics["k"] >= 3)]
    if valid.empty:
        valid = metrics
    return int(valid.sort_values("silhouette", ascending=False).iloc[0]["k"])


def fit_kmeans(Xs: pd.DataFrame, k: int, random_state=config.RANDOM_STATE) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(Xs)


def explain_segments(Xs: pd.DataFrame, seg_labels, max_depth=3, random_state=config.RANDOM_STATE) -> dict:
    """Explain the segments with a shallow decision tree and Random Forest importance.

    A tree that predicts the segment from the features shows the segments are separable
    (cross-validated accuracy) and gives plain-English rules. Thresholds are in z-scores.
    """
    y = np.asarray(seg_labels, dtype=str)  # plain array avoids a pandas 3 / sklearn dtype error
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    acc = cross_val_score(tree, Xs, y, cv=5, scoring="accuracy").mean()
    tree.fit(Xs, y)
    rules = export_text(tree, feature_names=list(Xs.columns))
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1).fit(Xs, y)
    importance = pd.Series(rf.feature_importances_, index=Xs.columns).sort_values(ascending=False)
    return {"tree_accuracy": float(acc), "rules": rules, "importance": importance}


def stability_score(Xs: pd.DataFrame, k: int, reference_labels, n_boot=20, frac=0.8,
                    random_state=config.RANDOM_STATE) -> dict:
    """Refit on random 80% subsamples and compare predictions on all rows (ARI)."""
    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(Xs), size=int(len(Xs) * frac), replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(0, 10_000))).fit(Xs.iloc[idx])
        scores.append(adjusted_rand_score(reference_labels, km.predict(Xs)))
    return {"mean_ari": float(np.mean(scores)), "min_ari": float(np.min(scores)), "n_boot": n_boot}
