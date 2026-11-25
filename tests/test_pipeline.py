import numpy as np
import pandas as pd
import pytest

from src import clustering, config, features, profiling
from src.data_prep import clean
from src.generate_data import generate


@pytest.fixture(scope="module")
def raw():
    return generate(n=600, seed=1)


@pytest.fixture(scope="module")
def df(raw):
    return features.add_features(clean(raw))


def test_clean_removes_duplicates_and_nulls(raw):
    out = clean(raw)
    assert out["customer_id"].is_unique
    assert out[["income", "clv", "monthly_premium"]].isna().sum().sum() == 0
    assert (out["monthly_premium"] > 0).all()


def test_income_zero_is_kept(df):
    # Zero income is a real value (unemployed / retired), not missing data
    assert (df["income"] == 0).any()


def test_claim_ratio_feature(df):
    expected = df["total_claim_amount"] / (df["monthly_premium"] * 12)
    assert np.allclose(df["claim_to_premium_ratio"], expected)


def test_matrix_is_finite_and_scaled(df):
    X = features.build_matrix(df)
    Xs, _ = features.scale(X)
    assert np.isfinite(Xs.values).all()
    assert np.allclose(Xs.mean().values, 0, atol=1e-8)


def test_choose_k_respects_min_segment_share():
    metrics = pd.DataFrame({
        "k": [2, 3, 4], "silhouette": [0.50, 0.45, 0.40],
        "min_segment_share": [0.30, 0.02, 0.20]})
    # k=3 has a tiny segment, k=2 is too coarse -> k=4 expected
    assert clustering.choose_k(metrics, forced_k=None) == 4


def test_forced_k_overrides():
    metrics = pd.DataFrame({"k": [2, 3], "silhouette": [0.5, 0.4], "min_segment_share": [0.3, 0.3]})
    assert clustering.choose_k(metrics, forced_k=7) == 7


def test_profile_names_are_unique_and_actionable(df):
    Xs, _ = features.scale(features.build_matrix(df))
    km = clustering.fit_kmeans(Xs, 4)
    prof = profiling.profile_segments(df, km.labels_)
    assert prof["segment"].is_unique
    assert prof["recommended_action"].str.len().gt(10).all()
    assert abs(prof["share_of_customers"].sum() - 1) < 1e-9
    assert abs(prof["share_of_premium"].sum() - 1) < 1e-9


def test_explain_segments_returns_accuracy_rules_importance(df):
    Xs, _ = features.scale(features.build_matrix(df))
    km = clustering.fit_kmeans(Xs, 4)
    prof = profiling.profile_segments(df, km.labels_)
    seg = pd.Series(km.labels_).map(prof["segment"].to_dict()).values
    out = clustering.explain_segments(Xs, seg)
    assert out["tree_accuracy"] > 0.5
    assert "class:" in out["rules"]
    assert set(out["importance"].index) == set(Xs.columns)
