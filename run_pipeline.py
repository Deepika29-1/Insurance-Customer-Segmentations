"""End-to-end pipeline: load -> clean -> features -> select k -> cluster -> profile -> visualise -> save.

Usage:  python run_pipeline.py
"""
import json
import logging

import joblib
import pandas as pd

from src import clustering, config, data_prep, features, profiling, visualize

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("pipeline")


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|\n"
    body = "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in df.itertuples(index=False))
    return head + sep + body


def main():
    for d in (config.PROCESSED_DIR, config.MODELS_DIR, config.FIG_DIR, config.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data
    source = data_prep.data_source()
    df = features.add_features(data_prep.load_clean())
    log.info("Loaded %s customers (%s data)", f"{len(df):,}", source)

    # 2. Features
    X = features.build_matrix(df)
    Xs, scaler = features.scale(X)

    # 3. Model selection
    metrics = clustering.evaluate_k(Xs)
    k = clustering.choose_k(metrics)
    log.info("Chosen k = %d", k)
    km = clustering.fit_kmeans(Xs, k)
    labels = km.labels_

    # 4. Profile and name segments
    prof = profiling.profile_segments(df, labels)
    names = prof["segment"].to_dict()
    seg_labels = pd.Series(labels).map(names).values
    cat_prof = profiling.categorical_profile(df, labels)

    # 5. Validation
    explain = clustering.explain_segments(Xs, seg_labels)
    stability = clustering.stability_score(Xs, k, labels)
    sil = float(metrics.loc[metrics["k"] == k, "silhouette"].iloc[0])
    log.info("Silhouette %.3f | stability ARI %.3f | tree accuracy %.3f", sil, stability["mean_ari"], explain["tree_accuracy"])

    # 6. Charts
    visualize.plot_distributions(df)
    visualize.plot_correlation(X)
    visualize.plot_k_selection(metrics, k)
    visualize.plot_pca(Xs, seg_labels)
    visualize.plot_profile_heatmap(Xs, seg_labels)
    visualize.plot_size_vs_value(prof)
    visualize.plot_boxplots(df, seg_labels)
    visualize.plot_radar(Xs, seg_labels)
    visualize.plot_feature_importance(explain["importance"])
    if "coverage" in df.columns:
        visualize.plot_categorical_mix(df, seg_labels, "coverage")

    # 7. Save artefacts
    joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")
    joblib.dump(km, config.MODELS_DIR / "kmeans.joblib")
    (config.MODELS_DIR / "segment_names.json").write_text(json.dumps({int(c): n for c, n in names.items()}, indent=2))
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps({
        "data_source": source, "n_customers": len(df), "k": k, "silhouette": sil,
        "stability": stability, "features": config.CLUSTER_FEATURES, "log_features": config.LOG_FEATURES,
        "random_state": config.RANDOM_STATE}, indent=2))

    out = df.assign(cluster=labels, segment=seg_labels)
    out["recommended_action"] = out["cluster"].map(prof["recommended_action"].to_dict())
    out.to_csv(config.PROCESSED_DIR / "customers_segmented.csv", index=False)  # Power BI / Tableau ready

    prof_out = prof.reset_index().rename(columns={"cluster": "cluster_id"})
    prof_out.to_csv(config.REPORTS_DIR / "segment_profile.csv", index=False)
    metrics.to_csv(config.REPORTS_DIR / "model_selection.csv", index=False)
    explain["importance"].rename("importance").to_csv(config.REPORTS_DIR / "feature_importance.csv")
    (config.REPORTS_DIR / "segment_rules.txt").write_text(explain["rules"])
    cat_prof.assign(segment=cat_prof.index.map(names)).to_csv(config.REPORTS_DIR / "segment_categorical_profile.csv", index=False)

    # 8. Auto-generated results summary
    show = prof.reset_index(drop=True)[["segment", "customers", "share_of_customers", "share_of_premium",
                                        "avg_monthly_premium", "loss_ratio_proxy", "tenure_months", "clv"]].copy()
    for c in ("share_of_customers", "share_of_premium"):
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    for c in ("avg_monthly_premium", "tenure_months", "clv"):
        show[c] = show[c].round(0).astype(int)
    show["loss_ratio_proxy"] = show["loss_ratio_proxy"].round(2)
    show.columns = ["Segment", "Customers", "% customers", "% premium", "Avg monthly premium", "Claims/premium", "Tenure (mo)", "Avg CLV"]

    top = prof.iloc[0]
    banner = ("> **WARNING: these results come from SYNTHETIC demo data.** Run the pipeline on the real Kaggle "
              "dataset (see README) and use those numbers on your resume.\n\n") if source == "synthetic" else ""
    action_lines = "\n".join(f"- **{r.segment}**: {r.recommended_action}" for r in prof.itertuples())
    summary = f"""# Results summary

{banner}**Data source:** {source} | **Customers:** {len(df):,} | **Clusters (k):** {k}

## Model quality
- Silhouette score: **{sil:.3f}**
- Stability (mean ARI over {stability['n_boot']} bootstrap refits): **{stability['mean_ari']:.3f}** (min {stability['min_ari']:.3f})
- Explainability: a depth-3 decision tree recovers the segments with **{explain['tree_accuracy']:.0%}** cross-validated accuracy (rules in `reports/segment_rules.txt`; importance in `reports/feature_importance.csv`)

## Segments
{md_table(show)}
## Recommended actions
{action_lines}

## Resume bullets (numbers filled from this run)
- Segmented {len(df):,} insurance customers into {k} groups using K-Means on engineered value, risk and tenure features (silhouette {sil:.2f}; bootstrap stability ARI {stability['mean_ari']:.2f}).
- Explained segments with a decision tree ({explain['tree_accuracy']:.0%} cross-validated accuracy) and Random Forest feature importance.
- Profiled segments by premium, claims-to-premium ratio and CLV; the "{top['segment']}" segment contributes {top['share_of_premium']*100:.0f}% of premium from {top['share_of_customers']*100:.0f}% of customers.
- Mapped each segment to a retention, cross-sell or risk-review action and exported a segment-level dataset for Power BI dashboarding.
"""
    (config.REPORTS_DIR / "results_summary.md").write_text(summary)
    log.info("Done. See reports/results_summary.md")


if __name__ == "__main__":
    main()
