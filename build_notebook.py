"""Builds notebooks/customer_segmentation.ipynb (run once; the notebook calls the src/ modules)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md("""# Insurance Customer Segmentation

**Business question:** Which customer groups exist in our policyholder base, and what should we do differently for each (retain, upsell, monitor)?

**Approach:** clean data, engineer insurance features, select k with several metrics, cluster with K-Means, validate against other methods and bootstrap stability, profile segments, and map each to an action.

> If `data/raw/auto_insurance_clv.csv` is missing, this notebook uses **synthetic** data. Replace it with the Kaggle file for real results."""),

code("""import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd
from IPython.display import Image, display
from src import config, data_prep, features, clustering, profiling, visualize
pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.width", 200)"""),

md("## 1. Load and clean"),
code("""source = data_prep.data_source()
print("Data source:", source)
df = features.add_features(data_prep.load_clean())
print(df.shape)
df[["monthly_premium","total_claim_amount","income","tenure_months","num_policies","clv"]].describe().T"""),

md("## 2. Explore"),
code("""display(Image(filename=str(visualize.plot_distributions(df))))"""),
md("Premium, claims, income and CLV are right-skewed, so they are log-transformed before scaling. Zero income is kept as a real value (unemployed or retired customers)."),

md("## 3. Feature engineering and scaling"),
code("""X = features.build_matrix(df)
Xs, scaler = features.scale(X)
display(Image(filename=str(visualize.plot_correlation(X))))"""),
md("Premium and CLV are correlated, which is expected. K-Means is distance based, so scaling matters more than dropping correlated columns here."),

md("## 4. Choose the number of clusters"),
code("""metrics = clustering.evaluate_k(Xs)
k = clustering.choose_k(metrics)
display(metrics.round(3))
print("Chosen k:", k)
display(Image(filename=str(visualize.plot_k_selection(metrics, k))))"""),
md("k is chosen by the highest silhouette among values with k >= 3 and no segment under 5% of customers. k=2 is too coarse to act on."),

md("## 5. Fit K-Means and profile segments"),
code("""km = clustering.fit_kmeans(Xs, k)
prof = profiling.profile_segments(df, km.labels_)
names = prof["segment"].to_dict()
seg = pd.Series(km.labels_).map(names).values
prof.reset_index(drop=True)[["segment","customers","share_of_customers","share_of_premium","avg_monthly_premium","loss_ratio_proxy","tenure_months","clv","recommended_action"]]"""),

md("## 6. Validate and explain the segmentation"),
code("""stab = clustering.stability_score(Xs, k, km.labels_)
print("Bootstrap stability:", {k_: round(v, 3) if isinstance(v, float) else v for k_, v in stab.items()})"""),
md("A high Adjusted Rand Index across bootstrap refits means the segments are not an artefact of one sample."),
code("""explain = clustering.explain_segments(Xs, seg)
print(f"Decision tree (depth 3) cross-validated accuracy: {explain['tree_accuracy']:.1%}")
print(explain["rules"])"""),
md("A shallow tree can predict the segment from the features, so the segments are separable and describable in plain rules. Thresholds are z-scores (0 = average customer)."),
code("""display(explain["importance"].round(3).to_frame("importance"))
display(Image(filename=str(visualize.plot_feature_importance(explain["importance"]))))"""),
md("Random Forest importance shows which features drive the segmentation most."),

md("## 7. Visualise the segments"),
code("""for fn in (visualize.plot_pca, visualize.plot_profile_heatmap, visualize.plot_radar):
    display(Image(filename=str(fn(Xs, seg))))
display(Image(filename=str(visualize.plot_size_vs_value(prof))))
display(Image(filename=str(visualize.plot_boxplots(df, seg))))
display(Image(filename=str(visualize.plot_categorical_mix(df, seg, "coverage"))))"""),

md("## 8. Business recommendations"),
code("""for r in prof.itertuples():
    print(f"{r.segment:<24} {r.share_of_customers:5.1%} of customers | {r.share_of_premium:5.1%} of premium\\n    -> {r.recommended_action}\\n")"""),

md("## 9. Score new customers with the saved model\nRun `python run_pipeline.py` first so `models/` exists."),
code("""from src.generate_data import generate
from src.predict import assign_segments
new = assign_segments(generate(n=300, seed=7))
new["segment"].value_counts()"""),

md("""## 10. Limitations and next steps
- Segments are descriptive; validate them with a business test (for example A/B test retention offers per segment).
- Join the churn model's predicted probability per segment to prioritise outreach.
- Add categorical features (coverage, vehicle class) with K-Prototypes or Gower distance as an extension.
- Re-run quarterly and monitor segment drift."""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/customer_segmentation.ipynb")
print("notebook written")
