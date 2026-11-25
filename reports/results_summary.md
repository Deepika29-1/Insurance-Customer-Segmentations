# Results summary

**Data source:** real | **Customers:** 9,134 | **Clusters (k):** 6

## Model quality
- Silhouette score: **0.168**
- Stability (mean ARI over 20 bootstrap refits): **0.980** (min 0.962)
- Explainability: a depth-3 decision tree recovers the segments with **71%** cross-validated accuracy (rules in `reports/segment_rules.txt`; importance in `reports/feature_importance.csv`)

## Segments
| Segment | Customers | % customers | % premium | Avg monthly premium | Claims/premium | Tenure (mo) | Avg CLV |
|---|---|---|---|---|---|---|---|
| Core Mid-Value | 1726 | 18.9% | 26.7% | 132 | 0.4 | 47 | 15730 |
| Low-Value Steady | 2450 | 26.8% | 22.6% | 78 | 0.4 | 48 | 4744 |
| High-Claim Risk Watch | 1828 | 20.0% | 19.6% | 91 | 0.56 | 49 | 7094 |
| Core Mid-Value (2) | 1262 | 13.8% | 12.5% | 84 | 0.08 | 49 | 7376 |
| Low-Value Steady (3) | 1209 | 13.2% | 11.9% | 84 | 0.38 | 46 | 6507 |
| Low-Value Steady (2) | 659 | 7.2% | 6.7% | 87 | 0.39 | 48 | 6376 |

## Recommended actions
- **Core Mid-Value**: Nurture: targeted upgrade offers (coverage tiers, extra policies) to move them up in value.
- **Low-Value Steady**: Serve efficiently: low-cost digital engagement, selective upsell, avoid heavy discounting.
- **High-Claim Risk Watch**: Manage risk: underwriting review, repricing at renewal, claims-prevention programmes.
- **Core Mid-Value (2)**: Nurture: targeted upgrade offers (coverage tiers, extra policies) to move them up in value.
- **Low-Value Steady (3)**: Serve efficiently: low-cost digital engagement, selective upsell, avoid heavy discounting.
- **Low-Value Steady (2)**: Serve efficiently: low-cost digital engagement, selective upsell, avoid heavy discounting.

## Resume bullets (numbers filled from this run)
- Segmented 9,134 insurance customers into 6 groups using K-Means on engineered value, risk and tenure features (silhouette 0.17; bootstrap stability ARI 0.98).
- Explained segments with a decision tree (71% cross-validated accuracy) and Random Forest feature importance.
- Profiled segments by premium, claims-to-premium ratio and CLV; the "Core Mid-Value" segment contributes 27% of premium from 19% of customers.
- Mapped each segment to a retention, cross-sell or risk-review action and exported a segment-level dataset for Power BI dashboarding.
