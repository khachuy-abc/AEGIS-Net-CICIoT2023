# Split robustness - UNSW-NB15 (multiclass, F1 macro, raw decision)

Three independent train/validation/test splits (data seeds 1, 2, 3) of the same data, 60,000 fit rows, training seed 42, one run per cell. The benchmark itself uses one more split (data seed 42) with 3 training seeds.

| Model | split 1 | split 2 | split 3 | mean | std over splits | best of XGBoost / LightGBM / Aegis_net (levels 12) on |
|---|---|---|---|---|---|---|
| XGBoost | 0.7320 | 0.7288 | 0.7177 | 0.7262 | 0.0061 | 0 of 3 |
| LightGBM | 0.7337 | 0.7336 | 0.7246 | 0.7306 | 0.0043 | 0 of 3 |
| **Aegis_net** (levels 12, published) | 0.7575 | 0.7631 | 0.7400 | 0.7535 | 0.0098 | 3 of 3 |
| Aegis_net, levels 8 (first MRBE configuration) | 0.7322 | 0.7313 | 0.7195 | 0.7277 | 0.0058 | - |

These splits were also used during development of Aegis_net (for hyper-parameters other than `levels`), so they are not a fully untouched test.
