# Split robustness - NSL-KDD (multiclass, F1 macro, raw decision)

Three independent train/validation/test splits (data seeds 1, 2, 3) of the same data, 60,000 fit rows, training seed 42, one run per cell. The benchmark itself uses one more split (data seed 42) with 3 training seeds.

| Model | split 1 | split 2 | split 3 | mean | std over splits | best of XGBoost / LightGBM / Aegis_net (levels 12) on |
|---|---|---|---|---|---|---|
| XGBoost | 0.9504 | 0.9524 | 0.9209 | 0.9412 | 0.0144 | 1 of 3 |
| LightGBM | 0.9642 | 0.9488 | 0.9133 | 0.9421 | 0.0213 | 1 of 3 |
| **Aegis_net** (levels 12, published) | 0.9544 | 0.9243 | 0.9218 | 0.9335 | 0.0148 | 1 of 3 |
| Aegis_net, levels 8 (first MRBE configuration) | 0.9394 | 0.9334 | 0.9267 | 0.9332 | 0.0052 | - |

NSL-KDD was not used to tune any hyper-parameter of Aegis_net (it was only looked at to confirm the choice of `levels`, where it made no difference), so this is the cleaner check.
