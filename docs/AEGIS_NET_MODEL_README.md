# Aegis_net

A deep-learning model for tabular network-intrusion data, built for the **rare attack classes** that other neural networks lose,
together with a benchmark that trains **15 models** under one identical protocol and compares the strongest ones.

```
python main.py --data_path your_data.csv --out outputs
```

That one command prepares the data, trains every model on 3 tasks, writes the tables, and ends with the **final comparison**:
the 2 strongest machine-learning models + the 2 strongest deep-learning models + Aegis_net (`outputs/results/finalists.md`).

---

## 1. What Aegis_net is (and what is new)

**The problem.** Rare intrusion classes are usually recognised by an *exact, narrow value band* of one or two features
(one attack family always sends payloads inside a band 1 % wide). Gradient-boosted trees isolate such a band with two splits.
A neural network that sees the raw number, or a smooth function of it (piece-wise-linear, periodic, Chebyshev encodings), has to learn a
very sharp function of a continuous input. It blurs the band into its neighbours, and the rare class is lost.
Measured on UNSW-NB15: raising XGBoost's number of histogram bins from 16 to 256 alone gives +0.04 F1 macro, and giving Aegis_net 16 instead of 4,096 bins costs 0.09 -
*threshold resolution* is the gap between trees and networks on rare classes.

**The model** (`aegis_net/models/dl/aegis_net.py`, one file):

1. **MRBE - Multi-Resolution Bin Embedding.** Each feature is cut into 2^12 = 4,096 quantile bins. The embedding of a value is the **sum over the dyadic
   hierarchy** of its bin (whole range, halves, quarters, ... finest bin) of one learned vector per level:
   `e(x) = W x + sum_{l=0..12} T_l[ bin(x) >> (12 - l) ]`. Coarse levels share statistical strength, fine levels memorise narrow bands,
   and a value is encoded *exactly* (a lookup, not a smooth function).
2. **Coarse-to-fine shrinkage.** The penalty `sum_l lambda * 2^l * mean(T_l^2)` pulls finer levels harder towards zero, so a fine bin that only a few
   training rows touch falls back on its parent bin instead of over-fitting. Rare classes get a prior "look like the coarse neighbourhood unless the data
   proves otherwise" - precise where there is evidence, smooth where there is none.
3. **BatchEnsemble of 4 members** on top (shared weights, per-member rank-1 scalings), whose averaged probabilities steady the rare-class decisions
   from one seed to the next.

**Novelty, stated honestly.** The pieces have ancestors: quantile-bin embeddings (Gorishniy et al., NeurIPS 2022), multi-resolution lookup tables summed over
levels (Instant-NGP, Mueller et al. 2022, for 3-D graphics), BatchEnsemble / TabM (Wen et al. 2020; Gorishniy et al., ICLR 2025). The combination - dyadic
multi-resolution bins with hierarchical shrinkage as the tabular feature embedding, ensembled - is, to the author's knowledge, not published, but that is not a
proof of uniqueness. The earlier version of Aegis_net (a Chebyshev/threshold encoder plus a class-balanced retrieval memory) was **removed**: measured
ablations showed that the retrieval memory added ~0 F1 and caused rare-class false alarms.

Rejected during development (all measured, none kept): retrieval / boundary-importance memory, hard-negative loss, hashed cross-feature memory, soft decision
forests, leaf embeddings, balanced bagging, GLU blocks, pair-grid embeddings, level dropout, class-balanced bin placement, a wider (512) network.

## 2. The 15 models (one file each)

| Family | Model | File |
|---|---|---|
| ML | Logistic Regression | `aegis_net/models/ml/logreg.py` |
| ML | Gaussian Naive Bayes | `aegis_net/models/ml/naive_bayes.py` |
| ML | Decision Tree | `aegis_net/models/ml/decision_tree.py` |
| ML | Random Forest | `aegis_net/models/ml/random_forest.py` |
| ML | Extra Trees | `aegis_net/models/ml/extra_trees.py` |
| ML | XGBoost | `aegis_net/models/ml/xgboost.py` |
| ML | LightGBM | `aegis_net/models/ml/lightgbm.py` |
| ML | CatBoost | `aegis_net/models/ml/catboost.py` |
| DL | Residual MLP | `aegis_net/models/dl/mlp_res.py` |
| DL | 1-D CNN | `aegis_net/models/dl/cnn1d.py` |
| DL | Bi-GRU | `aegis_net/models/dl/bigru.py` |
| DL | ChebyKAN | `aegis_net/models/dl/chebykan.py` |
| DL | FT-Transformer | `aegis_net/models/dl/ft_transformer.py` |
| DL | TabM (the `tabm` package) | `aegis_net/models/dl/tabm.py` |
| DL | **Aegis_net** | `aegis_net/models/dl/aegis_net.py` |

`aegis_net/models/registry.py` lists them; adding a model = adding one file and one line there.
There are many more models in the literature (TabNet, TabTransformer, SAINT, NODE, TabR, ModernNCA, TabPFN, RealMLP, ...); the 15 above are the ones that
run in this benchmark, and no claim is made about models that were not run.

## 3. Protocol (identical for every model)

* Same split (stratified blocks, seed 42), same fit rows, same sample weights (`weight ~ class_frequency^-0.5`).
* Deep models: **exactly 20 epochs**, AdamW + cosine schedule, weighted focal loss + label smoothing 0.05, EMA weights, the best epoch is chosen by macro-F1
  on the validation split. Trees: boosting rounds chosen by early stopping on the same validation split.
* Three tasks: `binary` (benign vs attack), `attack_type` (attacks only), `multiclass` (everything at once).
* Metric: **F1 macro** (raw argmax) plus a *calibrated* F1 (a per-class log-bias fitted on the validation split, the same for every model).
  Also accuracy, balanced accuracy, MCC, and the F1 of every class.
* Final comparison: the **2 strongest ML + 2 strongest DL models** (by mean rank of F1 macro over the tasks, Aegis_net excluded) + Aegis_net.

## 4. Results

Run on a CPU-only machine (2 cores), so the **reduced protocol** below was used; the official one (500,000 fit rows, 5 seeds) is what `settings.py` defaults to.
Every model: 20 epochs (deep) / early-stopped rounds (trees), **60,000 fit rows, batch size 512, 3 seeds (42, 43, 44)**, one data split (seed 42).
F1 macro, raw argmax decision, mean ± std over the 3 seeds, `(#k)` = rank among the 15 models on that task. ★ = picked for the final comparison.
Full tables (accuracy, balanced accuracy, MCC, per-class F1, fit time, parameters): `benchmark_outputs/<dataset>/results/summary.md`, `benchmark_outputs/<dataset>/results/per_model/`.

### UNSW-NB15 (10 classes; rarest class Worms: 23 test rows)

| # | Model | Type | binary | attack_type | multiclass | mean rank |
|---|---|---|---|---|---|---|
| 1 | **Aegis_net** | DL | 0.9064 ± 0.0013 (#1) | 0.8183 ± 0.0017 (#1) | 0.7297 ± 0.0007 (#1) | 1.00 |
| 2 | XGBoost ★ | ML | 0.9021 ± 0.0032 (#2) | 0.7856 ± 0.0018 (#3) | 0.7098 ± 0.0025 (#2) | 2.33 |
| 3 | LightGBM ★ | ML | 0.9011 ± 0.0018 (#3) | 0.7907 ± 0.0050 (#2) | 0.7007 ± 0.0035 (#3) | 2.67 |
| 4 | CatBoost | ML | 0.8988 ± 0.0005 (#5) | 0.7491 ± 0.0061 (#4) | 0.6705 ± 0.0027 (#4) | 4.33 |
| 5 | RandomForest | ML | 0.8993 ± 0.0017 (#4) | 0.7482 ± 0.0036 (#5) | 0.6696 ± 0.0012 (#5) | 4.67 |
| 6 | ExtraTrees | ML | 0.8962 ± 0.0018 (#6) | 0.6822 ± 0.0149 (#7) | 0.6181 ± 0.0103 (#7) | 6.67 |
| 7 | DecisionTree | ML | 0.8750 ± 0.0012 (#13) | 0.7152 ± 0.0008 (#6) | 0.6305 ± 0.0017 (#6) | 8.33 |
| 8 | CNN1D ★ | DL | 0.8943 ± 0.0007 (#8) | 0.6685 ± 0.0046 (#8) | 0.6028 ± 0.0031 (#9) | 8.33 |
| 9 | FT-Transformer ★ | DL | 0.8933 ± 0.0049 (#9) | 0.6672 ± 0.0048 (#9) | 0.6031 ± 0.0067 (#8) | 8.67 |
| 10 | MLP-Res | DL | 0.8925 ± 0.0007 (#10) | 0.6633 ± 0.0014 (#10) | 0.6015 ± 0.0004 (#10) | 10.00 |
| 11 | TabM | DL | 0.8956 ± 0.0022 (#7) | 0.6542 ± 0.0012 (#12) | 0.5947 ± 0.0015 (#12) | 10.33 |
| 12 | ChebyKAN | DL | 0.8910 ± 0.0013 (#11) | 0.6621 ± 0.0054 (#11) | 0.5961 ± 0.0027 (#11) | 11.00 |
| 13 | BiGRU | DL | 0.8785 ± 0.0046 (#12) | 0.4810 ± 0.0110 (#14) | 0.4765 ± 0.0232 (#14) | 13.33 |
| 14 | LogReg | ML | 0.8722 ± 0.0000 (#14) | 0.5455 ± 0.0000 (#13) | 0.4836 ± 0.0000 (#13) | 13.33 |
| 15 | NaiveBayes | ML | 0.6958 ± 0.0000 (#15) | 0.2069 ± 0.0000 (#15) | 0.2103 ± 0.0000 (#15) | 15.00 |

Aegis_net is **rank 1 of 15 on all three tasks**: +0.004 over XGBoost on binary, +0.028 over LightGBM on attack_type, +0.020 over XGBoost on multiclass.
The final comparison (`benchmark_outputs/unsw_nb15/results/finalists.md`, per-class chart `finalists_per_class.png`) has XGBoost, LightGBM, CNN1D, FT-Transformer and Aegis_net;
on the rare classes of the multiclass task Aegis_net gets F1 0.711 / 0.771 / 0.233 / 0.776 on Worms / Backdoor / Analysis / Shellcode
(XGBoost 0.712 / 0.694 / 0.285 / 0.657, LightGBM 0.645 / 0.642 / 0.259 / 0.715; the two neural finalists, CNN1D and FT-Transformer, stay below 0.54 on all four).
Analysis stays hard for every model (best 0.285). Three other splits of the same data: Aegis_net 0.754 vs XGBoost 0.726 / LightGBM 0.731 (`benchmark_outputs/unsw_nb15/results/split_check.md`).

### NSL-KDD (5 classes; rarest class U2R: 17 test rows)

| # | Model | Type | binary | attack_type | multiclass | mean rank |
|---|---|---|---|---|---|---|
| 1 | XGBoost ★ | ML | 0.9969 ± 0.0000 (#1) | 0.9452 ± 0.0062 (#3) | 0.9661 ± 0.0026 (#1) | 1.67 |
| 2 | LightGBM ★ | ML | 0.9964 ± 0.0002 (#2) | 0.9636 ± 0.0057 (#1) | 0.9499 ± 0.0048 (#2) | 1.67 |
| 3 | CatBoost | ML | 0.9956 ± 0.0000 (#4) | 0.9587 ± 0.0057 (#2) | 0.9474 ± 0.0000 (#4) | 3.33 |
| 4 | **Aegis_net** | DL | 0.9964 ± 0.0002 (#3) | 0.9429 ± 0.0040 (#6) | 0.9468 ± 0.0020 (#5) | 4.67 |
| 5 | RandomForest | ML | 0.9946 ± 0.0002 (#7) | 0.9430 ± 0.0001 (#5) | 0.9492 ± 0.0021 (#3) | 5.00 |
| 6 | CNN1D ★ | DL | 0.9942 ± 0.0003 (#9) | 0.9451 ± 0.0117 (#4) | 0.9123 ± 0.0026 (#8) | 7.00 |
| 7 | ExtraTrees | ML | 0.9931 ± 0.0002 (#11) | 0.9269 ± 0.0001 (#7) | 0.9187 ± 0.0043 (#7) | 8.33 |
| 8 | MLP-Res ★ | DL | 0.9954 ± 0.0001 (#5) | 0.9202 ± 0.0061 (#11) | 0.9007 ± 0.0038 (#10) | 8.67 |
| 9 | TabM | DL | 0.9942 ± 0.0001 (#8) | 0.9267 ± 0.0029 (#8) | 0.8990 ± 0.0023 (#11) | 9.00 |
| 10 | ChebyKAN | DL | 0.9946 ± 0.0003 (#6) | 0.9197 ± 0.0055 (#12) | 0.9119 ± 0.0050 (#9) | 9.00 |
| 11 | DecisionTree | ML | 0.9935 ± 0.0002 (#10) | 0.9117 ± 0.0037 (#13) | 0.9456 ± 0.0039 (#6) | 9.67 |
| 12 | FT-Transformer | DL | 0.9928 ± 0.0005 (#12) | 0.9243 ± 0.0106 (#10) | 0.8589 ± 0.0097 (#12) | 11.33 |
| 13 | BiGRU | DL | 0.9829 ± 0.0017 (#13) | 0.9262 ± 0.0080 (#9) | 0.8480 ± 0.0129 (#13) | 11.67 |
| 14 | LogReg | ML | 0.9583 ± 0.0000 (#14) | 0.9042 ± 0.0000 (#14) | 0.8250 ± 0.0000 (#14) | 14.00 |
| 15 | NaiveBayes | ML | 0.9382 ± 0.0000 (#15) | 0.6541 ± 0.0000 (#15) | 0.4793 ± 0.0000 (#15) | 15.00 |

Here Aegis_net is **not** the best model: binary 0.9964 (#3, XGBoost 0.9969), attack_type 0.9429 (#6, LightGBM 0.9636), multiclass 0.9468 (#5, XGBoost 0.9661).
Gradient-boosted trees stay ahead on this dataset, by 0.0005 to 0.02. Part of that gap is U2R (17 test rows: one row is worth ~0.03 F1);
on three other splits the mean is Aegis_net 0.9335 vs XGBoost 0.9412 / LightGBM 0.9421, with a split-to-split std of 0.015-0.02
(`benchmark_outputs/nsl_kdd/results/split_check.md`). Aegis_net is still the best neural network on binary and multiclass, and has the highest balanced accuracy of all 15 models on the
multiclass task (0.980).

### What the numbers do and do not say

* **Aegis_net is far ahead of every other neural network on UNSW-NB15** (+0.13 F1 macro on multiclass and +0.15 on attack_type over the best of them) and **ahead of XGBoost, LightGBM and CatBoost there**.
  It is **not** ahead of the boosted trees on NSL-KDD. It was **not run on CICIoT2023** (not available here): run it with the command in section 6.
* **The comparison is tilted towards Aegis_net on UNSW-NB15**: Aegis_net was developed and its hyper-parameters (`levels`, `shrink`, learning-rate multiplier) were chosen on UNSW-NB15,
  while XGBoost, LightGBM and the neural baselines use fixed, untuned defaults (`settings.py`). `levels` was raised from 8 to 12 after looking at the benchmark split; the three other
  splits (not used to choose `levels`) show the same gain. NSL-KDD was not used to tune anything and is the cleaner check; there Aegis_net does not win.
* Reduced protocol: 60k fit rows and 3 seeds, one data split. Differences below ~0.01 F1 macro on small classes are within split-to-split noise.
* Cost: Aegis_net has ~4.1M parameters (almost all in the bin tables) and trains in ~4 minutes on CPU per run versus ~10 s for XGBoost / LightGBM; inference ~21 ms per 1,000 rows.
* Ablation (UNSW-NB15 multiclass, `benchmark_outputs/unsw_nb15/results/ablation.md`): without bins at all 0.616, 16 bins 0.637, 256 bins 0.699, 4,096 bins (default) 0.730, 16,384 bins 0.737 (4x slower);
  without the shrinkage 0.686 (-0.044); a single member instead of 4: 0.716 (-0.014).


## 5. Project layout

```
main.py                         the only command you need
aegis_net/
  settings.py                   every number that defines the protocol (one place)
  data/                         reading, cleaning, splitting, scaling the CSVs
  models/
    registry.py                 the list of the 15 models
    ml/<model>.py               one file per machine-learning model
    dl/<model>.py               one file per deep-learning model (aegis_net.py = the proposed one)
  training/trainer.py           the one training loop shared by every deep model
  evaluation/                   runner, metrics, calibration, results tables, final comparison
tools/                          dataset preparation (NSL-KDD, UNSW-NB15), synthetic data, ablation.py, split_check.py
benchmark_outputs/              the finished runs and tables quoted in this README (unsw_nb15/, nsl_kdd/: runs/*.json + results/)
tests/                          pytest (every model is trained for 2 epochs and must predict valid probabilities)
```

## 6. Running it

```bash
pip install -r requirements.txt

# any CSV with numeric / text features and a label column
python main.py --data_path data.csv --label_column label --out outputs

# CICIoT2023 (folder of CSVs; the 34 labels are grouped into 8 families automatically)
python main.py --data_path CICIoT2023/ --out outputs_ciciot

# UNSW-NB15
python tools/prepare_unsw_nb15.py --train UNSW_NB15_training-set.csv --test UNSW_NB15_testing-set.csv --out unsw_nb15/unsw_nb15.csv
python main.py --data_path unsw_nb15 --label_column attack_cat --out outputs_unsw

# NSL-KDD
python tools/prepare_nsl_kdd.py --train KDDTrain+.txt --out nsl_kdd/nsl_kdd.csv
python main.py --data_path nsl_kdd --label_column label --out outputs_nsl

python main.py --report_only --out outputs_unsw          # rebuild the tables from finished runs
python tools/ablation.py --cache outputs_unsw/cache --fit_rows 60000   # what each ingredient of Aegis_net is worth
pytest -q                                                # tests
```

Every finished run is stored as `<out>/runs/*.json`; re-running the command after a disconnect resumes where it stopped.
The official protocol is 20 epochs and 5 seeds (42-46); `--seeds`, `--fit_rows`, `--batch_size` exist for smaller machines and **must be reported** when
you quote the numbers.
