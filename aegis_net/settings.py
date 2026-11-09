"""
settings.py - the ONE place where every default number of the project lives.

Nothing else in the code base hard-codes epochs, seeds, row budgets or learning rates.
If you want to change the experiment protocol, change it here (or override it on the command line of main.py).

THE PROTOCOL (identical for every model, so the comparison is fair)
-------------------------------------------------------------------
* Every neural model trains for exactly EPOCHS = 20 epochs (no early stopping; the best epoch on the monitor split is kept).
* Every model (neural, tree, linear, Aegis_net) is run with exactly the 5 SEEDS below. No model gets more or fewer seeds.
* Every model learns from the SAME rows (FIT_ROWS rows, class-aware sample of the train split, identical for all models and seeds).
* Every model uses the SAME class weights, the SAME monitor split and is scored on the SAME test split.
* No model receives individual hyper-parameter tuning: each uses the fixed defaults written in this file / in its own class.
* Multi-class tasks are reported twice for EVERY model: with the raw argmax decision, and after one shared decision-calibration step
  (one log-bias per class fitted on the validation 'select' split, see evaluation/calibration.py). The binary task always uses a threshold
  picked on the same split. The test split is never used for any choice.
"""
from dataclasses import dataclass
from typing import Tuple

# ----------------------------------------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------------------------------------
EPOCHS = 20                                   # epochs of EVERY neural model (Aegis_net included)
SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)  # exactly five training seeds for EVERY model
DATA_SEED = 42                                # seed of everything that must NOT change between training seeds (split, fit rows)
FIT_ROWS = 500_000                            # rows every model trains on (class-aware sample of the train split)
MONITOR_ROWS = 200_000                        # validation rows used to pick the best epoch / boosting round
SELECT_ROWS = 500_000                         # validation rows used to pick the decision threshold of the binary task
MAX_TEST_ROWS = 0                             # 0 = score on the whole test split; >0 = random sample (faster smoke runs)

# ----------------------------------------------------------------------------------------------------------
# Loss / class weighting (shared by every model)
# ----------------------------------------------------------------------------------------------------------
CLASS_WEIGHT_POWER = 0.5                      # class weight ~ (1 / class frequency) ** power
FOCAL_GAMMA = 2.0                             # focal loss exponent (neural models)
LABEL_SMOOTHING = 0.05                        # neural models

# ----------------------------------------------------------------------------------------------------------
# DL training loop (shared by every DL model)
# ----------------------------------------------------------------------------------------------------------
AEGIS_NET_KWARGS = {}                         # optional constructor overrides of Aegis_net ONLY (main.py --aegis_net_kwargs, used by tools/ablation.py); {} = defaults
BATCH_SIZE = 2048
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
WARMUP_FRACTION = 0.05                        # first 5% of the steps: linear warm-up, then cosine decay
EMA_DECAY = 0.999                             # exponential moving average of the weights (evaluation uses the EMA copy)
GRAD_CLIP = 1.0

# ----------------------------------------------------------------------------------------------------------
# Tree models (they have no "epochs": boosting rounds are chosen by early stopping on the monitor split)
# ----------------------------------------------------------------------------------------------------------
BOOST_MAX_ROUNDS = 2000
BOOST_PATIENCE = 50
XGB_ETA = 0.08
XGB_MAX_DEPTH = 8
LGB_LEARNING_RATE = 0.08
LGB_NUM_LEAVES = 127
FOREST_TREES = 100
CATBOOST_ROUNDS = 1500
CATBOOST_LR = 0.1
CATBOOST_DEPTH = 8


# ----------------------------------------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------------------------------------
@dataclass
class DataSettings:
    label_column: str = ""                    # "" = auto-detect (label, class, attack_cat, ...)
    drop_columns: Tuple[str, ...] = ()        # columns that must never be used as features (ids, timestamps, leaking columns)
    group_labels: str = "auto"                # "auto": group the 34 CICIoT2023 labels into 8 families only if the data looks like CICIoT2023
    chunk_rows: int = 500_000                 # CSV rows read at a time (controls peak RAM)
    max_rows: int = 0                         # 0 = read everything; >0 = stop after this many rows (quick tests)
    dedup: bool = True                        # drop exact duplicate rows (keep the first)
    collision_policy: str = "drop"            # identical features with different labels: "drop" | "majority" | "keep"
    onehot_max_categories: int = 12           # text columns: one-hot of the N most frequent values (+ "other")
    corr_threshold: float = 0.98              # drop one feature of every pair with |correlation| above this
    clip_quantiles: Tuple[float, float] = (1e-4, 1 - 1e-4)
    skew_log_threshold: float = 2.0           # non-negative features with |skew| above this get a signed log transform
    zscore_clip: float = 10.0
    fit_rows_per_class: int = 100_000         # rows per class used to estimate the scaler
    split_mode: str = "block"                 # "block": consecutive-row blocks stay together | "row": random rows
    block_size: int = 512                     # maximum block size (shrinks automatically on small datasets)
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = DATA_SEED


# ----------------------------------------------------------------------------------------------------------
# The three tasks every model is scored on
# ----------------------------------------------------------------------------------------------------------
TASK_BINARY = "binary"                        # Benign vs attack (or "largest class vs the rest" if the data has no benign class)
TASK_ATTACK_TYPE = "attack_type"              # which attack is it? (only the non-benign rows)
TASK_MULTICLASS = "multiclass"                # every class at once
TASKS = (TASK_BINARY, TASK_ATTACK_TYPE, TASK_MULTICLASS)

# Order in which models appear in every table
# (the order itself lives in models/registry.py; this tuple only fixes the order of the tables)
MODEL_ORDER = ("LogReg", "NaiveBayes", "DecisionTree", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost",
               "MLP-Res", "CNN1D", "BiGRU", "ChebyKAN", "FT-Transformer", "TabM", "Aegis_net")

# How many of the strongest baselines of each family go into the final comparison next to Aegis_net (evaluation/finalists.py)
FINALISTS_PER_FAMILY = 2
