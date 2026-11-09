"""
Turns the UNSW-NB15 "training-set" and "testing-set" CSV partitions into ONE CSV for main.py.
Both partitions are merged because main.py makes its own stratified train / validation / test split (the official split is not used).
The row id and the binary `label` column are removed: `label` is just "attack_cat != Normal" and would leak the answer of the binary task.
The target column that stays is `attack_cat` (10 classes: Normal + 9 attack families; Worms has only 174 rows - the rarest class).

    python tools/prepare_unsw_nb15.py --train UNSW_NB15_training-set.csv --test UNSW_NB15_testing-set.csv --out unsw_nb15/unsw_nb15.csv
Download: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Then:     python main.py --data_path unsw_nb15 --label_column attack_cat --out results_unsw
"""
import argparse
import os

import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(a.train), pd.read_csv(a.test)], ignore_index=True)
    df = df.drop(columns=[c for c in ("id", "label") if c in df.columns])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df["attack_cat"].value_counts().to_string())
