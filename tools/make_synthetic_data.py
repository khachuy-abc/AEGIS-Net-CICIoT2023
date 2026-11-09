"""
Writes a small fake IDS dataset (CICIoT2023-like: 30 numeric flow features, 6 classes, heavy imbalance, skewed features,
sharp thresholds, overlapping classes, near-duplicate flows). For smoke tests and demos only - NOT real network traffic.

    python tools/make_synthetic_data.py --out synthetic_data --rows 60000
"""
import argparse
import os

import numpy as np
import pandas as pd

CLASSES = [("BenignTraffic", .20), ("DDoS-ICMP_Flood", .40), ("DoS-TCP_Flood", .20), ("Recon-PortScan", .10), ("BruteForce", .06), ("SqlInjection", .04)]


def make(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    F, parts = 30, []
    centres = rng.normal(0, 1.2, (len(CLASSES), F))
    for k, (name, frac) in enumerate(CLASSES):
        n = int(rows * frac)
        z = centres[k] + rng.normal(0, 1.0, (n, F))
        z[:, :4] = np.where(rng.random((n, 4)) < .5, centres[k, :4] + rng.normal(0, .3, (n, 4)), z[:, :4])      # near-duplicate-ish flows
        x = np.expm1(np.abs(z) * 1.5) * (rng.random((n, F)) > .1)                                               # skewed, zero-inflated
        x[:, 5] = (z[:, 5] > 0.3 * k - 0.5).astype(float)                                                          # sharp threshold feature
        df = pd.DataFrame(x, columns=[f"feat_{i}" for i in range(F)])
        df["label"] = name
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    flip = rng.random(len(df)) < 0.01                                                                             # 1% label noise
    df.loc[flip, "label"] = rng.choice([c for c, _ in CLASSES], flip.sum())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_data")
    ap.add_argument("--rows", type=int, default=60000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = make(a.rows)
    for i, idx in enumerate(np.array_split(np.arange(len(df)), 3)):
        ch = df.iloc[idx]                      # several files, like the real CICIoT2023 folder
        ch.to_csv(os.path.join(a.out, f"part-{i:02d}.csv"), index=False)
    print(df["label"].value_counts().to_string())
