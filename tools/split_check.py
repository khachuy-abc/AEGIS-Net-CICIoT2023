"""
Split robustness check: the benchmark uses ONE train/validation/test split, and on rare classes with 20-60 test rows a single row moves F1 by 0.02-0.03.
This tool repeats the multiclass comparison on several independent splits of the same data (each <cache> is a folder made by main.py with another --data_seed)
and reports mean +- std over the splits, so a gap smaller than the split-to-split spread is not mistaken for a real difference.

    python main.py --data_path unsw_nb15 --label_column attack_cat --out splits/unsw_s1 --data_seed 1 --prepare_only      # repeat for 2, 3, ...
    python tools/split_check.py --caches splits/unsw_s1/cache splits/unsw_s2/cache splits/unsw_s3/cache --fit_rows 60000 --out split_check_unsw.md
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import f1_score

from aegis_net import settings as S
from aegis_net.data.dataset import Bundle, Dataset
from aegis_net.models.registry import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--models", nargs="*", default=["XGBoost", "LightGBM", "Aegis_net"])
    ap.add_argument("--fit_rows", type=int, default=S.FIT_ROWS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--out", default="split_check.md")
    ap.add_argument("--aegis_net_kwargs", default="", help='JSON overrides of Aegis_net, e.g. \'{"levels": 12}\'')
    a = ap.parse_args()
    if a.aegis_net_kwargs:
        import json
        S.AEGIS_NET_KWARGS = json.loads(a.aegis_net_kwargs)
    S.BATCH_SIZE = a.batch_size
    scores = {m: [] for m in a.models}
    for cache in a.caches:
        bundle = Bundle(Dataset(cache), fit_rows=a.fit_rows)
        td = bundle.task_view(bundle.tasks[S.TASK_MULTICLASS])
        for m in a.models:
            p = build(m).fit(td, a.seed).predict_proba(td.X["test"])
            scores[m].append(f1_score(td.y["test"], p.argmax(1), average="macro", labels=range(td.n_classes), zero_division=0))
            print(f"{cache}  {m:10s} F1 macro {scores[m][-1]:.4f}", flush=True)
    best = np.max([scores[m] for m in a.models], axis=0)
    md = ["# Split robustness (multiclass, F1 macro, raw decision)", "", f"{len(a.caches)} independent splits, seed {a.seed}, fit rows {a.fit_rows:,}.", "",
          "| Model | " + " | ".join(f"split {i + 1}" for i in range(len(a.caches))) + " | mean | std | best on |", "|---|" + "---|" * (len(a.caches) + 3)]
    for m in a.models:
        s = np.array(scores[m])
        md.append(f"| {m} | " + " | ".join(f"{v:.4f}" for v in s) + f" | {s.mean():.4f} | {s.std():.4f} | {int((s >= best - 1e-12).sum())} of {len(s)} |")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
