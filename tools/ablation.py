"""
Ablation of Aegis_net: switch its ingredients off one at a time and see what each one is worth (multiclass task, F1 macro).

    python tools/ablation.py --cache outputs_unsw/cache --fit_rows 60000 --seeds 42 43 44 --out outputs_unsw/results/ablation.md

Variants (all are ordinary constructor arguments of aegis_net/models/dl/aegis_net.py):
    full                 the published configuration
    no shrinkage         shrink = 0      (fine bins are not pulled towards their parents)
    no hierarchy         levels = 0      (no bins at all: only the linear term + one vector per feature -> a plain BatchEnsemble MLP)
    4 levels / 6 levels  coarser bins (16 / 64 per feature)
    1 member             no ensemble
    8 members            a bigger ensemble
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
from aegis_net.evaluation.calibration import fit_class_bias
from aegis_net.models.registry import build

VARIANTS = {
    "full": {},
    "no shrinkage (shrink=0)": {"shrink": 0.0},
    "no bins at all (levels=0)": {"levels": 0},
    "coarser bins (levels=4)": {"levels": 4},
    "coarser bins (levels=6)": {"levels": 6},
    "finer bins (levels=10)": {"levels": 10},
    "stronger shrinkage (shrink=0.2)": {"shrink": 0.2},
    "1 member (no ensemble)": {"members": 1},
    "8 members": {"members": 8},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="<out>/cache of a finished main.py run")
    ap.add_argument("--fit_rows", type=int, default=S.FIT_ROWS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--out", default="ablation.md")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--extra", default="", help='JSON {"name": {kwargs}} of additional variants to run (named in --variants too)')
    a = ap.parse_args()
    if a.extra:
        import json
        VARIANTS.update(json.loads(a.extra))
    S.BATCH_SIZE = a.batch_size
    bundle = Bundle(Dataset(a.cache), fit_rows=a.fit_rows)
    td = bundle.task_view(bundle.tasks[S.TASK_MULTICLASS])
    order = np.argsort(np.bincount(td.y["test"], minlength=td.n_classes))          # rarest class first
    rare = [int(c) for c in order[:3]]
    rows = []
    for name in a.variants:
        S.AEGIS_NET_KWARGS = dict(VARIANTS[name])
        raw, cal, rar = [], [], []
        for seed in a.seeds:
            m = build("Aegis_net").fit(td, seed)
            p, y = m.predict_proba(td.X["test"]), td.y["test"]
            bias = fit_class_bias(m.predict_proba(td.X["select"]), td.y["select"], td.n_classes)
            f_raw = f1_score(y, p.argmax(1), average=None, labels=range(td.n_classes), zero_division=0)
            f_cal = f1_score(y, (np.log(np.clip(p, 1e-9, 1)) + bias).argmax(1), average=None, labels=range(td.n_classes), zero_division=0)
            raw.append(f_raw.mean()), cal.append(f_cal.mean()), rar.append(f_raw[rare].mean())
        rows.append((name, np.mean(raw), np.std(raw), np.mean(cal), np.mean(rar)))
        print(f"{name:34s} F1 raw {rows[-1][1]:.4f} +- {rows[-1][2]:.4f} | cal {rows[-1][3]:.4f} | 3 rarest classes {rows[-1][4]:.4f}", flush=True)
    S.AEGIS_NET_KWARGS = {}
    names = [td.task.class_names[c] for c in rare]
    md = ["# Aegis_net ablation (multiclass)", "", f"Seeds {a.seeds}, fit rows {a.fit_rows:,}, batch size {a.batch_size}. Rarest classes: {', '.join(names)}.", "",
          "| Variant | F1 macro (raw) | std | F1 macro (calibrated) | F1 of the 3 rarest classes |", "|---|---|---|---|---|"]
    md += [f"| {n} | {r:.4f} | {s:.4f} | {c:.4f} | {q:.4f} |" for n, r, s, c, q in rows]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
