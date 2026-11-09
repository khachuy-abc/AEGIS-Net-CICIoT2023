"""
runner.py - runs every (task, model, seed) combination and stores one small JSON per combination in <out>/runs/.

Resumable: a combination whose JSON already exists (and was produced with the same fingerprint) is skipped, so a disconnected
Colab session continues where it stopped. Nothing here knows about a particular dataset.

Every multi-class run stores TWO scores: "test" (raw argmax) and "test_calibrated" (after the one shared decision-calibration step, fitted on the
validation 'select' split only). The binary task always uses a threshold picked on the same split.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import settings as S
from ..models.registry import PROPOSED, build
from .calibration import fit_class_bias
from .metrics import best_threshold, binary_metrics, multiclass_metrics

RUN_VERSION = 5          # bump when the meaning of a stored run changes


def _fingerprint(bundle, task_name: str) -> dict:
    return dict(fit_rows=bundle.fit_digest, n_fit=int(len(bundle.rows["fit"])), epochs=S.EPOCHS, task=task_name, n_test=int(len(bundle.rows["test"])),
                batch=S.BATCH_SIZE, lr=S.LEARNING_RATE, rounds=S.BOOST_MAX_ROUNDS, version=RUN_VERSION)


def run_all(bundle, out_dir: str, models: Sequence[str], seeds: Sequence[int] = S.SEEDS, tasks: Sequence[str] | None = None) -> None:
    runs = Path(out_dir) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    seeds = tuple(seeds)
    wanted = [t for t in (tasks or S.TASKS) if t in bundle.tasks]
    total = len(wanted) * len(models) * len(seeds)
    done = 0
    for tname in wanted:
        td = bundle.task_view(bundle.tasks[tname])
        fp0 = _fingerprint(bundle, tname)
        print(f"\n=== task: {tname} | classes: {td.task.class_names} | fit rows {len(td.y['fit']):,} | test rows {len(td.y['test']):,}")
        for name in models:
            for seed in seeds:
                done += 1
                path = runs / f"{tname}__{name}__seed{seed}.json"
                fp = dict(fp0, aegis_net_kwargs=dict(S.AEGIS_NET_KWARGS)) if (name == PROPOSED and S.AEGIS_NET_KWARGS) else fp0   # other settings -> not a stale run
                if path.exists():
                    old = json.load(open(path))
                    if old.get("fingerprint") == fp and "test" in old:
                        print(f"[{done}/{total}] skip   {tname} / {name} / seed {seed} (already done)")
                        continue
                try:
                    t0 = time.time()
                    model = build(name).fit(td, seed)
                    fit_s = time.time() - t0
                    t0 = time.time()
                    proba = model.predict_proba(td.X["test"])
                    infer_ms = (time.time() - t0) / max(len(proba), 1) * 1e6          # milliseconds per 1000 rows
                    res = dict(task=tname, model=name, seed=seed, fingerprint=fp, fit_seconds=fit_s, infer_ms_per_1k=infer_ms,
                               n_params=model.n_params(), class_names=td.task.class_names, info=model.info)
                    if tname == S.TASK_BINARY:
                        thr = best_threshold(td.y["select"], model.predict_proba(td.X["select"])[:, 1])
                        res["test"] = binary_metrics(td.y["test"], proba[:, 1], thr)
                        shown = res["test"]
                    else:
                        res["test"] = multiclass_metrics(td.y["test"], proba.argmax(1), td.n_classes)
                        bias = fit_class_bias(model.predict_proba(td.X["select"]), td.y["select"], td.n_classes)     # shared step, validation split only
                        cal = multiclass_metrics(td.y["test"], (np.log(np.clip(proba, 1e-9, 1.0)) + bias).argmax(1), td.n_classes)
                        cal["class_bias"] = [float(v) for v in bias]
                        res["test_calibrated"] = cal
                        shown = res["test"]
                    json.dump(res, open(path, "w"), indent=1)
                    extra = f"  calibrated {res['test_calibrated']['f1_macro']:.4f}" if "test_calibrated" in res else ""
                    print(f"[{done}/{total}] {tname} / {name:14s} seed {seed}: F1 macro {shown['f1_macro']:.4f}{extra}  acc {shown['accuracy']:.4f}  ({fit_s:.0f}s)", flush=True)
                    del model
                except Exception as e:                                      # one failing model must not stop the benchmark
                    traceback.print_exc()
                    json.dump(dict(task=tname, model=name, seed=seed, error=str(e)), open(path, "w"))
