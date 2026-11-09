"""
dataset.py - what the models actually see.

Dataset      : the cache written by prepare.py (memory-mapped, so 46M rows do not need 46M rows of RAM).
Task         : one of the three classification problems (binary / attack_type / multiclass).
Bundle       : the arrays every model uses: the SAME fit rows, monitor rows, select rows and test rows for all models and seeds.
TaskData     : a Bundle seen through one Task (labels remapped, rows of other classes removed, sample weights computed).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .. import settings as S
from .labels import is_benign
from .prepare import ROLE_TEST, ROLE_TRAIN, ROLE_VAL, balanced_subsample


class Dataset:
    def __init__(self, cache_dir: str):
        d = Path(cache_dir)
        self.meta: Dict = json.load(open(d / "meta.json"))
        self.X = np.load(d / "X.npy", mmap_mode="r")
        self.y = np.load(d / "y.npy").astype(np.int64)
        self.role = np.load(d / "role.npy")
        self.block = np.load(d / "block.npy")
        self.classes: List[str] = self.meta["classes"]
        self.features: List[str] = self.meta["features"]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def rows(self, role: int) -> np.ndarray:
        return np.flatnonzero(self.role == role)

    def read(self, rows: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.X[np.asarray(rows)], dtype=np.float32)


# ------------------------------------------------------------------------------------------------ tasks
@dataclass
class Task:
    name: str
    lut: np.ndarray            # class code of the dataset -> label of this task (-1 = row does not belong to this task)
    class_names: List[str]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def labels(self, y: np.ndarray) -> np.ndarray:
        return self.lut[y]


def make_tasks(classes: List[str], train_counts: np.ndarray) -> Dict[str, Task]:
    """binary: Benign (or the largest class if there is no Benign class) vs the rest | attack_type: the rest only | multiclass: all."""
    C = len(classes)
    terminal = 0 if is_benign(classes[0]) else int(np.argmax(train_counts))
    rest = [i for i in range(C) if i != terminal]
    tasks = {}
    lut = np.ones(C, np.int64)
    lut[terminal] = 0
    tasks[S.TASK_BINARY] = Task(S.TASK_BINARY, lut, [classes[terminal], "Attack" if is_benign(classes[terminal]) else "Other"])
    if len(rest) >= 2:
        lut = np.full(C, -1, np.int64)
        lut[rest] = np.arange(len(rest))
        tasks[S.TASK_ATTACK_TYPE] = Task(S.TASK_ATTACK_TYPE, lut, [classes[i] for i in rest])
    if C >= 3:
        tasks[S.TASK_MULTICLASS] = Task(S.TASK_MULTICLASS, np.arange(C, dtype=np.int64), list(classes))
    return tasks


def class_weights(counts: np.ndarray, power: float) -> np.ndarray:
    """(1/frequency)^power, scaled so the mean over the classes that exist is 1. Missing classes get weight 0."""
    c = np.asarray(counts, np.float64)
    w = np.zeros_like(c)
    ok = c > 0
    w[ok] = c[ok] ** -power
    return (w * ok.sum() / w.sum()).astype(np.float32)


# ------------------------------------------------------------------------------------------------ bundle
class Bundle:
    """All arrays are loaded ONCE and shared by every model. Row choices use DATA_SEED only, so they never depend on the training seed."""

    def __init__(self, ds: Dataset, fit_rows: int = S.FIT_ROWS, monitor_rows: int = S.MONITOR_ROWS,
                 select_rows: int = S.SELECT_ROWS, max_test_rows: int = S.MAX_TEST_ROWS, seed: int = S.DATA_SEED):
        self.ds, y = ds, ds.y
        rng = np.random.default_rng(seed)
        train, val, test = ds.rows(ROLE_TRAIN), ds.rows(ROLE_VAL), ds.rows(ROLE_TEST)
        self.train_counts = np.bincount(y[train], minlength=len(ds.classes))
        # the validation split is cut in two disjoint halves (by block): one picks epochs/rounds, the other picks the binary threshold
        half = (ds.block[val] % 2) == 0
        mon_pool, sel_pool = val[half], val[~half]
        if len(mon_pool) == 0 or len(sel_pool) == 0:
            mon_pool = sel_pool = val
        fit = train[balanced_subsample(y[train], fit_rows, seed)]
        mon = mon_pool[balanced_subsample(y[mon_pool], monitor_rows, seed)]
        sel = np.sort(rng.choice(sel_pool, select_rows, replace=False)) if 0 < select_rows < len(sel_pool) else sel_pool
        tst = np.sort(rng.choice(test, max_test_rows, replace=False)) if 0 < max_test_rows < len(test) else test
        self.rows = dict(fit=fit, monitor=mon, select=sel, test=tst)
        self.X = {k: ds.read(r) for k, r in self.rows.items()}
        self.y = {k: y[r] for k, r in self.rows.items()}
        self.tasks = make_tasks(ds.classes, self.train_counts)
        self.fit_digest = __import__("hashlib").sha1(fit.tobytes()).hexdigest()[:12]

    def task_view(self, task: Task) -> "TaskData":
        return TaskData(self, task)


class TaskData:
    def __init__(self, b: Bundle, task: Task):
        self.task, self.n_classes, self.n_features = task, task.n_classes, b.ds.n_features
        counts = np.bincount(task.lut[np.arange(len(b.train_counts))].clip(min=0), weights=b.train_counts * (task.lut >= 0), minlength=task.n_classes)
        self.class_weight = class_weights(counts, S.CLASS_WEIGHT_POWER)
        self.X, self.y = {}, {}
        for split in b.X:
            lab = task.labels(b.y[split])
            ok = lab >= 0
            self.X[split] = b.X[split] if ok.all() else b.X[split][ok]
            self.y[split] = lab[ok]
        # sample weights of the fit rows: class weight x (rows of that class in the whole train split / rows kept in the fit sample),
        # so a model trained on the sample "sees" the class-weighted distribution of the entire train split
        yf = self.y["fit"]
        kept = np.bincount(yf, minlength=task.n_classes).astype(np.float64)
        w = self.class_weight[yf] * (counts[yf] / np.maximum(kept[yf], 1))
        self.w_fit = (w / w.mean()).astype(np.float32)
        self.w_monitor = self.class_weight[self.y["monitor"]]
