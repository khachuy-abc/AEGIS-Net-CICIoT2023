"""
calibration.py - one shared decision-calibration step for the multi-class tasks (attack_type, multiclass).

WHY: on imbalanced intrusion data a class-weighted model over-predicts the rare classes (high recall, low precision: on CICIoT2023 BruteForce
and Web-based have precision ~0.2 and recall ~0.45). A stacking ensemble that was fitted on validation predictions repaired exactly that
(+0.036 F1 macro over its own members). The binary task already picks its decision threshold on the validation 'select' split for EVERY model;
this does the same for the multi-class tasks: one additive log-bias per class, fitted on the 'select' split, identical procedure for every model.

HOW: coordinate ascent over a small grid of biases maximising macro-F1 on one random half of the select rows; the biases are kept only if they
also improve macro-F1 on the other half (otherwise all biases stay 0 = no calibration), then they are re-fitted on all select rows.
The test split is never touched.
"""
from __future__ import annotations

import numpy as np

GRID = np.linspace(-3.0, 3.0, 25)


def _macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    cm = np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes).astype(np.float64)
    tp, fp, fn = np.diag(cm), cm.sum(0) - np.diag(cm), cm.sum(1) - np.diag(cm)
    denom = 2 * tp + fp + fn
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def _ascent(logp: np.ndarray, y: np.ndarray, C: int, passes: int = 3) -> np.ndarray:
    b = np.zeros(C, np.float32)
    best = _macro_f1(y, (logp + b).argmax(1), C)
    for _ in range(passes):
        improved = False
        for c in range(C):
            keep = b[c]
            for g in GRID:
                b[c] = g
                f = _macro_f1(y, (logp + b).argmax(1), C)
                if f > best + 1e-6:
                    best, keep, improved = f, g, True
            b[c] = keep
        if not improved:
            break
    return b - b.mean()


def fit_class_bias(proba: np.ndarray, y: np.ndarray, n_classes: int, seed: int = 0, max_rows: int = 300_000) -> np.ndarray:
    """Per-class additive log-bias (zeros when the gain does not hold on held-out validation rows)."""
    C = n_classes
    rng = np.random.default_rng(seed)
    if len(y) > max_rows:
        keep = rng.choice(len(y), max_rows, replace=False)
        proba, y = proba[keep], y[keep]
    logp = np.log(np.clip(proba, 1e-9, 1.0)).astype(np.float32)
    perm = rng.permutation(len(y))
    a, h = perm[: len(y) // 2], perm[len(y) // 2:]
    b = _ascent(logp[a], y[a], C)
    gain = _macro_f1(y[h], (logp[h] + b).argmax(1), C) - _macro_f1(y[h], logp[h].argmax(1), C)
    if gain <= 0.001:
        return np.zeros(C, np.float32)
    return _ascent(logp, y, C)
