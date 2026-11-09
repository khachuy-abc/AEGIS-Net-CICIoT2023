"""metrics.py - every number reported in the tables is computed here."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)


def multiclass_metrics(y, pred, n_classes: int) -> dict:
    labels = np.arange(n_classes)
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    present = sup > 0
    return dict(accuracy=float(accuracy_score(y, pred)), f1_macro=float(f.mean()), f1_macro_present=float(f[present].mean()),
                f1_weighted=float(f1_score(y, pred, average="weighted", zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)),
                per_class_f1=[float(v) for v in f], per_class_recall=[float(v) for v in r], per_class_support=[int(v) for v in sup])


def best_threshold(y, p_pos, grid: int = 199) -> float:
    """Threshold on P(positive) that maximises macro-F1 on the validation 'select' split."""
    best, best_t = -1.0, 0.5
    for t in np.linspace(0.005, 0.995, grid):
        f = f1_score(y, (p_pos >= t).astype(int), average="macro", zero_division=0)
        if f > best:
            best, best_t = f, float(t)
    return best_t


def binary_metrics(y, p_pos, threshold: float) -> dict:
    pred = (p_pos >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(threshold=float(threshold), accuracy=float(accuracy_score(y, pred)), f1_macro=float(f1_score(y, pred, average="macro", zero_division=0)),
               precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)), fpr=float(fp / max(fp + tn, 1)),
               balanced_accuracy=float(balanced_accuracy_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)))
    if len(np.unique(y)) == 2:
        out.update(auc_roc=float(roc_auc_score(y, p_pos)), auc_pr=float(average_precision_score(y, p_pos)))
    return out
