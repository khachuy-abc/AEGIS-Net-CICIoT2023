"""
_common.py - the shared plumbing of every machine-learning (ML) model.

All ML models learn from the same fit rows with the same sample weights as the neural models, and implement _fit(...) only.
A class that is missing from the fit rows would break the boosting libraries (labels must be 0..k-1): classes are compressed to the ones that exist
and expanded again in predict_proba, so every model returns probabilities for ALL classes of the task.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..base import Model


class MLModel(Model):
    name = "ML"
    description = ""

    def fit(self, data, seed: int):
        X, y, w = data.X["fit"], data.y["fit"], data.w_fit
        Xv, yv = data.X["monitor"], data.y["monitor"]
        self.n_classes = data.n_classes
        self.present = np.unique(y)
        lut = np.full(self.n_classes, -1, np.int64)
        lut[self.present] = np.arange(len(self.present))
        ok = lut[yv] >= 0
        self.info = {}
        self._fit(X, lut[y], w, Xv[ok], lut[yv[ok]], data.w_monitor[ok], seed, len(self.present))
        return self

    def _fit(self, X, y, w, Xv, yv, wv, seed, k):
        raise NotImplementedError

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            p = np.asarray(self.m.predict_proba(X), dtype=np.float32)
        if len(self.present) != self.n_classes:
            full = np.zeros((len(X), self.n_classes), np.float32)
            full[:, self.present] = p
            p = full
        return p

    def n_params(self) -> int:
        """Parameters for linear / Naive Bayes, tree nodes for trees and boosting (not comparable with neural parameter counts)."""
        m = self.m
        if hasattr(m, "coef_"):
            return int(m.coef_.size + m.intercept_.size)
        if hasattr(m, "theta_"):
            return int(m.theta_.size + m.var_.size + m.class_prior_.size)
        if hasattr(m, "estimators_"):
            return int(sum(t.tree_.node_count for t in np.ravel(m.estimators_)))
        if hasattr(m, "tree_"):
            return int(m.tree_.node_count)
        if hasattr(m, "get_booster"):
            return int(len(m.get_booster().trees_to_dataframe()))
        if hasattr(m, "booster_"):
            return int(len(m.booster_.trees_to_dataframe()))
        if hasattr(m, "tree_count_"):
            return int(m.tree_count_ * (2 ** 6))
        return 0
