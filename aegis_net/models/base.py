"""
base.py - the interface every model implements, so the benchmark treats all of them the same way.

    model = build(...)                                   # see models/registry.py
    model.fit(data, seed)                                # data: TaskData (fit rows + sample weights, monitor rows)
    proba = model.predict_proba(X)                       # (n, n_classes) float32
    model.n_params(), model.info                         # size and facts worth reporting (best epoch, best round, ...)
"""
from __future__ import annotations

import numpy as np


class Model:
    name = "model"
    info: dict

    def fit(self, data, seed: int) -> "Model":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def n_params(self) -> int:
        return 0
