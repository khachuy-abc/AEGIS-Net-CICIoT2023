"""
registry.py - the list of every model in the benchmark. ONE FILE PER MODEL:

    models/ml/<model>.py    machine learning (linear, Bayes, trees, boosting)      -> class with fit(data, seed) / predict_proba(X)
    models/dl/<model>.py    deep learning (neural networks, Aegis_net included)    -> NAME, DESCRIPTION, Network (an nn.Module)

Adding a model = adding one file and one line below.
"""
from __future__ import annotations

from importlib import import_module

# name -> (family, module inside aegis_net.models)
_MODELS = {
    "LogReg": ("ML", "ml.logreg"), "NaiveBayes": ("ML", "ml.naive_bayes"), "DecisionTree": ("ML", "ml.decision_tree"),
    "RandomForest": ("ML", "ml.random_forest"), "ExtraTrees": ("ML", "ml.extra_trees"), "XGBoost": ("ML", "ml.xgboost"),
    "LightGBM": ("ML", "ml.lightgbm"), "CatBoost": ("ML", "ml.catboost"),
    "MLP-Res": ("DL", "dl.mlp_res"), "CNN1D": ("DL", "dl.cnn1d"), "BiGRU": ("DL", "dl.bigru"), "ChebyKAN": ("DL", "dl.chebykan"),
    "FT-Transformer": ("DL", "dl.ft_transformer"), "TabM": ("DL", "dl.tabm"),
    "Aegis_net": ("DL", "dl.aegis_net"),
}
PROPOSED = "Aegis_net"
MODEL_NAMES = tuple(_MODELS)


def _module(name: str):
    if name not in _MODELS:
        raise KeyError(f"Unknown model {name!r}. Available: {list(_MODELS)}")
    return import_module(f"{__package__}.{_MODELS[name][1]}")


def model_type(name: str) -> str:
    """"ML" or "DL"."""
    return _MODELS[name][0]


def describe(name: str) -> str:
    m = _module(name)
    return getattr(m, "DESCRIPTION", None) or getattr(next(v for k, v in vars(m).items() if isinstance(v, type) and getattr(v, "name", None) == name), "description")


def build(name: str):
    """A fresh, untrained model object with fit(data, seed) / predict_proba(X) / n_params()."""
    m = _module(name)
    if model_type(name) == "ML":
        return next(v for v in vars(m).values() if isinstance(v, type) and getattr(v, "name", None) == name)()
    from .. import settings as S
    from ..training.trainer import DeepModel            # imported lazily so tree-only runs never need torch
    kw = dict(S.AEGIS_NET_KWARGS) if name == PROPOSED else {}      # only Aegis_net takes overrides; every baseline keeps its fixed defaults
    return DeepModel(name, m.Network, **kw)
