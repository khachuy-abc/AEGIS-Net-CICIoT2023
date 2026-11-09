"""Run with:  pytest -q   (a few minutes on CPU: every one of the 15 models is trained for 2 epochs on a tiny synthetic dataset)"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aegis_net import settings as S
from aegis_net.data import labels as L
from aegis_net.data.dataset import Bundle, Dataset
from aegis_net.data.prepare import balanced_subsample, prepare
from aegis_net.models import registry
from aegis_net.models.registry import MODEL_NAMES, PROPOSED, build, describe, model_type
from tools.make_synthetic_data import make


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    d = tmp_path_factory.mktemp("csv")
    make(6000).to_csv(d / "a.csv", index=False)
    cache = prepare(str(d), str(tmp_path_factory.mktemp("cache")), S.DataSettings())
    return Bundle(Dataset(cache), fit_rows=3000, monitor_rows=1000, select_rows=1000)


def test_protocol_is_the_requested_one():
    assert S.EPOCHS == 20 and len(S.SEEDS) == 5 and len(set(S.SEEDS)) == 5
    assert set(S.MODEL_ORDER) == set(MODEL_NAMES)                      # every model appears in the tables, nothing hidden
    assert sum(model_type(m) == "ML" for m in MODEL_NAMES) == 8 and sum(model_type(m) == "DL" for m in MODEL_NAMES) == 7
    assert PROPOSED == "Aegis_net" and PROPOSED in MODEL_NAMES and not any("Lite" in m for m in MODEL_NAMES)
    assert S.FINALISTS_PER_FAMILY == 2


def test_one_file_per_model():
    """Every model lives in its own file, named after it, and nothing else is left in the model folders."""
    for name, (family, module) in registry._MODELS.items():
        path = ROOT / "aegis_net" / "models" / (module.replace(".", "/") + ".py")
        assert path.exists(), (name, path)
        assert path.stem.replace("_", "") == name.lower().replace("-", "").replace("_", "") or name in ("FT-Transformer", "MLP-Res", "BiGRU", "CNN1D"), name
        assert describe(name)
    files = {p.stem for fam in ("ml", "dl") for p in (ROOT / "aegis_net" / "models" / fam).glob("*.py")} - {"__init__", "_common"}
    assert files == {module.split(".")[1] for _, module in registry._MODELS.values()}          # no orphan model files


def test_no_stale_names_left_in_the_project():
    bad = ("tune_aegis_net", "example_results", "Aegis_net_Lite", "DualEncoder", "CALIBRATE")
    for p in list((ROOT / "aegis_net").rglob("*.py")) + [ROOT / "main.py", ROOT / "README.md"]:
        text = p.read_text(encoding="utf-8")
        for b in bad:
            assert b not in text, f"{b!r} still appears in {p.relative_to(ROOT)}"


def test_labels():
    assert L.resolve_class_names(["BenignTraffic", "DDoS-ICMP_Flood", "Mirai-udpplain"]) == ["Benign", "DDoS", "Mirai"]
    assert L.resolve_class_names(["normal", "neptune"]) == ["Benign", "neptune"]          # not CICIoT-like: names untouched
    assert L.class_order(["DoS", "Benign", "A"])[0] == "Benign"


def test_balanced_subsample_keeps_rare_classes():
    y = np.array([0] * 1000 + [1] * 10)
    idx = balanced_subsample(y, 100)
    assert len(idx) <= 100 and (y[idx] == 1).sum() == 10


def test_bundle_tasks_and_no_leakage(bundle):
    assert set(bundle.tasks) == {"binary", "attack_type", "multiclass"}
    fit, test = set(bundle.rows["fit"]), set(bundle.rows["test"])
    assert not fit & test
    for t in bundle.tasks.values():
        td = bundle.task_view(t)
        assert td.y["fit"].max() < td.n_classes and np.isclose(td.w_fit.mean(), 1.0, atol=1e-4)


@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_every_model_trains_and_predicts(bundle, name):
    old = S.EPOCHS, S.BATCH_SIZE
    S.EPOCHS, S.BATCH_SIZE = 2, 256                      # smoke test only
    try:
        td = bundle.task_view(bundle.tasks["multiclass"])
        m = build(name).fit(td, seed=42)
        p = m.predict_proba(td.X["test"])
        assert p.shape == (len(td.X["test"]), td.n_classes) and np.allclose(p.sum(1), 1, atol=1e-3) and m.n_params() > 0
        assert np.isfinite(p).all()
    finally:
        S.EPOCHS, S.BATCH_SIZE = old


def test_mrbe_embedding_is_hierarchical_and_exact():
    import torch
    from aegis_net.models.dl.aegis_net import MultiResBinEmbedding
    torch.manual_seed(0)
    emb = MultiResBinEmbedding(3, levels=4, d=2, shrink=0.1, growth=2.0)
    X = torch.randn(2000, 3)
    emb.init_from_data(X)
    out = emb(X)
    assert out.shape == (2000, 6)
    # two values inside the same finest bin get the same table part (they differ only by the linear term)
    a = torch.zeros(1, 3)
    b = a + 1e-9
    assert torch.allclose(emb(a), emb(b), atol=1e-6)
    # shrinkage grows with the level: the same tensor costs more at a finer level
    emb.shrink = 0.0
    assert float(torch.as_tensor(emb.penalty()).detach()) == 0.0
    emb.shrink = 0.1
    assert float(torch.as_tensor(emb.penalty()).detach()) > 0


def test_aegis_net_members_are_diverse_and_probabilities_valid():
    import torch
    from aegis_net.models.dl.aegis_net import Network
    net = Network(5, 3, levels=3, d=2, hidden=16, blocks=1, members=4)
    X = torch.randn(64, 5)
    net.init_from_data(X)
    net.eval()
    lg = net._member_logits(X)
    assert lg.shape == (64, 4, 3)
    assert (lg[:, 0] - lg[:, 1]).abs().max() > 1e-4                        # members differ (rank-1 scalings)
    assert torch.allclose(net(X).exp().sum(-1), torch.ones(64), atol=1e-4)  # forward() is a proper log-probability


def test_calibration_fixes_an_overpredicted_rare_class():
    from aegis_net.evaluation.calibration import _macro_f1, fit_class_bias
    rng = np.random.default_rng(0)
    y = np.array([0] * 9000 + [1] * 300)
    score = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 1.0, len(y))           # overlapping classes
    p1 = 1 / (1 + np.exp(-(score + 2.0)))                                      # a model biased towards the rare class
    proba = np.stack([1 - p1, p1], 1)
    b = fit_class_bias(proba, y, 2)
    assert _macro_f1(y, (np.log(proba) + b).argmax(1), 2) > _macro_f1(y, proba.argmax(1), 2) + 0.05
    assert np.allclose(fit_class_bias(np.full((200, 2), 0.5), np.array([0, 1] * 100), 2), 0)      # nothing to fix -> no bias


def test_aegis_net_kwargs_reach_only_aegis_net():
    S.AEGIS_NET_KWARGS = {"hidden": 32, "blocks": 1}
    try:
        assert build("Aegis_net").arch_kwargs == {"hidden": 32, "blocks": 1}
        assert build("MLP-Res").arch_kwargs == {}              # baselines always keep their fixed defaults
    finally:
        S.AEGIS_NET_KWARGS = {}
    assert build("Aegis_net").arch_kwargs == {}


def test_finalists_are_two_ml_two_dl_without_aegis_net():
    from aegis_net.evaluation.finalists import pick_finalists
    rows = []
    for task in S.TASKS:
        for i, m in enumerate(MODEL_NAMES):
            rows.append(dict(task=task, model=m, f1_macro_mean=0.5 + 0.01 * ((i * 7) % 13), f1_cal_mean=0.5))
    fin = pick_finalists(pd.DataFrame(rows))
    assert len(fin) == 4 and PROPOSED not in fin
    assert sum(model_type(m) == "ML" for m in fin) == 2 and sum(model_type(m) == "DL" for m in fin) == 2
