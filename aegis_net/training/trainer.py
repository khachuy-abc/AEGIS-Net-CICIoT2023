"""
trainer.py - the ONE training loop used by every neural model (the five DL baselines AND Aegis_net).

Same for everybody: EPOCHS epochs, AdamW, linear warm-up + cosine decay, weighted focal loss with label smoothing, gradient clipping,
EMA of the weights, and "keep the epoch with the best macro-F1 on the monitor split". No early stopping: all models run all epochs.

A network may optionally provide these hooks (Aegis_net does; plain networks do not need them):
    init_from_data(X, y, w)            called once before training with a random sample of the fit rows (Aegis_net places its quantile bins with it)
    attach_training_set(X, y, C)       gives the network the fit rows (needed to sample retrieval candidates)
    loss(xb, yb, wb, idx, loss_fn)     custom training loss (default: loss_fn(net(xb), yb, wb))
    refresh_memory()                   rebuild whatever the network caches from the training set; called whenever the weights changed
    param_groups(weight_decay)         optional list of AdamW parameter groups (dicts may carry "lr_mult" and their own "weight_decay")
"""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .. import settings as S
from ..models.base import Model


INIT_ROWS = 200_000      # rows shown to net.init_from_data (a random sample of the fit rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_focal_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, gamma: float = S.FOCAL_GAMMA,
                        smoothing: float = S.LABEL_SMOOTHING) -> torch.Tensor:
    """Focal cross-entropy with label smoothing; w are per-row sample weights (class weights already folded in)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    lp_true = logp.gather(1, y[:, None]).squeeze(1)
    ce = (1 - smoothing) * (-lp_true) + smoothing * (-logp.mean(-1))
    focal = (1 - lp_true.exp()).clamp_min(0) ** gamma
    return (focal * ce * w).mean()


class _EMA:
    def __init__(self, net: nn.Module, decay: float):
        self.decay, self.n = decay, 0
        self.module = copy.deepcopy(net).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net: nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))     # warm-up: early steps use a smaller decay
        e, p = list(self.module.parameters()), [q.detach() for q in net.parameters()]
        torch._foreach_mul_(e, d)
        torch._foreach_add_(e, p, alpha=1.0 - d)
        for be, b in zip(self.module.buffers(), net.buffers()):
            be.copy_(b)


class DeepModel(Model):
    def __init__(self, name: str, arch, **arch_kwargs):
        self.name, self.arch, self.arch_kwargs = name, arch, arch_kwargs
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.info = {}
        self._mem_ready = False          # True while net.refresh_memory() is up to date with the current weights

    # -------------------------------------------------------------------------------------------- fit
    def fit(self, data, seed: int):
        seed_everything(seed)
        dev, C = self.dev, data.n_classes
        X = torch.from_numpy(data.X["fit"]).to(dev)
        y = torch.from_numpy(data.y["fit"]).long().to(dev)
        w = torch.from_numpy(data.w_fit).to(dev)
        net = self.arch(data.n_features, C, **self.arch_kwargs).to(dev)
        if hasattr(net, "init_from_data"):
            pick = torch.randperm(len(X), device=dev)[:INIT_ROWS]                              # natural (not class-balanced) sample: bins follow the real data density
            net.init_from_data(X[pick], y[pick], w[pick])
        ema = _EMA(net, S.EMA_DECAY)
        for m in (net, ema.module):                        # the EMA copy shares the same training tensors (no second copy in memory)
            if hasattr(m, "attach_training_set"):
                m.attach_training_set(X, y, C)
        n = len(X)
        bs = int(min(S.BATCH_SIZE, max(32, n // 8)))
        steps_per_epoch = max(n // bs, 1)
        total = S.EPOCHS * steps_per_epoch
        warm = max(1, int(round(total * S.WARMUP_FRACTION)))
        groups = net.param_groups(S.WEIGHT_DECAY) if hasattr(net, "param_groups") else [dict(params=list(net.parameters()), weight_decay=S.WEIGHT_DECAY)]
        opt = torch.optim.AdamW(groups, lr=S.LEARNING_RATE)
        for grp in opt.param_groups:
            grp.setdefault("lr_mult", 1.0)
        best, best_state, best_epoch, step, history = -1.0, None, 0, 0, []
        t0 = time.time()
        for epoch in range(S.EPOCHS):
            net.train()
            g = torch.Generator().manual_seed(seed * 100_003 + epoch)
            order = torch.randperm(n, generator=g).to(dev)
            for s in range(steps_per_epoch):
                idx = order[s * bs:(s + 1) * bs]
                lr = S.LEARNING_RATE * ((step + 1) / warm if step < warm else
                                        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1))))
                for grp in opt.param_groups:
                    grp["lr"] = lr * grp["lr_mult"]
                xb, yb, wb = X[idx], y[idx], w[idx]
                loss = net.loss(xb, yb, wb, idx, weighted_focal_loss) if hasattr(net, "loss") else weighted_focal_loss(net(xb), yb, wb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), S.GRAD_CLIP)
                opt.step()
                ema.update(net)
                step += 1
            self.net, self._mem_ready = ema.module, False          # the weights changed -> the memory must be rebuilt for this evaluation
            f1 = f1_score(data.y["monitor"], self.predict_proba(data.X["monitor"]).argmax(1), average="macro", zero_division=0)
            history.append(round(float(f1), 5))
            if f1 > best:
                best, best_epoch, best_state = f1, epoch + 1, copy.deepcopy(ema.module.state_dict())
        self.net = ema.module
        self.net.load_state_dict(best_state)
        self._mem_ready = False
        if hasattr(self.net, "refresh_memory"):
            self.net.refresh_memory()
            self._mem_ready = True
        self.info = dict(epochs_run=S.EPOCHS, best_epoch=best_epoch, monitor_f1_per_epoch=history, fit_seconds_loop=round(time.time() - t0, 1))
        return self

    # -------------------------------------------------------------------------------------------- predict
    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        if hasattr(self.net, "refresh_memory") and not self._mem_ready:      # rebuilt only when the weights changed (not once per call)
            self.net.refresh_memory()
            self._mem_ready = True
        out = []
        for s in range(0, len(X), 8192):
            xb = torch.from_numpy(np.ascontiguousarray(X[s:s + 8192], dtype=np.float32)).to(self.dev)
            out.append(F.softmax(self.net(xb).float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))
