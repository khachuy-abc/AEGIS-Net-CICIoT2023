"""
Aegis_net - the proposed model (deep learning). ONE FILE: everything specific to Aegis_net is in here.

THE PROBLEM IT TARGETS
    Tabular intrusion data is full of RARE classes whose signature is an EXACT, NARROW value band of one or two features (for example one attack
    type always uses a payload size inside a band that is 1% wide). Gradient-boosted trees isolate such a band with two splits. A neural network
    that sees the raw number (or a smooth function of it: periodic / piece-wise-linear / Chebyshev encodings) must learn a very sharp function
    of a continuous input, which it does badly, so it blurs the band into its neighbours and the rare class is lost (false alarms and misses).

WHAT IS NEW (stated honestly - see README "Novelty and prior art")
    1. MRBE, Multi-Resolution Bin Embedding.  Every feature is cut into 2^L quantile bins (L = 12 -> 4096 bins, narrow where the data is dense).
       The embedding of a value is the SUM, over the dyadic hierarchy of that bin (whole range, halves, quarters ... the finest bin), of one
       learned vector per level:   e(x) = W x + sum_{l=0..L} T_l[ bin_L(x) >> (L - l) ].
       Coarse levels share statistical strength between many rows, fine levels can memorise a narrow band, and the value is encoded exactly
       (a lookup, not a smooth function), so the sharpness problem disappears while the layers above stay an ordinary smooth network.
    2. Hierarchical shrinkage.  A penalty  sum_l  lambda * g^l * mean(T_l^2)  with g > 1 pulls fine levels harder towards zero, so a fine bin
       that only a handful of training rows ever touch falls back on its parent bin instead of over-fitting.  Together with the sum-over-levels
       parametrisation this is a prior "a rare fine bin looks like its coarse parent unless the data proves otherwise" (hierarchical-Bayes style).
       Rare classes get exactly the behaviour they need: precise where there is evidence, smooth where there is none.
    3. Both ideas run inside a BatchEnsemble of K members (one shared weight matrix, per-member rank-1 scalings and biases), whose averaged
       probabilities reduce the variance that makes rare-class predictions unstable from one seed to the next.

WHAT IS BORROWED (and where it is not new)
    * Per-feature quantile bins feeding learned embeddings: Piece-wise Linear / Target-Bin encodings (Gorishniy et al., NeurIPS 2022).
    * Multi-resolution lookup tables summed over levels: Instant-NGP multiresolution hash encoding (Mueller et al., 2022), used there for 3-D graphics.
    * BatchEnsemble / implicit ensembles for tabular MLPs: Wen et al. 2020, TabM (Gorishniy et al., ICLR 2025).
    * Residual LayerNorm-GELU blocks, weighted focal loss, EMA weights: standard and shared by every deep model of this benchmark.
    The combination "dyadic multi-resolution bins + coarse-to-fine shrinkage on tabular features, ensembled", as far as the author could find, has
    not been published; this is NOT proven to be unique and is not claimed to be.

All the knobs below are constructor arguments; main.py --aegis_net_kwargs '{"levels": 6}' overrides them (tools/ablation.py uses that).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "Aegis_net"
DESCRIPTION = "Multi-Resolution Bin Embedding (dyadic quantile bins, coarse-to-fine shrinkage) + BatchEnsemble MLP, 4 members"


class MultiResBinEmbedding(nn.Module):
    """MRBE: e(x) = W x + sum over levels l of T_l[feature, bin(x) >> (L - l)]  ->  (B, F * d)."""

    def __init__(self, n_features, levels=12, d=8, shrink=0.03, growth=2.0):
        super().__init__()
        self.F, self.L, self.d, self.shrink, self.growth = n_features, levels, d, shrink, growth
        self.register_buffer("edges", torch.zeros(n_features, 2 ** levels - 1))          # per-feature quantile edges (set in init_from_data)
        scale = 0.3 / math.sqrt(levels + 1)
        self.tables = nn.ParameterList([nn.Parameter(torch.randn(n_features, 2 ** l, d) * scale) for l in range(levels + 1)])   # level l: 2^l bins
        self.lin_w = nn.Parameter(torch.randn(n_features, d) * 0.1)                       # the raw value itself, so extrapolation stays linear
        self.out_dim = n_features * d

    @torch.no_grad()
    def init_from_data(self, X):
        q = torch.linspace(0, 1, 2 ** self.L + 1, device=X.device)[1:-1]
        rows = min(len(X), 200_000, 15_000_000 // self.F)                                   # torch.quantile accepts at most 16M values
        edges = torch.quantile(X[:rows], q, dim=0).T.contiguous()
        self.edges.copy_(edges + torch.arange(len(q), device=X.device) * 1e-7)           # tiny ramp: strictly increasing even for constant features

    def forward(self, x):
        idx = torch.searchsorted(self.edges, x.t().contiguous()).t()                      # (B, F) index of the finest bin
        feat = torch.arange(self.F, device=x.device)[None, :]
        h = x.unsqueeze(-1) * self.lin_w
        for level, table in enumerate(self.tables):
            h = h + table[feat, idx >> (self.L - level)]                                  # the ancestor of the finest bin at this level
        return h.flatten(1)

    def penalty(self):
        """Coarse-to-fine shrinkage: finer levels (larger l) are penalised more, so rare fine bins revert to their parents."""
        if self.shrink <= 0:
            return 0.0
        return sum(self.shrink * (self.growth ** l) * t.pow(2).mean() for l, t in enumerate(self.tables))


class _BatchEnsembleLinear(nn.Module):
    """Linear layer shared by K members; member k scales its input by r_k, its output by s_k and adds its own bias (Wen et al., 2020)."""

    def __init__(self, d_in, d_out, k, first=False):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        r = torch.randint(0, 2, (k, d_in)).float() * 2 - 1 if first else torch.ones(k, d_in) + 0.1 * torch.randn(k, d_in)   # first layer: random signs -> diverse members
        self.r = nn.Parameter(r)
        self.s = nn.Parameter(torch.ones(k, d_out) + 0.1 * torch.randn(k, d_out))
        self.b = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):                                                                 # (B, K, d_in) -> (B, K, d_out)
        return self.lin(x * self.r) * self.s + self.b


class Network(nn.Module):
    def __init__(self, n_features, n_classes, levels=12, d=8, hidden=256, blocks=2, members=4, p=0.1, shrink=0.03, growth=2.0, enc_lr_mult=10.0):
        super().__init__()
        self.K, self.enc_lr_mult = members, enc_lr_mult
        self.embed = MultiResBinEmbedding(n_features, levels, d, shrink, growth)
        self.stem = _BatchEnsembleLinear(self.embed.out_dim, hidden, members, first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(blocks)])
        self.mix = nn.ModuleList([_BatchEnsembleLinear(hidden, hidden, members) for _ in range(blocks)])
        self.drop = nn.Dropout(p)
        self.final_norm = nn.LayerNorm(hidden)
        self.head = _BatchEnsembleLinear(hidden, n_classes, members)

    # -- hooks used by training/trainer.py ---------------------------------------------------------------
    def init_from_data(self, X, y=None, w=None):
        self.embed.init_from_data(X)

    def param_groups(self, weight_decay):
        """Embedding tables: no weight decay (the shrinkage penalty does that job per level) and a 10x larger learning rate (they are sparse look-ups)."""
        emb, decay, no_decay = [], [], []
        for name, q in self.named_parameters():
            if name.startswith("embed."):
                emb.append(q)
            else:
                (no_decay if q.ndim <= 1 or name.endswith((".r", ".s", ".b")) else decay).append(q)
        groups = [dict(params=decay, weight_decay=weight_decay, lr_mult=1.0), dict(params=no_decay, weight_decay=0.0, lr_mult=1.0),
                  dict(params=emb, weight_decay=0.0, lr_mult=self.enc_lr_mult)]
        return [g for g in groups if g["params"]]

    def _member_logits(self, x):                                                          # (B, K, C)
        h = F.gelu(self.stem(self.embed(x).unsqueeze(1).expand(-1, self.K, -1)))
        for norm, mix in zip(self.norms, self.mix):
            h = h + self.drop(F.gelu(mix(norm(h))))
        return self.head(self.final_norm(h))

    def loss(self, xb, yb, wb, idx, loss_fn):
        logp = F.log_softmax(self._member_logits(xb), dim=-1)
        return sum(loss_fn(logp[:, k], yb, wb) for k in range(self.K)) / self.K + self.embed.penalty()    # every member learns on its own + shrinkage

    def forward(self, x):                                                                 # log of the mean member probability
        logp = F.log_softmax(self._member_logits(x), dim=-1)
        return torch.logsumexp(logp, dim=1) - math.log(self.K)
