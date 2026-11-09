"""
TabM - parameter-efficient implicit ensemble of MLPs (Gorishniy et al., ICLR 2025), via the authors' `tabm` package.
K sub-networks share most weights (BatchEnsemble); every one is trained on the same loss, the prediction is the mean of their probabilities.
This is the strongest plain-MLP family in the public TabArena benchmark, so it is the "hardest" neural baseline of this project.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME, DESCRIPTION = "TabM", "BatchEnsemble MLP, 16 members (Gorishniy et al., ICLR 2025)"


class Network(nn.Module):
    def __init__(self, n_features, n_classes, k=16, blocks=3, hidden=256, p=0.1):
        super().__init__()
        import tabm
        self.k = k
        self.net = tabm.TabM.make(n_num_features=n_features, d_out=n_classes, k=k, n_blocks=blocks, d_block=hidden, dropout=p, arch_type="tabm")

    def forward(self, x):                                   # log of the mean member probability (a valid log-probability: softmax() of it is the identity)
        logp = F.log_softmax(self.net(x).float(), dim=-1)  # (B, k, C)
        return torch.logsumexp(logp, dim=1) - math.log(self.k)

    def loss(self, xb, yb, wb, idx, loss_fn):               # every member gets its own loss (the TabM training recipe)
        logits = self.net(xb).float()
        return sum(loss_fn(logits[:, i], yb, wb) for i in range(self.k)) / self.k
