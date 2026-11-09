"""MLP-Res - a plain MLP with residual blocks (BatchNorm -> GELU -> Dropout -> Linear)."""
import torch.nn as nn

NAME, DESCRIPTION = "MLP-Res", "MLP with residual blocks"


class Network(nn.Module):
    def __init__(self, n_features, n_classes, hidden=256, blocks=3, p=0.15):
        super().__init__()
        self.inp = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, hidden)) for _ in range(blocks)])
        self.out = nn.Sequential(nn.BatchNorm1d(hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)
