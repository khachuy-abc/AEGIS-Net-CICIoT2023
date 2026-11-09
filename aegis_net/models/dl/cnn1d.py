"""CNN1D - 1-D convolutions along the feature axis (value channel + learnable position channel), average + max pooling."""
import torch
import torch.nn as nn
import torch.nn.functional as F

NAME, DESCRIPTION = "CNN1D", "1-D CNN along the feature axis"


class _ResConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.skip = nn.Identity() if cin == cout else nn.Conv1d(cin, cout, 1, bias=False)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)) + self.skip(x))


class Network(nn.Module):
    def __init__(self, n_features, n_classes, channels=(32, 64, 96), head=128, p=0.15):
        super().__init__()
        self.pos = nn.Parameter(torch.linspace(-1, 1, n_features).view(1, 1, -1))
        blocks, cin = [], 2
        for c in channels:
            blocks.append(_ResConv(cin, c))
            cin = c
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(2 * channels[-1], head), nn.GELU(), nn.Dropout(p), nn.Linear(head, n_classes))

    def forward(self, x):
        h = self.conv(torch.cat([x.unsqueeze(1), self.pos.expand(len(x), -1, -1)], dim=1))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=-1))
