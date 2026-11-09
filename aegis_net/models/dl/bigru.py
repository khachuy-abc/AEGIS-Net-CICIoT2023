"""BiGRU - a bidirectional GRU that reads the features as a sequence (one value per step)."""
import torch.nn as nn

NAME, DESCRIPTION = "BiGRU", "Bidirectional GRU reading the features as a sequence"


class Network(nn.Module):
    def __init__(self, n_features, n_classes, hidden=48, p=0.15):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 128), nn.GELU(), nn.Dropout(p), nn.Linear(128, n_classes))

    def forward(self, x):
        self.gru.flatten_parameters()          # avoids the cuDNN 'weights are not part of single contiguous chunk' warning after deepcopy
        o, _ = self.gru(x.unsqueeze(-1))
        return self.head(o.mean(1))
