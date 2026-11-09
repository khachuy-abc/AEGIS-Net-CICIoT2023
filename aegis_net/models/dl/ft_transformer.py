"""FT-Transformer - feature-tokenizer Transformer (Gorishniy et al. 2021), small: every feature is one token, plus a CLS token."""
import torch
import torch.nn as nn

NAME, DESCRIPTION = "FT-Transformer", "Feature-tokenizer Transformer"


class Network(nn.Module):
    def __init__(self, n_features, n_classes, d=32, layers=2, heads=4, p=0.1):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.randn(n_features, d) * 0.1), nn.Parameter(torch.zeros(n_features, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x):
        t = torch.cat([self.cls.expand(len(x), -1, -1), x.unsqueeze(-1) * self.w + self.b], 1)
        return self.head(self.enc(t)[:, 0])
