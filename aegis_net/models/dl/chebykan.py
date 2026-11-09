"""ChebyKAN - Kolmogorov-Arnold network with Chebyshev polynomials, a gate and a linear residual path."""
import torch
import torch.nn as nn
import torch.nn.functional as F

NAME, DESCRIPTION = "ChebyKAN", "Kolmogorov-Arnold network with Chebyshev polynomials"


class _ChebyKANLayer(nn.Module):
    def __init__(self, n_in, n_out, degree):
        super().__init__()
        self.degree = degree
        self.coef = nn.Parameter(torch.randn(n_in, n_out, degree + 1) / (n_in * (degree + 1)) ** 0.5)

    def forward(self, x):
        x = torch.tanh(x)
        T = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            T.append(2 * x * T[-1] - T[-2])
        return torch.einsum("bik,ijk->bj", torch.stack(T[: self.degree + 1], -1), self.coef)


class Network(nn.Module):
    def __init__(self, n_features, n_classes, hidden=128, degree=4, p=0.15):
        super().__init__()
        self.in_norm, self.norm = nn.LayerNorm(n_features), nn.LayerNorm(hidden)
        self.kan1, self.kan2 = _ChebyKANLayer(n_features, hidden, degree), _ChebyKANLayer(hidden, n_classes, degree)
        self.res = nn.Linear(n_features, n_classes)
        self.gate = nn.Sequential(nn.Linear(n_features, n_classes), nn.Sigmoid())
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.drop(F.gelu(self.norm(self.kan1(self.in_norm(x)))))
        g = self.gate(x)
        return g * self.kan2(h) + (1 - g) * self.res(x)
