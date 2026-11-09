"""
scaler.py - turns raw numeric columns into the scale every model sees.

Fitted on TRAIN rows only (never on validation or test), then applied to all rows:
  1. a column with < 0.1% negative cells is treated as "non-negative": its negative cells become NaN (sensor glitches)
  2. NaN -> column median
  3. clip to the [1e-4, 1 - 1e-4] quantiles (kills absurd outliers)
  4. strongly skewed non-negative columns get a signed log transform
  5. constant columns are dropped; of every pair with |corr| > 0.98 one column is dropped
  6. z-score, then clip to [-10, 10]
"""
from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
from scipy.stats import skew as _skew


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Scaler:
    def __init__(self, clip_quantiles=(1e-4, 1 - 1e-4), skew_log_threshold=2.0, corr_threshold=0.98, zscore_clip=10.0):
        self.clip_quantiles, self.skew_log_threshold = clip_quantiles, skew_log_threshold
        self.corr_threshold, self.zscore_clip = corr_threshold, zscore_clip
        self.report: Dict = {}

    def fit(self, S: np.ndarray, features: List[str]) -> "Scaler":
        S = np.array(S, dtype=np.float64)
        S[~np.isfinite(S)] = np.nan
        n_features = S.shape[1]
        self.nonneg = (np.nan_to_num(S, nan=0.0) < 0).mean(0) < 1e-3
        S[:, self.nonneg] = np.where(S[:, self.nonneg] < 0, np.nan, S[:, self.nonneg])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median = np.nan_to_num(np.nanmedian(S, axis=0), nan=0.0)
        S = np.where(np.isnan(S), self.median, S)
        if self.clip_quantiles:
            self.lo, self.hi = (np.quantile(S, q, axis=0) for q in self.clip_quantiles)
        else:
            self.lo, self.hi = np.full(n_features, -np.inf), np.full(n_features, np.inf)
        S = np.clip(S, self.lo, self.hi)
        sk = np.nan_to_num(_skew(S, axis=0))
        self.log_cols = (np.abs(sk) > self.skew_log_threshold) & self.nonneg
        S[:, self.log_cols] = _signed_log(S[:, self.log_cols])
        keep = S.std(0) > 1e-6
        constant = [features[i] for i in np.flatnonzero(~keep)]
        if not keep.any():
            raise ValueError("Every feature is constant on the train split: check the label column and the CSV files.")
        active, dropped_corr = np.flatnonzero(keep), []
        if len(active) > 1:
            C = np.nan_to_num(np.abs(np.corrcoef(S[:, active], rowvar=False)))
            np.fill_diagonal(C, 0.0)
            var, alive = S[:, active].var(0), np.ones(len(active), bool)
            for i in np.argsort(-var):                      # keep the feature with the larger variance
                if not alive[i]:
                    continue
                for j in np.flatnonzero((C[i] > self.corr_threshold) & alive):
                    alive[j] = False
                    dropped_corr.append(f"{features[active[j]]} ~ {features[active[i]]}")
            keep[active[~alive]] = False
        self.kept = np.flatnonzero(keep)
        self.mu, self.sd = S[:, self.kept].mean(0), S[:, self.kept].std(0) + 1e-9
        self.kept_features = [features[i] for i in self.kept]
        self.report = dict(features_in=n_features, features_out=len(self.kept), constant_dropped=constant,
                           correlated_dropped=dropped_corr, log_transformed=[features[i] for i in np.flatnonzero(self.log_cols & keep)])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        B = np.array(X, dtype=np.float64)
        B[~np.isfinite(B)] = np.nan
        B[:, self.nonneg] = np.where(B[:, self.nonneg] < 0, np.nan, B[:, self.nonneg])
        B = np.where(np.isnan(B), self.median, B)
        B = np.clip(B, self.lo, self.hi)
        B[:, self.log_cols] = _signed_log(B[:, self.log_cols])
        Z = (B[:, self.kept] - self.mu) / self.sd
        return np.clip(Z, -self.zscore_clip, self.zscore_clip).astype(np.float32)
