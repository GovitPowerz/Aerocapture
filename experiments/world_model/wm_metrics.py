"""Scoring metrics for the world-model experiment (#113): ensemble CRPS, interval coverage, rank AUC."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def crps_ensemble(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of the empirical CDF of `samples` (axis 0 = members) at y: E|X - y| - E|X - X'| / 2."""
    m = samples.shape[0]
    x = np.sort(samples, axis=0)
    w = (2.0 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * (x.ndim - 1))
    spread = 2.0 / m**2 * (w * x).sum(axis=0)
    return np.asarray(np.abs(samples - y).mean(axis=0) - 0.5 * spread)


def central_coverage(samples: np.ndarray, y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Fraction of y inside each central interval of the ensemble (axis 0 = members), pooled over the rest."""
    out = np.empty(len(levels))
    for i, level in enumerate(levels):
        lo, hi = np.quantile(samples, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
        out[i] = np.mean((y >= lo) & (y <= hi))
    return out


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count one half)."""
    ranks = rankdata(scores)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
