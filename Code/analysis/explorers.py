"""Fiber-length centrality and the explorer / villager partition (main text, Eq. 3)."""
import numpy as np

ALPHA = 0.3


def centrality(D, A):
    """f_i: mean distance spanned by the links of node i (NaN for isolated nodes)."""
    degree = A.sum(1)
    total = np.where(A, D, 0).sum(1)
    return np.where(degree > 0, total / np.maximum(degree, 1), np.nan)


def explorers(D, A, alpha=ALPHA):
    """Boolean mask of the top-alpha nodes by fiber-length centrality."""
    f = centrality(D, A)
    return f > np.nanquantile(f, 1 - alpha)


def pair_classes(expl, i, j):
    """Masks of explorer-explorer and villager-villager pairs."""
    return expl[i] & expl[j], ~expl[i] & ~expl[j]
