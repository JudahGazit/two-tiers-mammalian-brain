"""
Fixed-width histogram estimate of the fiber-length density over the full range x in [0, 1]:
rho(x) = P(A_ij = 1 | x) / S, evaluated in 12 bins. Bins with fewer than 20 pairs are left empty.
"""
import numpy as np

BINS = np.linspace(0, 1, 13)
CENTERS = 0.5 * (BINS[1:] + BINS[:-1])
MIN_PAIRS = 20


def binned_rho(x, linked):
    """x: pair distances (flat); linked: bool of the same shape. Returns rho at CENTERS."""
    keep = np.isfinite(x) & (x > 0)
    x, linked = np.clip(x[keep], 0, 1), linked[keep]
    bins = np.clip(np.digitize(x, BINS[1:-1]), 0, len(CENTERS) - 1)
    return np.array([_bin_value(linked[bins == b], linked.mean()) for b in range(len(CENTERS))])


def _bin_value(linked_in_bin, density):
    if len(linked_in_bin) < MIN_PAIRS:
        return np.nan
    return linked_in_bin.mean() / density


def ensemble_mean(curves):
    """Average of per-brain curves; bins defined in fewer than half of the brains are dropped."""
    Y = np.array(curves)
    mean = np.nanmean(Y, 0)
    mean[np.isfinite(Y).mean(0) <= 0.5] = np.nan
    return mean
