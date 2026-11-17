"""
Synthetic controls for node-level linking profiles (Supplementary Section 5.2).

Every node keeps its position, degree k_i and potential partners, and selects k_i partners from
    w_ij ~ (1 - theta_i) exp(-lam d_ij) / Z_i + theta_i / (N - 1),
a mixture of a cost-driven and a cost-agnostic rule. lam is fitted to the brain's short-range decay.
    continuum control     theta_i ~ Uniform(0, 1)
    two-classes control   theta_i = 1 for explorers, 0 for villagers
"""
import numpy as np

from analysis.connectome import upper_pairs
from analysis.tail_test import fit_exponential, LEFT


def short_range_rate(D, A):
    i, j = upper_pairs(len(D))
    x, linked = D[i, j], A[i, j]
    return max(fit_exponential(x[x < LEFT], linked[x < LEFT])[1], 0.0)


def continuum_control(D, A, lam, rng):
    return mixture_network(D, A, rng.uniform(0, 1, len(D)), lam, rng)


def two_classes_control(D, A, expl, lam, rng):
    return mixture_network(D, A, expl.astype(float), lam, rng)


def mixture_network(D, A, theta, lam, rng):
    """Directed network: row i holds the k_i partners chosen by node i."""
    S = np.zeros_like(A)
    for i, k in enumerate(A.sum(1)):
        partners = np.flatnonzero(np.arange(len(D)) != i)
        weights = _mixture_weights(D[i, partners], theta[i], lam)
        S[i, rng.choice(partners, size=min(k, len(partners)), replace=False, p=weights)] = True
    return S


def _mixture_weights(d, theta, lam):
    decay = np.exp(-lam * d)
    w = (1 - theta) * decay / decay.sum() + theta / len(d)
    return w / w.sum()
