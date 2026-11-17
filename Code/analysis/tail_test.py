"""
Left-fit / right-test of the long-range tail (Supplementary Section 3.4).

1. Fit a single exponential P(A = 1 | x) = exp(a - b x) by maximum likelihood on pairs with x < 0.5.
2. On pairs with x > 0.7:
   excess   observed links / links predicted by the exponential, with a one-sided p-value
   rise     slope of a logistic regression of A on x, with a one-sided p-value for slope > 0
Per-brain p-values are combined with Stouffer's method.
"""
import numpy as np
import scipy.optimize
import scipy.stats

LEFT, RIGHT = 0.5, 0.7
EPS = 1e-9


def tail_test(x, linked):
    keep = np.isfinite(x)
    x, linked = x[keep], linked[keep]
    left, right = x < LEFT, x > RIGHT
    if linked[left].sum() < 10 or right.sum() < 50:
        return dict(excess_ratio=np.nan, excess_p=np.nan, rise_slope=np.nan, rise_p=np.nan)
    theta = fit_exponential(x[left], linked[left])
    ratio, p_excess = excess(x[right], linked[right], theta)
    slope, p_rise = rise(x[right], linked[right])
    return dict(excess_ratio=ratio, excess_p=p_excess, rise_slope=slope, rise_p=p_rise)


def fit_exponential(x, linked):
    def nll(t):
        p = np.clip(np.exp(t[0] - t[1] * x), EPS, 1 - EPS)
        return -(linked * np.log(p) + ~linked * np.log1p(-p)).sum()
    start = [np.log(max(linked.mean(), 1e-3)) + 1, 3]
    return scipy.optimize.minimize(nll, start, method='L-BFGS-B', bounds=[(-15, 5), (-50, 50)]).x


def excess(x, linked, theta):
    p = np.clip(np.exp(theta[0] - theta[1] * x), EPS, 1 - EPS)
    observed, expected, var = linked.sum(), p.sum(), (p * (1 - p)).sum()
    return observed / expected, scipy.stats.norm.sf((observed - expected) / np.sqrt(var))


def rise(x, linked):
    if linked.sum() < 3 or (~linked).sum() < 3:
        return np.nan, np.nan
    X = np.column_stack([np.ones_like(x), x - x.mean()])
    beta = scipy.optimize.minimize(lambda b: _logistic_nll(b, X, linked), [0, 0], method='BFGS').x
    se = _slope_standard_error(beta, X)
    return beta[1], scipy.stats.norm.sf(beta[1] / se)


def _logistic_nll(beta, X, y):
    z = X @ beta
    return np.sum(np.logaddexp(0, z)) - np.sum(y * z)


def _slope_standard_error(beta, X):
    p = 1 / (1 + np.exp(-(X @ beta)))
    fisher = X.T @ (X * (p * (1 - p))[:, None])
    return np.sqrt(np.linalg.inv(fisher)[1, 1])


def pooled_z(p_values):
    """Stouffer's combination of one-sided p-values (NaNs ignored)."""
    p = np.asarray(p_values, float)
    p = np.clip(p[np.isfinite(p)], 1e-300, 1 - 1e-16)
    return scipy.stats.norm.isf(p).sum() / np.sqrt(len(p))
