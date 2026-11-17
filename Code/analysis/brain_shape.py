"""Brain elongation and the composition of the most distant node pairs (Supplementary Section 4.2)."""
import numpy as np

FAR = 0.9


def elongation(positions, hemi):
    """Anterior-posterior extent divided by left-right width, both from the node coordinates."""
    left_right = _unit(positions[hemi].mean(0) - positions[~hemi].mean(0))
    centered = positions - positions.mean(0)
    in_plane = centered - np.outer(centered @ left_right, left_right)
    long_axis = np.linalg.svd(in_plane, full_matrices=False)[2][0]
    return np.ptp(in_plane @ long_axis) / np.ptp(centered @ left_right)


def far_pairs_across(x, hemi, i, j):
    """Fraction of the most distant pairs (x > FAR) that lie in opposite hemispheres."""
    far = x > FAR
    return (hemi[i] != hemi[j])[far].mean()


def _unit(v):
    return v / np.linalg.norm(v)
