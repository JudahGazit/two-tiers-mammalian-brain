"""Access to the MaMI connectomes and to their node pairs."""
from publication_paths import data_path, output_path
import os
from dataclasses import dataclass

import numpy as np
import scipy.io
import scipy.spatial

from dataloaders.brain_net import BrainNet

MAT_DIR = data_path('brain_nets_mat')
CACHE_DIR = data_path('')


@dataclass
class Connectome:
    """A binary connectome with its fiber-length distances and node coordinates."""
    name: str
    adjacency: np.ndarray       # bool, symmetric, empty diagonal
    distances: np.ndarray       # normalized fiber-length distances d_ij
    positions: np.ndarray       # node centroids

    @property
    def size(self):
        return len(self.adjacency)


def load(name):
    """MaMI connectome as used in the main text (link if seen in >= 1 of 20 realizations)."""
    brain = BrainNet(name)
    return Connectome(name, adjacency_of(brain), brain.distances.astype(float), brain.positions)


def adjacency_of(brain):
    A = np.nan_to_num(brain.adjacency) > 0
    np.fill_diagonal(A, False)
    return A


def all_names():
    return list(BrainNet.options())


def realization_counts(name):
    """m_ij: number of the 20 tractography realizations in which i and j are linked."""
    conmat = scipy.io.loadmat(os.path.join(MAT_DIR, f'{name}_FinalFinal2.mat'))['conmatT']
    return (conmat > 0).sum(-1)


def euclidean(positions):
    """Euclidean distances normalized by the largest inter-node distance."""
    E = scipy.spatial.distance.cdist(positions, positions)
    return E / E.max()


def hemisphere(n):
    """Hemisphere label of each node: the first half of the nodes lies in one hemisphere."""
    return np.arange(n) >= n // 2


def upper_pairs(n):
    return np.triu_indices(n, 1)


def same_hemisphere(hemi, i, j):
    return hemi[i] == hemi[j]
