"""
Modularity of the explorer and villager sub-networks (Supplementary Section 5.5).

Each connectome is partitioned once (Louvain, best modularity of 10 runs). A sub-network is scored by
the modularity of its own links under that partition, and compared with two baselines:
    random subsets   random node sets of the same size
    rewiring         degree-preserving rewiring of the sub-network (keeps nodes, density and degrees)
"""
import networkx as nx
import numpy as np

LOUVAIN_RUNS = 10
N_BASELINE = 20


def louvain(A):
    G = nx.from_numpy_array(A.astype(int))
    runs = [nx.community.louvain_communities(G, seed=s) for s in range(LOUVAIN_RUNS)]
    best = max(runs, key=lambda c: nx.community.modularity(G, c))
    return _labels(best, len(A)), nx.community.modularity(G, best)


def _labels(communities, n):
    labels = np.empty(n, int)
    for m, nodes in enumerate(communities):
        labels[list(nodes)] = m
    return labels


def subnetwork(A, nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from((i, j) for i in nodes for j in nodes if i < j and A[i, j])
    return G


def modularity(G, labels):
    """Modularity of G under the partition `labels` (isolated nodes ignored)."""
    G = G.subgraph([n for n in G if G.degree(n) > 0])
    groups = {}
    for n in G:
        groups.setdefault(labels[n], set()).add(n)
    return nx.community.modularity(G, list(groups.values()))


def random_subset_baseline(A, labels, size, rng):
    draws = [rng.choice(len(A), size, replace=False) for _ in range(N_BASELINE)]
    return np.mean([modularity(subnetwork(A, d), labels) for d in draws])


def rewiring_baseline(G, labels):
    return np.mean([modularity(_rewired(G, seed), labels) for seed in range(N_BASELINE)])


def _rewired(G, seed):
    H = G.copy()
    m = H.number_of_edges()
    try:
        nx.double_edge_swap(H, nswap=10 * m, max_tries=100 * m, seed=seed)
    except nx.NetworkXAlgorithmError:
        pass
    return H


def participation(A, labels):
    """Participation coefficient and share of links leaving the node's own module."""
    degree = np.maximum(A.sum(1), 1)
    shares = np.array([A[:, labels == m].sum(1) / degree for m in np.unique(labels)])
    leaving = 1 - shares[np.searchsorted(np.unique(labels), labels), np.arange(len(A))]
    return 1 - (shares ** 2).sum(0), leaving
