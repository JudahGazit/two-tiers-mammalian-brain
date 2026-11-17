"""
Cumulative linking profiles (Supplementary Section 5.2).

For node i, partner j gets its percentile u_ij among i's own distances (0 = nearest, 1 = farthest).
    G_i(u) = fraction of i's links with u_ij <= u,    u = 0.1, ..., 0.9
A cost-agnostic node follows the diagonal G_i(u) = u. Profiles are standardized per bin within a brain
and compared with the cosine distance.
"""
import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

U = np.linspace(0.1, 0.9, 9)
MIN_LINKS = 3


def cumulative_profiles(D, A):
    """Raw profiles G (nodes x len(U)) and the indices of the nodes with at least MIN_LINKS links."""
    nodes = np.flatnonzero(A.sum(1) >= MIN_LINKS)
    return np.array([_profile(D, A, i) for i in nodes]), nodes


def _profile(D, A, i):
    others = np.arange(len(D)) != i
    u = scipy.stats.rankdata(D[i, others]) / others.sum()
    linked_u = np.sort(u[A[i, others]])
    return np.searchsorted(linked_u, U, side='right') / len(linked_u)


def standardize(G):
    sd = G.std(0)
    return (G - G.mean(0)) / np.where(sd > 0, sd, 1)


def separation(Z, labels):
    """Silhouette score of a binary partition under the cosine distance."""
    return silhouette_score(Z, labels.astype(int), metric='cosine')


def flat_cluster(Z):
    """Cosine k-means (k = 2): the cluster with the flatter profiles (fewer short-range links)."""
    unit = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    c = KMeans(2, n_init=20, random_state=0).fit_predict(unit)
    return c == np.argmin([Z[c == k].mean() for k in (0, 1)])


def dominant_axis(Z, f):
    """Variance explained by the first principal component and its |Spearman| correlation with f_i."""
    pca = PCA(2).fit(Z)
    pc1 = pca.transform(Z)[:, 0]
    return pca.explained_variance_ratio_[0], abs(scipy.stats.spearmanr(pc1, f).statistic)
