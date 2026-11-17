"""
Supplementary Section 5.5 (explorers, villagers and network modules).

Printed: Supplementary Table "Modularity of the explorer and villager networks" and the
participation statistics quoted in the text.
"""
from publication_paths import data_path, output_path
import numpy as np
import pandas as pd
import scipy.spatial
import scipy.stats
from tqdm.auto import tqdm

from analysis import connectome, explorers, modules


def analyze(name, rng):
    C = connectome.load(name)
    labels, q = modules.louvain(C.adjacency)
    expl = explorers.explorers(C.distances, C.adjacency)
    row = dict(brain=name, modularity=q, n_modules=len(np.unique(labels)), compactness=compactness(C, labels))
    for part, nodes in (('V', np.flatnonzero(~expl)), ('E', np.flatnonzero(expl))):
        row.update(class_modularity(C.adjacency, labels, nodes, part, rng))
    row.update(participation_stats(C, labels, expl))
    return row


def compactness(C, labels):
    """Mean distance between nodes of the same module, relative to the mean distance overall."""
    i, j = connectome.upper_pairs(C.size)
    E = scipy.spatial.distance.cdist(C.positions, C.positions)[i, j]
    return E[labels[i] == labels[j]].mean() / E.mean()


def class_modularity(A, labels, nodes, part, rng):
    G = modules.subnetwork(A, nodes)
    return {f'density_{part}': len(G.edges) / (len(nodes) * (len(nodes) - 1) / 2),
            f'Q_{part}': modules.modularity(G, labels),
            f'Q_random_{part}': modules.random_subset_baseline(A, labels, len(nodes), rng),
            f'Q_rewired_{part}': modules.rewiring_baseline(G, labels)}


def participation_stats(C, labels, expl):
    P, leaving = modules.participation(C.adjacency, labels)
    top = P > np.quantile(P, 1 - explorers.ALPHA)
    f = explorers.centrality(C.distances, C.adjacency)
    return dict(P_E=np.median(P[expl]), P_V=np.median(P[~expl]), leaving_E=np.median(leaving[expl]),
                leaving_V=np.median(leaving[~expl]), jaccard_E_topP=(expl & top).sum() / (expl | top).sum(),
                spearman_f_P=scipy.stats.spearmanr(f, P, nan_policy='omit').statistic)


def table(df):
    return pd.DataFrame({part: {'density': df[f'density_{part}'].median(), 'modularity': df[f'Q_{part}'].median(),
                                'random subsets': df[f'Q_random_{part}'].median(),
                                'degree-preserving rewiring': df[f'Q_rewired_{part}'].median()}
                         for part in ('V', 'E')}).T.round(3)


def comparisons(df):
    n = len(df)
    margin = (df.Q_V - df.Q_rewired_V) > (df.Q_E - df.Q_rewired_E)
    return {'Q_V > Q_E': (df.Q_V > df.Q_E).sum(), 'Q_V > random': (df.Q_V > df.Q_random_V).sum(),
            'Q_E < random': (df.Q_E < df.Q_random_E).sum(), 'margin over rewiring V > E': margin.sum(),
            'P_E > P_V': (df.P_E > df.P_V).sum(), 'brains': n}


def main():
    rng = np.random.default_rng(0)
    df = pd.DataFrame([analyze(n, rng) for n in tqdm(connectome.all_names())])
    df.to_csv(output_path('SuppTables/S_modules.csv'), index=False)
    print(table(df).to_string())
    print(comparisons(df))
    print(df[['modularity', 'n_modules', 'compactness', 'P_E', 'P_V', 'leaving_E', 'leaving_V',
              'jaccard_E_topP', 'spearman_f_P']].median().round(2).to_string())
    print('modules per brain, 10th-90th percentile:', df.n_modules.quantile([0.1, 0.9]).tolist())


if __name__ == '__main__':
    main()
