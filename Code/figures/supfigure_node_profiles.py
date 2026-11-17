"""
Supplementary Section 5.2 (node-level linking profiles).

Figure S_node_profiles.png
    (a) cumulative profiles G_i(u) of one brain, colored by class
    (b-d) pooled profiles of all brains on the first two principal components of the observed profiles:
          observed, continuum control, two-classes control
    (e) silhouette score of the explorer / villager partition across brains
Printed: silhouettes, cosine k-means vs. explorers, PCA, PC1 overlap between classes.
"""
import os
from publication_paths import data_path, output_path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from analysis import connectome, explorers, synthetic, node_profiles as profiles
from figures.revision_plots import shaded_density, class_legend, clean, COLOR_E, COLOR_V
from figures.utils import label_axes

OUT = os.path.dirname(output_path('SuppFigures/_'))
EXAMPLE = 'Macaque1'
NETWORKS = ('observed', 'continuum control', 'two-classes control')
HIST_COLORS = {'continuum control': '0.6', 'observed': 'C2', 'two-classes control': 'C0'}


def networks(C, expl, rng):
    """The observed network and its two synthetic controls (same order of random draws as before)."""
    lam = synthetic.short_range_rate(C.distances, C.adjacency)
    return {'observed': C.adjacency,
            'continuum control': synthetic.continuum_control(C.distances, C.adjacency, lam, rng),
            'two-classes control': synthetic.two_classes_control(C.distances, C.adjacency, expl, lam, rng)}


def analyze(name, rng):
    C = connectome.load(name)
    expl = explorers.explorers(C.distances, C.adjacency)
    row, pooled = dict(brain=name), {}
    for label, A in networks(C, expl, rng).items():
        G, nodes = profiles.cumulative_profiles(C.distances, A)
        Z, labels = profiles.standardize(G), expl[nodes]
        row[f'silhouette {label}'] = profiles.separation(Z, labels)
        row.update(flat_cluster_stats(Z, labels, label))
        pooled[label] = (G, labels)
        if label == 'observed':
            row['pc1_variance'], row['pc1_spearman_f'] = profiles.dominant_axis(Z, explorers.centrality(C.distances, A)[nodes])
    return row, pooled


def flat_cluster_stats(Z, labels, label):
    flat = profiles.flat_cluster(Z)
    return {f'flat share {label}': flat.mean(), f'explorers in flat {label}': (flat & labels).sum() / labels.sum(),
            f'jaccard {label}': (flat & labels).sum() / (flat | labels).sum()}


def run(seed=0):
    rng = np.random.default_rng(seed)
    rows, pooled = [], {label: ([], []) for label in NETWORKS}
    for name in tqdm(connectome.all_names()):
        row, brain_pooled = analyze(name, rng)
        rows.append(row)
        for label, (G, labels) in brain_pooled.items():
            pooled[label][0].append(G)
            pooled[label][1].append(labels)
    return pd.DataFrame(rows), {k: (np.vstack(G), np.concatenate(L)) for k, (G, L) in pooled.items()}


def figure(df, pooled):
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.3)
    axes = [example_panel(fig.add_subplot(gs[0, 0]))]
    pca = PCA(2).fit(pooled['observed'][0])
    lims = pca_limits(pca.transform(pooled['observed'][0]))
    for col, label in enumerate(NETWORKS):
        axes.append(pca_panel(fig.add_subplot(gs[0, col + 1]), pca, *pooled[label], lims, label))
    class_legend(axes[-1], outside=True)
    axes.append(silhouette_panel(fig.add_subplot(gs[1, 1:3]), df))
    label_axes(fig, axes, size=24, d=0.03)
    return fig


def example_panel(ax):
    C = connectome.load(EXAMPLE)
    G, nodes = profiles.cumulative_profiles(C.distances, C.adjacency)
    expl = explorers.explorers(C.distances, C.adjacency)[nodes]
    for g, e in zip(G, expl):
        ax.plot(profiles.U, g, color=COLOR_E if e else COLOR_V, alpha=0.35 if e else 0.2, lw=1)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.setp(ax, xlabel='$u$', ylabel='$G_i(u)$', xlim=(0, 1), ylim=(0, 1))
    ax.set_title(EXAMPLE, size=15)
    clean(ax)
    return ax


def pca_limits(Y):
    return [np.quantile(Y[:, 0], [0.002, 0.998]), np.quantile(Y[:, 1], [0.002, 0.998])]


def pca_panel(ax, pca, G, labels, lims, title):
    Y = pca.transform(G)
    shaded_density(ax, Y[~labels], COLOR_V, lims)
    shaded_density(ax, Y[labels], COLOR_E, lims)
    ax.set_title(title, size=15)
    plt.setp(ax, xlabel='PC1', ylabel='PC2', xlim=lims[0], ylim=lims[1])
    clean(ax)
    return ax


def silhouette_panel(ax, df):
    bins = np.linspace(-0.2, 0.9, 45)
    for label, color in HIST_COLORS.items():
        ax.hist(df[f'silhouette {label}'], bins, color=color, alpha=0.7, label=label)
    plt.setp(ax, xlabel='silhouette score', ylabel='frequency')
    ax.legend(frameon=False, fontsize=13)
    clean(ax)
    return ax


def pc1_overlap(pca, G, labels, lims, bins=80):
    pc1 = pca.transform(G)[:, 0]
    h = [np.histogram(pc1[s], bins, range=lims[0], density=True)[0] for s in (labels, ~labels)]
    return np.minimum(*h).sum() * np.diff(lims[0])[0] / bins


def summary(df, pooled):
    columns = [c for c in df.columns if c != 'brain']
    print(df[columns].median().round(3).to_string())
    o, c = df['silhouette observed'], df['silhouette continuum control']
    print(f'silhouette observed > continuum in {(o > c).sum()}/{len(df)}; IQR observed [{o.quantile(.25):.3f}, {o.quantile(.75):.3f}]')
    pca = PCA(2).fit(pooled['observed'][0])
    lims = pca_limits(pca.transform(pooled['observed'][0]))
    print(f'pooled: {len(pooled["observed"][0]):,} nodes; PCA variance {pca.explained_variance_ratio_.round(3)}')
    for label in NETWORKS:
        print(f'   PC1 overlap {label}: {pc1_overlap(pca, *pooled[label], lims):.2f}')


def main():
    df, pooled = run()
    figure(df, pooled).savefig(f'{OUT}/S_node_profiles.png', dpi=200, bbox_inches='tight')
    summary(df, pooled)


if __name__ == '__main__':
    main()
