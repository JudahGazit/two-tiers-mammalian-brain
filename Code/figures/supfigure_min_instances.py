"""
Supplementary Section 3.5 (minimal number of tractography realizations).

A pair is linked only if it appears in at least m of the 20 realizations, m_ij >= m. Distances are
fixed; the explorer / villager partition is recomputed for each m.
Figure: S_min_instances.png (rho, rho_E, rho_V for m = 1, 2, 5, 10, 20).
Printed: Supplementary Table "Tail test under increasing link thresholds".
"""
import os
from publication_paths import data_path, output_path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from analysis import connectome, explorers
from analysis.density import binned_rho
from analysis.tail_test import tail_test, pooled_z
from figures.revision_plots import density_axes, ensemble_curve
from figures.utils import label_axes

OUT = os.path.dirname(output_path('SuppFigures/_'))
THRESHOLDS = (1, 2, 5, 10, 20)
TABLE_THRESHOLDS = (1, 2, 5, 10)
MIN_LINKS_FOR_PARTITION = 100


def analyze(name):
    C = connectome.load(name)
    counts = connectome.realization_counts(name)
    i, j = connectome.upper_pairs(C.size)
    rows, curves = [], {}
    for m in THRESHOLDS:
        A = counts >= m
        np.fill_diagonal(A, False)
        rows.append(dict(brain=name, m=m, density=A[i, j].mean(), **tail_test(C.distances[i, j], A[i, j])))
        curves.update({(m, part): c for part, c in class_curves(C.distances, A, i, j).items()})
    return rows, curves


def class_curves(D, A, i, j):
    curves = {'all': binned_rho(D[i, j], A[i, j])}
    if A[i, j].sum() < MIN_LINKS_FOR_PARTITION:
        return curves
    active = np.isfinite(explorers.centrality(D, A))
    expl = explorers.explorers(D, A)
    ee, vv = explorers.pair_classes(expl, i, j)
    vv &= active[i] & active[j]
    curves.update(E=binned_rho(D[i, j][ee], A[i, j][ee]), V=binned_rho(D[i, j][vv], A[i, j][vv]))
    return curves


def run():
    results = [analyze(n) for n in tqdm(connectome.all_names())]
    return pd.DataFrame([r for rows, _ in results for r in rows]), [c for _, c in results]


def figure(curves):
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(wspace=0.35))
    ylabels = {'all': r'$\rho(x)$', 'E': r'$\rho_{\mathcal{E}}(x)$', 'V': r'$\rho_{\mathcal{V}}(x)$'}
    for ax, (part, ylabel) in zip(axes, ylabels.items()):
        for n, m in enumerate(THRESHOLDS):
            label = rf'$m_{{ij}} \geq {m}$' + (' (main text)' if m == 1 else '')
            available = [c[(m, part)] for c in curves if (m, part) in c]
            ensemble_curve(ax, available, cmap(n / (len(THRESHOLDS) - 1)), label, individual=False, lw=3 if m == 1 else 2)
        density_axes(ax, ylabel, (0, 4))
    axes[0].legend(frameon=False, fontsize=12)
    label_axes(fig, axes, size=26, d=0.04)
    return fig


def table(df):
    return pd.DataFrame([table_row(m, df[df.m == m]) for m in TABLE_THRESHOLDS]).set_index('m')


def table_row(m, g):
    return dict(m=m, density=round(g.density.median(), 3),
                excess_significant=f'{(g.excess_p < 0.01).sum()}/{len(g)}', excess_ratio=round(g.excess_ratio.median(), 1),
                rise_z=round(pooled_z(g.rise_p), 1), positive_slopes=f'{(g.rise_slope > 0).sum()}/{len(g)}')


def main():
    df, curves = run()
    figure(curves).savefig(f'{OUT}/S_min_instances.png', dpi=200, bbox_inches='tight')
    pd.set_option('display.width', 200)
    print(table(df).to_string())
    print('median density at m >= 20:', round(df[df.m == 20].density.median(), 3))


if __name__ == '__main__':
    main()
