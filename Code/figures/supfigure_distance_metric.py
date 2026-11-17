"""
Supplementary Section 4 (robustness to distance metric) and 5.3 (explorers and villagers under
Euclidean distance).

Figures (Paper/SuppFigures)
    S_euclid_orders.png   rho^delta(x) per order, whole brain and within hemispheres
    S_euclid_shape.png    brain elongation vs. far pairs across hemispheres and vs. the tail slope
    S_euclid_ev.png       rho^delta, rho^delta_E, rho^delta_V, whole brain and within hemispheres
Printed: Supplementary Table "Euclidean fiber-length density by order".
"""
import os
from publication_paths import data_path, output_path
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from analysis import connectome, explorers, brain_shape
from analysis.density import binned_rho
from analysis.tail_test import tail_test, pooled_z
from figures.revision_plots import density_axes, ensemble_curve, class_densities, clean, COLOR_ALL
from figures.utils import label_axes

OUT = os.path.dirname(output_path('SuppFigures/_'))
ORDERS = ('Primates', 'Carnivora', 'Artiodactyla', 'Chiroptera', 'Rodentia')
SUBSETS = {'whole': 'whole brain', 'intra': 'within hemispheres'}


def euclidean_pairs(C):
    """Pair distances for the whole brain and within hemispheres (each normalized by its maximum)."""
    i, j = connectome.upper_pairs(C.size)
    x = connectome.euclidean(C.positions)[i, j]
    same = connectome.same_hemisphere(connectome.hemisphere(C.size), i, j)
    return {'whole': (x, i, j), 'intra': (x[same] / x[same].max(), i[same], j[same])}


def analyze(name, order):
    C = connectome.load(name)
    expl = explorers.explorers(C.distances, C.adjacency)
    hemi = connectome.hemisphere(C.size)
    row = dict(brain=name, order=order, elongation=brain_shape.elongation(C.positions, hemi))
    curves = {}
    for subset, (x, i, j) in euclidean_pairs(C).items():
        linked = C.adjacency[i, j]
        row.update({f'{subset}_{k}': v for k, v in tail_test(x, linked).items()})
        curves.update(class_curves(subset, x, linked, expl, i, j))
    x, i, j = euclidean_pairs(C)['whole']
    row['far_across'] = brain_shape.far_pairs_across(x, hemi, i, j)
    return row, curves


def class_curves(subset, x, linked, expl, i, j):
    ee, vv = explorers.pair_classes(expl, i, j)
    masks = {'all': np.ones(len(x), bool), 'E': ee, 'V': vv}
    return {(subset, part): binned_rho(x[m], linked[m]) for part, m in masks.items()}


def run():
    orders = pd.read_csv(data_path('MamiNum_formatted.csv')).set_index('brain_name').Order.to_dict()
    results = [analyze(n, orders.get(n, 'other')) for n in tqdm(connectome.all_names())]
    df = pd.DataFrame([r for r, _ in results])
    curves = {name: c for name, (_, c) in zip(df.brain, results)}
    return df, curves


def orders_figure(df, curves):
    fig, axes = plt.subplots(2, len(ORDERS), figsize=(4.2 * len(ORDERS), 8.5), gridspec_kw=dict(hspace=0.55, wspace=0.35))
    for col, order in enumerate(ORDERS):
        group = df[df.order == order]
        for row, (subset, title) in enumerate(SUBSETS.items()):
            ax = axes[row, col]
            ensemble_curve(ax, [curves[n][(subset, 'all')] for n in group.brain], COLOR_ALL)
            ax.set_title(f'{order} ({len(group)})\n{title}', size=16)
            ax.text(0.95, 0.92, f'$z={pooled_z(group[f"{subset}_rise_p"]):.1f}$', transform=ax.transAxes, ha='right', size=15)
            density_axes(ax, r'$\rho^{\delta}(x)$')
    label_axes(fig, axes.ravel(), size=26, d=0.04)
    return fig


def shape_figure(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(wspace=0.4))
    panels = (('far_across', 'share of farthest pairs\nacross hemispheres'), ('whole_rise_slope', 'tail slope (whole brain)'))
    for ax, (column, ylabel) in zip(axes, panels):
        scatter_by_order(ax, df, column)
        rho = scipy.stats.spearmanr(df.elongation, df[column], nan_policy='omit').statistic
        ax.text(0.95, 0.92, f'Spearman $={rho:.2f}$', transform=ax.transAxes, ha='right', size=14)
        plt.setp(ax, xlabel='brain length / width', ylabel=ylabel)
    axes[1].axhline(0, color='0.7', ls=':', lw=1)
    axes[0].legend(frameon=False, fontsize=11, loc='lower left')
    label_axes(fig, axes, size=26, d=0.04)
    return fig


def scatter_by_order(ax, df, column):
    colors = {o: f'C{k}' for k, o in enumerate(ORDERS)}
    for order, group in df.groupby('order'):
        ax.scatter(group.elongation, group[column], s=22, alpha=0.75, edgecolor='none',
                   color=colors.get(order, '0.75'), label=order if order in colors else None)
    clean(ax)


def ev_figure(df, curves):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(wspace=0.35))
    for ax, (subset, title) in zip(axes, SUBSETS.items()):
        by_part = {p: [curves[n][(subset, p)] for n in df.brain] for p in ('all', 'E', 'V')}
        class_densities(ax, by_part, superscript='^{\\delta}')
        ax.set_title(f'{title} ({len(df)})', size=16)
        density_axes(ax, r'$\rho^{\delta}(x)$')
    axes[0].legend(frameon=False, fontsize=14)
    label_axes(fig, axes, size=26, d=0.04)
    return fig


def order_table(df):
    groups = [(o, df[df.order == o]) for o in ORDERS] + [('All brains', df)]
    return pd.DataFrame([table_row(name, g) for name, g in groups]).set_index('order')


def table_row(name, g):
    rise = {s: f'{pooled_z(g[f"{s}_rise_p"]):.1f} ({(g[f"{s}_rise_slope"] > 0).sum()}/{len(g)})' for s in SUBSETS}
    return dict(order=name, brains=len(g), elongation=round(g.elongation.median(), 2),
                far_pairs_across=round(g.far_across.median(), 2), rise_whole=rise['whole'], rise_within=rise['intra'],
                excess_ratio_whole=round(g.whole_excess_ratio.median(), 2), excess_ratio_within=round(g.intra_excess_ratio.median(), 2))


def main():
    df, curves = run()
    orders_figure(df, curves).savefig(f'{OUT}/S_euclid_orders.png', dpi=200, bbox_inches='tight')
    shape_figure(df).savefig(f'{OUT}/S_euclid_shape.png', dpi=200, bbox_inches='tight')
    ev_figure(df, curves).savefig(f'{OUT}/S_euclid_ev.png', dpi=200, bbox_inches='tight')
    pd.set_option('display.width', 250)
    print(order_table(df).to_string())
    print('excess significant (p < 0.01), whole brain:', (df.whole_excess_p < 0.01).sum(), '/', len(df))
    for column in ('far_across', 'whole_rise_slope'):
        print(f'Spearman(elongation, {column}) =', round(scipy.stats.spearmanr(df.elongation, df[column], nan_policy='omit').statistic, 2))


if __name__ == '__main__':
    main()
