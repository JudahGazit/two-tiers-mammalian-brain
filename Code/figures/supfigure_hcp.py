"""
Supplementary Section 11 (external validation on the Human Connectome Project).

Atlas connectomes (Schaefer 200 and 400) of 100 HCP subjects; Euclidean distances between region
centroids, pairs within the same hemisphere; explorers by the mean Euclidean length of their links.
Figure: S_hcp.png. Printed: densities and tail tests quoted in the text.
"""
import os
from publication_paths import data_path, output_path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from analysis import hcp, explorers
from analysis.connectome import euclidean, upper_pairs
from analysis.density import binned_rho
from analysis.tail_test import tail_test, pooled_z
from figures.revision_plots import density_axes, ensemble_curve, class_densities, COLOR_ALL
from figures.utils import label_axes

OUT = os.path.dirname(output_path('SuppFigures/_'))


class AtlasGeometry:
    """Pairs of regions of an atlas, whole brain and within hemispheres, with their distances."""

    def __init__(self, atlas):
        self.regions, centroids = hcp.centroids(atlas)
        self.E = euclidean(centroids)
        self.i, self.j = upper_pairs(len(centroids))
        right = centroids[:, 0] > 0                      # MNI space: x > 0 is the right hemisphere
        self.same = right[self.i] == right[self.j]

    def subsets(self):
        x = self.E[self.i, self.j]
        return {'whole': (x, np.ones(len(x), bool)), 'intra': (x / x[self.same].max(), self.same)}


def analyze(subject, atlas, geometry):
    A = hcp.adjacency(subject, atlas, geometry.regions)
    expl = explorers.explorers(geometry.E, A)
    ee, vv = explorers.pair_classes(expl, geometry.i, geometry.j)
    linked = A[geometry.i, geometry.j]
    row, curves = dict(atlas=atlas, subject=subject, density=linked.mean()), {}
    for subset, (x, sel) in geometry.subsets().items():
        for part, mask in (('all', sel), ('E', sel & ee), ('V', sel & vv)):
            row.update({f'{subset}_{part}_{k}': v for k, v in tail_test(x[mask], linked[mask]).items()})
            curves[(subset, part)] = binned_rho(x[mask], linked[mask])
    return row, curves


def run():
    rows, curves = [], {}
    for atlas in hcp.ATLASES:
        geometry = AtlasGeometry(atlas)
        for subject in tqdm(hcp.subjects(), desc=atlas):
            row, curves[(atlas, subject)] = analyze(subject, atlas, geometry)
            rows.append(row)
    return pd.DataFrame(rows), curves


def figure(df, curves):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), gridspec_kw=dict(hspace=0.45, wspace=0.3))
    for row, atlas in zip(axes, hcp.ATLASES):
        subjects = df.subject[df.atlas == atlas]
        get = lambda part: [curves[(atlas, s)][('intra', part)] for s in subjects]
        ensemble_curve(row[0], get('all'), COLOR_ALL)
        class_densities(row[1], {p: get(p) for p in ('E', 'V')}, parts=('E', 'V'), superscript='^\\delta')
        for ax in row:
            ax.set_title(f'HCP, {atlas} ({len(subjects)})', size=15)
            density_axes(ax, r'$\rho^\delta(x)$')
    axes[0, 1].legend(frameon=False, fontsize=14)
    label_axes(fig, axes.ravel(), size=26, d=0.04)
    return fig


def summary(df):
    for atlas, g in df.groupby('atlas'):
        print(f'{atlas} ({len(g)} subjects): median density {g.density.median():.2f}')
        for subset in ('intra', 'whole'):
            print(f'   {subset}: ' + ' | '.join(part_summary(g, f'{subset}_{p}', p) for p in ('all', 'E', 'V')))


def part_summary(g, prefix, part):
    n = len(g)
    return (f'{part}: rise z={pooled_z(g[f"{prefix}_rise_p"]):.1f} ({(g[f"{prefix}_rise_slope"] > 0).sum()}/{n}), '
            f'excess {(g[f"{prefix}_excess_p"] < 0.01).sum()}/{n} x{g[f"{prefix}_excess_ratio"].median():.2f}')


def main():
    df, curves = run()
    figure(df, curves).savefig(f'{OUT}/S_hcp.png', dpi=200, bbox_inches='tight')
    summary(df)


if __name__ == '__main__':
    main()
