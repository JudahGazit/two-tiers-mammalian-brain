from publication_paths import data_path, output_path
from dataloaders.brain_net import BrainNet
from projections.brain_stack import *
import projections.brain_tracts
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import scipy
import seaborn as sns
from style import *
from figures.utils import *
from figures.figure6 import illustrate_brain_as_two_hemispheres, illustrate_shape_from_points
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev

def spicies_datasets():
    df = pd.DataFrame({'name': BrainNet.options()})
    df['spicie'] = df['name'].str.replace('[0-9]', '', regex=True)
    df = df.groupby('spicie').agg(list)
    df['count'] = df['name'].apply(len)
    df = df[df['count'].gt(3)]
    return df


def load_aligned_brain(brain_name):
    mri_scan = load_mri(brain_name)
    dataset = BrainNet(brain_name, location=HUMAN_BRAINS_DIR if 'Human' in brain_name else BRAINS_LOCATIONS)
    flip = -1 if brain_name in FLIP_Z else 1
    positions = align_brain_hemispheres(dataset.positions, mri_scan) * [flip, 1, flip]
    return positions


def projection_validity_figure():
    errs = {}
    all_projections = {}
    datasets = spicies_datasets()
    for spicie in datasets.index:
        try:
            brains = datasets.loc[spicie]['name']
            aligned_brains = []
            for proj_brain_name in brains:
                proj_brain_aligned = load_aligned_brain(proj_brain_name)
                aligned_brains.append(proj_brain_aligned)
            for i, representative in enumerate(brains):
                projection = project_multiple_brains(brains, aligned_brains[i], cluster_method=lambda v, c: v)
                all_projections[representative] = {b: p for b, p in zip(brains, projection)}
                for j, (before, after) in enumerate(zip(aligned_brains, projection)):
                    if i != j:
                        mean_shortest_distance = scipy.spatial.distance.cdist(before, before).max()
                        err = np.sqrt(np.square((before - after)).sum(1)) / mean_shortest_distance
                        errs[(brains[i], brains[j])] = err
        except: pass

    fig, axes = plt.subplot_mosaic([['demo', 'demo_density', 'ensemble']], figsize=(15, 5), tight_layout=True)
    before_points = load_aligned_brain('Cat2')
    after_points = all_projections['Cat1']['Cat2']
    axes['demo'].scatter(*before_points[:, :2].T, label='Original')
    axes['demo'].scatter(*after_points[:, :2].T, label='Projected')
    annotate_point = np.stack([before_points, after_points])[:, 187, :2]
    axes['demo'].annotate('', annotate_point[1], annotate_point[0],
                          arrowprops=dict(arrowstyle="-|>, head_width=0.1, head_length=0.1", lw=2, color='k', shrinkA=0, shrinkB=0))
    axes['demo'].text(*annotate_point.mean(0), '$d_i$', va='top', ha='right')
    axes['demo'].legend(loc='upper center', bbox_to_anchor=(0.5, 0), frameon=False)
    axes['demo'].axis('off')
    axes['demo'].set_aspect('equal')

    sns.histplot(errs[('Cat1', 'Cat2')], ax=axes['demo_density'], kde=True, stat='count', line_kws=dict(color='k', lw=3))
    plt.setp(axes['demo_density'], xlabel='$\\tilde{d}_i$', ylabel='Frequency', xlim=(0, 0.3))

    errs_array = np.array(list(errs.values()))
    qs = np.array([np.quantile(e, 0.9) for e in errs_array])
    sns.kdeplot(errs_array[qs < 0.2].T, cumulative=True, legend=False, alpha=0.5, common_norm=False, ax=axes['ensemble'])
    x = np.linspace(-0, 0.3, 200)
    y = np.cumsum(scipy.stats.gaussian_kde(errs_array.flatten())(x)) * np.diff(x).mean()
    q90_point = x[(y > 0.9).argmax()]
    axes['ensemble'].plot(x, y, c='k', lw=3)
    axes['ensemble'].scatter(q90_point, 0.9, c='k', zorder=100)
    axes['ensemble'].plot([0, q90_point], [0.9] * 2, '--k')
    axes['ensemble'].plot([q90_point] * 2, [0, 0.9], '--k')
    axes['ensemble'].text(q90_point, 0.9, f' ({q90_point:.2f}, 0.9)', va='top', size=16)
    plt.setp(axes['ensemble'], xlabel='$\\tilde{d}_i$', ylabel='Density (CDF)', xlim=(0, 0.3))

    label_axes(fig, axes.values(), d=0.1)

    return fig


kde_values_cache = {}
kde_values_all_into_one_cache = {}
def projection_kde_heatmaps():
    orders = load_orders()
    all_brains = np.concatenate(list(orders.values())).tolist()

    fig, axes = plt.subplots(len(PROJ_BRAIN_NAMES), 3, figsize=(15, 3 * len(PROJ_BRAIN_NAMES)))

    for (order_name, representative), ax_row in zip(PROJ_BRAIN_NAMES.items(), axes):
        order_brains = orders[order_name]
        target_brain = load_aligned_brain(representative)
        if order_name not in kde_values_cache:
            kde_values_cache[order_name] = project_and_kde(order_brains, target_brain)
        if order_name not in kde_values_all_into_one_cache:
            kde_values_all_into_one_cache[order_name] = project_and_kde(all_brains, target_brain)
        positions, kde_values = kde_values_cache[order_name]
        positions_all, kde_values_all = kde_values_all_into_one_cache[order_name]
        top_kde_values = kde_values_all > np.quantile(kde_values_all, 0.9)
        top_nodes_heatmap = top_kde_values.reshape(positions_all.shape[1:])

        colorbar_images = []
        for ax, dim in zip(ax_row, ([0, 1], [1, 2], [0, 2])):
            node_heatmap_bounds = positions.reshape(3, -1).min(1)[dim], positions.reshape(3, -1).max(1)[dim]
            node_heatmap_bounds_img = (node_heatmap_bounds[0][0], node_heatmap_bounds[1][0], node_heatmap_bounds[0][1], node_heatmap_bounds[1][1])
            kde_values_plane = kde_values.reshape(*positions.shape[1:]).mean(3 - np.sum(dim)).T

            nodes_heatmap = (top_nodes_heatmap + top_nodes_heatmap[::-1]).mean(3 - np.sum(dim))
            nodes_heatmap = np.argwhere(nodes_heatmap > np.nanquantile(nodes_heatmap, 0.8)) / nodes_heatmap.shape
            nodes_heatmap = nodes_heatmap * (node_heatmap_bounds[1] - node_heatmap_bounds[0]) + node_heatmap_bounds[0]
            nodes_heatmap = alphashape.alphashape(nodes_heatmap, 1)
            nodes_heatmap = list(nodes_heatmap.geoms) if nodes_heatmap.geom_type == 'MultiPolygon' else [nodes_heatmap]

            ax_inner = ax.inset_axes((0, 0, 1, 1))
            image = ax_inner.imshow(kde_values_plane, extent=node_heatmap_bounds_img, origin='lower')
            colorbar_images.append(image)
            ax_inner.axis('off')
            ax.axis('off')
            illustrate_brain_as_two_hemispheres(target_brain, 'k', dim=dim, ax=ax_inner, alpha=0, border=True)

        title = f'{order_name}\n${{}}_{{(N = {len(order_brains)})}}$'
        ax_row[0].set_title(title, weight='bold', size=35, rotation='vertical', x=-0.15, y=0.5, va='center')
        cbar = fig.colorbar(colorbar_images[-1], ax=ax_row[-1], label='Density')
        cbar.ax.set_yticks([])
    titles = ['XY-plane', 'YZ-plane', 'XZ-plane']
    for title, ax in zip(titles, axes[0]):
        ax.text(0.5, 1.05, title, weight='bold', size=24, transform=ax.transAxes, ha='center')
    return fig

def project_and_kde(order_brains, target_brain):
    positions = generate_meshgrid(target_brain, factor=1.1)
    projections = project_multiple_brains(order_brains, target_brain, cluster_method=cluster_null)
    kde = scipy.stats.gaussian_kde(np.concatenate(projections).T)
    kde_values = kde(positions.reshape(3, -1))
    return positions, kde_values

def demo_without_projections():
    # brain_names = ['Eonycteris1', 'Mara', 'Lama2', 'FallowDeer1', 'BrownLemur1']
    brain_names = BrainNet.options()
    fig, axes = plt.subplots(len(brain_names), 3, figsize=(15, 3 * len(brain_names)))
    for brain_name, ax_row in zip(brain_names, axes):
        aligned_brain = load_aligned_brain(brain_name)
        positions, kde_values = project_and_kde([brain_name], aligned_brain)
        colorbar_images = []
        for ax, dim in zip(ax_row, ([0, 1], [1, 2], [0, 2])):
            node_heatmap_bounds = positions.reshape(3, -1).min(1)[dim], positions.reshape(3, -1).max(1)[dim]
            node_heatmap_bounds_img = (node_heatmap_bounds[0][0], node_heatmap_bounds[1][0], node_heatmap_bounds[0][1], node_heatmap_bounds[1][1])
            kde_values_plane = kde_values.reshape(*positions.shape[1:]).mean(3 - np.sum(dim)).T
            ax_inner = ax.inset_axes((0, 0, 1, 1))
            coreness = core_periphery_average_weight(BrainNet(brain_name))
            ax_inner.scatter(*aligned_brain[coreness][:, dim].T, c='C0', lw=1, ec='k')
            image = ax_inner.imshow(kde_values_plane, extent=node_heatmap_bounds_img, origin='lower')
            illustrate_brain_as_two_hemispheres(aligned_brain, 'k', dim=dim, ax=ax_inner, alpha=0, border=True)
            colorbar_images.append(image)
            ax_inner.set_aspect('equal')
            ax_inner.axis('off')
            ax.axis('off')
        title = brain_name
        ax_row[0].set_title(title, weight='bold', size=25, rotation='vertical', x=-0.15, y=0.5, va='center')
        cbar = fig.colorbar(colorbar_images[-1], ax=ax_row[-1], label='Density')
        cbar.ax.set_yticks([])
    titles = ['XY-plane', 'YZ-plane', 'XZ-plane']
    for title, ax in zip(titles, axes[0]):
        ax.text(0.5, 1.05, title, weight='bold', size=24, transform=ax.transAxes, ha='center')
    return fig

def sup_figure6():
    figs = [
        projection_validity_figure(),
        projection_kde_heatmaps(),
        demo_without_projections()
    ]
    return figs

if __name__ == '__main__':
    figs = sup_figure6()
    for i, fig in enumerate(figs):
        if fig is not None:
            fig.savefig(output_path(f'SuppFigures/figure6_{i}.png'), dpi=300, bbox_inches='tight')
