from publication_paths import data_path, output_path
from dataloaders.brain_net import BrainNet
import matplotlib.pyplot as plt
import numpy as np
import PIL
from urllib.request import urlopen
from figures.utils import *
from style import *
from core_periphery.core_periphery import core_periphery_average_weight
import matplotlib.patches as patches
import networkx as nx
import igraph
from tqdm.contrib.concurrent import thread_map
import functools


def all_shortest_paths(graph, i, group):
    paths = graph.get_all_shortest_paths(i, group)
    paths_ = {}
    for path in paths:
        target = path[-1]
        if target in paths_:
            paths_[target].append(path)
        else:
            paths_[target] = [path]
    return paths_


@functools.lru_cache(maxsize=280)
def fraction_of_paths_through_core(brain):
    try:
        brain = BrainNet(brain) if isinstance(brain, str) else brain
        coreness = core_periphery_average_weight(brain)
        core, periphery = np.where(coreness)[0], np.where(~coreness)[0]
        core_set = set(core)
        euclidean = scipy.spatial.distance.cdist(brain.positions, brain.positions)
        euclidean = euclidean / euclidean.max()
        paths_of_periphery = iter(all_shortest_paths(igraph.Graph.from_networkx(brain.graph), i, periphery) for i in periphery)
        apsp_is_through_core = {i: {j: [1 if len(core_set.intersection(path)) else 0 for path in paths[j]]
                                    for j in periphery if i < j}
                                for i, paths in zip(periphery, paths_of_periphery)}
        data = [[euclidean[i, j], np.mean(apsp_is_through_core[i][j])]
                for i in periphery for j in periphery if i < j]
        data = pd.DataFrame(data, columns=['dist', 'frac'])
        return data
    except KeyboardInterrupt: raise
    except: pass


def plot_number_of_paths_through_core_single(ax=None, animal='Giraffe3', n_bins=20):
    ax = ax or plt.gca()
    brain = BrainNet(animal)
    data = fraction_of_paths_through_core(brain)
    data['distance'] = pd.cut(data.dist, n_bins).apply(lambda v: v.mid).astype(float)
    data = data.groupby('distance').agg(['mean', 'std'])
    ax.fill_between(data.index, data['frac', 'mean'] - data['frac', 'std'],
                    data['frac', 'mean'] + data['frac', 'std'], alpha=0.2)
    ax.plot(data.index, data['frac', 'mean'])
    plt.setp(ax, xlabel='$d_{ij}$', ylabel='$\\Phi_{ij}$', xticks=np.linspace(0, 1, 3))

    aximage = ax.inset_axes((0.0, 0.8, 0.4, 0.4), )
    aximage.imshow(load_image(animal))
    aximage.axis('off')


def plot_number_of_paths_through_core_ensemble(ax=None, n_bins=20, n_brains=10, highlight=[], highlight_h=0.2):
    ax = ax or plt.gca()
    datas = []
    random_brains = np.random.choice(BrainNet.options(), n_brains, replace=False) if n_brains != 'all' else BrainNet.options()

    datas = thread_map(fraction_of_paths_through_core, random_brains)
    lines = {}
    for brain_name, data in zip(random_brains, datas):
        if data is not None:
            data = data.assign(name=brain_name)
            distance_bins = pd.cut(data.dist, n_bins).apply(lambda v: v.mid).astype(float)
            data = data.groupby(distance_bins)['frac'].mean()
            data.plot(ax=ax, c='k', alpha=0.05)
            lines[brain_name] = data

    data = pd.concat([data for data in datas if data is not None])
    distance_bins = pd.cut(data.dist, n_bins).apply(lambda v: v.mid).astype(float)
    data = data.groupby(distance_bins)['frac'].mean()
    data.plot(ax=ax, c='C2', lw=5)
    ax.legend([ax.get_lines()[-1]], ['Ensemble'], fontsize=24, frameon=False)
    plt.setp(ax, xlabel='$d_{ij}$', ylabel='$\\Phi_{ij}$')

    highlight_pd = pd.DataFrame([dict(name=name, y=lines[name].iloc[-1])
                                 for name in highlight if name in lines])
    highlight_pd = highlight_pd.sort_values('y').reset_index()

    for i, row in highlight_pd.iterrows():
        yimage = i / (len(highlight) - 1)
        ax.annotate('', (1, row['y']), (1.1, yimage), xycoords='data',
                    arrowprops=dict(arrowstyle='->',
                                    connectionstyle='arc,angleA=180,angleB=0,armA=20,armB=20,rad=10',
                                    color=(0.3, ) * 3, lw=2))
        aximage = ax.inset_axes((1.1, yimage - highlight_h / 2, 0.1, highlight_h), transform=ax.transData)
        aximage.imshow(load_image(row['name']))
        aximage.axis('off')



def core_periphery_in_circles(brain, n_bins):
    graph_density = nx.density(brain.graph)
    bin_size = brain.number_of_nodes // n_bins
    bins = range(0, brain.number_of_nodes + bin_size, bin_size)
    x = np.array(bins[1:])

    coreness = core_periphery_average_weight(brain)
    core, periphery = np.where(coreness)[0], np.where(~coreness)[0]
    euclidean = scipy.spatial.distance.cdist(brain.positions, brain.positions)
    all_links_to_core_by_euc = []
    all_links_to_periphery_by_euc = []
    for i in periphery:
        by_euclidean = euclidean[i].argsort()
        links_to_core = np.isin(by_euclidean, core) * brain.adjacency[i][by_euclidean] > 0
        links_to_periphery = np.isin(by_euclidean, periphery) * brain.adjacency[i][by_euclidean] > 0
        all_links_to_core_by_euc.append(links_to_core)
        all_links_to_periphery_by_euc.append(links_to_periphery)
    all_links_to_core_by_euc = np.array(all_links_to_core_by_euc)
    all_links_to_periphery_by_euc = np.array(all_links_to_periphery_by_euc)
    y_core = [all_links_to_core_by_euc[:, bins[i]:bins[i+1]].mean() / graph_density for i in range(len(bins) - 1)]
    y_periphery = [all_links_to_periphery_by_euc[:, bins[i]:bins[i+1]].mean() / graph_density for i in range(len(bins) - 1)]
    return x, y_core, y_periphery


def plot_core_periphery_in_circles(brains, n_bins=20, lw=3, legend=True, ax=None, individual=False):
    ax = plt.gca() if ax is None else ax
    brain_name = brains[0] if len(brains) == 1 and isinstance(brains[0], str) else None
    brains = brains if brains != 'all' else BrainNet.options()
    brains = [BrainNet(b) if isinstance(b, str) else b for b in brains]
    x = []
    ys = []
    for brain in brains:
        try:
            x, y_core_, y_periphery_ = core_periphery_in_circles(brain, n_bins)
            ys.append(dict(brain_name=brain.brain_name, core=y_core_, periphery=y_periphery_))
        except KeyboardInterrupt: raise
        except: pass

    mean_core = np.mean([y['core'] for y in ys], 0)
    mean_periphery = np.mean([y['periphery'] for y in ys], 0)
    x = np.arange(len(x)) + 1
    if individual:
        for y_data in ys:
            ax.plot(x, y_data['core'], c='C0', alpha=0.1)
            ax.plot(x, y_data['periphery'], c='C1', alpha=0.1)

        ax.plot(x, mean_core, c='k', lw=lw + 3)
        ax.plot(x, mean_periphery, c='k', lw=lw + 3)

    ax.plot(x, mean_core, c='C0', lw=lw, label='$\\mathcal{E}$')
    ax.plot(x, mean_periphery, c='C1', lw=lw, label='$\\mathcal{V}$')
    plt.setp(ax, xlabel='$r$', ylabel='$\\sigma(r)$', ylim=(0, 2))
    if legend:
        ax.legend(frameon=False)

    if brain_name is not None:
        ax_image = ax.inset_axes((1 - 0.4, 1, 0.4, 0.4))
        ax_image.imshow(load_image(brain_name))
        ax_image.axis('off')

def circles_frequency(brain_name, n_circles=8, radius=9, sample_ratio=1, ax=None):
    from figures.figure6 import illustrate_brain_as_two_hemispheres
    ax = plt.gca() if ax is None else ax
    ax.axis('off')
    ax.set_aspect('equal')

    im, brain, aligned = create_brain_background(brain_name)
    coreness = core_periphery_average_weight(brain)
    center_node = brain.degrees.argsort()[26]

    distances = scipy.spatial.distance.cdist(aligned, aligned)[center_node]
    by_distances = distances.argsort()
    circle_index = np.arange(brain.number_of_nodes) // (brain.number_of_nodes // n_circles)

    aligned_on_center = (aligned)[:, :2]
    extent_on_aligned = np.concatenate([aligned_on_center.min(0), aligned_on_center.max(0)])[[0, 2, 1, 3]]
    mask = np.ones_like(im)
    mask[:, :, -1] = (im == 1).all(2)

    for i in range(n_circles):
        radius = distances[by_distances[circle_index == i]].max()
        ax.add_patch(patches.Circle(aligned_on_center[center_node], radius, fill=False, zorder=0))

    ax.scatter(*aligned_on_center[center_node], s=200, color='k', zorder=20)
    points = illustrate_brain_as_two_hemispheres(aligned, 'k', ax, alpha=0.2)

    core_nei = aligned_on_center[coreness & (brain.adjacency[center_node] > 0)] * 0.9
    per_nei = aligned_on_center[~coreness & (brain.adjacency[center_node] > 0)] * 0.9
    ax.scatter(*core_nei.T, c='C0', zorder=15)
    ax.scatter(*per_nei.T, c='C1', zorder=15)
    for color, nei in zip(['C0', 'C1'], [core_nei, per_nei]):
        for n in nei:
            ax.plot(*np.stack([aligned_on_center[center_node], n]).T, c=color, zorder=15, lw=0.5)
    plt.setp(ax, xlim=extent_on_aligned[:2], ylim=extent_on_aligned[2:])
    ax.axis('equal')

def display_illustration(uri, ax):
    ax.imshow(np.array(PIL.Image.open(uri)))
    ax.set_aspect('equal', anchor='C')
    ax.axis('off')

def figure4():
    ILLUSTRATION_URL = data_path("images/figure3_illustration.png")
    ILLUSTRATION_PATHS_URL = data_path("images/figure4_path_illustraion.png")
    ILLUSTRATION_CIRCLES_URL = data_path("images/figure4_illustration.png")
    ILLUSTRATION_3D_SPHERES = data_path("images/figure4_3d_illustration.png")
    ILLUSTRATION_2D_SPHERES = data_path("images/figure4_2d_illustration.png")

    fig, axes = plt.subplot_mosaic([
        ['shortest_path_illustration', 'shortest_path_illustration', 'highways_illustration', 'highways_illustration'],
        ['shortest_path_illustration', 'shortest_path_illustration', 'highways_illustration', 'highways_illustration'],
        ['phi_a', 'phi_b', 'phi_all', 'phi_all'],
        ['phi_c', 'phi_d', 'phi_all', 'phi_all'],
        ['.'] * 4,
        ['illustrations'] * 4,
        ['illustrations'] * 4,
        ['sigma_a', 'sigma_b', 'sigma_all', 'sigma_all'],
        ['sigma_c', 'sigma_d', 'sigma_all', 'sigma_all'],
    ], figsize=(20, 35), facecolor='w', tight_layout=True,
    gridspec_kw=dict(wspace=0.8, hspace=0.8, height_ratios=[1, 1, 1, 1, 0, 1, 1, 1, 1]))

    for box_origin in ['phi_c', 'sigma_c']:
        box_position = axes[box_origin].get_position().bounds[:2] + np.array([-0.06, -0.03])
        fig.add_artist(patches.FancyBboxPatch(box_position, 0.9, 0.412,
                                            facecolor="w", lw=3, ec=(0.6, ) * 3, clip_on=False,
                                            zorder=-5,
                                            boxstyle='round,pad=0,rounding_size=0.02'))

    demo_brains = ('Deer1', 'BrownLemur3', 'Cynnopterus', 'Otter3')

    display_illustration(ILLUSTRATION_PATHS_URL, axes['shortest_path_illustration'])
    display_illustration(ILLUSTRATION_URL, axes['highways_illustration'])
    phi_ax = [v for k, v in axes.items() if 'phi_' in k and '_all' not in k]
    for ax_brain_distance, brain in zip(phi_ax, demo_brains):
        plot_number_of_paths_through_core_single(ax_brain_distance, brain, 15)
    plot_number_of_paths_through_core_ensemble(axes['phi_all'], 15, n_brains='all', highlight=demo_brains)

    sigma_ax = [v for k, v in axes.items() if 'sigma_' in k and '_all' not in k]
    for ax_brain_circle, brain in zip(sigma_ax, demo_brains):
        plot_core_periphery_in_circles([brain], 10, legend=False, ax=ax_brain_circle)
    plot_core_periphery_in_circles('all', 20, ax=axes['sigma_all'], lw=5, individual=True)

    illustration_axes = [axes['illustrations'].inset_axes((i / 3, 0.05, 0.25, 1)) for i in range(3)]
    display_illustration(ILLUSTRATION_3D_SPHERES, illustration_axes[0])
    display_illustration(ILLUSTRATION_2D_SPHERES, illustration_axes[1])
    display_illustration(ILLUSTRATION_CIRCLES_URL, illustration_axes[2])
    axes['illustrations'].axis('off')

    label_axes(fig, [
        axes['shortest_path_illustration'], axes['highways_illustration'],
        *[axes[f'phi_{i}'] for i in ('a', 'b', 'c', 'd', 'all')],
        *illustration_axes, *[axes[f'sigma_{i}'] for i in ('a', 'b', 'c', 'd', 'all')]
    ], remove_spines=True, d=0.01)
    return fig


if __name__ == "__main__":
    fig = figure4()
    fig.savefig(output_path('Figures/figure4.pdf'), dpi=300, bbox_inches='tight')
