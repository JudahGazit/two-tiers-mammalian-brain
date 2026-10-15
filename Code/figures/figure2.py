from publication_paths import data_path, output_path
from dataloaders.brain_net import BrainNet
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import PIL
from urllib.request import urlopen
from figures.utils import *
from style import *
from core_periphery.core_periphery import core_periphery_average_weight
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

CORE_PERIPHERY_ILLUSTRATIONS = dict(core=data_path('images/figure_2_core_illustration.png'),
                                    periphery=data_path('images/figure_2_periphery_illustration.png'))

def create_aligned_background(brain_name, ax):
    im, brain, aligned = create_brain_background(brain_name)
    aligned_min, aligned_max = (aligned / 0.9).min(0), (aligned / 0.9).max(0)

    ax.imshow(im, extent=(aligned_min[0], aligned_max[0], aligned_min[1], aligned_max[1]), origin='lower')
    return aligned

def _pick_representative_nodes(brain):
    coreness = core_periphery_average_weight(brain)
    nodes = pd.DataFrame(dict(node=np.arange(brain.number_of_nodes),
                              degree=brain.degrees,
                              coreness=coreness.astype(int),
                              fblncent=np.nanmean(brain.fibers, 1)))
    core_nodes = nodes[nodes['coreness'].eq(1) & nodes['node'].lt(100)]
    periphery_nodes = nodes[nodes['coreness'].eq(0) & nodes['node'].gt(100)]
    nodes = core_nodes.merge(periphery_nodes, on='degree')
    nodes['diff'] = nodes.fblncent_x / nodes.fblncent_y
    c, p = nodes.sort_values('diff', ascending=False)[['node_x', 'node_y']].values[0]
    return c,p


def fibers_of_nodes_illustration(brain_name='Macaque10', ax=None):
    ax = plt.gca() if ax is None else ax
    aligned = create_aligned_background(brain_name, ax)
    brain = BrainNet(brain_name)

    gamma = np.linspace(0, 1, 50)
    ygamma = (np.sin(gamma * np.pi / 2 * 4) ** 2 +
              5 * np.sin(gamma * np.pi / 2 * 1))
    ygamma = ygamma / ygamma[-1]

    c, p = _pick_representative_nodes(brain)

    c_neighbors, p_neighbors = np.where(brain.fibers[c] > 0)[0], np.where(brain.fibers[p] > 0)[0]
    c_neighbors, p_neighbors = c_neighbors[brain.fibers[c][c_neighbors].argsort()], p_neighbors[brain.fibers[p][p_neighbors].argsort()]
    c_neighbors, p_neighbors = c_neighbors[-10:].tolist(), p_neighbors[:10].tolist()
    ax.scatter(*aligned[[c, p], :2].transpose(),
               c=['C0', 'C1'], s=300, linewidths=1, edgecolor='k', zorder=15)
    ax.scatter(*aligned[c_neighbors, :2].transpose(),
               c='w', s=70, linewidths=1, edgecolor='C0', zorder=10)
    ax.scatter(*aligned[p_neighbors, :2].transpose(),
               c='w', s=70, linewidths=1, edgecolor='C1', zorder=10)
    ax.annotate('2', aligned[c, :2] + np.array([-30, -30]), zorder=15,
                va='center', ha='center', fontsize=25, fontweight='bold', color='C0')
    ax.annotate('1', aligned[p, :2] + np.array([30, -30]), zorder=15,
                va='center', ha='center', fontsize=25, fontweight='bold', color='C1')
    for i, (node, neighbors, color) in enumerate(zip([c, p], [c_neighbors, p_neighbors], ['C0', 'C1'])):
        for neighbor in neighbors:
            x, y = aligned[[node, neighbor], :2].transpose()
            ax.plot(x[0] * (1 - gamma) + gamma * x[1],
                    y[0] * (1 - ygamma) + ygamma * y[1], color, alpha=0.8, lw=3)
    ax.axis('off')


def illustrate_nodes_by_colors(brain_name, nodes, ax=None, **kwargs):
    ax = plt.gca() if ax is None else ax
    aligned = create_aligned_background(brain_name, ax)
    scatter = ax.scatter(*aligned[nodes, :2].T, **kwargs)
    ax.axis('off')
    return scatter

def illustrate_nodes_fbcent_gradient(brain_name, ax=None):
    ax = plt.gca() if ax is None else ax
    colorbar_ax = ax.inset_axes((-0.2, 0, 0.1, 1))
    brain = BrainNet(brain_name)
    cmap = _cmap_between_main_colors()

    fbcent = np.nansum(brain.fibers, 0) / np.nansum(brain.adjacency, 0)
    degree = np.nansum(brain.adjacency, 0)
    i = illustrate_nodes_by_colors(brain_name, np.arange(brain.adjacency.shape[0]), ax=ax,
                                   c=fbcent, cmap=cmap, s=degree ** 1.3, lw=1, ec='k')
    colorbar = plt.colorbar(i, colorbar_ax, location='left')
    plt.setp(colorbar.ax, ylabel='$f_i$')

def _cmap_between_main_colors():
    rgb_from = list(plt.rcParams['axes.prop_cycle'])[1]['color']
    rgb_to = list(plt.rcParams['axes.prop_cycle'])[0]['color']
    cmap = LinearSegmentedColormap.from_list("my_cmap", [rgb_from, rgb_from, rgb_to, rgb_to], gamma=1)
    return cmap

def illustrate_nodes_core_periphery(brain_name, axes=None):
    axes = plt.subplots(1, 2, figsize=(10, 3))[1] if axes is None else axes
    brain = BrainNet(brain_name)
    degree = np.nansum(brain.adjacency, 0)
    coreness = core_periphery_average_weight(brain)
    illustrate_nodes_by_colors(brain_name, coreness, axes[0], c='C0',  s=(degree[coreness] ** 1.3) / 2, lw=1, ec='k')
    illustrate_nodes_by_colors(brain_name, ~coreness, axes[1], c='C1', s=(degree[~coreness] ** 1.3) / 2, lw=1, ec='k')

def plot_rho_in_subnetworks(brain_name, axes=None):
    axes = plt.subplots(1, 2, figsize=(10, 3))[1] if axes is None else axes
    brain = BrainNet(brain_name)
    coreness = core_periphery_average_weight(brain)
    plot_params = dict(bin_type='kde', style='-', lw=3, n_bins=8, trim_q=0.05)
    plot_fbln_dist(brain.distances[coreness][:, coreness], brain.fibers[coreness][:, coreness],
                   ax=axes[0], color='C0', **plot_params)
    plot_fbln_dist(brain.distances[~coreness][:, ~coreness], brain.fibers[~coreness][:, ~coreness],
                   ax=axes[1], color='C1', **plot_params)
    plt.setp(axes, xlabel='$x$', xlim=(None, 1), ylim=(0, 3.5))
    axes[0].set_ylabel('$\\rho_{\\mathcal{E}}(x)$')
    axes[1].set_ylabel('$\\rho_{\\mathcal{V}}(x)$')
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[1].spines[['top', 'right']].set_visible(False)
    for ax, kind in zip(axes, ['core', 'periphery']):
        illustration_ax = ax.inset_axes((0.7, 0.7, 0.3, 0.3))
        illustration_ax.axis('off')
        illustration_ax.imshow(plt.imread(CORE_PERIPHERY_ILLUSTRATIONS[kind]))

def _collect_connection_patterns(dataset, normalize_y=True, n_bins=10):
    distances = dataset.distances
    fibers = dataset.fibers
    all_bins, connection_patterns = [], []
    x = np.linspace(distances.min(), 0.9, n_bins)
    d_kde = logit_kde(distances[distances > 0])
    for i in range(dataset.number_of_nodes):
        e_kde = logit_kde(fibers[i][fibers[i] > 0])
        y = e_kde(x) / d_kde(x)
        y = (y / (np.diff(x).mean() * y.sum())) if normalize_y else y
        all_bins.append(x)
        connection_patterns.append(y)
    return np.array(all_bins), np.array(connection_patterns)


def plot_connection_patterns(dataset, color_core_periphery=True,
                             normalize_x=False, ax=None, alpha=0.2, coreness=None, default_color='C2'):
    ax = ax or plt.gca()
    bins, freqs = _collect_connection_patterns(dataset)
    min_distance, max_distance = bins[:, 0].mean(), bins[:, -1].mean()
    if normalize_x:
        bins = np.stack([np.linspace(0, 1, bins.shape[1])
                        for _ in range(len(bins))])
    colors = np.where(coreness, 'C0', 'C1') if color_core_periphery else default_color
    colors = [colors] * len(coreness) if isinstance(colors, (str, tuple)) else colors
    for i in range(len(coreness)):
        ax.plot(bins[i], freqs[i], c=colors[i], alpha=alpha)
    ax.legend(['Node Connections'])
    plt.setp(ax, xlabel='FbLn', ylabel='p', ylim=(-0.05, None),
             xlim=(0, 1) if normalize_x else (min_distance, max_distance))


def fiberlength_distributions_per_fiber(brain_name, n_bins=8, axes=None):
    axes = plt.subplots(1, 2, figsize=(10, 3))[1] if axes is None else axes
    cmap = _cmap_between_main_colors()
    brain = BrainNet(brain_name)
    fblen_centrality = np.nansum(brain.fibers, 1) / brain.degrees
    coreness = core_periphery_average_weight(brain)
    plot_connection_patterns(brain, ax=axes[0], normalize_x=False, color_core_periphery=False,
                                coreness=coreness, default_color=(0.5, ) * 3, alpha=0.1)
    plot_connection_patterns(brain, ax=axes[1], normalize_x=False, color_core_periphery=False, coreness=coreness,
                                default_color=cmap((fblen_centrality - fblen_centrality.min()) / np.ptp(fblen_centrality)))

    bins, freqs = _collect_connection_patterns(brain)
    nice_looking_fibers = np.where((freqs == 0).sum(1) == 0)[0]
    core_demo, periphery_demo = nice_looking_fibers[fblen_centrality[nice_looking_fibers].argmax()], nice_looking_fibers[fblen_centrality[nice_looking_fibers].argmin()]
    axes[0].plot(bins[core_demo], freqs[core_demo], 'C0', lw=4)
    axes[0].plot(bins[periphery_demo], freqs[periphery_demo], 'C1', lw=4)
    def arrow_style(c): return dict(transform=axes[0].transData, va='center',
                                    ha='center', arrowprops=dict(arrowstyle='->', color=c, lw=5))
    axes[0].annotate(f'$f_2 = {fblen_centrality[core_demo]:.2f}$',
                    (bins[core_demo][-3], freqs[core_demo][-3]), (0.7, 5), **arrow_style('C0'))
    axes[0].annotate(f'$f_1 = {fblen_centrality[periphery_demo]:.2f}$',
                    (bins[periphery_demo][:2].mean(), freqs[periphery_demo][:2].mean()), (0.55, 6), **arrow_style('C1'))

    plt.setp(axes, xlabel='$x$', ylabel='$\\rho_i(x)$',
             ylim=(0, 8), xlim=(None, 0.95))
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend().remove()

def fiberlength_distributions_split_global(axes, n_bins=10, core_bins=8, periphery_bins=10, finder=core_periphery_average_weight, n_brains=20):
    core_bins = core_bins or n_bins
    periphery_bins = periphery_bins or n_bins

    ax_villagers, ax_explorers = axes
    plt.setp([ax_villagers, ax_explorers], xlim=(0.2, 1), ylim=(0, 3),
             xticks=np.linspace(0.2, 1, 3), xlabel='$\\it{x}$', ylabel='$\\rho(x)$')
    ax_villagers.set_title('$\\mathcal{V}$', loc='right', size=35, color='C1')
    ax_explorers.set_title('$\\mathcal{E}$', loc='right', size=35, color='C0')

    all_es = {}
    random_brains = np.random.choice(BrainNet.options(), n_brains, replace=False) if n_brains != 'all' else BrainNet.options()
    for brain_name in random_brains:
        brain = BrainNet(brain_name)
        try:
            coreness = finder(brain)
            core, periphery = np.where(coreness)[0], np.where(~coreness)[0]
            distances, fibers = brain.distances, brain.fibers
            plot_params = dict(bin_type='kde', return_xy=True, alpha=0.05, style='-')
            x_core, y_core = plot_fbln_dist(distances[core][:, core], fibers[core][:, core], n_bins=core_bins,
                                            ax=ax_explorers, color='C0', trim_q=0.05, **plot_params)
            x_periphery, y_periphery = plot_fbln_dist(distances[periphery][:, periphery], fibers[periphery][:, periphery], n_bins=periphery_bins,
                                                      ax=ax_villagers, color='C1', **plot_params)
            all_es['core'] = all_es.get('core', []) + [(x_core, y_core)]
            all_es['periphery'] = all_es.get('periphery', []) + [(x_periphery, y_periphery)]
        except KeyboardInterrupt:
            raise
        except:
            pass

    for kind, ax, color in zip(['core', 'periphery'], [ax_explorers, ax_villagers], ['C0', 'C1']):
        x, y = zip(*all_es[kind])
        ax.plot(np.mean(x, 0), np.mean(y, 0), f'-{color}', lw=3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        illustration_ax = ax.inset_axes((0.82, 0.7, 0.3, 0.3))
        illustration_ax.axis('off')
        illustration_ax.imshow(plt.imread(CORE_PERIPHERY_ILLUSTRATIONS[kind]))

    return ax_villagers, ax_explorers


def network_illustration(ax=None, num_nodes=8, kind='core', color='C2'):
    ax = ax or plt.gca()
    thetas = np.linspace(0, 1, num_nodes + 1) * np.pi * 2
    positions = np.stack([np.sin(thetas), np.cos(thetas)]).T
    adjacency = np.zeros((len(positions), len(positions)))
    if kind == 'periphery':
        color = 'C1'
        for i in range(adjacency.shape[0] - 1):
            adjacency[i + 1, i] = adjacency[i, i + 1] = 1
    elif kind == 'core':
        color = 'C0'
        for i in range(adjacency.shape[0]):
            js = np.random.choice(adjacency.shape[0], 3, replace=False)
            js = js[(js != i) & (js != (i + 1) % num_nodes) & (js != (i - 1) % num_nodes)]
            adjacency[js, i] = adjacency[i, js] = 1
    nx.draw_networkx(nx.from_numpy_array(adjacency), pos=positions, with_labels=False,
                     node_color=color, node_size=1000, edge_color=color, width=5, ax=ax)
    ax.set_aspect('equal')
    ax.axis('off')


def figure2():
    fig, axes = plt.subplot_mosaic([
        ['fiber_illustration', 'two_nodes', '.', 'nodes_by_fbcent', 'rho_i_split'],
        ['.'] * 5,
        ['villager_nodes', 'villagers', '.', 'explorer_nodes', 'explorers'],
        ['.'] * 5,
        ['split_villagers', 'split_villagers', '.', 'split_explorers', 'split_explorers'],
        ],
        figsize=(20, 20), facecolor='w', tight_layout=True,
        gridspec_kw=dict(hspace=0.6, wspace=0.5, width_ratios=(1, 1, 0.2, 1, 1), height_ratios=(1, 0, 1, 0, 2)))
    box_position = axes['villager_nodes'].get_position().bounds[:2] + np.array([-0.06, -0.05])
    fig.add_artist(patches.FancyBboxPatch(box_position, 0.85, 0.5,
                                          facecolor="w", lw=3, ec=(0.6, ) * 3, clip_on=False,
                                          zorder=-5,
                                          boxstyle='round,pad=0,rounding_size=0.02'))

    demo_brain = 'Macaque10'
    img_axes = axes['rho_i_split'].inset_axes((0.82, 0.85, 0.08, 0.08), transform=fig.transFigure)
    img_axes.imshow(load_image(demo_brain))
    img_axes.axis('off')

    fibers_of_nodes_illustration(demo_brain, ax=axes['fiber_illustration'])
    illustrate_nodes_fbcent_gradient(demo_brain, ax=axes['nodes_by_fbcent'])
    illustrate_nodes_core_periphery(demo_brain, axes=[axes['explorer_nodes'], axes['villager_nodes']])
    plot_rho_in_subnetworks(demo_brain, axes=[axes['explorers'], axes['villagers']])
    fiberlength_distributions_per_fiber(demo_brain, axes=[axes['two_nodes'], axes['rho_i_split']])
    axsplit_axes = fiberlength_distributions_split_global(axes=[axes['split_villagers'], axes['split_explorers']], n_brains='all')
    label_axes(fig, list(axes.values())[:-1] + list(axsplit_axes), d=0.01)

    return fig


if __name__ == '__main__':
    fig = figure2()
    fig.savefig(output_path('Figures/figure2.pdf'), dpi=300, bbox_inches='tight')
