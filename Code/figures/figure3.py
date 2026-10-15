from publication_paths import data_path, output_path
from dataloaders.brain_net import BrainNet
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import PIL
from urllib.request import urlopen
from figures.utils import *
from style import *
from core_periphery.core_periphery import core_periphery_average_weight
from projections.brain_alignment import align_brain_hemispheres
from metrics.wiring_cost import WiringCost
from metrics.routing_cost import RoutingCost
import networkx as nx

GREY_COLOR = (0.4, 0.4, 0.4, 0.2)
CORE_PERIPHERY_COLOR = '#275317'


def skew(points, scale_x=0.7, scale_y=0.3, skewness=1.6, rotation=-15):
    rotation = np.array([[np.cos(rotation / 180 * np.pi), -np.sin(rotation / 180 * np.pi)],
                         [np.sin(rotation / 180 * np.pi), np.cos(rotation / 180 * np.pi)]])
    scaling = np.array([[scale_x, 0], [0, scale_y]])
    shear = np.array([[1, skewness], [0, 1]])
    transform = np.array([[10], [1]])
    skewed = rotation @ shear @ scaling @ points[:, :2].transpose() + transform
    return skewed.transpose()


def __plot_brain_shape(brain, pos, brain_shape, ax: plt.Axes):
    for hemisphere in brain_shape:
        ax.fill(*hemisphere, c=GREY_COLOR)
    nx.draw_networkx_nodes(brain.graph, pos=pos, node_color='grey', node_size=brain.degrees, ax=ax,
                           alpha=0.1)

def __brain_splines(points, hemispheres):
    hemis = [points[hemispheres == hemi] for hemi in range(2)]
    hemi_points = []
    for i in range(2):
        points_of_hemi = np.concatenate([hemis[i], hemis[1 - i] * np.array([-1, 1])])
        hemi_points.append(spline_shape_of_points(points_of_hemi, force_sign=True))
    return hemi_points

def __plot_brain_network(brain, brain_shape, hemispheres, xys, node_list=None, node_color='C0', edge_color='C0', ax=None, nodes_kwargs={}, edges_kwargs={}):
    node_list = node_list if node_list is not None else range(brain.number_of_nodes)
    pos = {i: xys[i] for i in range(brain.number_of_nodes)}
    edge_list = [(u, v)
                 for u, v in brain.graph.edges if u in node_list and v in node_list]

    __plot_brain_shape(brain, pos, brain_shape, ax)

    plot_params = dict(G=brain.graph, nodelist=node_list, pos=pos, ax=ax, node_size=brain.degrees[node_list])
    nx.draw_networkx_nodes(node_color=node_color, linewidths=1.5, edgecolors='k', **plot_params, **nodes_kwargs)
    nx.draw_networkx_edges(edgelist=edge_list, width=0.1, **plot_params, **edges_kwargs,
                           edge_color=[edge_color if hemispheres[u] == hemispheres[v] else GREY_COLOR for u, v in edge_list], alpha=0.1)


def brain_separation_illustration(ax_whole, ax_split, brain, coreness, h=1.2):
    for subax in (ax_whole, ax_split):
        subax.axis('off')
    coreness = coreness
    core, periphery = np.where(coreness)[0], np.where(~coreness)[0]
    positions_aligned = align_brain_hemispheres(brain.positions)
    xys = positions_aligned[:, :2]
    hemispheres = np.arange(len(coreness)) < len(coreness) // 2
    core_shift = [0, 60]

    brain_shape = __brain_splines(xys, hemispheres)
    brain_shape_skew = [skew(p.T).T for p in brain_shape]
    brain_shape_skew_shift = [(skew(p.T) + core_shift).T for p in brain_shape]

    __plot_brain_network(brain, brain_shape, hemispheres, xys, node_color='grey', edge_color='grey', ax=ax_whole)
    __plot_brain_network(brain, brain_shape_skew_shift, hemispheres, skew(xys) + core_shift,  node_list=core, node_color='C0', edge_color='C0', ax=ax_split)
    __plot_brain_network(brain, brain_shape_skew, hemispheres, skew(xys), node_list=periphery, node_color='C1', edge_color='C1', ax=ax_split)
    core_periphery_links = np.argwhere(brain.subset_edges(core, periphery).adjacency)
    nx.draw_networkx_edges(brain.graph,
                           pos=skew(xys) + coreness[:, None] * core_shift,
                           edgelist=core_periphery_links[np.random.choice(len(core_periphery_links), 1000, False)],
                           width=0.1, edge_color=CORE_PERIPHERY_COLOR, alpha=0.5, ax=ax_split)
    text_x = (skew(xys) + core_shift).min(0)[0] - 10
    ax_split.text(text_x, (skew(xys) + core_shift).mean(0)[1],
                  '$\\mathcal{E}$', color='C0', va='bottom', ha='center', size=25)
    ax_split.text(text_x, (skew(xys)).mean(0)[1],
                  '$\\mathcal{V}$', color='C1', va='top', ha='center', size=25)

def brain_separation_metrics(ax_illustration, axes, brain_name='Deer1', q_trim=0.01):
    brain = BrainNet(brain_name)
    coreness = core_periphery_average_weight(brain)
    brain_separation_illustration(*ax_illustration, brain, coreness)
    core, periphery = np.where(coreness)[0], np.where(~coreness)[0]
    core_density, periphery_density = nx.density(brain.subset(core).graph), nx.density(brain.subset(periphery).graph)
    axes[0].bar(range(2), [core_density, periphery_density],
                color=['C0', 'C1'])
    axes[0].text(0, core_density, '$S_{\\mathcal{E}}$',
                 va='bottom', ha='center', color='C0')
    axes[0].text(1, periphery_density, '$S_{\\mathcal{V}}$',
                 va='bottom', ha='center', color='C1')
    plt.setp(axes[0], ylim=(0, 0.6), xlim=(-1, 2), xticks=[], ylabel='Density')
    axes[0].set_yticks(np.linspace(*axes[0].get_ylim(), 3))

    wiring_core = 1 - WiringCost(brain.subset(core)).cost().normalized_value
    wiring_periphery = 1 - WiringCost(brain.subset(periphery)).cost().normalized_value
    axes[1].bar(range(2), [wiring_core, wiring_periphery], color=['C0', 'C1'])
    axes[1].text(0, wiring_core, '$w_{\\mathcal{E}}$',
                 va='bottom', ha='center', color='C0')
    axes[1].text(1, wiring_periphery, '$w_{\\mathcal{V}}$',
                 va='bottom', ha='center', color='C1')
    plt.setp(axes[1], ylim=(0, 0.6), xlim=(-1, 2),
             xticks=[], ylabel='Wiring')
    axes[1].set_yticks(np.linspace(0, 0.6, 3))

    routing_core = RoutingCost(brain.subset(core)).cost().normalized_value
    routing_periphery = RoutingCost(brain.subset(periphery)).cost().normalized_value
    axes[2].bar(range(2), [routing_core, routing_periphery],
                color=['C0', 'C1'])
    axes[2].text(0, routing_core, '$r_{\\mathcal{E}}$',
                 va='bottom', ha='center', color='C0')
    axes[2].text(1, routing_periphery, '$r_{\\mathcal{V}}$',
                 va='bottom', ha='center', color='C1')

    plt.setp(axes[2], ylim=(0, 1.1), xlim=(-1, 2),
             xticks=[], ylabel='Routing')
    axes[2].set_yticks(np.linspace(0, 1, 3))

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)


@functools.lru_cache()
def collect_ensemble_metrics(n_brains):
    S, W, R = [], [], []
    annotations = {}
    random_brains = np.random.choice(BrainNet.options(), n_brains, replace=False) if n_brains != 'all' else BrainNet.options()
    for brain_name in random_brains:
        brain = BrainNet(brain_name)
        try:
            coreness = core_periphery_average_weight(brain)
            core, periphery = np.where(coreness)[0], np.where(~coreness)[0]
            density_core = 2 * \
                brain.subset(core).number_of_edges / \
                ((len(core)) * (len(core) - 1))
            density_periphery = 2 * \
                brain.subset(periphery).number_of_edges / \
                ((len(periphery)) * (len(periphery) - 1))

            wiring_core = 1 - WiringCost(brain.subset(core)).cost().normalized_value
            wiring_periphery = 1 - WiringCost(brain.subset(periphery)).cost().normalized_value
            routing_core = RoutingCost(brain.subset(core)).cost().normalized_value
            routing_periphery = RoutingCost(brain.subset(periphery)).cost().normalized_value
            if brain_name in brains_to_silhouette:
                annotations[brain_name] = len(S)
            S.append([density_periphery, density_core])
            W.append([wiring_periphery, wiring_core])
            R.append([routing_periphery, routing_core])
        except KeyboardInterrupt:
            raise
        except:
            pass

    S, W, R = np.array(S), np.array(W), np.array(R)
    return S, W, R, annotations


def brain_ensemble_metrics(ax_ensemble_metrics, n_brains='all'):
    S, W, R, annotations = collect_ensemble_metrics(n_brains)

    demo_brains = np.random.choice(list(annotations.keys()), 3, replace=False)
    for ax, data, lim, letter, brain_name in zip(ax_ensemble_metrics, [S, W, R], [(0, 0.6), (0, 0.7), (0.5, 1)], ['S', 'w', 'r'], demo_brains):
        h = (lim[1] - lim[0]) * 0.3
        ax.scatter(*data.transpose(), color='C2', alpha=0.5)
        ax.plot([0, 2], [0, 2], '-k')
        ax.text(lim[0], lim[0] + 0.03 * (lim[1] - lim[0]), '$y=x$',
                rotation=45, va='bottom')
        index = annotations[brain_name]
        ax.scatter([data[index][0]], [data[index][1]], color='C2',
                   s=150, alpha=1, linewidths=3, edgecolors='k', zorder=5)
        image_position = (0.6 * (lim[1] - lim[0]) + lim[0],
                          0.3 * (lim[1] - lim[0]) + lim[0])
        aximage = ax.inset_axes((image_position[0] - h / 2, image_position[1] - h / 2, h, h),
                                transform=ax.transData)
        image = np.array(load_image(brain_name))
        if image.shape[-1] == 3:
            image = np.stack([image[:, :, 0], image[:, :, 1], image[:, :, 2],
                              255 * (~(image == 255).all(2))], axis=2)
        aximage.imshow(image)
        aximage.axis('off')
        ax.annotate('', image_position, tuple(data[index]), xycoords='data', textcoords='data', arrowprops=dict(
            arrowstyle='<|-', color='k', shrinkA=10), zorder=5)
        plt.setp(ax, xlim=lim, ylim=lim, xlabel=f'${letter}_{{\\mathcal{{V}}}}$',
                 ylabel=f'${letter}_{{\\mathcal{{E}}}}$', aspect='equal')


def figure3():
    fig, axes = plt.subplot_mosaic([
        ['illustration_a', 'illustration_a', 'illustration_a', 'separation_a', 'separation_a', 'separation_a', '.', 'global_density', 'global_density', 'global_density'],
        ['illustration_a', 'illustration_a', 'illustration_a', 'separation_a', 'separation_a', 'separation_a', '.', 'global_density', 'global_density', 'global_density'],
        ['density_a', 'density_a', 'wiring_a', 'wiring_a', 'routing_a', 'routing_a',                           '.', 'global_wiring', 'global_wiring', 'global_wiring'],
        ['illustration_b', 'illustration_b', 'illustration_b', 'separation_b', 'separation_b', 'separation_b', '.', 'global_wiring', 'global_wiring', 'global_wiring'],
        ['illustration_b', 'illustration_b', 'illustration_b', 'separation_b', 'separation_b', 'separation_b', '.', 'global_routing', 'global_routing', 'global_routing'],
        ['density_b', 'density_b', 'wiring_b', 'wiring_b', 'routing_b', 'routing_b',                           '.', 'global_routing', 'global_routing', 'global_routing'],
    ], figsize=(20, 26), facecolor='w', tight_layout=True,
    gridspec_kw=dict(wspace=0.9, hspace=0.8, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    for box_origin in ['density_a', 'density_b']:
        box_position = axes[box_origin].get_position().bounds[:2] + np.array([-0.06, -0.02])
        fig.add_artist(patches.FancyBboxPatch(box_position, 0.53, 0.41,
                                            facecolor="w", lw=3, ec=(0.6, ) * 3, clip_on=False,
                                            zorder=-5,
                                            boxstyle='round,pad=0,rounding_size=0.02'))
    label_axes(fig,
               [v for k, v in axes.items() if k.endswith('_a')] +
               [v for k, v in axes.items() if k.endswith('_b')] +
               [v for k, v in axes.items() if k.startswith('global')],
               d=0.01, remove_spines=True)

    for letter, brain_name in zip(('a', 'b'), ('BrownLemur3', 'Cynnopterus')):
        ax_image = axes[f'separation_{letter}'].inset_axes((0.8, 0.92, 0.25, 0.25))
        ax_image.imshow(load_image(brain_name))
        ax_image.axis('off')
        brain_separation_metrics([axes[f'illustration_{letter}'], axes[f'separation_{letter}']],
                                [axes[f'density_{letter}'], axes[f'wiring_{letter}'], axes[f'routing_{letter}']],
                                brain_name)

    brain_ensemble_metrics([axes['global_density'], axes['global_wiring'], axes['global_routing']])
    return fig

if __name__ == "__main__":
    fig = figure3()
    fig.savefig(output_path('Figures/figure3.pdf'), dpi=300, bbox_inches='tight')
