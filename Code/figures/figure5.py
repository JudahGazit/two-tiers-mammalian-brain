from publication_paths import data_path, output_path
from dataloaders.brain_net import BrainNet
import matplotlib.pyplot as plt
import numpy as np
import PIL
from urllib.request import urlopen
from figures.utils import *
from style import *
from core_periphery.core_periphery import core_periphery_average_weight
from projections.brain_alignment import align_brain_hemispheres
from projections.brain_stack import load_mri
import networkx as nx
import sklearn.cluster
from community import best_partition
import warnings
import seaborn as sns
import os
import matplotlib.patches as patches


CONTROL_NETS = data_path('brain_net_control/')

MODULES_TYPES = [
    dict(t=None, title='Global', k=None, cmap=plt.cm.Oranges),
    dict(t='hemi', title='Hemi', k=2, cmap=plt.cm.Reds),
    dict(t='hemi', title='Quad', k=4, cmap=plt.cm.Purples),
    dict(t='hemi', title='Octo', k=8, cmap=plt.cm.autumn),
]


def random_variant_of_color(color, size=1, std=50):
    if isinstance(color, str):
        if color.startswith('C'):
            color_index = int(color[1:])
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][color_index]
        color = color.lstrip('#')
        color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    variants = np.linspace(-std, std, size).reshape(-1, 1)
    return [tuple(var_color) for var_color in np.clip(variants + color, 0, 255) / 255]


def hemispheres_clusters(dataset, mri, k):
    aligned = align_brain_hemispheres(dataset.positions, mri)
    aligned = (aligned - aligned.min(0)) / (aligned.max(0) - aligned.min(0))
    aligned = aligned[:, [0, 2, 1]]
    parts = int(np.log2(k))
    clustered = np.array([int(''.join(map(str, x)), base=2)
                         for x in (aligned > 0.5).astype(int)[:, :parts]])
    return clustered


def fiber_length_distribution_in_modules(dataset, modules_type, ax=None, k=8, normalize=True,
                                         alpha=1, subset=None, n_bins=20, mri=None, trim_q=0.01, **kwargs):
    """modules_type: ['kmeans', 'hemi', 'louvain', None]"""
    ax = ax or plt.gca()
    x, y = None, None
    fbln_config = dict(n_bins=n_bins, bin_type='kde', norm=normalize, trim_q=0.01,
                       return_xy=True, style='-', color='k', lw=1, alpha=alpha)
    fbln_config.update(kwargs)

    clustered = np.ones(dataset.number_of_nodes)
    if modules_type and k > 1:
        if modules_type == 'kmeans':
            clustered = sklearn.cluster.KMeans(k).fit_predict(dataset.positions)
        if modules_type == 'hemi':
            clustered = hemispheres_clusters(dataset, mri, k)
        if modules_type == 'louvain':
            adj = np.reciprocal(dataset.adjacency,
                                where=dataset.adjacency != 0, out=np.zeros_like(dataset.adjacency))
            clustered = np.array([v for k, v in sorted(best_partition(nx.from_numpy_matrix(adj)).items())])
            k = clustered.max() + 1

    inner_cluster_links = clustered == clustered.reshape(-1, 1)
    inner_subset_links = np.zeros(dataset.number_of_nodes)
    inner_subset_links[subset if subset is not None else range(dataset.number_of_nodes)] = 1
    inner_subset_links = inner_subset_links * inner_subset_links.reshape(-1, 1)
    inner_subset_and_clusters = inner_cluster_links * inner_subset_links

    distances = dataset.distances * inner_subset_and_clusters
    fibers = dataset.fibers * inner_subset_and_clusters
    q = np.quantile(distances[distances > 0], [trim_q, 1 - trim_q])
    distances = np.where((distances > q[0]) & (dataset.distances < q[1]), dataset.distances, 0)
    fibers = np.where((fibers > q[0]) & (dataset.fibers < q[1]), dataset.fibers, 0)

    x, y = plot_fbln_dist(distances / distances.max(), fibers / distances.max(), ax=ax, **fbln_config)

    return x, y


def plot_modules_fiber_lengths_for_dataset(dataset, axes, mri=None, types=MODULES_TYPES, normalize=True,
                                           subset=None, titles=False,
                                           n_bins=20, alpha=1):
    if isinstance(dataset, str):
        dataset = BrainNet(dataset)
        mri = load_mri(dataset)
    xs, ys = [], []
    for t, ax in zip(types, axes):
        y = None
        try:
            x, y = fiber_length_distribution_in_modules(dataset, t['t'], ax, t['k'],
                                                        normalize=normalize, alpha=alpha,
                                                        subset=subset, n_bins=n_bins, mri=mri)
            if titles:
                ax.set_title(t['title'], fontsize=16)
        except:
            raise
        xs.append(x)
        ys.append(y)
    return xs, ys


def _plot_split_grid_individial(n_brains, n_bins, titles, axes):
    res_all, res_core, res_periphery = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        random_brains = BrainNet.options()[:n_brains] if n_brains != 'all' else BrainNet.options()
        for brain_name in random_brains:
            try:
                dataset = BrainNet(brain_name)
                mri = load_mri(brain_name)
                params = dict(titles=titles, alpha=0.05, n_bins=n_bins, mri=mri)
                coreness = core_periphery_average_weight(dataset)
                core, periphery = np.where(coreness)[0], np.where(~coreness)[0]
                r_all = plot_modules_fiber_lengths_for_dataset(dataset, axes[2], **params)
                r_core = plot_modules_fiber_lengths_for_dataset(dataset, axes[1], subset=core, **params)
                r_periphery = plot_modules_fiber_lengths_for_dataset(dataset, axes[0], subset=periphery, **params)

                for r, rs in zip((r_all, r_core, r_periphery), (res_all, res_core, res_periphery)):
                    rs.append(r)
            except KeyboardInterrupt:
                raise
            except:
                pass
    return res_all, res_core, res_periphery

def plot_split_grid(n_brains, n_bins=20, titles=False, modules_type=MODULES_TYPES,
                    v_axes=None, e_axes=None, all_axes=None):
    if v_axes is None:
        v_axes, e_axes, all_axes = v_axes or plt.subplots(3, 4, figsize=(15, 9))[1]
    axes = [v_axes, e_axes, all_axes]
    res_all, res_core, res_periphery = _plot_split_grid_individial(n_brains, n_bins, titles, axes)
    for ax_row, rs, color in zip(axes, (res_periphery, res_core, res_all), ['C1', 'C0', 'C2']):
        xs, ys = zip(*rs)
        xs, ys = list(zip(*xs)), list(zip(*ys))
        for ax, x, y in zip(ax_row, xs, ys):
            x = np.concatenate([v if len(v.shape) > 1 else np.expand_dims(v, 0) for v in x if v is not None])
            y = np.concatenate([v if len(v.shape) > 1 else np.expand_dims(v, 0) for v in y if v is not None])
            x = x if len(x.shape) < 2 else np.nanmean(x, 0)
            y = y if len(y.shape) < 2 else np.nanmean(y, 0)
            ax.plot(x, y, c=color, lw=3)

    v_axes[0].text(0, 1.1, '$\\mathcal{V}$', fontsize=35, transform=v_axes[0].transAxes, color='C1')
    e_axes[0].text(0, 1.1, '$\\mathcal{E}$', fontsize=35, transform=e_axes[0].transAxes, color='C0')

    plt.setp(axes, xlim=(0.2, 1), ylim=(0, 6),
             xticks=np.linspace(0.2, 1, 3), xlabel='$x$', ylabel='$\\rho(x)$')


def display_illustration(uri, ax):
    ax.imshow(np.array(PIL.Image.open(uri).convert('RGBA')))
    ax.axis('off')

def rel_fbdist(distances, fibers, eps=1e-6):
    d = distances[distances > 0].clip(eps, 1 - eps)
    w = fibers[fibers > 0].clip(eps, 1 - eps)
    d_kde = logit_kde(d)
    w_kde = logit_kde(w)
    def density(x): return (w_kde(x.flatten()) / d_kde(x.flatten())).reshape(*x.shape)
    return density

ref_brains_p_cache = {}
def generate_control_brain(ref_brain_name):
    ref_brain = BrainNet(ref_brain_name)
    degs = np.nansum(ref_brain.adjacency, 0).astype(int)

    if ref_brain_name not in ref_brains_p_cache:
        w = rel_fbdist(ref_brain.distances, ref_brain.fibers)
        p = w(ref_brain.distances)
        p = p / p.sum(1, keepdims=True)
        ref_brains_p_cache[ref_brain_name] = p
    p = ref_brains_p_cache[ref_brain_name]

    control = [np.random.choice(len(degs), size=degs[i] // 2, p=p[i], replace=False) for i in range(len(degs))]
    control_A = np.zeros_like(ref_brain.adjacency)
    for i in range(len(degs)):
        control_A[i, control[i]] = 1
    control_A = np.maximum(control_A, control_A.T)
    np.fill_diagonal(control_A, 0)
    ref_brain.fibers = (control_A * ref_brain.distances)
    ref_brain.adjacency = control_A
    return ref_brain

def generate_control_brains(ref_brain_name, n=10):
    res = [generate_control_brain(ref_brain_name) for i in range(n)]
    adjacencies = np.stack([r.adjacency for r in res])
    np.save(os.path.join(CONTROL_NETS, f'{ref_brain_name}'), adjacencies)


def _process_controls_and_actuals():
    def extract_xy_per_k(dataset):
        for k in range(4):
            x, y = fiber_length_distribution_in_modules(dataset, 'hemi', plt.gca(), 2 ** k, plot=False, n_bins=8)
            yield [x, y]

    control_per_k, actual_per_k = [], []
    for brain_name in BrainNet.options():
        actual_brain = BrainNet(brain_name)
        controls = np.load(os.path.join(CONTROL_NETS, f'{brain_name}.npy'))
        actual_per_k.append(list(extract_xy_per_k(actual_brain)))
        for control in controls:
            actual_brain.adjacency = control
            actual_brain.fibers = control * actual_brain.distances
            control_per_k.append(list(extract_xy_per_k(actual_brain)))

    return np.stack(control_per_k), np.stack(actual_per_k)

def compare_to_control_brain(axes=None):
    ax_control, ax_actual = axes if axes is not None else plt.subplots(1, 2, figsize=(10, 3))[1]
    cache_file = os.path.join(CONTROL_NETS, 'results.mat')
    if not os.path.isfile(cache_file):
        control_per_k, actual_per_k = _process_controls_and_actuals()
        scipy.io.savemat(cache_file, dict(control_per_k=control_per_k, actual_per_k=actual_per_k))
    cache_data = scipy.io.loadmat(cache_file)
    control_per_k, actual_per_k = cache_data['control_per_k'], cache_data['actual_per_k']

    control_mean, actual_mean = np.stack(control_per_k).mean(0), np.stack(actual_per_k).mean(0)
    styles = ['-', '--', ':', '-.']
    names = ['Full', 'Hemi', 'Quad', 'Octo']
    for ax, arr in zip([ax_control, ax_actual], [control_mean, actual_mean]):
        for k in range(4):
            ax.plot(*arr[k], c='C2', linestyle=styles[k], label=names[k])
        ax.legend(loc='upper right', frameon=False, fontsize=16)
        ax.spines[['top', 'right']].set_visible(False)
        plt.setp(ax, ylim=(0, 6), xlim=(0.2, 1), xlabel='$x$', ylabel='Density')

def plot_split_grid_control(axes=None):
    axes = axes or plt.subplots(1, 4, figsize=(20, 3))[1]
    cache_file = os.path.join(CONTROL_NETS, 'results.mat')
    if not os.path.isfile(cache_file):
        control_per_k, actual_per_k = _process_controls_and_actuals()
        scipy.io.savemat(cache_file, dict(control_per_k=control_per_k, actual_per_k=actual_per_k))
    cache_data = scipy.io.loadmat(cache_file)
    control_per_k = cache_data['control_per_k'].transpose(1, 2, 3, 0)
    actual_watermark_k = np.stack(cache_data['actual_per_k']).mean(0)
    for ax, control, actual_watermark in zip(axes, control_per_k, actual_watermark_k):
        ax.plot(*control, c=(0.3, ) * 3, alpha=0.05, lw=0.1)
        ax.plot(*control.mean(2), c='k', lw=3, label='Control')
        ax.plot(*actual_watermark, c='C2', lw=3, alpha=0.5, label='Empirical')
    plt.setp(axes, xlim=(0.2, 1), ylim=(0, 4), xlabel='$x$', ylabel='$\\rho(x)$')
    axes[-1].legend(loc='upper right', frameon=False)

def figure5():
    CUT_ILLUSTRATIONS = [data_path("images/full-brain-illustration.png"),
                         data_path("images/hemi-brain-illustration.png"),
                         data_path("images/quad-brain-illustration.png"),
                         data_path("images/octo-brain-illustration.png"),]
    CONTROL_ILLUSTRATIONS = [
        data_path(r'images/figure5_illustration_cuts_control.png'),
        data_path(r'images/figure5_illustration_cuts_actual.png')
    ]

    fig, axes = plt.subplot_mosaic([
        ['control_rho'] * 2 + ['actual_rho'] * 2,
        ['control_rho'] * 2 + ['actual_rho'] * 2,
        [f'illustration_{i}' for i in range(4)],
        [f'v_sub_{i}' for i in range(4)],
        [f'e_sub_{i}' for i in range(4)],
        [f'all_sub_{i}' for i in range(4)],
        ['.'] * 4,
        [f'control_sub_{i}' for i in range(4)],
    ], figsize=(20, 30), facecolor='w',
    gridspec_kw=dict(wspace=0.8, hspace=0.5, height_ratios=[1, 1, 1, 1, 1, 1, 0, 1]))

    box_position = axes['control_rho'].get_position().corners()[0] + np.array([-0.06, 0.00])
    w, h = axes['actual_rho'].get_position().corners()[-1] - box_position - np.array([-0.03, -0.04])
    fig.add_artist(patches.FancyBboxPatch(box_position, w, h,
                                          facecolor="w", lw=3, ec=(0.6, ) * 3, clip_on=False,
                                          zorder=-5,
                                          boxstyle='round,pad=0,rounding_size=0.02'))
    v_axes = [ax for k, ax in axes.items() if 'v_sub' in k]
    e_axes = [ax for k, ax in axes.items() if 'e_sub' in k]
    all_axes = [ax for k, ax in axes.items() if 'all_sub' in k]
    control_axes = [ax for k, ax in axes.items() if 'control_sub' in k]

    for i in range(len(CUT_ILLUSTRATIONS)):
        display_illustration(CUT_ILLUSTRATIONS[i], axes[f'illustration_{i}'])

    for i, ax in enumerate([axes['control_rho'], axes['actual_rho']]):
        display_illustration(CONTROL_ILLUSTRATIONS[i], ax)

    plot_split_grid('all', v_axes=v_axes, e_axes=e_axes, all_axes=all_axes)
    plt.setp(all_axes, ylim=(0, 4))
    plot_split_grid_control(control_axes)

    label_axes(fig, (axes['control_rho'], axes['actual_rho'],
                     axes['illustration_0'],
                     *v_axes, *e_axes, *all_axes, *control_axes), remove_spines=True, d=0.01)
    return fig


if __name__ == "__main__":
    fig = figure5()
    fig.savefig(output_path('Figures/figure5.pdf'), dpi=300, bbox_inches='tight')
