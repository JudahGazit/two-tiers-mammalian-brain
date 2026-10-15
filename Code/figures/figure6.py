from publication_paths import data_path, output_path
from dataloaders.brain_net import BrainNet
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL
from urllib.request import urlopen
from figures.utils import *
from style import *
from projections.brain_alignment import align_brain_hemispheres
import projections.brain_tracts
from projections.brain_tracts import TRACTS_DIR, HUMAN_TRACTS_DIR
from projections.brain_stack import (
    load_mri, cluster_null, HUMAN_BRAINS_DIR, BRAINS_LOCATIONS, PROJ_BRAIN_NAMES,
    rotate_mri_by_aligned, load_orders, generate_meshgrid, create_mesh_convex_hull
)
from sklearn.decomposition import PCA
from projections.brain_projection import BrainProjection
import os
from matplotlib.legend_handler import HandlerBase
from matplotlib.offsetbox import OffsetImage
from projections.brain_alignment import BrainAlignment
from PIL import ImageColor
from core_periphery.core_periphery import core_periphery_average_weight
import pickle

CACHE_DIR = data_path(r'images/figure6_pickle/')

class HandlerImage(HandlerBase):
    def __init__(self, img, space=10):
        super().__init__()
        self.img = img
        self.space = space

    def create_artists(self, legend, artist, x0, y0, width, height, fontsize, trans):
        oi = OffsetImage(self.img, zoom=1)
        oi.set_transform(trans)
        # center the image in the legend box
        oi.set_offset((x0 + width/2, y0 + height/2))
        return [oi]


def radial_lines(s1, s2, M=100, ax=None):
    ax = ax if ax is not None else plt.gca()
    x1, y1 = s1.T
    x2, y2 = s2.T
    z1 = x1 + 1j*y1
    z2 = x2 + 1j*y2
    thetas = np.linspace(0, 2*np.pi, M, endpoint=False)

    for th in thetas:
        d = np.exp(1j*th)

        # projection coefficient in direction d
        t1 = np.real(z1 / d)
        t2 = np.real(z2 / d)

        # farthest boundary point along this direction
        p1 = d * t1.max()
        p2 = d * t2.max()

        ax.annotate(
            "",
            xy=(p2.real, p2.imag),    # arrow head
            xytext=(p1.real, p1.imag),# tail
            arrowprops=dict(
                arrowstyle="-|>, head_width=0.05, head_length=0.05",
                lw=1,
                color="k",
                shrinkA=0, shrinkB=0,
            )
        )

def illustrate_shape_from_points(points, color, ax, alpha=0.5, border=False, force_hemi_gap=False):
    points_to_plot = spline_shape_of_points(points, force_hemi_gap)
    ax.fill(*points_to_plot, c=color, alpha=alpha)
    if border:
        ax.plot(*points_to_plot, c=color, lw=2)
    return points_to_plot

def illustrate_brain_as_two_hemispheres(points, color, ax, dim=[0, 1], alpha=0.5, border=False):
    N = points.shape[0]
    hemi_size = N // 2
    hemis = [points[hemi_size * hemi:hemi_size * (hemi + 1)] for hemi in range(2)]
    hemi_points = []
    for i in range(2):
        points_of_hemi = np.concatenate([hemis[i], hemis[1 - i] * np.array([-1, 1, 1])])[:, dim]
        points = illustrate_shape_from_points(points_of_hemi, color, ax, alpha, border, force_hemi_gap=0 in dim)
        hemi_points.append(points)
    return hemi_points

def illustrate_projection_rotation(ax, brain_name, color):
    brain =  BrainNet(brain_name)
    positions_centered = brain.positions - brain.positions.mean(0)
    positions_centered = positions_centered[:, [1, 0, 2]]
    illustrate_brain_as_two_hemispheres(positions_centered, color, ax, border=True)

    brain_alignment = BrainAlignment(positions_centered)
    rotated, hemi_cut = brain_alignment._rotate_by_hemispheres_cut(positions_centered, return_hemispheres_cut=True)
    pca = PCA(2).fit(rotated[:, 1:]).components_
    pca = np.column_stack([np.zeros((2, 1)), pca])
    rot = scipy.spatial.transform.Rotation.align_vectors(np.eye(3)[:1], hemi_cut)[0]
    pca = rot.inv().apply(pca)
    ax.axline(np.zeros(2), hemi_cut[0, :2], c=color, linestyle='--', lw=3)
    ax.axline(np.zeros(2), pca[0, :2], c=color, linestyle='--', lw=3)
    rad = int(-1 * np.sign(hemi_cut[0, 1]))
    max_pos = positions_centered.max(0) * 1
    ax.annotate("", xy=max_pos[0] * (hemi_cut[0, :2] / hemi_cut[0, 0]),
                xytext=max_pos[1] * (pca[0, :2] / pca[0, 1]), xycoords="data",
                    arrowprops=dict(arrowstyle="-|>", connectionstyle=f"arc3,rad={rad}", lw=3, color=color,
                                     shrinkA=20, shrinkB=20))

    ax.axline([0, 0], [0, 1], c='k')
    ax.axline([0, 0], [1, 0], c='k')

    ax.set_aspect('equal')
    ax.axis('off')

def illustrate_projection_stacking(ax, order, color):
    order_brains = load_orders()[order]
    for brain_name in order_brains:
        brain =  BrainNet(brain_name)
        mri = load_mri(brain_name)
        aligned_brain = align_brain_hemispheres(brain.positions, mri)
        illustrate_brain_as_two_hemispheres(aligned_brain, color, ax)

    ax.set_aspect('equal')
    ax.axis('off')

def illustrate_projection_image_legend(ax, brain_names):
    legend_patches = [ax.patches[0], ax.patches[2]]
    for i, (patch, brain_name) in enumerate(zip(legend_patches, brain_names)):
        alpha = load_image(brain_name)[:, :, -1] == 255
        img = alpha[:, :, None] * patch.get_facecolor()[:3]
        img = np.concatenate([img, alpha[:, :, None]], axis=2)
        mini_ax = ax.inset_axes([1.35, 0.5 + 0.3 * i, 0.3, 0.3])
        mini_ax.imshow(img)
        mini_ax.axis('off')

def illustrate_projection_transformation(ax, brain_names, colors):
    aligned_brains = []
    sample_tracts = []
    convex_hulls = []
    for brain_name, color in zip(brain_names, colors):
        brain = BrainNet(brain_name)
        mri = load_mri(brain_name)
        aligned_brain = align_brain_hemispheres(brain.positions, mri)
        illustrate_brain_as_two_hemispheres(aligned_brain, color, ax, alpha=0.5, border=True)
        aligned_brains.append(aligned_brain)

        tracts = scipy.io.loadmat(os.path.join(TRACTS_DIR, f'{brain_name}_core.mat'))['Tracts'][0]
        tracts = tracts[:500][-1:]
        tracts = [rotate_mri_by_aligned(tract, brain, aligned_brain) for tract in tracts]
        sample_tracts.append(tracts)

        aligned_brain_mirrored = np.concatenate([aligned_brain[:, :2], aligned_brain[:, :2] * [-1, 1]])
        convex_hull = ConvexHull(aligned_brain_mirrored)
        convex_hull_points = convex_hull.points[convex_hull.vertices]
        convex_hulls.append(convex_hull_points)

    projector = BrainProjection(*aligned_brains)
    projection = projector.project(aligned_brains[0])
    projection = np.stack([aligned_brains[0], projection])[:, :, :2]
    projection_tracts = [projector.project(tract)[:, :2] for tract in sample_tracts[0]]
    radial_lines(convex_hulls[0], convex_hulls[1], ax=ax)
    demo_node = 71
    ax.scatter(*projection[0, demo_node], c='C0', lw=0.5, ec='w')
    ax.scatter(*projection[1, demo_node], c='C2', lw=0.5, ec='w')
    ax.annotate('', *projection[::-1, demo_node],
                    arrowprops=dict(arrowstyle="-|>, head_width=0.1, head_length=0.2", lw=1, color='k', shrinkA=0, shrinkB=0))
    ax.plot(*sample_tracts[0][0][:, :2].T, c='C0', lw=2)
    ax.plot(*projection_tracts[0][:, :2].T, c='C2', lw=2)
    for source, target in list(zip(sample_tracts[0][0][:, :2], projection_tracts[0][:, :2]))[::20]:
        ax.annotate('', target, source,
                        arrowprops=dict(arrowstyle="-|>, head_width=0.05, head_length=0.05", lw=1, color='k', shrinkA=0, shrinkB=0, alpha=0.5))

    illustrate_projection_image_legend(ax, brain_names)

    ax.set_aspect('equal')
    ax.axis('off')


def illustrate_projection_old(axes=None, brain_names=['Macaque1', 'Dog7'], order='Primates'):
    axes = plt.subplots(1, 3, figsize=(15, 3))[1] if axes is None else axes
    illustrate_projection_rotation(axes[0], brain_names[0], 'C0')
    illustrate_projection_stacking(axes[1], order, 'grey')
    illustrate_projection_transformation(axes[2], brain_names, ['C0', 'C2'])


def illustrate_projection(ax):
    images = [plt.imread(data_path(f'images/brain_projection_illustration_{letter}.png'))
              for letter in ['a', 'b', 'c']]
    ratios = [image.shape[1] / image.shape[0] for image in images]
    start_points = (np.array(ratios) / np.sum(ratios)).cumsum()
    start_points = np.concatenate([[0], start_points])
    titles = ['I. Rotate', 'II. Rescale', 'III. Morph']
    subaxes = []
    for i in range(len(images)):
        x, w = start_points[i], np.diff(start_points)[i] - 0.05
        axinset = ax.inset_axes((x, 0, w, 1))
        axinset.imshow(images[i])
        axinset.axis('off')
        axinset.set_aspect('equal', anchor='C')
        ax.text(x + w/2, 1.1, titles[i], weight='bold', size=30, ha='center')
        subaxes.append(axinset)
    ax.axis('off')
    return subaxes


def show_stack_3d(volume, B, ax, cmap="Reds", alpha=1.0, gap=1.0, pow=2.0):
    ax = ax.inset_axes((0, -0.5, 1, 2), projection='3d')
    volume = volume.copy()
    volume = volume / volume.sum()
    volume[volume == 0] = np.nan
    volume = np.log(volume)
    volume = (volume - np.nanmin(volume)) / (np.nanmax(volume) - np.nanmin(volume))
    Z, Y, X = volume.shape

    Xg, Yg = np.meshgrid(B[0][:-1], B[1][:-1])

    for i in range(Z):
        Z_plane = np.full_like(Xg, i * gap)
        volume_color = volume[i]
        volume_color = volume_color ** pow
        volume_color = plt.cm.get_cmap(cmap)(volume_color)
        volume_color[(volume_color == 0).all(2)] = np.array([1, 1, 1, 1])
        ax.plot_surface(Xg, Yg, Z_plane, rstride=1, cstride=1, facecolors=volume_color, shade=False, alpha=alpha)
        ax.plot_surface(Xg, Yg, Z_plane-1, rstride=Y-1, cstride=X-1, facecolors='k', shade=False, alpha=1)
        ax.plot_wireframe(Xg, Yg, Z_plane, rstride=Y-1, cstride=X-1, color='k', linewidth=0.8)

    ax.set_box_aspect((X, Y, Z * gap))
    ax.set_axis_off()
    ax.view_init(elev=30, azim=-60)
    ax.set_proj_type('persp', focal_length=0.5)
    ax.set_facecolor([0, 0, 0, 0])

def load_cache_from_disk(cache_dir=CACHE_DIR):
    file_names = ['nodes_cache', 'tracts_cache', 'data_cache']
    content = []
    for file_name in file_names:
        with open(os.path.join(cache_dir, f'{file_name}.pkl'), 'rb') as f:
            content.append(pickle.load(f))
    return content

def save_cache_to_disk(nodes_cache, tracts_cache, data_cache, cache_dir=CACHE_DIR):
    file_names = ['nodes_cache', 'tracts_cache', 'data_cache']
    for arr, fname in zip([nodes_cache, tracts_cache, data_cache], file_names):
        with open(os.path.join(cache_dir, f'{fname}.pkl'), 'wb') as f:
            f.write(pickle.dumps(arr))

nodes_cache, tracts_cache, data_cache = load_cache_from_disk()

def plot_mri_stack(title, brains, proj_brain_name, axes=None, n_samples=500, pow=2.0,
                   show_3d=True, num_axes=3, title_y=1.3):
    axes = axes if axes is not None else plt.subplots(1, 4, figsize=(15, 5))[1]
    axes_stack = axes[0] if show_3d else None
    axes_dims = axes[1:] if show_3d else axes
    proj_mri = load_mri(proj_brain_name)
    proj_brain_dataset = BrainNet(proj_brain_name, location=HUMAN_BRAINS_DIR if 'Human' in proj_brain_name else BRAINS_LOCATIONS)
    proj_brain_aligned = align_brain_hemispheres(proj_brain_dataset.positions, proj_mri)

    cache_key = (proj_brain_name, tuple(sorted(brains)))
    if cache_key not in nodes_cache:
        projection_nodes = projections.brain_stack.project_multiple_brains(brains, proj_brain_aligned, cluster_method=cluster_null)
        projection_tracts = projections.brain_tracts.project_all_brains(proj_brain_name, brains, n_samples)[1]
        nodes_cache[cache_key], tracts_cache[cache_key] = projection_nodes, projection_tracts
    projection_nodes, projection_tracts = nodes_cache[cache_key], tracts_cache[cache_key]

    if cache_key not in data_cache:
        nodes_kde = scipy.stats.gaussian_kde(np.concatenate(projection_nodes).T)
        kde_pos = generate_meshgrid(proj_brain_aligned, 1)
        node_kde_values = nodes_kde(kde_pos.reshape(3, -1))
        H, B = np.histogramdd(projection_tracts.values[:, :3].astype(float), bins=100)
        B = np.stack(B)
        top_node_kde = node_kde_values > np.quantile(node_kde_values, 0.9)
        top_tract_kde = H > np.quantile(H, 0.98)
        Hn = top_node_kde.reshape(kde_pos.shape[1:])
        data_cache[cache_key] = (kde_pos, H, B, top_tract_kde, Hn)
    kde_pos, H, B, top_tract_kde, Hn = data_cache[cache_key]

    dims = ([0, 1], [1, 2], [0, 2])[:num_axes]
    for ax, dim in zip(axes_dims, dims):
        tracts_points = np.stack(np.meshgrid(*B[dim]))[:, :-1, :-1].transpose([0, 2, 1])
        top_tract_kde_color = top_tract_kde.mean(3 - np.sum(dim))
        top_tract_kde_color = np.log(top_tract_kde_color, where= top_tract_kde_color > 0, out=np.zeros_like(top_tract_kde_color))
        top_tract_kde_color[top_tract_kde_color == 0] = np.nan
        im_region = [tracts_points[0].min(), tracts_points[0].max(), tracts_points[1].min(), tracts_points[1].max()]
        ax.imshow(top_tract_kde_color.T, cmap='Reds', origin='lower', extent=im_region)
        nodes_heatmap = Hn.mean(3 - np.sum(dim)).T
        nodes_heatmap = np.where(nodes_heatmap > np.nanquantile(nodes_heatmap, 0.8), nodes_heatmap, np.nan)
        node_heatmap_bounds = kde_pos.reshape(3, -1).min(1)[dim], kde_pos.reshape(3, -1).max(1)[dim]
        node_heatmap_bounds = (node_heatmap_bounds[0][0], node_heatmap_bounds[1][0], node_heatmap_bounds[0][1], node_heatmap_bounds[1][1])
        ax.imshow(np.log(nodes_heatmap), cmap='Blues', origin='lower', extent=node_heatmap_bounds, alpha=0.5)

        illustrate_brain_as_two_hemispheres(proj_brain_aligned, 'k', dim=dim, ax=ax, alpha=0, border=True)
        ax.axis('off')
        ax.set_aspect('equal', anchor='W')

    if show_3d:
        show_stack_3d(H[:, :, 10::20].transpose([2, 1, 0]), B, alpha=1, gap=45, ax=axes_stack, pow=pow)
        axes_stack.axis('off')
    title_ax = axes[0] if len(axes) == 1 else axes[1]
    title_ax.set_title(title, weight='bold', size=30, y=title_y)

def display_all_core_tracts(brain_name, ax=None):
    brain = BrainNet(brain_name, location=HUMAN_BRAINS_DIR if 'Human' in brain_name else BRAINS_LOCATIONS)
    coreness = core_periphery_average_weight(brain)
    mesh = create_mesh_convex_hull(brain.positions, 0.01)
    tracts_dir = HUMAN_TRACTS_DIR if 'Human' in brain_name else TRACTS_DIR
    tracts = scipy.io.loadmat(os.path.join(tracts_dir, f'{brain_name}_core.mat'))['Tracts'][0]
    tracts_all_points = np.concatenate(tracts)

    ax_global = plt.gca() if ax is None else ax
    ax = ax_global.inset_axes((0, 0, 1, 1), projection='3d', proj_type='persp', focal_length=1)

    ax.plot_trisurf(mesh['x'], mesh['y'], mesh['z'],
                    triangles=np.column_stack([mesh['i'], mesh['j'], mesh['k']]),
                    linewidth=0.05, ec=[0, 0, 0, 0.2], antialiased=True, shade=True, color=[0.8, 0.8, 0.8, 0.05])

    rgb_color = np.array(ImageColor.getrgb(list(plt.rcParams['axes.prop_cycle'])[0]['color'])) / 255

    for t in tracts:
        ax.plot(*t[::5].T, c=np.clip(rgb_color + 0.2 * (np.random.rand(3) - 0.5), 0, 1), alpha=0.1, lw=0.1)
    ax.scatter(*brain.positions[coreness].T, ec='k')

    ax.view_init(elev=45, azim=45 + 30)
    ax.set_box_aspect(np.ptp(tracts_all_points, axis=0))
    ticks = np.stack([tracts_all_points.min(0), tracts_all_points.max(0)])
    plt.setp(ax,
             xticks=ticks[:, 0], yticks=ticks[:, 1], zticks=ticks[:, 2],
             xticklabels=[], yticklabels=[], zticklabels=[],
             xlabel='$x$', ylabel='$y$', zlabel='$z$')

    im_axis = ax_global.inset_axes((0, 0.9, 0.2, 0.2))
    im_axis.imshow(load_image(brain_name))
    im_axis.axis('off')
    ax_global.axis('off')


def figure6():
    fig, axes = plt.subplot_mosaic([
            [f'tracts_{l}' for l in ('a', 'b', 'c') for _ in range(5)],
            [f'tracts_{l}' for l in ('a', 'b', 'c') for _ in range(5)],
            ['.'] * 15,
            ['illustration'] * 15,
            ['illustration'] * 15,
            ['.'] * 15,
            [f'all_a'] * 5 + ['all_b'] * 4 + ['all_c'] * 3 + ['all_d'] * 3,
            [f'all_a'] * 5 + ['all_b'] * 4 + ['all_c'] * 3 + ['all_d'] * 3,
            ['.'] * 15,
            [f'order_{l}' for l in range(5) for _ in range(3)],
            [f'order_{l}' for l in range(5) for _ in range(3)],
            ['.'] * 15,
            [f'humans_a'] * 5 + ['humans_b'] * 4 + ['humans_c'] * 3 + ['humans_d'] * 3,
            [f'humans_a'] * 5 + ['humans_b'] * 4 + ['humans_c'] * 3 + ['humans_d'] * 3,
        ],
        figsize=(22, 35), facecolor='w',
        gridspec_kw=dict(wspace=0.8, hspace=0.8, height_ratios=[1, 1, 0.5, 1.5, 1.5, 1, 1, 1, 0.5, 1, 1, 0.5, 1, 1])
    )

    for ax_left, ax_right in [[axes['illustration']] * 2, [axes['humans_a'], axes['humans_d']]]:
        pos_left, pos_right = ax_left.get_position(), ax_right.get_position()
        box_position = [pos_left.x0 - 0.06, pos_left.y0 - 0.03]
        w, h = pos_right.x1 - pos_left.x0 + 0.1, pos_right.y1 - pos_left.y0 + 0.07
        fig.add_artist(patches.FancyBboxPatch(box_position, w, h,
                                                    facecolor="w", lw=3, ec=(0.6, ) * 3, clip_on=False,
                                                    zorder=-5,
                                                    boxstyle='round,pad=0,rounding_size=0.02'))

    sub_illustrations = illustrate_projection(axes['illustration'])
    axes_3d_tracts = [v for k, v in axes.items() if 'tracts_' in k]
    for ax, brain_name in zip(axes_3d_tracts, ('FruitBat4', 'Dog7', 'Human1')):
        display_all_core_tracts(brain_name, ax)
    orders_to_plot = [
        dict(title='All Mammals', axes=[axes['all_a'], axes['all_b'], axes['all_c'], axes['all_d']],
             brains=BrainNet.options().tolist(), proj_brain_name='Hyrax1'),
        *[dict(title=order_name, axes=[axes[f'order_{i}']],
               brains=load_orders()[order_name], proj_brain_name=PROJ_BRAIN_NAMES[order_name],
               show_3d=False, num_axes=1, title_y=-0.3)
               for i, order_name in enumerate(['Primates', 'Chiroptera', 'Rodentia', 'Artiodactyla', 'Carnivora'])],
        dict(title='Humans', axes=[axes['humans_a'], axes['humans_b'], axes['humans_c'], axes['humans_d']],
             brains=load_orders()['Humans'], proj_brain_name='Human1', n_samples=1500, pow=0.5),
    ]
    for order_to_plot in orders_to_plot:
        plot_mri_stack(**order_to_plot)

    label_axes(fig, [axes['tracts_a'], axes['tracts_b'], axes['tracts_c'],
                     *sub_illustrations,
                     axes['all_a'], axes['all_b'], axes['all_c'], axes['all_d'],
                     axes['order_4'], axes['order_0'], axes['order_1'], axes['order_2'], axes['order_3'],
                     axes['humans_a'], axes['humans_b'], axes['humans_c'], axes['humans_d']],
                     remove_spines=False, d=0)
    return fig

if __name__ == '__main__':
    fig = figure6()
    fig.savefig(output_path('Figures/figure6.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(output_path('Figures/figure6.png'), dpi=300, bbox_inches='tight')
