from publication_paths import data_path, output_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import functools
import io
from dataloaders.brain_net import BrainNet
import projections.brain_stack as projections
import re
from style import brains_to_silhouette
import PIL
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev


def label_axes(fig, axes, remove_spines=False, d=0.02, h=0.02, size=40):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for ax, title in zip(axes, alphabet):
        pos = ax.get_subplotspec()
        pos = pos.get_position(fig) if pos is not None else ax.get_position()

        x = pos.x0 - h
        y = pos.y1 + d
        fig.text(x, y, f"({title})",
            ha='right', va='bottom', fontsize=size, fontweight='bold')

        if remove_spines:
            ax.spines[['top', 'right']].set_visible(False)


def save_ax_nosave(ax, **kwargs):
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    buff = io.BytesIO()
    plt.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches=bbox)
    ax.axis("on")
    buff.seek(0)
    im = plt.imread(buff)
    return im


@functools.lru_cache()
def create_brain_background(brain_name, elev=90, azim=-90):
    brain = BrainNet(brain_name)
    mri = projections.load_mri(brain_name)
    aligned = projections.align_brain_hemispheres(brain.positions, mri)
    mesh = projections.create_mesh(brain, aligned, mri, brain_name, kind='mri')

    fig = plt.figure(figsize=(10, 10))
    ax3d = fig.add_subplot(projection='3d', proj_type='ortho')
    ax3d.plot_trisurf(mesh['x'], mesh['y'], mesh['z'],
                      triangles=np.column_stack([mesh['i'], mesh['j'], mesh['k']]),
                      color=(0.9, 0.9, 0.9), linewidth=0)
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.dist = 7
    ax3d.set_box_aspect((np.ptp(mesh['x']), np.ptp(mesh['y']), np.ptp(mesh['z'])))
    ax3d.axis('off')

    im = save_ax_nosave(ax3d)
    fig.clf()
    im, aligned = _fix_brain_background_coordinates(im, aligned)
    return im, brain, aligned

def _fix_brain_background_coordinates(im, aligned):
    content_location = np.argwhere(im[:, :, 0] != 1)
    content_min, content_max = content_location.min(0), content_location.max(0)
    aligned = aligned * 0.9

    im = im[content_min[0]:content_max[0], content_min[1]:content_max[1]][::-1]
    return im, aligned

def logit_kde(data):
    eps = 1e-6
    data = data.clip(eps, 1 - eps)
    kde = scipy.stats.gaussian_kde(np.log(data) - np.log1p(-data))

    def density(x):
        x = np.clip(x, eps, 1 - eps)
        return kde(np.log(x) - np.log1p(-x)) / (x * (1 - x))

    return density


def trim_q_distances(distances, fibers, trim_q=0.01):
    if trim_q > 0:
        q = np.quantile(distances, [trim_q, 1 - trim_q])
        fibers = fibers[(fibers > q[0]) & (fibers < q[1])]
        distances = distances[(distances > q[0]) & (distances < q[1])]
    return distances, fibers


def plot_fbln_dist(distances, fibers, n_bins=8, bin_type='qcut', quantiles=False, norm=False, return_xy=False,
                   plot=True, style='-o', lw=1, trim_q=0.01, x_range=('auto', 'auto'), color=None, alpha=1, ax=None):
    distances = distances[distances > 0]
    fibers = fibers[fibers > 0]

    distances, fibers = trim_q_distances(distances, fibers, trim_q)

    if bin_type == 'kde':
        eps = 1e-6
        d = distances.clip(eps, 1 - eps)
        w = fibers.clip(eps, 1 - eps)
        d_kde = logit_kde(d)
        w_kde = logit_kde(w)
        def density(x): return w_kde(x) / d_kde(x)

        x_range_auto = [np.nanmean(d[d <= np.nanquantile(d, 0.1)]), np.nanmean(d[d >= np.nanquantile(d, 0.9)])]
        x_range = [x_range_auto[i] if x_range[i] == 'auto' else x_range[i] for i in range(2)]
        x = np.linspace(*x_range, n_bins)
        y = density(x)
        y = y / (np.diff(x).mean() * y.sum()) if norm else y
    else:
        _, bins = getattr(pd, bin_type)(distances, n_bins, retbins=True)
        freq, _ = np.histogram(distances, bins)
        freq_e, _ = np.histogram(fibers, bins)

        pretty_bins = 0.5 * (bins[1:] + bins[:-1])
        x = np.linspace(0, 1, len(pretty_bins)) if quantiles else pretty_bins

        y = freq_e / freq
        y = (y / (np.diff(bins) * y.sum())) if norm else y

    if plot:
        ax = ax if ax is not None else plt.gca()
        ax.plot(x, y, style, alpha=alpha, color=color, lw=lw)
    if return_xy:
        return x, y

def load_image(brain_name, animals_folder=data_path('images/animals')):
    animal_name = re.sub('[0-9]', '', brain_name)
    image_uri = brains_to_silhouette[animal_name]
    image_uri = f'{animals_folder}/{animal_name}.png'
    image = PIL.Image.open(image_uri).convert('RGBA')
    image = np.array(image)
    return image

def spline_shape_of_points(points, force_sign=False):
    convex_hull = ConvexHull(points)
    convex_hull_points = convex_hull.points[convex_hull.vertices]
    convex_hull_points = np.concatenate([convex_hull_points, convex_hull_points[:1]])
    tck, u = splprep(convex_hull_points.T, s=2, per=True)
    u_new = np.linspace(0, 1, 400)
    points_to_plot = splev(u_new, tck)

    if force_sign:
        main_sign = np.sign(np.sign(points_to_plot[0]).mean())
        points_to_plot[0] = points_to_plot[0].clip(None, -1) if main_sign < 0 else points_to_plot[0].clip(1, None)
    return np.stack(points_to_plot)
