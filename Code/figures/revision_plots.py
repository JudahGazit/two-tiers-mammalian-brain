"""Plotting helpers shared by the supplementary figures of the revision."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from analysis.density import CENTERS, ensemble_mean
from style import *  # noqa: F401,F403  (project-wide matplotlib style)

COLOR_ALL, COLOR_E, COLOR_V = 'C2', 'C0', 'C1'
PART_STYLE = {'all': (COLOR_ALL, r'$\rho$'), 'E': (COLOR_E, r'$\rho_{\mathcal{E}}$'), 'V': (COLOR_V, r'$\rho_{\mathcal{V}}$')}


def density_axes(ax, ylabel=r'$\rho(x)$', ylim=(0, 3)):
    plt.setp(ax, xlabel='$x$', ylabel=ylabel, xlim=(0, 1), ylim=ylim)
    ax.axhline(1, color='0.7', ls=':', lw=1)
    ax.spines[['top', 'right']].set_visible(False)


def ensemble_curve(ax, curves, color, label=None, individual=True, lw=3):
    """Thin per-brain curves (optional) and their ensemble mean."""
    if individual:
        for y in curves:
            ax.plot(CENTERS, y, color=color, alpha=0.06, lw=0.8)
    mean = ensemble_mean(curves)
    keep = np.isfinite(mean)
    ax.plot(CENTERS[keep], mean[keep], '-o', ms=4, color=color, lw=lw, label=label)


def class_densities(ax, curves_by_part, parts=('all', 'E', 'V'), superscript=''):
    for part in parts:
        color, label = PART_STYLE[part]
        ensemble_curve(ax, curves_by_part[part], color, label.replace(r'\rho', r'\rho' + superscript))


def shaded_density(ax, points, color, lims, bins=70, smooth=1.5):
    """Filled contours enclosing the densest 50, 25, 10 and 3% of a 2D point cloud."""
    H, xe, ye = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=lims, density=True)
    H = gaussian_filter(H, smooth)
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    levels = np.quantile(H[H > H.max() * 1e-3], [0.5, 0.75, 0.9, 0.97])
    for level in levels:
        ax.contourf(xc, yc, H.T, levels=[level, H.max() * 1.01], colors=[color], alpha=0.14)
    ax.contour(xc, yc, H.T, levels=levels, colors=color, linewidths=1.4)


def class_legend(ax, outside=False):
    handles = [Line2D([], [], color=COLOR_V, lw=2), Line2D([], [], color=COLOR_E, lw=2)]
    where = dict(loc='center left', bbox_to_anchor=(1, 0.5)) if outside else {}
    ax.legend(handles, ['villagers', 'explorers'], frameon=False, fontsize=13, **where)


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
