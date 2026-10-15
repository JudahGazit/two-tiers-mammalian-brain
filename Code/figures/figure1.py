from publication_paths import data_path, output_path
from dataloaders.brain_net import BrainNet
import projections.brain_projection as brain_projection
import matplotlib.pyplot as plt
import numpy as np
import functools
from multiprocessing import Pool
from matplotlib.patches import Circle
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import PIL
from urllib.request import urlopen
from figures.utils import *
from style import *
import io

@functools.lru_cache
def physical_virtual_fibers_():
    all_distances, all_links = [], []
    for brain_name in BrainNet.options():
        brain = BrainNet(brain_name)
        try:
            all_distances.append(brain.distances[brain.distances > 0])
            all_links.append(brain.fibers[brain.fibers > 0])
        except:
            pass
    all_distances, all_links = np.concatenate(all_distances), np.concatenate(all_links)
    return all_distances, all_links


def physical_virtual_fibers(n_bins=15, ax=None):
    ax = ax if ax is not None else plt.gca()
    bins = np.linspace(0, 1, n_bins + 1)

    all_distances, all_links = physical_virtual_fibers_()

    freq_d, _, freq_d_bars = ax.hist(all_distances, bins,
                                     color=(208 / 255, 206 / 255, 206 / 255), edgecolor='k', label='$S_d(x)$')
    freq_l, _, freq_l_bars = ax.hist(all_links, bins,
                                     color=(197 / 255, 224 / 255, 180 / 255), edgecolor='k', label='$S_l(x)$')

    ax.legend(['$\\mathscr{d}$', '$\\mathscr{l}$'])
    ax.get_legend().get_frame().set_linewidth(0)
    plt.setp(ax, xlabel='$\\it{x}$', ylabel='Frequency')
    plt.setp([ax.spines['right'], ax.spines['top']], visible=False)
    ax.tick_params(axis='both', which='major')
    ax.ticklabel_format(axis='y', style='sci', useMathText=True)

    bold_bar = -5
    plt.setp(freq_d_bars[bold_bar], linewidth=3, zorder=10)
    plt.setp(freq_l_bars[bold_bar], linewidth=3, zorder=10)
    smallax = ax.inset_axes((0.7, 0.5, 0.3, 0.3))
    rho_explained = np.array(PIL.Image.open(data_path(r"images/figure1_rho_explained.png")).convert('RGBA'))
    smallax.imshow(rho_explained)
    annotation = plt.annotate('', (0.5 * (bins[bold_bar - 1] + bins[bold_bar]), freq_d[bold_bar]), xycoords=ax.transData, textcoords=ax.transAxes,
                              xytext=(0.7, 0.6),
                              size=20,
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color=(0.3, ) * 3, lw=5))
    smallax.axis('off')


def fiberlength_distributions(brains=('Deer1', 'Koala', 'Rat1', 'Gerbil'), n_bins=50, silhouettes=None, axes=None):
    silhouettes = brains_to_silhouette if silhouettes is None else silhouettes
    if axes is None:
        _, axes = plt.subplots(1, len(brains), figsize=(5 * 3, len(brains)))
    for b, ax in zip(brains, axes):
        brain = BrainNet(b)
        plot_fbln_dist(brain.distances, brain.fibers, n_bins=n_bins, ax=ax,
                       color=(0.3, ) * 3, bin_type='kde', norm=True, lw=3, style='-')

        plt.setp(ax, xlabel='$\\it{x}$', ylabel='$\\rho(x)$', ylim=(
            0, 3.5), xticks=np.linspace(0, 1, 5))

        aximage = ax.inset_axes((0.7, 0.8, 0.4, 0.4), )
        aximage.imshow(load_image(b))
        aximage.axis('off')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(np.linspace(0, 1, 3))


def fiberlength_distribution_ensemble(ax=None, n_bins=8, illustrations=('Deer1', 'Koala', 'Rat1', 'Gerbil')):
    ax = ax if ax is not None else plt.gca()

    data = []

    all_distances, all_fibers = [], []
    for brain_name in BrainNet.options():
        brain = BrainNet(brain_name)
        try:
            x, y = plot_fbln_dist(brain.distances, brain.fibers, n_bins=n_bins, x_range=('auto', 0.9),
                                  bin_type='kde', norm=True, return_xy=True, plot=False, style='-')
            data.append(dict(name=brain_name, x=x, y=y))
            distances, fibers = brain.distances[brain.distances > 0], brain.fibers[brain.adjacency > 0]
            all_distances.append(distances)
            all_fibers.append(fibers)
        except KeyboardInterrupt:
            raise
        except:
            pass

    data = pd.DataFrame(data)
    ax.plot(np.column_stack(data.x), np.column_stack(data.y), color=(0.5, 0.5, 0.5), alpha=0.1)
    black_line_result = ax.plot(np.column_stack(data.x).mean(1), np.column_stack(data.y).mean(1), c='C2', lw=5)

    data_with_images = data[data.name.isin(illustrations)].copy()
    data_with_images['image'] = data_with_images['name'].apply(load_image)
    data_with_images['end_x'] = np.stack(data_with_images.x)[:, -1]
    data_with_images['end_y'] = np.stack(data_with_images.y)[:, -1]
    data_with_images = data_with_images.sort_values('end_y')

    x_max = np.stack(data.x).max()
    y_max = np.stack(data.y).max()

    h = 0.2 * 4
    ximage = 1.1
    for index, y in zip(data_with_images.index, np.linspace(0, y_max, len(data_with_images))):
        row = data_with_images.loc[index]
        ax.annotate('', (x_max, row['end_y']), (ximage, y), xycoords='data',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc,angleA=180,angleB=0,armA=50,armB=50,rad=10', color=(0.3, ) * 3, lw=2))
        aximage = ax.inset_axes((ximage, y - h / 2, ximage - 1, h), transform=ax.transData)
        aximage.imshow(row['image'])
        aximage.axis('off')

    ax.legend([black_line_result[0]], ['$\\rho_O(x)$'], loc='upper left')
    ax.get_legend().get_frame().set_linewidth(0)
    ax.set_xlabel('$\\it{x}$')
    ax.set_ylabel('$\\rho(x)$')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def demo_fbln(ax):
    ax.hlines(1 / 0.8, 0.1, 0.9, lw=5, colors='C0', label='Cost-Agnostic')
    x = np.linspace(0.1, 0.9, 20)
    y = np.exp(- 8 * x)
    ax.plot(x, y / (y.sum() * np.diff(x).mean()),
            lw=5, c='C1', label='Cost-Driven')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax.get_legend().get_frame().set_linewidth(0)

    plt.setp(ax, ylim=(0, 5), xlabel='$\\it{x}$', ylabel='$\\rho(x)$')
    plt.setp([ax.spines['right'], ax.spines['top']], visible=False)


def figure1():
    fig = plt.figure(figsize=(20, 15), facecolor='w')
    gs = fig.add_gridspec(3, 9, height_ratios=[2.5, 1, 1])
    axbrain = fig.add_subplot(gs[0, :3])
    axphysical_virtual = fig.add_subplot(gs[0, 3:-3])
    ax_fblen_demo = fig.add_subplot(gs[0, -3:])
    ax_fblens = [fig.add_subplot(gs[-2, :2]), fig.add_subplot(gs[-2, 2:4]),
                 fig.add_subplot(gs[-1, :2]), fig.add_subplot(gs[-1, 2:4])]
    ax_fblen_ensemble = fig.add_subplot(gs[-2:, 4:-1])

    gs.tight_layout(fig, h_pad=5.5, w_pad=5)

    physical_virtual_fibers(ax=axphysical_virtual)
    axbrain.imshow(np.array(PIL.Image.open(data_path(r"images/figure1_illustration.png")).convert('RGBA')))
    axbrain.axis('off')
    fiberlength_distributions(axes=ax_fblens)
    fiberlength_distribution_ensemble(ax=ax_fblen_ensemble)
    demo_fbln(ax_fblen_demo)

    label_axes(fig, (axbrain, axphysical_virtual,
               ax_fblen_demo, *ax_fblens, ax_fblen_ensemble))
    return fig


if __name__ == "__main__":
    fig = figure1()
    fig.savefig(output_path('Figures/figure1.pdf'), dpi=300, bbox_inches='tight')
