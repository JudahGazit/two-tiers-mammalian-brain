from publication_paths import data_path, output_path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from figures.utils import *
from style import *
from dataloaders.brain_net import BrainNet
from core_periphery.core_periphery import core_periphery_average_weight
from tqdm.auto import tqdm
import seaborn as sns
import re

def get_distances_from_brain(brain, distance_type):
    if distance_type == 'fibers':
        return brain.distances, brain.fibers
    if distance_type == 'euclidean':
        euc = scipy.spatial.distance.cdist(brain.positions, brain.positions)
        euc = euc / euc.max()
        return euc, np.where(brain.fibers > 0, euc, np.nan)

def plot_three_fblen_dists(bin_type, bin_count, core_threshold=0.3, axes=None,
                           distance_type='fibers', core_periphery_finder=core_periphery_average_weight):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
    all_es = {}
    names = ['global', 'core', 'periphery']
    colors = [(0.5, 0.5, 0.5), 'C0', 'C1']
    ylabels = ['$\\rho(x)$', '$\\rho_{\\mathcal{E}}(x)$', '$\\rho_{\\mathcal{V}}(x)$']
    mean_colors = ['C2', *colors[1:]]
    for brain_name in tqdm(BrainNet.options()):
        try:
            brain = BrainNet(brain_name)
            coreness = core_periphery_finder(brain, core_threshold)
            core, periphery = np.where(coreness)[0], np.where(~coreness)[0]
            distances, fibers = get_distances_from_brain(brain, distance_type)
            plot_params = dict(bin_type=bin_type, return_xy=True, alpha=0.05, style='-', n_bins=bin_count)
            subsets = [np.arange(len(distances)), core, periphery]
            subsets = [(distances[subset][:, subset], fibers[subset][:, subset]) for subset in subsets]
            for i, ax in enumerate(axes):
                append_to_es = []
                if ax is not None:
                    x, y = plot_fbln_dist(*subsets[i], ax=ax, color=colors[i], **plot_params)
                    append_to_es = [(x, y)]
                all_es[names[i]] = all_es.get(names[i], []) + append_to_es

        except KeyboardInterrupt: raise
        except: raise

    averages = []
    for ax, kind, color, ylabel in zip(axes, names, mean_colors, ylabels):
        if ax is not None:
            x, y = zip(*all_es[kind])
            x, y = np.mean(x, 0), np.mean(y, 0)
            averages.append([x, y])
            ax.plot(x, y, f'-{color}', lw=3)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.setp(ax, ylabel=ylabel, xlabel='$x$')

    plt.setp([ax for ax in axes if ax is not None], xlim=(None, 1), ylim=(0, 1))
    return averages


def histogram_fblens():
    fig, axes = plt.subplot_mosaic([
        ['all_cut_low', 'all_cut_high',     '.', 'E_cut_low', 'E_cut_high',     '.', 'V_cut_low', 'V_cut_high'],
        ['all_qcut_low', 'all_qcut_high',   '.', 'E_qcut_low', 'E_qcut_high',   '.', 'V_qcut_low', 'V_qcut_high']
    ], figsize=(20, 5), width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1, 1], gridspec_kw=dict(hspace=0.8, wspace=0.8))
    cut_low = [v for k, v in axes.items() if '_cut_low' in k]
    cut_high = [v for k, v in axes.items() if '_cut_high' in k]
    qcut_low = [v for k, v in axes.items() if '_qcut_low' in k]
    qcut_high = [v for k, v in axes.items() if '_qcut_high' in k]

    bins_low, bins_high = 7, 15

    cut_low[0].text(-1, 0.5, 'Equal Width', va='center', ha='center', rotation=90, weight='bold', size=28, transform=cut_low[0].transAxes)
    qcut_low[0].text(-1, 0.5, 'Equal Count', va='center', ha='center', rotation=90, weight='bold', size=28, transform=qcut_low[0].transAxes)

    for axes, bins in zip((cut_low, cut_high), (bins_low, bins_high)):
        for ax in axes:
            ax.text(0.5, 1.5, f'k={bins}', va='center', ha='center', weight='bold', size=28, transform=ax.transAxes)
    label_axes(fig, [ax for axes in zip(cut_low, cut_high, qcut_low, qcut_high) for ax in axes])

    plot_three_fblen_dists('cut', bins_low, cut_low)
    plot_three_fblen_dists('cut', bins_high, cut_high)
    plot_three_fblen_dists('qcut', bins_low, qcut_low)
    plot_three_fblen_dists('qcut', bins_high, qcut_high)
    return fig


def threhold_selection():
    thresholds = [20, 30, 40]
    fig, axes = plt.subplot_mosaic(
        [[*[f'{prefix}_{t}' for t in thresholds], '.', f'{prefix}_agg'] for prefix in ['e', 'v']],
        tight_layout=True, figsize=(10, 5), width_ratios=[1, 1, 1, 0.1, 1], gridspec_kw=dict(hspace=0.8, wspace=0.8))
    label_axes(fig, axes.values())
    averages = []
    for threshold in thresholds:
        threshold_axes = [v for k, v in axes.items() if str(threshold) in k]
        threshold_averages = plot_three_fblen_dists('kde', 8, threshold / 100, [None] + threshold_axes)
        for ax in threshold_axes:
            ax.set_ylabel(re.sub(r'(\\mathcal{.})', r'\1_{\\alpha}', ax.get_ylabel()))
        averages.append(threshold_averages)

    averages = np.stack(averages)
    agg_axes = [v for k, v in axes.items() if '_agg' in k]
    linestyles = ['--', '-', ':', '-.']
    for i, (ax, color) in enumerate(zip(agg_axes, ['C0', 'C1'])):
        for threshold, r, s in zip(thresholds, averages[:, i], linestyles):
            ax.plot(*r, s, c=color, label=f'{threshold / 100:.1f}')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.setp(ax, xticks=np.linspace(0, 1, 3))
    plt.setp(axes.values(), ylim=(0, 4))

    top_axes = [v for k, v in axes.items() if 'e_' in k]
    for threshold, ax in zip(thresholds, top_axes):
        ax.set_title(f'α={threshold / 100:.1f}', weight='bold')
    return fig


def nitpicking_figure_template(**kwargs):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3), gridspec_kw=dict(wspace=0.5))
    averages = plot_three_fblen_dists(**kwargs, axes=axes[:3])
    for xy, color in zip(averages, ['C2', 'C0', 'C1']):
        axes[-1].plot(*xy, color=color, lw=3)
    plt.setp(axes[-1], xlabel='$x$', ylabel='$\\rho(x)$')
    plt.setp(axes, xlim=(0, 1), ylim=(0, 3))

    label_axes(fig, axes, True, 0.05)
    return fig

def fblens_under_euclidean_geometry():
    fig = nitpicking_figure_template(bin_type='kde', bin_count=8, distance_type='euclidean')
    for ax in fig.get_axes():
        ax.set_ylabel(re.sub(r'\\rho', r'{\\rho}^{\\delta}', ax.get_ylabel()))
    return fig

def separation_using_degrees():
    def separate_using_degrees(brain, threshold):
        degrees = np.nansum(brain.adjacency, axis=0)
        return degrees > np.quantile(degrees, 1 - threshold)

    fig = nitpicking_figure_template(bin_type='kde', bin_count=8, core_periphery_finder=separate_using_degrees)
    for ax in fig.get_axes():
        ax.set_ylabel(re.sub(r'(\\mathcal{.})', r"\\hat{\1}", ax.get_ylabel()))
    return fig


def relation_between_fbcent_and_degree():
    fig, axes = plt.subplots(1, 3, figsize=(15, 3), gridspec_kw=dict(wspace=0.5))
    label_axes(fig, axes, True)
    degrees = []
    fbcents = []
    for brain_name in BrainNet.options():
        brain = BrainNet(brain_name)
        fbcent = np.nansum(brain.fibers, axis=0) / np.nansum(brain.adjacency, axis=0)
        degrees.append(np.nansum(brain.adjacency, axis=0))
        fbcents.append(fbcent)
    degrees, fbcents = np.array(degrees), np.array(fbcents)
    degree_zscore = ((degrees - degrees.mean(1, keepdims=True)) / degrees.std(1, keepdims=True))
    fbcent_zscore = ((fbcents - fbcents.mean(1, keepdims=True)) / fbcents.std(1, keepdims=True))
    spearmans = np.array([scipy.stats.spearmanr(d, f).statistic for d, f in zip(degrees, fbcents)])

    sns.regplot(x=degree_zscore.flatten(), y=fbcent_zscore.flatten(),
                color='C2', scatter_kws=dict(alpha=0.5, linewidths=0.5, ec='k'),
                line_kws=dict(zorder=10, color='k'), ax=axes[0])
    plt.setp(axes[0], xlabel='$\\bar{k}_i$', ylabel='$\\bar{f}_i$')

    sns.histplot(spearmans, bins='auto', stat='density', ax=axes[1])
    axes[1].axvline(np.nanmean(spearmans), linestyle='--', color='k')
    axes[1].text(np.nanmean(spearmans), axes[1].get_ylim()[1], f'{np.nanmean(spearmans):.3f}', ha='center', size=14)
    plt.setp(axes[1], xlabel='Spearman($k, f$)', xlim=(0, 1))

    top_degrees = degrees.argsort(1)[:, -int(0.3 * degrees.shape[1]):]
    top_fbcents = fbcents.argsort(1)[:, -int(0.3 * fbcents.shape[1]):]
    jaccards = [np.intersect1d(d, f).size / np.union1d(d, f).size for d, f in zip(top_degrees, top_fbcents)]

    sns.histplot(jaccards, bins='auto', stat='density', ax=axes[2])
    axes[2].axvline(np.nanmean(jaccards), linestyle='--', color='k')
    axes[2].text(np.nanmean(jaccards), axes[2].get_ylim()[1], f'{np.nanmean(jaccards):.3f}', ha='center', size=14)
    plt.setp(axes[2], xlabel='$J_\\alpha$', xlim=(0, 1))
    return fig


def sup_figure2():
    figs = [
        None, #histogram_fblens(),
        None, #threhold_selection(),
        None, #fblens_under_euclidean_geometry(),
        relation_between_fbcent_and_degree(),
        None, # separation_using_degrees()
    ]
    return figs

if __name__ == '__main__':
    figs = sup_figure2()
    for i, fig in enumerate(figs):
        if fig is not None:
            fig.savefig(output_path(f'SuppFigures/figure2_{i}.png'), dpi=300, bbox_inches='tight')
