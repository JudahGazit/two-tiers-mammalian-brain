from publication_paths import data_path, output_path
from figures.figure4 import *

def sup_figure4():
    n_bins = [5, 10, 20, 40]
    fig, axes = plt.subplot_mosaic(
        [[f'r_{n}' for n in n_bins[:2]],
         [f'r_{n}' for n in n_bins[2:]]],
        figsize=(10, 8), tight_layout=True,
    )
    for n in n_bins:
        ax = axes[f'r_{n}']
        plot_core_periphery_in_circles('all', n_bins=n, ax=ax)
        ax.set_title(f'R = {n}', weight='bold', size=28)
    return fig

if __name__ == '__main__':
    fig = sup_figure4()
    fig.savefig(output_path(f'SuppFigures/figure4_0.png'), dpi=300, bbox_inches='tight')
