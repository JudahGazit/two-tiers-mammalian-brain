"""Run the paper's figures or a self-contained synthetic demonstration."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
FIGURES = [f'figure{i}' for i in range(1, 7)] + [f'supfigure{i}' for i in (2, 4, 6)]
REVISION = ['supfigure_distance_metric', 'supfigure_min_instances', 'supfigure_node_profiles',
            'supfigure_hcp', 'suptable_modules']
FIGURES += REVISION
REVISION_INPUTS = {'supfigure_distance_metric': ['MamiNum_formatted.csv'],
                   'supfigure_min_instances': ['brain_nets_mat/*_FinalFinal2.mat']}


def check_inputs(name, data):
    # A coarse preflight. It cannot prove that every subject/file is present.
    patterns = ['brain_nets_fixed_means_knn/*.connections.npy',
                'brain_nets_fixed_means_knn/*.distances.npy',
                'brain_nets_fixed_means_knn/*.positions.npy']
    if name == 'supfigure_hcp':
        return []  # downloads the HCP connectomes from Open NeuroData into <data-dir>/hcp_atlas
    if name in REVISION:
        return [p for p in patterns + REVISION_INPUTS.get(name, []) if not any(data.glob(p))]
    if name not in ('supfigure2', 'supfigure4'):
        patterns += ['images/animals/*.png']
    if name in ('figure2', 'figure5', 'figure6', 'supfigure6'):
        patterns += ['brain_nets_raw/*.nii']
    assets = {
        'figure1': ['figure1_illustration.png', 'figure1_rho_explained.png'],
        'figure2': ['figure_2_core_illustration.png', 'figure_2_periphery_illustration.png'],
        'figure4': ['figure3_illustration.png', 'figure4_path_illustraion.png',
                    'figure4_illustration.png', 'figure4_3d_illustration.png', 'figure4_2d_illustration.png'],
        'figure5': ['full-brain-illustration.png', 'hemi-brain-illustration.png',
                    'quad-brain-illustration.png', 'octo-brain-illustration.png',
                    'figure5_illustration_cuts_control.png', 'figure5_illustration_cuts_actual.png'],
        'figure6': [f'brain_projection_illustration_{letter}.png' for letter in 'abc'],
    }
    patterns += ['images/' + asset for asset in assets.get(name, [])]
    if name == 'figure5':
        patterns += ['brain_net_control/results.mat'] if (data / 'brain_net_control/results.mat').is_file() else ['brain_net_control/*.npy']
    if name in ('figure6', 'supfigure6'):
        patterns += ['MamiNum_formatted.csv', 'human_brains/processed/*.connections.npy',
                     'human_brains/mri/*.nii', 'brain_nets_tracts/streamlines_core/*_core.mat',
                     'human_brains/core_tracts_cleaned/*_core.mat']
        patterns += [f'images/figure6_pickle/{stem}_cache.pkl' for stem in ('nodes', 'tracts', 'data')]
    return [pattern for pattern in patterns if not any(data.glob(pattern))]


def check_environment():
    problems = []
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f'{name}: missing (expected {expected})')
        else:
            if actual != expected:
                problems.append(f'{name}: installed {actual}, expected {expected}')
    return problems


def demo(output, seed):
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    from core_periphery.core_periphery import core_periphery_average_weight
    from dataloaders.brain_net import BrainNet
    from metrics.routing_cost import RoutingCost
    from projections.brain_projection import BrainProjection

    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(40, 3))
    positions -= positions.mean(axis=0)
    distances = cdist(positions, positions)
    adjacency = (rng.random((40, 40)) < 0.15).astype(float)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0)
    for i in range(40):
        adjacency[i, (i + 1) % 40] = adjacency[(i + 1) % 40, i] = 1
    brain = BrainNet('synthetic', adjacency=adjacency.copy(), distances=distances.copy(),
                     positions=positions, fibers=distances * adjacency)
    coreness = core_periphery_average_weight(brain)
    routing = RoutingCost(brain).cost().value
    expected_routing = nx.average_shortest_path_length(nx.from_numpy_array(adjacency))
    np.testing.assert_allclose(routing, expected_routing)
    projected = BrainProjection(positions, positions).project()
    np.testing.assert_allclose(projected, positions, atol=1e-10)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(*positions[:, :2].T, c=coreness, cmap='coolwarm')
    axes[0].set(title='Synthetic core / periphery', xlabel='x', ylabel='y')
    axes[1].hist(brain.fibers[np.isfinite(brain.fibers)], bins=15)
    axes[1].set(title='Synthetic normalized fiber lengths', xlabel='length', ylabel='count')
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output / 'demo.png', dpi=150)
    plt.close(fig)
    report = {'seed': seed, 'nodes': 40, 'core_nodes': int(coreness.sum()),
              'mean_shortest_path': float(routing), 'independent_routing_check': 'passed',
              'identity_projection_check': 'passed', 'note': 'Synthetic demonstration; not manuscript data.'}
    (output / 'demo.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figures', nargs='*', choices=FIGURES + ['all'])
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('BRAIN_NETS_DATA_DIR', ROOT / 'Datasets')))
    parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get('BRAIN_NETS_OUTPUT_DIR', ROOT / 'outputs')))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--check', action='store_true', help='Check dependency versions and coarse input presence')
    parser.add_argument('--demo', action='store_true', help='Run synthetic checks without manuscript data or artwork')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    selected = FIGURES if 'all' in args.figures else args.figures
    if not selected and not args.demo:
        selected = FIGURES
    data, output = args.data_dir.resolve(), args.output_dir.resolve()
    os.environ['BRAIN_NETS_DATA_DIR'] = str(data)
    os.environ['BRAIN_NETS_OUTPUT_DIR'] = str(output)
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(ROOT / 'Code'))
    if args.check:
        missing = {name: check_inputs(name, data) for name in selected}
        report = {'environment_problems': check_environment(), 'missing_inputs': missing,
                  'note': 'Presence check only: subject coverage, array contents and cache provenance require separate verification.'}
        print(json.dumps(report, indent=2))
        return int(bool(report['environment_problems']) or any(missing.values()))
    if args.worker or args.demo:
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        if args.demo:
            demo(output, args.seed)
        else:
            runpy.run_module('figures.' + selected[0], run_name='__main__')
        return 0
    failures = {name: check_inputs(name, data) for name in selected}
    if any(failures.values()):
        print(json.dumps({'missing_inputs': failures}, indent=2))
        print('See Datasets/README.md for input layout and access information.', file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in selected:
        start = time.perf_counter()
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), name,
                                 '--worker', '--data-dir', str(data), '--output-dir', str(output),
                                 '--seed', str(args.seed)])
        records.append({'figure': name, 'returncode': result.returncode,
                        'seconds': round(time.perf_counter() - start, 3), 'seed': args.seed})
        (output / 'run-report.json').write_text(json.dumps(records, indent=2) + '\n')
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
