from publication_paths import data_path, output_path
import plotly.express as px
from projections.brain_alignment import align_brain_hemispheres
from projections.brain_stack import load_mri, rotate_mri_by_aligned
from projections.brain_projection import BrainProjection
from scipy.signal import savgol_filter
import scipy.io
import numpy as np
import pandas as pd
from dataloaders.brain_net import BrainNet, BRAINS_LOCATIONS
import os
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from tqdm.contrib.concurrent import thread_map
import h5py
from sklearn.neighbors import KDTree
from core_periphery.core_periphery import core_periphery_average_weight

HUMAN_TRACTS_DIR = data_path('human_brains/core_tracts_cleaned')
TRACTS_DIR = data_path('brain_nets_tracts/streamlines_core')
# HUMAN_BRAINS_DIR = r'../Datasets/brain_nets_fixed_means_knn'
HUMAN_BRAINS_DIR = data_path(r'human_brains/processed')

def tracts_figure(avoid_figure=False, **tracts_lists):
    ready_tracts_list = []
    for j, (brain_name, tracts_list) in enumerate(tracts_lists.items()):
        tracts_list = np.column_stack([np.concatenate(tracts_list).astype(float),
                                       np.concatenate([[i] * len(tract) for i, tract in enumerate(tracts_list)]),
                                       np.concatenate([[brain_name] * len(tract) for i, tract in enumerate(tracts_list)]),
                                      ])
        tracts_list = pd.DataFrame(tracts_list, columns=['x', 'y', 'z', 'group', 'color'])
        ready_tracts_list.append(tracts_list)
    ready_tracts_list = pd.concat(ready_tracts_list).astype({'x': float, 'y': float, 'z': float, 'group': int, 'color': str})
    if avoid_figure:
        fig = None
    else:
        fig = px.line_3d(ready_tracts_list, x='x', y='y', z='z', line_group='group', color='color', height=768, )
    return fig, ready_tracts_list


def transform_tracts(input_brain_name, target_brain_name, n_tracts=500):
    input_brain = BrainNet(input_brain_name, location=HUMAN_BRAINS_DIR if 'Human' in input_brain_name else BRAINS_LOCATIONS)
    target_brain = BrainNet(target_brain_name, location=HUMAN_BRAINS_DIR if 'Human' in target_brain_name else BRAINS_LOCATIONS)

    input_brain_aligned = align_brain_hemispheres(input_brain.positions, load_mri(input_brain_name))
    target_brain_aligned = align_brain_hemispheres(target_brain.positions, load_mri(target_brain_name))

    projector = BrainProjection(input_brain_aligned, target_brain_aligned)
    projection =  projector.project()
    tracts_dir = HUMAN_TRACTS_DIR if 'Human' in input_brain_name else TRACTS_DIR
    tracts = scipy.io.loadmat(os.path.join(tracts_dir, f'{input_brain_name}_core.mat'))['Tracts'][0]
    random_tracts = tracts[np.random.choice(len(tracts), n_tracts)] if n_tracts > 0 else tracts
    random_tracts_aligned = rotate_mri_by_aligned(np.concatenate(random_tracts), input_brain, input_brain_aligned)
    random_tracts = np.array(np.split(random_tracts_aligned, np.cumsum([len(a) for a in random_tracts])), dtype=object)[:-1]
    transformed_tracts = []

    if input_brain_name != target_brain_name:
        for tract in random_tracts:
            proj = projector.project(tract) ## Project fibers
            x, y, z = tract.transpose()
            x_proj, y_proj, z_proj = savgol_filter(proj.transpose(), window_length=3, polyorder=1)
            transformed_tracts.append(np.column_stack([x_proj, y_proj, z_proj]))
    else:
        transformed_tracts = random_tracts
    return np.array(transformed_tracts, dtype=object)


def project_all_brains(target_brain_name='Hyrax1', brains=None, n_tracts=500):
    tracts_dir = HUMAN_TRACTS_DIR if 'Human' in target_brain_name else TRACTS_DIR
    brain_names = [brain_name.replace('_core.mat', '') for brain_name in os.listdir(tracts_dir) if '_core' in brain_name]
    brain_names = brain_names if brains is None else [brain for brain in brain_names if brain in brains]
    transformed = {}

    def process_brain(brain_name):
        try:
            return transform_tracts(brain_name, target_brain_name, n_tracts)
        except Exception as e:
            print(f'Error processing {brain_name}: {e}')
            pass

    transformed_results = thread_map(process_brain, brain_names, max_workers=8)
    transformed = {brain_name: result for brain_name, result in zip(brain_names, transformed_results) if result is not None}
    fig, df = tracts_figure(avoid_figure=True, **transformed)
    return fig, df


def hotspots_of_tracts(tracts):
    H, B = np.histogramdd(tracts.values[:, :3], 50, )
    B = np.array(B)[:, 1:]

    q = np.quantile(H, 0.98)
    pixels = pd.DataFrame(B[np.arange(3), np.argwhere(H > q)], columns=['x', 'y', 'z'])
    pixels['val'] = H[H > q]
    pixels['size'] = pixels['val'] / pixels['val'].max() * 20
    pixels['color'] = np.log(pixels['val'] / pixels['val'].max())

    pixels['c'] = DBSCAN(1, min_samples=1).fit_predict(pixels[['x', 'y', 'z']])
    pixels = pixels[pixels.c != -1]

    fig = go.Figure(data=[
        go.Scatter3d(x=pixels['x'], y=pixels['y'], z=pixels['z'],
                     mode='markers',
                     marker=dict(size=pixels['size'] * 2, color=pixels['color'], line=dict(width=0), opacity=0.1))],
                 layout=dict(height=768))
    return fig

def extract_core_tracts(brain_name):
    tracts = h5py.File(data_path(f'human_brains/tracts/{brain_name}_Tracts.mat'), 'r')
    brain = BrainNet(brain_name, location=HUMAN_BRAINS_DIR if 'Human' in brain_name else BRAINS_LOCATIONS)
    vdims = np.array(tracts['VDims'])
    all_tracts = [tracts[t[0]][:] / vdims for t in tracts['Tracts']]
    start_end_points = np.stack([np.stack([t[:, 0], t[:, -1]]) for t in all_tracts])
    nei = KDTree(brain.positions)
    _, ind_from = nei.query(start_end_points[:, 0, :])
    _, ind_to = nei.query(start_end_points[:, 1, :])
    ind_tracts = np.column_stack([ind_from, ind_to])
    coreness = core_periphery_average_weight(brain)
    core = np.where(coreness)[0]
    core_tracts = np.isin(ind_tracts, core).all(1)
    core_tracts = np.array([all_tracts[i].T for i in np.where(core_tracts)[0]], dtype=object)
    scipy.io.savemat(data_path(f'human_brains/core_tracts/{brain_name}_core.mat'), {'Tracts': core_tracts})

if __name__ == '__main__':
    # for i in range(5):
        # extract_core_tracts(f'Human{i + 1}')

    # _, df = project_all_brains('Human1')
    # df.to_csv('../Datasets/human_brains/core_tracts/aligned_streamlines.csv', index=False)
    from projections.brain_stack import *

    process_metadata = [
        dict(streamlines=data_path('brain_nets_tracts/aligned_streamlines.csv'), proj_brain_name='Macaque10', output_file='projections_html/primates.html'),
        dict(streamlines=data_path('human_brains/core_tracts/aligned_streamlines.csv'), proj_brain_name='Human1', output_file='projections_html/humans.html'),
    ]
    for metadata in process_metadata:
        df = pd.read_csv(metadata['streamlines'], dtype={'x': float, 'y': float, 'z': float})
        fig = hotspots_of_tracts(df)

        proj_brain_name = metadata['proj_brain_name']
        proj_mri = load_mri(proj_brain_name)
        proj_brain_dataset = BrainNet(proj_brain_name, location=HUMAN_BRAINS_DIR if 'Human' in proj_brain_name else BRAINS_LOCATIONS)
        proj_brain_aligned = align_brain_hemispheres(proj_brain_dataset.positions, proj_mri)
        mesh = create_mesh(proj_brain_dataset, proj_brain_aligned, proj_mri, proj_brain_name)
        mesh.opacity = 0.1

        fig.add_trace(mesh).write_html(metadata['output_file'])
