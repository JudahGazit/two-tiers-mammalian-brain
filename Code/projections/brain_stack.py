from publication_paths import data_path, output_path
import logging
import os
import traceback
from collections import defaultdict

import alphashape
import nibabel as nib
import nilearn
import nilearn.surface
import plotly.graph_objects as go
import scipy.optimize
import scipy.spatial
import sklearn.cluster
from tqdm.contrib.concurrent import thread_map
from core_periphery.core_periphery import core_periphery_average_weight
from projections.brain_alignment import align_brain_hemispheres
from dataloaders.brain_net import BrainNet, BRAINS_LOCATIONS
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import trimesh
from projections.brain_projection import BrainProjection


warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


COLORS = [[38, 70, 83], [157, 2, 8], [72, 12, 168], [233, 196, 106], [244, 162, 97], [231, 111, 81]] + \
         [[int(np.random.random() * 255), int(np.random.random() * 255), int(np.random.random() * 255)] for _ in range(20 + 1)]
FLIP_Z = {'HowlerMonkey2', 'Baboon5', 'GreenMonkey2', 'MacaqueLionTail', 'CommonFox', 'Leopard2'}
PROJ_BRAIN_NAMES = {'Rodentia': 'Rat1', 'Chiroptera': 'ArtibeusJamacien', 'Artiodactyla': 'CapraNubiana3',
                    'Primates': 'Macaque10', 'Carnivora': 'Meerkat3', 'Humans': 'Human1'}
BRAINS_BLACKLIST = {'Moongose', 'BottleNoseDolphin', 'StrippedDolphin1',
                    'StrippedDolphin2', 'Donkey', 'MyotisVivesi2', 'Asellia'}
HUMAN_BRAINS_DIR = data_path(r'human_brains/processed')
HUMAN_BRAINS_MRI = data_path('human_brains/mri')
MAMMALS_MRI = data_path('brain_nets_raw')


def load_orders():
    df = pd.read_csv(data_path('MamiNum_formatted.csv'))
    order_mapping = df[~df.brain_name.isin(BRAINS_BLACKLIST)].groupby('Order').agg({'brain_name': list, 'AnimalName': 'count'})
    order_mapping = order_mapping.sort_values('AnimalName', ascending=False).head(5)
    order_mapping = order_mapping.brain_name.to_dict()
    all_humans = np.unique([b.split('.')[0] for b in os.listdir(HUMAN_BRAINS_DIR)]).tolist()
    order_mapping['Humans'] = all_humans
    return order_mapping


def plot_over_brain(brain, points, colors):
    markers = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                           mode='markers', marker=dict(size=10, color=colors))
    scene_params = dict({axis: dict(visible=False) for axis in ('xaxis', 'yaxis', 'zaxis')})
    fig = go.Figure(layout=dict(height=800, width=800, scene=scene_params),
                    data=[brain, markers])
    return fig


def load_mri(brain_name):
    file_name_prefix = HUMAN_BRAINS_MRI if 'Human' in brain_name else MAMMALS_MRI
    file_name = os.path.join(file_name_prefix, f'{brain_name}Surface.nii')
    if not os.path.isfile(file_name):
        file_name = os.path.join(file_name_prefix, f'{brain_name}_1vol.nii')
    rsn = nib.load(file_name)
    rsn_pos = np.asanyarray(rsn.dataobj)
    rsn_pos = rsn_pos.max(3) if len(rsn_pos.shape) > 3 else rsn_pos
    rsn = nib.Nifti1Image(rsn_pos[::-1, ::-1, :], np.eye(4))
    return rsn


def rotate_mri_by_aligned(points, dataset, aligned):
    rot, _ = scipy.spatial.transform.Rotation.align_vectors(aligned, dataset.positions - dataset.positions.mean(0))
    rot = rot.as_matrix().flatten().tolist()
    x = scipy.optimize.minimize(lambda m: np.sum((aligned -
                                                  np.array([m.reshape(3, 4)[:, :3] @ loc.transpose() + m.reshape(3, 4)[:, 3]
                                                            for loc in dataset.positions])) ** 2),
                                rot + [0, 0, 0]).x
    x = x.reshape(3, 4)
    return [x[:, :3] @ loc for loc in points] + x[:, 3]


def create_mesh_from_mri(mri_scan, dataset=None, aligned=None, threshold=0.8, sample_points=8000):
    dataset = mri_scan if dataset is None and isinstance(mri_scan, str) else dataset
    mri_scan = load_mri(mri_scan) if isinstance(mri_scan, str) else mri_scan
    dataset = BrainNet(dataset) if isinstance(dataset, str) else dataset
    aligned = align_brain_hemispheres(dataset.positions, mri_scan) if aligned is None else aligned

    mri_data = np.asarray(mri_scan.dataobj)
    points = np.argwhere(mri_data > np.quantile(mri_data, threshold))
    points = points[np.random.choice(len(points), sample_points, False)]
    points = rotate_mri_by_aligned(points[:, [1, 0, 2]], dataset, aligned)
    mesh = alphashape.alphashape(points, 0.25)
    return go.Mesh3d(x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
                     i=mesh.faces[:, 0], j=mesh.faces[:,
                                                      1], k=mesh.faces[:, 2],
                     color='lightpink', opacity=0.50, hoverinfo='skip')


def create_mesh_from_surface(brain_name, dataset, aligned):
    file_name_prefix = HUMAN_BRAINS_MRI if 'Human' in brain_name else MAMMALS_MRI
    mesh = nilearn.surface.load_surf_mesh(os.path.join(file_name_prefix, f'{brain_name}Surface.surf.gii'))
    coor = (mesh.coordinates * [-1, -1, 1])[:, [1, 0, 2]]
    coor = rotate_mri_by_aligned(coor, dataset, aligned)
    coor = (coor - coor.min(0)) / (coor.max(0) - coor.min(0)) * \
        (aligned.max(0) - aligned.min(0)) + aligned.min(0)
    return go.Mesh3d(x=coor[:, 0], y=coor[:, 1], z=coor[:, 2],
                     i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                     color='lightgrey', opacity=0.3, hoverinfo='skip',
                     lighting=dict(ambient=0.1, diffuse=0.9, fresnel=4.5, specular=2),
                     lightposition=dict(x=0, y=0, z=100))


def create_mesh_convex_hull(aligned, alpha=0.1):
    mesh = alphashape.alphashape(aligned, alpha)
    return go.Mesh3d(x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
                     i=mesh.faces[:, 0], j=mesh.faces[:,
                                                      1], k=mesh.faces[:, 2],
                     color='lightpink', opacity=0.50, hoverinfo='skip')


def cluster_dbscan(positions, core):
    r = np.sort(scipy.spatial.distance.cdist(positions, positions), 1)[:, 1].max()
    preds = sklearn.cluster.DBSCAN(r).fit_predict(positions[core])
    d = defaultdict(list)
    for k, v in zip(preds, positions[core]):
        if k > -1:
            d[k].append(v)
    centroids = np.array([np.stack(v).mean(0) for v in d.values()])
    return centroids


def cluster_optics(positions, core):
    preds = sklearn.cluster.OPTICS().fit_predict(positions[core])
    d = defaultdict(list)
    for k, v in zip(preds, positions[core]):
        if k > -1:
            d[k].append(v)
    centroids = np.array([np.stack(v).mean(0) for v in d.values()])
    return centroids


def cluster_kmeans(positions, core):
    model = sklearn.cluster.KMeans(5).fit(positions[core])
    return model.cluster_centers_


def cluster_null(positions, core):
    return positions[core]

def project_multiple_brains(brains, target_brain, core_periphery_finder=core_periphery_average_weight,
                            cluster_method=cluster_optics, n_jobs=8):
    def project_single(brain):
        try:
            mri_scan = load_mri(brain)
            dataset = BrainNet(brain, location=HUMAN_BRAINS_DIR if 'Human' in brain else BRAINS_LOCATIONS) if isinstance(brain, str) else brain
            core = np.where(core_periphery_finder(dataset))[0]
            flip = -1 if brain in FLIP_Z else 1
            positions = align_brain_hemispheres(dataset.positions, mri_scan) * [flip, 1, flip]
            projector = BrainProjection(positions, target_brain)
            core_centroids = cluster_method(positions, core)
            projection = projector.project(core_centroids)
            return projection
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # raise
            print(e)

    results = thread_map(project_single, brains if brains != 'all' else BrainNet.options(), max_workers=n_jobs)
    return [res for res in results if res is not None]


def create_mesh(proj_brain_dataset, proj_brain, proj_mri, proj_brain_name, kind='convexhull'):
    if kind == 'mri':
        try:
            mesh = create_mesh_from_surface(proj_brain_name, proj_brain_dataset, proj_brain)
        except:
            traceback.print_exc()
            mesh = create_mesh_from_mri(proj_mri, proj_brain_dataset, proj_brain, 0.85, 10000)
        return mesh
    elif kind == 'convexhull':
        return create_mesh_convex_hull(proj_brain)


def inside_mesh_fast(mesh3d_obj, resolution=100):
    vertices = np.vstack((mesh3d_obj.x, mesh3d_obj.y, mesh3d_obj.z)).T
    faces = np.vstack((mesh3d_obj.i, mesh3d_obj.j, mesh3d_obj.k)).T
    tri = trimesh.Trimesh(vertices=vertices, faces=faces)

    voxelized = tri.voxelized(pitch=(tri.extents.max() / resolution))
    voxel_centers = voxelized.fill('holes').points
    return voxel_centers


def generate_meshgrid(positions, resolution=1, factor=1.0):
    mins, maxs = factor * positions.min(0), factor * positions.max(0)
    resolution_per_axes = np.ceil((maxs - mins) / resolution)
    meshgrid = np.meshgrid(*[np.linspace(mins[i], maxs[i], int(resolution_per_axes[i])) for i in range(positions.shape[1])])
    meshgrid = np.stack(meshgrid).transpose(0, 2, 1, 3)
    return meshgrid


def plot_order(order_name, order, proj_brain_name=None, cluster_method=cluster_kmeans, cluster_of_clusters_mean=7.5, colors=None):
    colors = colors if colors is not None else COLORS
    proj_brain_name = proj_brain_name if isinstance(proj_brain_name, str) else PROJ_BRAIN_NAMES[order_name]
    proj_mri = load_mri(proj_brain_name)
    proj_brain_dataset = BrainNet(proj_brain_name, location=HUMAN_BRAINS_DIR if 'Human' in proj_brain_name else BRAINS_LOCATIONS)
    proj_brain_aligned = align_brain_hemispheres(proj_brain_dataset.positions, proj_mri)
    proj = np.concatenate(project_multiple_brains(order, proj_brain_aligned, cluster_method=cluster_method))

    clustered = sklearn.cluster.DBSCAN(cluster_of_clusters_mean, min_samples=3).fit_predict(proj)

    cluster_name = cluster_method.__name__.replace("cluster_", "")
    mesh = create_mesh(proj_brain_dataset, proj_brain_aligned, proj_mri, proj_brain_name)
    fig = plot_over_brain(mesh, proj, np.array(colors)[clustered + 1] / 255)
    fig.layout.title.text = f"""Non-Localized nodes of <b>{order_name}</b> (N = {len(order)} brains) projected onto <b>{proj_brain_name}</b>.
    <br>Each point is a {cluster_name.upper()} centroid of the Non-Localized set, colored by DBSCAN"""
    return fig


def plot_order_kde(order_name, order, proj_brain_name=None, cluster_of_clusters_mean=7.5, colors=None, q=0.5):
    colors = colors if colors is not None else COLORS
    proj_brain_name = proj_brain_name if isinstance(proj_brain_name, str) else PROJ_BRAIN_NAMES[order_name]
    proj_mri = load_mri(proj_brain_name)
    proj_brain_dataset = BrainNet(proj_brain_name, location=HUMAN_BRAINS_DIR if 'Human' in proj_brain_name else BRAINS_LOCATIONS)
    proj_brain_aligned = align_brain_hemispheres(proj_brain_dataset.positions, proj_mri)
    proj = np.concatenate(project_multiple_brains(order, proj_brain_aligned, cluster_method=cluster_null))
    mesh = create_mesh(proj_brain_dataset, proj_brain_aligned, proj_mri, proj_brain_name)

    kde = scipy.stats.gaussian_kde(proj.T)
    positions = inside_mesh_fast(mesh)
    kde_values = kde(positions.T)
    kde_values = kde_values / kde_values.max()
    kde_values_th = np.quantile(kde_values, q)
    proj = positions[kde_values > kde_values_th]

    clustered = sklearn.cluster.DBSCAN(cluster_of_clusters_mean, min_samples=3).fit_predict(proj)
    c = np.array(colors)[clustered + 1] / 255
    fig = plot_over_brain(mesh, proj, c)
    fig.layout.title.text = f"""Non-Localized nodes of <b>{order_name}</b> (N = {len(order)} brains) projected onto <b>{proj_brain_name}</b>.
    <br>Showing the top {(1 - q)*100:.2f}% densest regions, colored by DBSCAN"""
    return fig


if __name__ == '__main__':
    order_mapping = load_orders()

    with open('projections_html/mammals_hotspots.html', 'w') as f:
        for order_name, order in order_mapping.items():
            fig = plot_order_kde(order_name, order)
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    all_humans = np.unique([b.split('.')[0] for b in os.listdir(HUMAN_BRAINS_DIR)]).tolist()
    fig = plot_order_kde('Humans', all_humans, proj_brain_name='Human1')
    fig.write_html('projections_html/nodes_hotspots_humans.html')

    import pandas as pd
    dfs = []
    for order_name, order in order_mapping.items():
        proj_brain_name = PROJ_BRAIN_NAMES[order_name]
        proj_mri = load_mri(proj_brain_name)
        proj_brain_dataset = BrainNet(proj_brain_name, location=HUMAN_BRAINS_DIR if 'Human' in proj_brain_name else BRAINS_LOCATIONS)
        proj_brain_aligned = align_brain_hemispheres(proj_brain_dataset.positions, proj_mri)
        proj = project_multiple_brains(order, proj_brain_aligned, cluster_method=lambda v, c: v)
        for brain_name, p in zip(order, proj):
            brain = BrainNet(brain_name)
            coreness = np.where(core_periphery_average_weight(brain), 'E', 'V')
            brain_df = pd.DataFrame(np.column_stack([np.arange(len(p)), coreness, brain.positions, p]),
                                    columns=['index', 'type', *[f'{prefix}_{suffix}' for prefix in ('original', 'projected') for suffix in ('x', 'y', 'z')]])
            brain_df['name'] = brain_name
            brain_df['order'] = order_name
            brain_df['projected_name'] = proj_brain_name
            dfs.append(brain_df)
    dfs = pd.concat(dfs)
