"""
Human Connectome Project connectomes (Supplementary Section 11).

Atlas connectomes of the HCP-1200 release, reconstructed by NeuroData and distributed through the
Open NeuroData repository (s3://open-neurodata/hcp1200). Each file lists streamline counts between the
regions of an atlas; node coordinates are the region centroids of the Neuroparc atlas files.
A_ij = 1 if at least one streamline links regions i and j.
"""
import os
import re
import urllib.parse
import urllib.request

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage

from analysis.connectome import CACHE_DIR

BUCKET = 'https://open-neurodata.s3.amazonaws.com'
CACHE = f'{CACHE_DIR}/hcp_atlas'
ATLASES = ('Schaefer200', 'Schaefer400')
N_SUBJECTS = 100


def subjects(n=N_SUBJECTS, atlases=ATLASES):
    """The first n subjects in the repository with non-empty connectomes for all atlases (cached)."""
    path = f'{CACHE}/subjects_{n}.txt'
    if not os.path.exists(path):
        chosen = [s for s in _all_subjects() if all(_has_connectome(s, a) for a in atlases)][:n]
        open(path, 'w').write('\n'.join(chosen))
    return open(path).read().split()


def centroids(atlas):
    img = nib.load(_cached(f'{atlas}_space-MNI152NLin6_res-1x1x1.nii.gz',
                           f'neuroparc/atlases/label/Human/{atlas}_space-MNI152NLin6_res-1x1x1.nii.gz'))
    labels = np.asarray(img.dataobj).astype(int)
    regions = np.unique(labels[labels > 0])
    voxels = scipy.ndimage.center_of_mass(np.ones_like(labels), labels, regions)
    return regions, nib.affines.apply_affine(img.affine, np.array(voxels))


def adjacency(subject, atlas, regions):
    counts = streamline_counts(subject, atlas, regions)
    A = (counts + counts.T) > 0
    np.fill_diagonal(A, False)
    return A


def streamline_counts(subject, atlas, regions):
    edges = pd.read_csv(_cached(*_connectome_paths(subject, atlas)), sep=r'\s+', header=None, names=['i', 'j', 'w'])
    index = {r: n for n, r in enumerate(regions)}
    C = np.zeros((len(regions), len(regions)))
    for i, j, w in edges.itertuples(index=False):
        if i in index and j in index:
            C[index[i], index[j]] += w
    return C


def _connectome_paths(subject, atlas):
    name = f'{subject}_ses-1_run-1_dwi_{atlas}_space-MNI152NLin6_res-1x1x1_connectome.csv'
    return f'{subject}_{atlas}.csv', f'hcp1200/{subject}/ses-1/connectomes/{atlas}_space-MNI152NLin6_res-1x1x1/{name}'


def _has_connectome(subject, atlas):
    try:
        path = _cached(*_connectome_paths(subject, atlas))
    except OSError:
        return False
    weights = pd.read_csv(path, sep=r'\s+', header=None).iloc[:, 2] if os.path.getsize(path) else []
    return len(weights) > 0 and weights.sum() > 0


def _cached(local_name, remote_key):
    path = f'{CACHE}/{local_name}'
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        urllib.request.urlretrieve(f'{BUCKET}/{remote_key}', path)
    return path


def _all_subjects():
    found, token = [], None
    while True:
        page = urllib.request.urlopen(_listing_url(token)).read().decode()
        found += re.findall(r'<Prefix>hcp1200/(sub-\d+)/</Prefix>', page)
        token = next(iter(re.findall(r'<NextContinuationToken>([^<]*)', page)), None)
        if token is None:
            return found


def _listing_url(token):
    url = f'{BUCKET}/?list-type=2&delimiter=/&prefix=hcp1200/'
    return url + ('&continuation-token=' + urllib.parse.quote(token) if token else '')
