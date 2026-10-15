import warnings

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pandas as pd
import scipy.spatial
import sklearn.svm
from sklearn.decomposition import PCA


class BrainAlignment:
    def __init__(self, positions: np.ndarray, mri_scan: nibabel.Nifti1Image = None):
        self.mri_scan = mri_scan
        self.positions = positions

    def align(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            positions, hemispheres_cut = self._rotate_by_hemispheres_cut(self.positions, return_hemispheres_cut=True)
            positions = self._rotate_by_max_variance(positions, hemispheres_cut)
            return positions

    def _rotate_by_hemispheres_cut(self, positions, hemispheres_cut=None, return_hemispheres_cut=False):
        if hemispheres_cut is None:
            hemispheres_cut = self._find_hemispheres_cut(positions)
        rot, _ = scipy.spatial.transform.Rotation.align_vectors(np.eye(3)[:1], hemispheres_cut)
        positions = rot.apply(positions - positions.mean(0))
        return (positions, hemispheres_cut) if return_hemispheres_cut else positions

    def _find_hemispheres_cut(self, positions):
        svm = sklearn.svm.LinearSVC(C=.5, max_iter=20000)
        which_hemisphere = np.array([0 if i < 100 else 1 for i in range(200)])
        svm.fit(positions - positions.mean(0), which_hemisphere)
        return svm.coef_

    def _rotate_by_max_variance(self, positions, hemispheres_cut=None, mri_threshold=0.8, mri_samples=5000):
        rotate_by = positions
        if self.mri_scan is not None:
            rotate_by = self._prepare_mri_scan(hemispheres_cut, mri_samples, mri_threshold)

        pca_vector = PCA(2).fit(rotate_by[:, 1:]).components_[0]
        positions = self._rotate_by_pca_vector(pca_vector, positions)
        positions = self._flip_y_if_neccesary(positions)
        return positions

    def _prepare_mri_scan(self, hemispheres_cut, mri_samples=5000, mri_threshold=0.8):
        mri_data = np.asarray(self.mri_scan.dataobj)
        mri_data = np.argwhere(mri_data > np.quantile(mri_data, mri_threshold))
        mri_data = mri_data[np.random.choice(len(mri_data), mri_samples, False)]
        rotate_by = self._rotate_by_hemispheres_cut(mri_data[:, [1, 0, 2]], hemispheres_cut)
        return rotate_by

    def _flip_y_if_neccesary(self, positions):
        variance_bottom = abs(positions[positions[:, 1] < np.quantile(positions[:, 1], 0.3), 0]).mean()
        variance_top = abs(positions[positions[:, 1] > np.quantile(positions[:, 1], 0.7), 0]).mean()
        if variance_bottom < variance_top:
            positions = positions * [1, -1, 1]
        return positions

    def _rotate_by_pca_vector(self, pca_vector, positions):
        rot, _ = scipy.spatial.transform.Rotation.align_vectors([[0, 1, 0]], [[0, *pca_vector]])
        positions = np.column_stack([positions[:, 0], rot.apply(positions)[:, 1:]])
        return positions


def align_brain_hemispheres(positions, mri_scan=None):
    return BrainAlignment(positions, mri_scan).align()
