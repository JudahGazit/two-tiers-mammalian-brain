import scipy
import scipy.optimize
import scipy.spatial

import sklearn
import sklearn.neighbors
import numpy as np

def interpolate_using_knn(fibers, euclidean_distances, k=5):
    def polyzero(x, a, b, c):
            return a * x ** b + c

    p, e = scipy.optimize.curve_fit(polyzero, euclidean_distances[fibers > 0], fibers[fibers > 0], method='trf', loss='soft_l1')
    reg = sklearn.neighbors.KNeighborsRegressor(k, weights='uniform')
    reg.fit(euclidean_distances[fibers > 0].flatten().reshape(-1, 1), fibers[fibers > 0].flatten())
    inter = reg.predict(euclidean_distances.reshape(-1, 1)).reshape(euclidean_distances.shape)
    cond = euclidean_distances > euclidean_distances[fibers > 0].min()
    inter = inter * cond + polyzero(euclidean_distances, *p) * (~cond)
    return inter

def interpolate_fiber_length(fibers, positions, neighborhood_threshold=.2):
    interpolated = fibers.copy()
    euclidean_distances = scipy.spatial.distance.cdist(positions, positions)
    knn_interpolation = interpolate_using_knn(fibers, euclidean_distances)

    for i, j in np.argwhere(np.isnan(fibers)):
        if i < j and np.isnan(interpolated[i, j]):
            distance = euclidean_distances[i, j]
            neighborhood_i = np.where(euclidean_distances[i] < neighborhood_threshold * distance)[0]
            neighborhood_j = np.where(euclidean_distances[j] < neighborhood_threshold * distance)[0]
            fiber = np.asarray(fibers[neighborhood_i][:, neighborhood_j])
            if np.nansum(fiber):
                interpolated[i, j] = interpolated[j, i] = fiber[fiber > 0].mean()
            else:
                interpolated[i, j] = interpolated[j, i] = knn_interpolation[i, j]
    return interpolated
