import numpy as np
import scipy.spatial


class BrainProjection:
    def __init__(self, source_brain_point_cloud, target_brain_point_cloud):
        self.source_brain_point_cloud = source_brain_point_cloud
        self.target_brain_point_cloud = target_brain_point_cloud
        self.source_convex_hull = self._convex_hull(self.source_brain_point_cloud)
        self.target_convex_hull = self._convex_hull(self.target_brain_point_cloud)

    def _get_scaling(self, points, brain_points, convex_hull=None):
        points = points / np.linalg.norm(points, axis=1)[:, None]
        equations, triangles = convex_hull if convex_hull is not None else self._convex_hull(brain_points)
        intersect = self._is_intersect(equations, points)
        return np.nanmin(intersect, 0)

    def _is_intersect(self, equations, points):
        a, d = equations[:, :3], equations[:, 3]
        intersect = -d[:, None] / (a @ points.transpose())
        intersect[intersect <= 0] = np.nan
        return intersect

    def _convex_hull(self, brain_points):
        mesh = scipy.spatial.ConvexHull(brain_points)
        triangles = np.array([mesh.points[s] for s in mesh.simplices])
        equations = mesh.equations
        return equations, triangles

    def _sphere_projection(self, positions, brain_points=None, convex_hull=None):
        scaling = self._get_scaling(positions, brain_points, convex_hull)
        return positions / scaling[:, None]

    def _brain_projections(self, sphere_projection, brain_points, convex_hull=None):
        scaling = self._get_scaling(sphere_projection, brain_points, convex_hull)
        return sphere_projection * scaling[:, None]

    def project(self, points=None):
        points = points if points is not None else self.source_brain_point_cloud
        sphere_projection = self._sphere_projection(points, self.source_brain_point_cloud, self.source_convex_hull)
        return self._brain_projections(sphere_projection, self.target_brain_point_cloud, self.target_convex_hull)
