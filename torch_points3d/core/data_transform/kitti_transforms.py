import open3d
import torch
import numpy as np
import numba

from .grid_transform import GridSampling3D
from .transforms import apply_mask

class ComputeNormals(object):

    def __init__(self, radius=0.7, max_nn=30):
        self.radius = radius
        self.max_nn = max_nn


    def __call__(self, data):
        search_param = open3d.geometry.KDTreeSearchParamHybrid(radius=self.radius, max_nn=self.max_nn)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(data.pos.detach().cpu().numpy())
        pcd.estimate_normals(search_param)
        data.norm = torch.from_numpy(np.asarray(pcd.normals)).float()
        return data

class RandomizePlane(object):

    """
    estimate the main plain and randomize points in it
    """

    def __init__(self, ransac_thresh=5e-7, iterations=1000, num_points_per_iteration=5, normal_thresh=1e-2, sigma_noise=1):
        self.ransac_thresh = ransac_thresh
        self.normal_thresh = normal_thresh
        self.iterations = iterations
        self.num_points_per_iteration = num_points_per_iteration
        self.sigma_noise = sigma_noise

    @staticmethod
    def _pca_compute(pos):
        centered = pos - pos.mean(0)
        H = (centered.T @ centered) / len(pos)
        e, w = torch.symeig(H, eigenvectors=True)
        normal = w[:, 0]
        return normal, pos.mean(0)

    @staticmethod
    def _count_agreements(data, normal, center, ransac_thresh=1e-2, normal_thresh=1e-2, return_mask=False):
        normalized_centered = (data.pos - center) / (torch.norm(data.pos-center, 0) + 1e-20)
        mask_0 = torch.abs(normalized_centered @ normal) < ransac_thresh
        if getattr(data, "norm", None) is None:
            mask = mask_0
        else:
            mask_1 = 1 - torch.abs(data.norm @ normal) < normal_thresh
            mask = mask_0 & mask_1
        if return_mask:
            return mask
        else:
            return mask.sum(0)

    @staticmethod
    def _random_plane(points, normal, center, range_p):
        points_n = points + torch.randn(points.shape[0], 3) * range_p
        centered_points_n = points_n - center

        projected_points = centered_points_n - (centered_points_n @ normal).view(-1, 1) * normal
        return projected_points + center

    def _compute_plane_ransac(self, data):
        final_normal = torch.zeros(3)
        final_center = torch.zeros(3)
        final_num_agreements = 0
        for _ in range(self.iterations):
            # select random points
            list_ind = torch.randint(0, len(data.pos), (self.num_points_per_iteration,))
            # compute The normal of that plane
            normal, center = self._pca_compute(data.pos[list_ind])
            num_agreements = self._count_agreements(
                data, normal, center, self.ransac_thresh, self.normal_thresh)
            if(num_agreements > final_num_agreements):
                final_normal = normal
                final_center = center
                final_num_agreements = num_agreements
        return final_normal, final_center, final_num_agreements

    def __call__(self, data):
        normal, center, num_agreements = self._compute_plane_ransac(data)
        mask = self._count_agreements(data,
                                      normal,
                                      center,
                                      self.ransac_thresh,
                                      return_mask=True)
        normal, center = self._pca_compute(data.pos[mask])
        plane = self._random_plane(data.pos[mask], normal, center,self.sigma_noise)
        data.pos[mask] = plane
        if getattr(data, "norm", None) is not None:
            data.norm[mask] = normal
        data.mask_randomize_plane = mask
        return data



class KittiPeriodicSampling(object):
    """
    sample point at a periodic distance Idea from J-E Deschaud
    """

    def __init__(self, period: float = 0.1, prop: float = 0.1,
                 remove_distance: float = 4.0,
                 pow_dist: float = 0.5,
                 grid_size_center: float = 0.01,
                 skip_keys=[]):

        self.pulse = 2 * np.pi / period
        self.thresh = np.cos(self.pulse * prop * period * 0.5)
        self.pow_dist = pow_dist
        self.remove_distance = remove_distance
        self.grid_sampling = GridSampling3D(grid_size_center, mode="last")
        self.skip_keys = skip_keys

    def __call__(self, data):

        data_c = self.grid_sampling(data.clone())
        i = torch.randint(0, len(data_c.pos), (1,))
        center = data_c.pos[i]
        d_p = torch.norm(data.pos - center, dim=1)
        mask_0 = d_p > self.remove_distance
        mask_1 = torch.cos(self.pulse * d_p**self.pow_dist) > self.thresh
        mask = torch.logical_and(mask_0, mask_1)
        data = apply_mask(data, mask, self.skip_keys)
        return data

    def __repr__(self):
        return "{}(pulse={}, thresh={}, remove_distance={}, grid_sampling={}, skip_keys={})".format(
            self.__class__.__name__, self.pulse, self.thresh, self.remove_distance, self.grid_sampling, self.skip_keys)
