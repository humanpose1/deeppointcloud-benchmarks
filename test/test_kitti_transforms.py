import unittest
import sys
import os
import numpy as np
import torch

from torch_geometric.data import Data

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, ".."))
torch.manual_seed(0)


from torch_points3d.core.data_transform import RandomizePlane, KittiPeriodicSampling


np.random.seed(0)


class Testhelpers(unittest.TestCase):
    def test_pca(self):
        trans = RandomizePlane()
        pos = torch.cat([torch.randn(100, 2), torch.zeros(100, 1)], 1)
        normal, _ = trans._pca_compute(pos)
        torch.testing.assert_allclose(normal, torch.tensor([0, 0, 1]).float())

    def test_count_aggrements(self):
        trans = RandomizePlane()
        pos = torch.cat([torch.randn(100, 2), torch.zeros(100, 1)], 1)
        normal, center = trans._pca_compute(pos)
        pos[[4, 8, 12, 16, 20], 2] = 10
        data = Data(pos=pos)
        num_agg = trans._count_agreements(data, normal, center, 0.01)  # pi/300 which is  0.6degree
        self.assertAlmostEqual(num_agg.item(), 95)

    def test_robust_plane_detection(self):
        trans = RandomizePlane(num_points_per_iteration=5, ransac_thresh=1e-6)
        pos = torch.cat([torch.randn(100, 2), torch.zeros(100, 1)], 1)
        pos = torch.cat([pos, torch.randn(200, 3)])
        data = Data(pos=pos)
        normal, center, num_agg = trans._compute_plane_ransac(data)

        torch.testing.assert_allclose(normal, torch.tensor([0, 0, 1]).float())

    def test_plane_generation(self):
        trans = RandomizePlane()
        center = torch.zeros(3)
        normal = torch.tensor([0, 0, 1]).float()
        point = torch.randn(100, 3)
        plane = trans._random_plane(point, normal, center, 100)
        estimated_normal, _ = trans._pca_compute(plane)
        torch.testing.assert_allclose(normal, estimated_normal)

    def test_kitti_periodic_sampling(self):
        tr = KittiPeriodicSampling()
        pos = torch.randn(500, 3)
        data = Data(pos=pos)
        data = tr(data)
        self.assertGreater(500, len(data.pos))


if __name__ == "__main__":
    unittest.main()
