import unittest
import torch
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)
from torch_points3d.core.losses.dirichlet_loss import (
    _variance_estimator_dense,
    dirichlet_loss,
    _variance_estimator_sparse,
)
from torch_points3d.core.losses.metric_losses import InfoNCELoss


class TestDirichletLoss(unittest.TestCase):
    def test_loss_dense(self):
        pos = torch.tensor([[[0, 0, 0], [1, 0, 0], [1.1, 0, 0]]], dtype=torch.float)
        f = torch.tensor([[1, 1, 3]], dtype=torch.float)

        var = _variance_estimator_dense(1.01, pos, f)
        torch.testing.assert_allclose(var, [[0, 4, 4]])

        loss = dirichlet_loss(1.01, pos, f)
        self.assertAlmostEqual(loss.item(), 4 / 3.0)

    def test_loss_sparse(self):
        pos = torch.tensor([[0, 0, 0], [1, 0, 0], [1.1, 0, 0], [0, 0, 0], [1, 0, 0], [1.1, 0, 0]], dtype=torch.float)
        f = torch.tensor([1, 1, 3, 0, 1, 0], dtype=torch.float)
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1])

        var = _variance_estimator_sparse(1.01, pos, f, batch_idx)
        torch.testing.assert_allclose(var, [0, 4, 4, 1, 2, 1])

        loss = dirichlet_loss(1.01, pos, f, batch_idx)
        self.assertAlmostEqual(loss.item(), sum([0, 4, 4, 1, 2, 1]) / (2 * 6))

    def test_info_nce_loss(self):
        loss = InfoNCELoss(temperature=1, num_pos=10, num_hn_samples=0)

        f0 = torch.tensor([[1.0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 1, 0]]).float()
        f1 = torch.tensor([[1.0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]]).float()
        f2 = torch.tensor([[1, 0.01, 0.5, 0.1], [1, 0, 1, 0], [0, 1, 1, 0]]).float()

        positive_pairs = torch.tensor([[0, 0], [1, 1]]).long()
        res1 = loss(f0, f1, positive_pairs)
        res2 = loss(f0, f2, positive_pairs)
        print(res1, res2)
        self.assertGreater(res2.item(), res1.item())


if __name__ == "__main__":
    unittest.main()
