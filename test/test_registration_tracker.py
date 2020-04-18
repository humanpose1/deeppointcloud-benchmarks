import unittest
import torch
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT)
from src.metrics.registration_metrics import rodrigues
from src.metrics.registration_tracker import FragmentRegistrationTracker


def rand_T():

    t = torch.randn(3)
    axis = torch.randn(3)
    theta = torch.norm(axis)
    axis = axis / theta
    R = rodrigues(axis, theta)
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class MockDataset:
    def __init__(self):
        self.num_classes = 2


class MockModel:
    def __init__(self):
        self.iter = 0
        self.losses = [
            {"loss_1": 1, "loss_2": 2},
            {"loss_1": 2, "loss_2": 2},
            {"loss_1": 1, "loss_2": 2},
            {"loss_1": 1, "loss_2": 2},
        ]

        list_xyz = [torch.randn(100, 3) for _ in range(16)]
        trans = [rand_T() for _ in range(16)]
        list_xyz_target = [list_xyz[i].mm(trans[i][:3, :3]) + trans[i][:3, 3] for i in range(16)]
        self.ind = [torch.arange(400) for _ in range(4)]
        self.ind_target = [torch.arange(400) for _ in range(4)]
        self.xyz = [torch.cat(list_xyz[4 * i : 4 * (i + 1)], 0) for i in range(4)]
        self.xyz_target = [torch.cat(list_xyz_target[4 * i : 4 * (i + 1)], 0) for i in range(4)]
        self.feat = self.xyz
        self.feat_target = self.xyz_target
        self.batch_idx = [torch.cat(tuple(torch.arange(4).repeat(100, 1).T)) for i in range(4)]
        self.batch_idx_target = [torch.cat(tuple(torch.arange(4).repeat(100, 1).T)) for i in range(4)]

        rang1 = torch.cat((torch.arange(10), torch.arange(100, 110), torch.arange(200, 210), torch.arange(300, 310)))
        rang2 = torch.cat((torch.arange(20), torch.arange(100, 120), torch.arange(200, 220), torch.arange(300, 320)))
        rang3 = torch.cat((torch.arange(10), torch.arange(100, 115), torch.arange(200, 220), torch.arange(300, 330)))

        inv1 = torch.cat(
            (torch.arange(9, -1, -1), torch.arange(109, 99, -1), torch.arange(209, 199, -1), torch.arange(309, 299, -1))
        )
        inv2 = torch.cat(
            (
                torch.arange(19, -1, -1),
                torch.arange(119, 99, -1),
                torch.arange(219, 199, -1),
                torch.arange(319, 299, -1),
            )
        )
        inv3 = torch.cat(
            (torch.arange(9, -1, -1), torch.arange(116, 99, -1), torch.arange(219, 199, -1), torch.arange(329, 299, -1))
        )

        self.feat[1][rang1] = self.feat[1][inv1]
        self.feat[2][rang2] = self.feat[2][inv2]
        self.feat[3][rang3] = self.feat[3][inv3]

    def get_outputs(self):
        return self.feat[self.iter], self.feat_target[self.iter]

    def get_xyz(self):
        return self.xyz[self.iter], self.xyz_target[self.iter]

    def get_ind(self):
        return self.ind[self.iter], self.ind_target[self.iter]

    def get_current_losses(self):
        return self.losses[self.iter]

    def get_batch_idx(self):
        return self.batch_idx[self.iter], self.batch_idx_target[self.iter]


class TestSegmentationTracker(unittest.TestCase):
    def test_track(self):
        tracker = FragmentRegistrationTracker(MockDataset(), stage="test", tau_2=0.83, num_points=100)
        model = MockModel()
        list_hit_ratio = [1.0, 0.9, 0.8, (0.9 + 0.85 + 0.8 + 0.7) / 4]
        list_feat_match_ratio = [1.0, 1.0, 0.0, 0.5]
        for i in range(4):
            tracker.track(model)
            metrics = tracker.get_metrics()
            # the most important metrics in registration
            self.assertAlmostEqual(metrics["test_hit_ratio"], list_hit_ratio[i])
            self.assertAlmostEqual(metrics["test_feat_match_ratio"], list_feat_match_ratio[i])
            tracker.reset("test")
            model.iter += 1
        tracker.reset("test")
        model.iter = 0
        for i in range(4):
            tracker.track(model)
            model.iter += 1
        metrics = tracker.get_metrics()
        self.assertAlmostEqual(metrics["test_hit_ratio"], (4 * 1.0 + 4 * 0.9 + 4 * 0.8 + 0.9 + 0.85 + 0.8 + 0.7) / 16)
        self.assertAlmostEqual(metrics["test_feat_match_ratio"], (4 * 1 + 4 * 1 + 4 * 0 + 2 * 1 + 2 * 0) / 16)


if __name__ == "__main__":
    unittest.main()
