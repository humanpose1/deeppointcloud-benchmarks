import numpy as np
import os
import os.path as osp
import torch


from torch_points3d.datasets.classification.modelnet import SampledModelNet
from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.registration.base_siamese_dataset import GeneralFragment
from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker


class SiameseModelNet(SampledModelNet, GeneralFragment):
    r"""
    the ModelNet Dataset from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing sampled CAD models of 40 categories. Each sample contains 10,000
    points uniformly sampled with their normal vector.

    But applied for registration.
    Only the self supervised mode is supported
    """

    def __init__(self, root,
                 name="10",
                 min_size_block=0.3,
                 max_size_block=2,
                 max_dist_overlap=0.1,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 num_pos_pairs=1024):
        SampledModelNet.__init__(self,
                                 root,
                                 name,
                                 train,
                                 transform,
                                 pre_transform,
                                 pre_filter)
        self.self_supervised = True
        self.num_pos_pairs = num_pos_pairs
        self.min_size_block = min_size_block
        self.max_size_block = max_size_block

    def get(self, idx):
        return self.get_fragment(idx)


class SiameseModelNetDataset(BaseSiameseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        pre_transform = self.pre_transform
        train_transform = self.train_transform
        test_transform = self.test_transform
        pre_filter = self.pre_filter
        self.tau_1 = dataset_opt.tau_1
        self.rot_thresh = dataset_opt.rot_thresh
        self.trans_thresh = dataset_opt.trans_thresh

        self.train_dataset = SiameseModelNet(
            root=self._data_path,
            train=True,
            min_size_block=dataset_opt.min_size_block,
            max_size_block=dataset_opt.max_size_block,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            pre_transform=pre_transform,
            transform=train_transform,
            pre_filter=pre_filter,
            num_pos_pairs=dataset_opt.num_pos_pairs)

        self.test_dataset = SiameseModelNet(
            root=self._data_path,
            train=False,
            self_supervised=True,
            min_size_block=dataset_opt.min_size_block,
            max_size_block=dataset_opt.max_size_block,
            pre_transform=pre_transform,
            transform=test_transform,
            pre_filter=pre_filter,
            num_pos_pairs=dataset_opt.num_pos_pairs)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """
        Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return FragmentRegistrationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, tau_1=self.tau_1, rot_thresh=self.rot_thresh, trans_thresh=self.trans_thresh)
