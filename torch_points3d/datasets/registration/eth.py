import numpy as np
import os
import os.path as osp
import random
import torch
import json
from torch_geometric.data import Data
from torch_points_kernels.points_cpu import ball_query

from torch_points3d.datasets.registration.pair import Pair, MultiScalePair
from torch_points3d.datasets.registration.basetest import BaseETHTest
from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.registration.base_siamese_dataset import GeneralFragment
from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker
from torch_points3d.datasets.registration.utils import tracked_matches
from torch_points3d.datasets.registration.utils import compute_overlap_and_matches


class SSETH(BaseETHTest):

    r"""
    ETH dataset training in self supervised mode
    """

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False,
                 min_size_block=0.3,
                 max_size_block=2,
                 max_dist_overlap=0.1,
                 num_pos_pairs=1024,
                 num_random_pt=5000):

        super(SSETH, self).__init__(root,
                                    transform,
                                    pre_transform,
                                    pre_filter,
                                    verbose,
                                    debug,
                                    num_random_pt)
        self.self_supervised = True
        self.min_size_block = min_size_block
        self.max_size_block = max_size_block
        self.max_dist_overlap = max_dist_overlap
        self.path_table = osp.join(self.processed_dir, 'fragment')
        with open(osp.join(self.path_table, 'table.json'), 'r') as f:
            self.table = json.load(f)

    def __getitem__(self, idx):

        data_source = torch.load(
            osp.join(self.path_table, 'fragment_{:06d}.pt'.format(idx)))
        data_target = torch.load(
            osp.join(self.path_table, 'fragment_{:06d}.pt'.format(idx)))

        pos = data_source.pos
        i = torch.randint(0, len(pos))
        size_block = random.random()*(self.max_size_block - self.min_size_block) + self.min_size_block
        point = pos[i].view(1, 3)
        ind, dist = ball_query(point,
                               pos,
                               radius=size_block,
                               max_num=-1,
                               mode=1)
        _, col = ind[dist[:, 0] > 0].t()
        new_pair = torch.stack((col, col)).T

        if self.transform is not None:
            data_source = self.transform(data_source)
            data_target = self.transform(data_target)

        if(hasattr(data_source, "multiscale")):
            batch = MultiScalePair.make_pair(data_source, data_target)
        else:
            batch = Pair.make_pair(data_source, data_target)
        if self.is_online_matching:
            new_match = compute_overlap_and_matches(
                Data(pos=data_source.pos),
                Data(pos=data_target.pos),
                self.max_dist_overlap
            )
            batch.pair_ind = torch.from_numpy(new_match["pair"].copy())
        else:
            pair = tracked_matches(data_source, data_target, new_pair)
            batch.pair_ind = pair

        num_pos_pairs = len(batch.pair_ind)
        if self.num_pos_pairs < len(batch.pair_ind):
            num_pos_pairs = self.num_pos_pairs

        rand_ind = torch.randperm(len(batch.pair_ind))[:num_pos_pairs]
        batch.pair_ind = batch.pair_ind[rand_ind]
        batch.size_pair_ind = torch.tensor([num_pos_pairs])
        return batch.contiguous()

    def __len__(self):
        return len(self.table)

    def get_table(self):
        return self.table


class SSETHDataset(BaseSiameseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        pre_transform = self.pre_transform
        train_transform = self.train_transform
        test_transform = self.test_transform
        pre_filter = self.pre_filter
        self.tau_1 = dataset_opt.tau_1
        self.rot_thresh = dataset_opt.rot_thresh
        self.trans_thresh = dataset_opt.trans_thresh

        self.train_dataset = SSETH(
            root=self._data_path,
            min_size_block=dataset_opt.min_size_block,
            max_size_block=dataset_opt.max_size_block,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            pre_transform=pre_transform,
            transform=train_transform,
            pre_filter=pre_filter,
            num_pos_pairs=dataset_opt.num_pos_pairs)

        self.test_dataset = SSETH(
            root=self._data_path,
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
