import json
import numpy as np
import os
import os.path as osp
import re
import torch
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.registration.detector import RandomDetector
from torch_points3d.datasets.registration.liffre import read_mesh_vertices, Liffre


class BaseLiffre(Liffre):
    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False,
                 num_random_pt=5000):
        pass


class TestLiffre(Liffre):
    def __init__(self,
                 root,
                 pre_transform=None,
                 pre_filter=None,
                 transform=None,
                 verbose=False,
                 debug=False,
                 num_random_pt=5000):

        super(TestLiffre, self).__init__(root,
                                         False,
                                         True,
                                         transform,
                                         pre_transform,
                                         pre_filter)

        self.num_random_pt = num_random_pt

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""

        path_match = osp.join(self.processed_dir, self.mode, 'matches')
        match = np.load(osp.join(path_match, "matches{:06d}.npy".format(idx)), allow_pickle=True).item()
        data = torch.load(match["path_source"]).to(torch.float)
        if(self.transform is not None):
            data = self.transform(data)
        if(self.num_random_pt > 0):
            detector = RandomDetector(self.num_random_pt)
            data = detector(data)
        return data

    def get_name(self, idx):
        path_match = osp.join(self.processed_dir, self.mode, 'matches')
        match = np.load(osp.join(path_match, "matches{:06d}.npy".format(idx)), allow_pickle=True).item()
        path = match["path_source"]
        split = osp.split(path)
        ind = int(re.search(r'\d+', split[1]).group())
        # name = split[1]
        name = "cloud_bin_{}".format(ind)
        scene = osp.split(split[0])[1]
        return scene, name


class TestLiffreDataset(BaseDataset):

    def __init__(self, dataset_opt):

        super().__init__(dataset_opt)
        pre_transform = self.pre_transform
        test_transform = self.test_transform

        self.base_dataset = TestLiffre(root=self._data_path,
                                       pre_transform=pre_transform,
                                       transform=test_transform,
                                       num_random_pt=dataset_opt.num_random_pt)
        if(dataset_opt.is_patch):
            raise NotImplementedError("Not the mode patch yet")
        else:
            self.test_dataset = self.base_dataset

    def get_name(self, idx):
        """
        return a pair of string which indicate the name of the scene and
        the name of the point cloud
        """
        return self.base_dataset.get_name(idx)


    @property
    def num_fragment(self):
        return len(self.base_dataset)
