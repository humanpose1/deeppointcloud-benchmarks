"""
KITTI Test for the benchmark
"""
import gdown
import os
import os.path as osp
import logging
import requests
import glob
import re
import sys
import csv

from zipfile import ZipFile


from torch_points3d.datasets.registration.basetest import BasePCRBTest
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset

from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker

from torch_points3d.datasets.registration.utils import files_exist
from torch_points3d.datasets.registration.utils import makedirs

log = logging.getLogger(__name__)


class TestPairKitti(BasePCRBTest):

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False,
                 num_pos_pairs=200,
                 max_dist_overlap=0.01,
                 self_supervised=False,
                 min_size_block=2,
                 max_size_block=3,
                 min_points=500,
                 ss_transform=None,
                 use_fps=False):
        self.url = "https://cloud.mines-paristech.fr/index.php/s/XsyjolyYE4jfFMB/download"
        BasePCRBTest.__init__(self,
                              root=root,
                              transform=transform,
                              pre_transform=pre_transform,
                              pre_filter=pre_filter,
                              verbose=verbose, debug=debug,
                              max_dist_overlap=max_dist_overlap,
                              num_pos_pairs=num_pos_pairs,
                              self_supervised=self_supervised,
                              min_size_block=min_size_block,
                              max_size_block=max_size_block,
                              min_points=min_points,
                              ss_transform=ss_transform,
                              use_fps=use_fps)

    def download(self):
        folder = osp.join(self.raw_dir, "test")
        print(folder)
        if files_exist([folder]):  # pragma: no cover
            log.warning("already downloaded {}".format("test"))
            return
        else:
            makedirs(folder)
        log.info("Download elements in the file {}...".format(folder))
        req = requests.get(self.url)
        with open(osp.join(folder, "testKitti.zip"), "wb") as archive:
            archive.write(req.content)
        with ZipFile(osp.join(folder, "testKitti.zip"), "r") as zip_obj:
            log.info("extracting dataset")
            zip_obj.extractall(folder)
        os.remove(osp.join(folder, "testKitti.zip"))

    def process(self):
        super().process()


class TestKITTIDataset(BaseSiameseDataset):
    """
    this class is a dataset for testing registration algorithm on KITTI dataset
    It is inspired by: https://github.com/iralabdisco/point_clouds_registration_benchmark.
    """


    def __init__(self, dataset_opt):

        super().__init__(dataset_opt)
        pre_transform = self.pre_transform
        train_transform = self.train_transform
        ss_transform = getattr(self, "ss_transform", None)
        test_transform = self.test_transform

        # training is similar to test but only unsupervised training is allowed XD
        self.train_dataset = TestPairKitti(root=self._data_path,
                                           pre_transform=pre_transform,
                                           transform=train_transform,
                                           max_dist_overlap=dataset_opt.max_dist_overlap,
                                           self_supervised=True,
                                           min_size_block=dataset_opt.min_size_block,
                                           max_size_block=dataset_opt.max_size_block,
                                           num_pos_pairs=dataset_opt.num_pos_pairs,
                                           min_points=dataset_opt.min_points,
                                           ss_transform=ss_transform,
                                           use_fps=dataset_opt.use_fps)
        self.test_dataset = TestPairKitti(root=self._data_path,
                                          pre_transform=pre_transform,
                                          transform=test_transform,
                                          max_dist_overlap=dataset_opt.max_dist_overlap,
                                          num_pos_pairs=dataset_opt.num_pos_pairs,
                                          self_supervised=False)
