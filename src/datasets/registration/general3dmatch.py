import numpy as np
import os
import os.path as osp
import torch
from src.datasets.base_dataset import BaseDataset
from src.datasets.registration.base3dmatch import Base3DMatch
from src.datasets.registration.utils import PatchExtractor
from src.metrics.registration_tracker import PatchRegistrationTracker
from src.core.data_transform.transforms import GridSampling
from torch_geometric.data import Batch


class General3DMatch(Base3DMatch):

    def __init__(self, root,
                 is_patch=True,
                 radius_patch=0.3,
                 num_frame_per_fragment=50,
                 mode='train_small',
                 min_overlap_ratio=0.3,
                 max_overlap_ratio=1.0,
                 max_dist_overlap=0.01,
                 tsdf_voxel_size=0.02,
                 depth_thresh=6,
                 is_fine=True,
                 transform=None,
                 pre_transform=None,
                 pre_transform_fragment=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False, detector=None):
        r"""
        Patch extracted from :the Princeton 3DMatch dataset\n
        `"3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions"
        <https://arxiv.org/pdf/1603.08182.pdf>`_
        paper, containing rgbd frames of the following dataset:
        `" SUN3D: A Database of Big Spaces Reconstructed using SfM and Object Labels
        "<http://sun3d.cs.princeton.edu/>`
        `"Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images
        "<https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images/>`
        `"Unsupervised Feature Learning for 3D Scene Labeling
        "<http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/>`
        `"BundleFusion: Real-time Globally Consistent 3D Reconstruction using Online
        Surface Re-integration
        "<http://graphics.stanford.edu/projects/bundlefusion/>`
        `"Learning to Navigate the Energy Landscape
        "<http://graphics.stanford.edu/projects/reloc/>`

        Args:

            root (string): Root directory where the dataset should be saved

            radius_patch(float, optional): the size of the patch

            is_patch(bool, optional): is it a patch mode or a

            num_frame_per_fragment (int, optional): indicate the number of frames
                we use to build fragments. If it is equal to 0, then we don't
                build fragments and use the raw frames.

            mode (string, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)

            min_overlap_ratio(float, optional): the minimum overlap we should have to match two fragments (overlap is the number of points in a fragment that matches in an other fragment divided by the number of points)
            max_overlap_ratio(float, optional): the maximum overlap we should have to match two fragments
            max_dist_overlap(float, optional): minimum distance to consider that a point match with an other.
            tsdf_voxel_size(float, optional): the size of the tsdf voxel grid to perform fine RGBD fusion to create fine fragments
            depth_thresh: threshold to remove depth pixel that are two far.

            is_fine: fine mode for the fragment fusion


            transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                every access. (default: :obj:`None`)

            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            detector: function that select a fixed number of point in the dataset for the testing
        """

        super(General3DMatch, self).__init__(root,
                                             num_frame_per_fragment,
                                             mode,
                                             min_overlap_ratio,
                                             max_overlap_ratio,
                                             max_dist_overlap,
                                             tsdf_voxel_size,
                                             depth_thresh,
                                             is_fine,
                                             transform,
                                             pre_transform,
                                             pre_filter,
                                             verbose,
                                             debug,
                                             detector)

        self.radius_patch = radius_patch
        self.is_patch = is_patch

    def get_patch(self, idx):
        if('train' in self.mode or 'val' in self.mode):
            match = np.load(
                osp.join(self.processed_dir,
                         self.mode, 'matches',
                         'matches{:06d}.npy'.format(idx)),
                allow_pickle=True).item()

            print(match['path_source'])
            data_source = torch.load(match['path_source'])
            data_target = torch.load(match['path_target'])
            p_extractor = PatchExtractor(self.radius_patch)
            # select a random match on the list of match.
            # It cannot be 0 because matches are filtered.
            rand = np.random.randint(0, len(match['pair']))

            data_source = p_extractor(data_source, match['pair'][rand][0])
            data_target = p_extractor(data_target, match['pair'][rand][1])

            if(self.transform is not None):
                data_source = self.transform(data_source)
                data_target = self.transform(data_target)
            batch = Batch.from_data_list([data_source, data_target])
            batch.pair = batch.batch
            batch.batch = None
            return batch.contiguous()

        else:
            raise NotImplementedError('Need to implement a testing dataset')

    def get_fragment(self, idx):

        if('train' in self.mode or 'val' in self.mode):
            match = np.load(
                osp.join(self.processed_dir,
                         self.mode, 'matches',
                         'matches{:06d}.npy'.format(idx)),
                allow_pickle=True).item()

            print(match['path_source'])
            data_source = torch.load(match['path_source'])
            data_target = torch.load(match['path_target'])
            if(self.transform is not None):
                data_source = self.transform(data_source)
                data_target = self.transform(data_target)
            batch = Batch.from_data_list([data_source, data_target])
            batch.pair = batch.batch
            batch.batch = None
            batch.y = torch.from_numpy(match['pair'])
            return batch.contiguous()

        else:
            raise NotImplementedError('Need to implement a testing dataset')


    def get(self, idx):
        if(self.is_patch):
            return self.get_patch(idx)
        else:
            return self.get_fragment(idx)

    def len(self):
        if('train' in self.mode or 'val' in self.mode):
            return len(os.listdir(osp.join(self.processed_dir,
                                           self.mode, 'matches')))
        else:
            raise NotImplementedError('need to implement testing dataset')


class General3DMatchDataset(BaseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        pre_transform = self.pre_transform
        train_transform = self.train_transform
        test_transform = self.test_transform

        self.train_dataset = General3DMatch(
            root=self._data_path,
            mode='train',
            radius_patch=dataset_opt.radius_patch,
            is_patch=dataset_opt.is_patch,
            num_frame_per_fragment=dataset_opt.num_frame_per_fragment,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            min_overlap_ratio=dataset_opt.min_overlap_ratio,
            tsdf_voxel_size=dataset_opt.tsdf_voxel_size,
            depth_thresh=dataset_opt.depth_thresh,
            pre_transform=pre_transform,
            transform=train_transform)

        self.test_dataset = General3DMatch(
            root=self._data_path,
            mode='test',
            radius_patch=dataset_opt.radius_patch,
            is_patch=dataset_opt.is_patch,
            num_frame_per_fragment=dataset_opt.num_frame_per_fragment,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            min_overlap_ratio=dataset_opt.min_overlap_ratio,
            tsdf_voxel_size=dataset_opt.tsdf_voxel_size,
            depth_thresh=dataset_opt.depth_thresh,
            pre_transform=pre_transform,
            transform=test_transform)

    @staticmethod
    def get_tracker(model, task: str, dataset, wandb_log: bool,
                    tensorboard_log: bool):
        """
        Factory method for the tracker

        Arguments:
            task {str} -- task description
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        return PatchRegistrationTracker(dataset, task, wandb_log,
                                        use_tensorboard=tensorboard_log)
