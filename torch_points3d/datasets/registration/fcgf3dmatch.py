#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License

import open3d as o3d

import copy
import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
from scipy.linalg import expm, norm
import pathlib

import torch_points3d.core.data_transform as t

import MinkowskiEngine as ME

from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker
from torch_points3d.datasets.registration.pair import Pair
from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset

def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)
    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def collate_pair_fn(list_data):
    xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans = list(
        zip(*list_data))
    xyz_batch0, xyz_batch1 = [], []
    matching_inds_batch, trans_batch, len_batch = [], [], []

    batch_id = 0
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f'Can not convert to torch tensor, {x}')

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]

        xyz_batch0.append(to_tensor(xyz0[batch_id]))
        xyz_batch1.append(to_tensor(xyz1[batch_id]))

        trans_batch.append(to_tensor(trans[batch_id]))

        matching_inds_batch.append(
            torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
        len_batch.append([N0, N1])

        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    xyz_batch0 = torch.cat(xyz_batch0, 0).float()
    xyz_batch1 = torch.cat(xyz_batch1, 0).float()
    trans_batch = torch.cat(trans_batch, 0).float()
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

    return {
        'pcd0': xyz_batch0,
        'pcd1': xyz_batch1,
        'sinput0_C': coords_batch0,
        'sinput0_F': feats_batch0.float(),
        'sinput1_C': coords_batch1,
        'sinput1_F': feats_batch1.float(),
        'correspondences': matching_inds_batch,
        'T_gt': trans_batch,
        'len_batch': len_batch
    }


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


class PairDataset(torch.utils.data.Dataset):
    AUGMENT = None

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        self.phase = phase
        self.files = []
        self.data_objects = []
        self.transform = transform
        self.voxel_size = config.voxel_size
        self.matching_search_voxel_size = \
            config.voxel_size * config.positive_pair_search_voxel_size_multiplier

        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def __len__(self):
        return len(self.files)


class IndoorPairDataset(PairDataset):
    OVERLAP_RATIO = None
    AUGMENT = None

    def __init__(self,
                 phase,
                 transform=None,
                 random_rotation=True,
                 random_scale=True,
                 manual_seed=False,
                 config=None):
        PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                             manual_seed, config)
        self.root = root = config.threed_match_dir
        logging.info(f"Loading the subset {phase} from {root}")
        self.num_features = 1
        subset_names = open(self.DATA_FILES[phase]).read().split()
        for name in subset_names:
            fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
            fnames_txt = glob.glob(root + "/" + fname)
            assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])

    def __getitem__(self, idx):
        file0 = os.path.join(self.root, self.files[idx][0])
        file1 = os.path.join(self.root, self.files[idx][1])
        data0 = np.load(file0)
        data1 = np.load(file1)
        xyz0 = data0["pcd"]
        xyz1 = data1["pcd"]
        color0 = data0["color"]
        color1 = data1["color"]
        matching_search_voxel_size = self.matching_search_voxel_size

        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz0 = scale * xyz0
            xyz1 = scale * xyz1

        if self.random_rotation:
            T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
            T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
            trans = T1 @ np.linalg.inv(T0)

            xyz0 = self.apply_transform(xyz0, T0)
            xyz1 = self.apply_transform(xyz1, T1)
        else:
            trans = np.identity(4)

        # Voxelization
        sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

        # Make point clouds using voxelized points
        pcd0 = make_open3d_point_cloud(xyz0)
        pcd1 = make_open3d_point_cloud(xyz1)

        # Select features and points using the returned voxelized indices
        pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
        pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
        # Get matches
        matches = np.array(get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size))
        # print(len(matches))
        # Get features
        npts0 = len(pcd0.colors)
        npts1 = len(pcd1.colors)

        feats_train0, feats_train1 = [], []

        feats_train0.append(np.ones((npts0, 1)))
        feats_train1.append(np.ones((npts1, 1)))

        feats0 = np.hstack(feats_train0)
        feats1 = np.hstack(feats_train1)

        # Get coords
        xyz0 = np.array(pcd0.points)
        xyz1 = np.array(pcd1.points)

        coords0 = np.floor(xyz0 / self.voxel_size)
        coords1 = np.floor(xyz1 / self.voxel_size)
        # print(coords0.shape, feats0.shape)
        # if self.transform:
        #     coords0, feats0 = self.transform(coords0, feats0)
        #     coords1, feats1 = self.transform(coords1, feats1)
        # coords0, feats0 = ME.utils.sparse_collate(coords0, feats0)
        # coords1, feats1 = ME.utils.sparse_collate(coords1, feats1)
        matches = torch.from_numpy(matches)
        feats0 = torch.from_numpy(feats0).to(torch.float)
        feats1 = torch.from_numpy(feats1).to(torch.float)

        data = Pair(pos_x=torch.from_numpy(xyz0[:, 0]).to(torch.float),
                    pos_y=torch.from_numpy(xyz0[:, 1]).to(torch.float),
                    pos_z=torch.from_numpy(xyz0[:, 2]).to(torch.float),
                    pos_x_target=torch.from_numpy(xyz1[:, 0]).to(torch.float),
                    pos_y_target=torch.from_numpy(xyz1[:, 1]).to(torch.float),
                    pos_z_target=torch.from_numpy(xyz1[:, 2]).to(torch.float),
                    pos=torch.from_numpy(coords0),
                    pos_target=torch.from_numpy(coords1),
                    x=feats0,
                    x_target=feats1,
                    pair_ind=matches,
                    size_pair_ind=torch.tensor([len(matches)]))
        # return xyz0, xyz1, coords0, coords1, feats0, feats1, matches
        return data





class ThreeDMatchPairDataset(IndoorPairDataset):
    OVERLAP_RATIO = 0.3
    DATA_FILES = {
        'train': '/media/admincaor/DataHDD2To/mines/code/FCGF/config/train_3dmatch.txt',
        'val': '/media/admincaor/DataHDD2To/mines/code/FCGF/config/val_3dmatch.txt',
        'test': '/media/admincaor/DataHDD2To/mines/code/FCGF/config/test_3dmatch.txt'
    }


ALL_DATASETS = [ThreeDMatchPairDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
    assert phase in ['train', 'trainval', 'val', 'test']
    if shuffle is None:
        shuffle = phase != 'test'

    if config.dataset not in dataset_str_mapping.keys():
        logging.error(f'Dataset {config.dataset}, does not exists in ' +
                      ', '.join(dataset_str_mapping.keys()))

    Dataset = dataset_str_mapping[config.dataset]

    use_random_scale = False
    use_random_rotation = False
    transforms = []
    if phase in ['train', 'trainval']:
        use_random_rotation = config.use_random_rotation
        use_random_scale = config.use_random_scale
        transforms += [t.Jitter()]

    dset = Dataset(
        phase,
        transform=t.Compose(transforms),
        random_scale=use_random_scale,
        random_rotation=use_random_rotation,
        config=config)

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=collate_pair_fn,
        pin_memory=False,
        drop_last=True)

    return loader


class FCGF3DMatchDataset(BaseSiameseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.train_dataset = ThreeDMatchPairDataset('train',
                                                    transform=t.Jitter(),
                                                    random_scale=True,
                                                    random_rotation=True,
                                                    config=dataset_opt)

        self.test_dataset = ThreeDMatchPairDataset('val',
                                                   transform=None,
                                                   random_scale=False,
                                                   random_rotation=False,
                                                   config=dataset_opt)


    @staticmethod
    def get_tracker(model, dataset, wandb_log: bool,
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

        return FragmentRegistrationTracker(dataset, wandb_log=wandb_log,
                                           use_tensorboard=tensorboard_log)
