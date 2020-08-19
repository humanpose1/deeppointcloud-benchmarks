import logging
import numpy as np
import os
import os.path as osp
from plyfile import PlyData
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_points_kernels.points_cpu import ball_query

from torch_points3d.datasets.registration.pair import Pair, MultiScalePair
from torch_points3d.datasets.registration.utils import tracked_matches
from torch_points3d.datasets.registration.utils import files_exist
from torch_points3d.datasets.registration.utils import makedirs

from torch_points3d.metrics.registration_tracker import FragmentRegistrationTracker
from torch_points3d.datasets.registration.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.registration.base_siamese_dataset import GeneralFragment

log = logging.getLogger(__name__)


def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        normals = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        normals[:, 0] = plydata["vertex"].data["nx"]
        normals[:, 1] = plydata["vertex"].data["ny"]
        normals[:, 2] = plydata["vertex"].data["nz"]
    return vertices, normals


class Liffre(Dataset):

    def __init__(self,
                 root,
                 train=True,
                 self_supervised=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 min_block_size=0.1,
                 max_block_size=0.5,
                 max_dist_overlap=5e-3,
                 num_pos_pairs=1024
    ):
        """
        Liffre Dataset for
        """

        self.self_supervised = self_supervised
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.max_dist_overlap = max_dist_overlap
        self.size = 0
        self.num_pos_pairs = num_pos_pairs
        if(train):
            self.mode = "train"
        else:
            self.mode = "val"
        super(Liffre, self).__init__(root, transform, pre_transform, pre_filter)
        self.size = len(os.listdir(osp.join(self.processed_dir, self.mode, 'matches')))

    @property
    def raw_file_names(self):
        return [self.mode]

    @property
    def processed_file_names(self):
        res = [
            osp.join(self.mode, 'fragment'),
            osp.join(self.mode, 'matches')]

        return res

    def _pre_transform_fragment(self, mod):
        """
        pre_transform raw fragments and save it into fragments
        """
        out_dir = osp.join(self.processed_dir,
                           mod, 'fragment')
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)
        for scene_path in os.listdir(osp.join(self.raw_dir, mod)):

            in_dir = osp.join(self.raw_dir,
                              mod,
                              scene_path)
            out_dir = osp.join(self.processed_dir,
                               mod, 'fragment',
                               scene_path)
            makedirs(out_dir)
            list_fragment_path = sorted([f
                                         for f in os.listdir(in_dir)
                                         if 'ply' in f])
            for path in list_fragment_path:
                pcd, normals = read_mesh_vertices(osp.join(in_dir, path))
                data = Data(pos=torch.from_numpy(pcd),
                            norm=torch.from_numpy(normals))
                if(self.pre_transform is not None):
                    data = self.pre_transform(data)
                torch.save(data, osp.join(out_dir, path.split(".")[0]+".pt"))

    def _list_pair_fragments(self, mod):

        out_dir = osp.join(self.processed_dir,
                           mod, 'matches')
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)
        ind = 0
        for scene_path in os.listdir(osp.join(self.raw_dir, mod)):

            log.info("{}".format(scene_path))
            fragment_dir = osp.join(self.processed_dir,
                                    mod, 'fragment',
                                    scene_path)
            list_fragment_path = sorted([osp.join(fragment_dir, f)
                                         for f in os.listdir(fragment_dir)])
            log.info("list all the matches")

            for path1 in list_fragment_path:
                if(not self.self_supervised):
                    for path2 in list_fragment_path:
                        if path1 <= path2:
                            out_path = osp.join(out_dir,
                                                'matches{:06d}.npy'.format(ind))
                            match = dict()
                            match['path_source'] = path1
                            match['path_target'] = path2
                            np.save(out_path, match)
                            ind += 1
                else:
                    out_path = osp.join(out_dir,
                                        'matches{:06d}.npy'.format(ind))
                    match = dict()
                    match['path_source'] = path1
                    match['path_target'] = path1
                    np.save(out_path, match)
                    ind += 1
        self.size = ind

    def process(self):
        log.info("pre_transform those fragments")
        self._pre_transform_fragment(self.mode)
        log.info("list_pairs")
        self._list_pair_fragments(self.mode)

    def __getitem__(self, idx):
        r"""get pairs of fragments"""
        path_match = osp.join(self.processed_dir, self.mode, 'matches')
        match = np.load(osp.join(path_match, "matches{:06d}.npy".format(idx)), allow_pickle=True).item()
        data_source = torch.load(match["path_source"]).to(torch.float)
        data_target = torch.load(match["path_target"]).to(torch.float)
        new_pair = torch.zeros(0, 2)
        while len(new_pair) < 100:
            i = np.random.randint(0, len(data_source.pos))
            radius = np.random.random() * (self.max_block_size - self.min_block_size) + self.min_block_size
            point = data_source.pos[i].view(1, 3)
            ind, dist = ball_query(point,
                                   data_source.pos,
                                   radius=radius,
                                   max_num=-1, mode=1, sorted=True)
            ind_s = ind[dist[:, 0] >= 0][:, 1]
            ind_, dist_ = ball_query(data_source.pos[ind_s],
                                     data_target.pos,
                                     radius=self.max_dist_overlap,
                                     max_num=1, mode=1, sorted=True)
            ind_ind_s = ind_[dist_[:, 0] >= 0][:, 0]
            ind_t = ind_[dist_[:, 0] >= 0][:, 1]

            new_pair = torch.stack((ind_s[ind_ind_s], ind_t)).T

        if self.transform is not None:
            data_source = self.transform(data_source)
            data_target = self.transform(data_target)

        if(hasattr(data_source, "multiscale")):
            batch = MultiScalePair.make_pair(data_source, data_target)
        else:
            batch = Pair.make_pair(data_source, data_target)
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
        return self.size


class LiffreDataset(BaseSiameseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.tau_1 = dataset_opt.tau_1
        pre_transform = self.pre_transform
        train_transform = self.train_transform
        test_transform = self.test_transform
        pre_filter = self.pre_filter

        self.train_dataset = Liffre(
            root=self._data_path,
            train=True,
            self_supervised=dataset_opt.self_supervised,
            pre_transform=pre_transform,
            transform=train_transform,
            pre_filter=pre_filter,
            min_block_size=dataset_opt.min_block_size,
            max_block_size=dataset_opt.max_block_size,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            num_pos_pairs=dataset_opt.num_pos_pairs
            )

        self.val_dataset = Liffre(
            root=self._data_path,
            train=False,
            self_supervised=False,
            pre_transform=pre_transform,
            transform=test_transform,
            pre_filter=pre_filter,
            min_block_size=dataset_opt.min_block_size,
            max_block_size=dataset_opt.max_block_size,
            max_dist_overlap=dataset_opt.max_dist_overlap,
            num_pos_pairs=dataset_opt.num_pos_pairs
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """
        Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """

        return FragmentRegistrationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, tau_1=self.tau_1)
