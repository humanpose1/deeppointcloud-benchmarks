import logging
import numpy as np
import os
import os.path as osp

import torch

from torch_geometric.data import Dataset, download_url, extract_zip
from src.datasets.registration.utils import rgbd2fragment_rough
from src.datasets.registration.utils import rgbd2fragment_fine
from src.datasets.registration.utils import compute_overlap_and_matches
from src.datasets.registration.utils import to_list
from src.datasets.registration.utils import files_exist
from src.datasets.registration.utils import makedirs
from src.datasets.registration.utils import get_urls



log = logging.getLogger(__name__)


class General3DMatch(Dataset):

    base = osp.abspath(osp.join(osp.realpath(__file__),
                                '..'))
    list_urls_train = get_urls(osp.join(base, 'urls', 'url_train.txt'))
    list_urls_train_small = get_urls(osp.join(base, 'urls', 'url_train_small.txt'))
    list_urls_train_tiny = get_urls(osp.join(base, 'urls', 'url_train_tiny.txt'))
    list_urls_test = get_urls(osp.join(base, 'urls', 'url_val.txt'))
    list_urls_test = get_urls(osp.join(base, 'urls', 'url_test.txt'))
    dict_urls = dict(train=list_urls_train,
                     train_small=list_urls_train_small,
                     train_tiny=list_urls_train_tiny,
                     test=list_urls_test)

    def __init__(self, root,
                 num_frame_per_fragment=50,
                 mode='train_small',
                 min_overlap_ratio=0.3,
                 max_overlap_ratio=1.0,
                 max_dist_overlap=0.01,
                 tsdf_voxel_size=0.01,
                 depth_thresh=6,
                 is_fine=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 verbose=False,
                 debug=False):
        r"""
        the Princeton 3DMatch dataset from the
        `"3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions"
        <https://arxiv.org/pdf/1603.08182.pdf>`_
        paper, containing rgbd frames of the following dataset:
        `" SUN3D: A Database of Big Spaces Reconstructed using SfM and Object Labels
        "<http://sun3d.cs.princeton.edu/>`
        `"Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images
        "<https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images/>`
        `"Unsupervised Feature Learning for 3D Scene Labeling
        "<http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/>`
        `"BundleFusion: Real-time Globally Consistent 3D Reconstruction using Online Surface Re-integration
        "<http://graphics.stanford.edu/projects/bundlefusion/>`
        `"Learning to Navigate the Energy Landscape
        "<http://graphics.stanford.edu/projects/reloc/>`

        Args:
            root (string): Root directory where the dataset should be saved

            num_frame_per_fragment (int, optional): indicate the number of frames
                we use to build fragments. If it is equal to 0, then we don't
                build fragments and use the raw frames.

            mode (string, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)

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
        """

        self.verbose = verbose
        self.debug = debug
        self.is_fine = is_fine
        self.num_frame_per_fragment = num_frame_per_fragment
        self.tsdf_voxel_size = tsdf_voxel_size
        self.depth_thresh = depth_thresh
        self.mode = mode
        # constant to compute overlap
        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_dist_overlap = max_dist_overlap
        if mode not in self.dict_urls.keys():
            raise RuntimeError('this mode {} does '
                               'not exist'
                               '(train_small|train_tiny|train|val|test)'.format(mode))
        super(General3DMatch, self).__init__(root,
                                             transform,
                                             pre_transform,
                                             pre_filter)

        # path = self.processed_paths[0] if train else self.processed_paths[1]
        # self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [self.mode]

    @property
    def processed_file_names(self):
        return [osp.join(self.mode, 'fragment'),
                osp.join(self.mode, 'matches')]

    def download(self):

        folder = osp.join(self.raw_dir, self.mode)
        log.info("Download elements in the file {}...".format(folder))
        for url in self.dict_urls[self.mode]:
            path = download_url(url, folder, self.verbose)
            extract_zip(path, folder, self.verbose)
            os.unlink(path)

    def _create_fragment(self, mod):
        r"""
        create fragments from the rgbd frames ie a partial reconstruction of
        the scene with some frames(usually 50).
        We will only use the first sequence for each scene
        """

        print("Create fragment from RGBD frames...")
        if files_exist([osp.join(self.processed_dir, mod, 'fragment')]):  # pragma: no cover
            print("the fragments on mode {} already exists".format(mod))
            return
        for scene_path in os.listdir(osp.join(self.raw_dir, mod)):
            # TODO list the right sequences.
            list_seq = [f for f in os.listdir(osp.join(self.raw_dir, mod,
                                                       scene_path)) if 'seq' in f]
            for seq in list_seq:
                frames_dir = osp.join(self.raw_dir, self.mode,
                                      scene_path, seq)
                out_dir = osp.join(self.processed_dir,
                                   mod, 'fragment',
                                   scene_path, seq)
                makedirs(out_dir)
                path_intrinsic = osp.join(self.raw_dir,
                                          self.mode, scene_path,
                                          'camera-intrinsics.txt')
                list_path_frames = sorted([osp.join(frames_dir, f)
                                           for f in os.listdir(frames_dir)
                                           if 'png' in f and 'depth' in f])
                list_path_color = sorted([osp.join(frames_dir, f)
                                          for f in os.listdir(frames_dir)
                                          if 'png' in f and 'color' in f])
                list_path_trans = sorted([osp.join(frames_dir, f)
                                          for f in os.listdir(frames_dir)
                                          if 'pose' in f and 'txt' in f])
                # compute each fragment and save it
                if(not self.is_fine):
                    rgbd2fragment_rough(list_path_frames, path_intrinsic,
                                        list_path_trans, out_dir,
                                        self.num_frame_per_fragment,
                                        pre_transform=self.pre_transform,
                                        list_path_color=list_path_color)
                else:
                    assert list_path_color is not None
                    rgbd2fragment_fine(list_path_frames, path_intrinsic,
                                       list_path_trans, list_path_color,
                                       out_dir, self.num_frame_per_fragment,
                                       voxel_size=self.tsdf_voxel_size,
                                       pre_transform=self.pre_transform,
                                       depth_thresh=self.depth_thresh)




    def _compute_matches_between_fragments(self, mod):

        out_dir = osp.join(self.processed_dir,
                           mod, 'matches')
        if files_exist([out_dir]):  # pragma: no cover
            return
        makedirs(out_dir)

        for scene_path in os.listdir(osp.join(self.raw_dir, mod)):

            list_seq = sorted([f for f in os.listdir(osp.join(self.raw_dir, mod,
                                                              scene_path)) if 'seq' in f])
            for seq in list_seq:
                print(seq)
                fragment_dir = osp.join(self.processed_dir,
                                        mod, 'fragment',
                                        scene_path, seq)
                list_fragment_path = sorted([osp.join(fragment_dir, f)
                                             for f in os.listdir(fragment_dir)
                                             if 'fragment' in f])
                print("compute_overlap_and_matches")
                ind = 0
                for path1 in list_fragment_path:
                    for path2 in list_fragment_path:
                        if path1 < path2:
                            out_path = osp.join(out_dir,
                                                'matches{:06d}.npy'.format(ind))

                            match = compute_overlap_and_matches(
                                path1, path2, self.max_dist_overlap)
                            if(self.verbose):
                                print(match['path_source'],
                                      match['path_target'],
                                      'overlap={}'.format(match['overlap']))
                            if(np.max(match['overlap']) > self.min_overlap_ratio and
                               np.max(match['overlap']) < self.max_overlap_ratio):
                                np.save(out_path, match)
                                ind += 1

    def process(self):
        print("create fragments")
        self._create_fragment(self.mode)
        print("compute matches")
        self._compute_matches_between_fragments(self.mode)

    def get(self, idx):
        raise NotImplementedError("implement class to get patch or fragment or more")
