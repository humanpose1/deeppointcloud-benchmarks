import open3d
import numpy as np
import torch
import hydra
import logging
from omegaconf import OmegaConf
import os
import os.path as osp
import sys
from plyfile import PlyData
from copy import deepcopy
import pandas as pd
from torch_geometric.data import Data, Batch

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..", "..")
sys.path.insert(0, ROOT)

from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from torch_points3d.datasets.registration.pair import Pair
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

from torch_points3d.datasets.registration.utils import tracked_matches
from torch_points3d.datasets.registration.utils import compute_overlap_and_matches


from torch_points3d.metrics.registration_metrics import estimate_transfo
from torch_points3d.metrics.registration_metrics import get_matches
from torch_points3d.metrics.registration_metrics import teaser_pp_registration
from torch_points3d.metrics.registration_metrics import ransac_registration
from torch_points3d.metrics.registration_metrics import compute_metrics
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

from scripts.test_registration_scripts.compute_and_visu_matches import match_visualizer

log = logging.getLogger(__name__)


def torch2o3d(xyz, color=[1, 0, 0], normal=0.5):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    pcd.paint_uniform_color(color)
    # pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=normal, max_nn=30))
    return pcd


def read_ply(path):
    with open(path, "rb") as f:
        data = PlyData.read(f)
        pos = [np.array(data["vertex"][axis]) for axis in ["x", "y", "z"]]
        pos = np.stack(pos, axis=-1)
    return Batch(pos=torch.from_numpy(pos), batch=torch.zeros(len(pos), dtype=torch.long))


def read_bin(path):
    xyzr = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
    return Batch(pos=torch.from_numpy(xyzr[:, :3]), batch=torch.zeros(len(xyzr), dtype=torch.long))


def read_pt(path):
    data = torch.load(path)
    return Batch(pos=data.pos, batch=torch.zeros(len(data.pos), dtype=torch.long))


def run(model: BaseModel, dataset, device, cfg):

    assert hasattr(cfg, "path_source")
    assert hasattr(cfg, "path_target")
    parent, name = osp.split(cfg.path_source)
    if name.split(".")[1] == "ply":
        data_source = read_ply(cfg.path_source)
        data_target = read_ply(cfg.path_target)
    elif name.split(".")[1] == "bin":
        data_source = read_bin(cfg.path_source)
        data_target = read_bin(cfg.path_target)
    elif name.split(".")[1] == "pt":
        data_source = read_pt(cfg.path_source)
        data_target = read_pt(cfg.path_target)

    new_match = compute_overlap_and_matches(
        Data(pos=data_source.pos), Data(pos=data_target.pos), cfg.data.max_dist_overlap
    )
    new_pair = torch.from_numpy(new_match["pair"].copy())
    data_source = dataset.inference_transform(data_source)
    data_target = dataset.inference_transform(data_target)

    pair = tracked_matches(data_source, data_target, new_pair)
    data = Pair.make_pair(data_source, data_target)
    # data.pair_ind = torch.zeros(1, 2).to(torch.long)
    # data.size_pair_ind = torch.tensor([0])
    data.pair_ind = pair
    data.size_pair_ind = torch.tensor([50])

    with torch.no_grad():
        model.set_input(data, device)
        model.forward()
        # input
        input, input_target = model.get_input()
        xyz, xyz_target = input.pos, input_target.pos
        ind, ind_target = input.ind, input_target.ind
        matches_gt = torch.stack([ind, ind_target]).transpose(0, 1)
        T_gt = estimate_transfo(xyz[matches_gt[:, 0]], xyz_target[matches_gt[:, 1]])
        # compute feature
        feat, feat_target = model.get_output()

        # select random points
        rand = torch.randperm(len(feat))[: cfg.data.num_points]
        rand_target = torch.randperm(len(feat_target))[: cfg.data.num_points]
        # compute transfo
        matches_pred = get_matches(feat[rand], feat_target[rand_target], sym=True)

        T_teaser = teaser_pp_registration(
            xyz[rand][matches_pred[:, 0]],
            xyz_target[rand_target][matches_pred[:, 1]],
            noise_bound=cfg.data.noise_bound_teaser,
        )

        print(torch.norm(T_gt[:3, 3] - T_teaser[:3, 3]))
        # print(T_gt[:3, 3] - T_ransac[:3, 3])
        # print(T_gt, T_gt.shape)
        # print(T_teaser, T_teaser.shape)
        pcd = torch2o3d(xyz, color=[0.7, 0.05, 0.1])
        kp = torch2o3d(xyz[rand][matches_pred[:, 0]])
        pcd_t = torch2o3d(xyz_target, color=[0.12, 0.7, 0.9])
        kp_t = torch2o3d(xyz_target[rand_target][matches_pred[:, 1]])
        open3d.visualization.draw_geometries([pcd, pcd_t])
        open3d.visualization.draw_geometries([deepcopy(pcd).transform(T_teaser.detach().cpu().numpy()), pcd_t])

        # open3d.visualization.draw_geometries([pcd, pcd_t])
        # open3d.visualization.draw_geometries([deepcopy(pcd).transform(T_gt.detach().cpu().numpy()), pcd_t])

        inliers = (
            torch.norm(
                xyz[rand][matches_pred[:, 0]] @ T_gt[:3, :3].T
                + T_gt[:3, 3]
                - xyz_target[rand_target][matches_pred[:, 1]],
                dim=1,
            )
            < 0.5
        )
        print(torch.mean(inliers.to(torch.float)))
        match_visualizer(pcd, kp, pcd_t, kp_t, t=200, radius=0.5, inliers=inliers.detach().cpu().numpy())


@hydra.main(config_path="../../conf/config.yaml", strict=False)
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    # Get device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.training.cuda) else "cpu")
    log.info("DEVICE : {}".format(device))

    # Enable CUDNN BACKEND
    torch.backends.cudnn.enabled = cfg.training.enable_cudnn

    # Checkpoint
    checkpoint = ModelCheckpoint(cfg.training.checkpoint_dir, cfg.model_name, cfg.training.weight_name, strict=True)

    # Setup the dataset config
    # Generic config

    dataset = instantiate_dataset(cfg.data)
    model = checkpoint.create_model(dataset, weight_name=cfg.training.weight_name)
    log.info(model)
    log.info("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    log.info(dataset)

    model.eval()
    if cfg.enable_dropout:
        model.enable_dropout_in_eval()
    model = model.to(device)

    run(model, dataset, device, cfg)


if __name__ == "__main__":
    main()
