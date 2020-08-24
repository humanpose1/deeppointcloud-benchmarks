import open3d as o3d
from copy import deepcopy
import torch
import numpy as np
import os
import os.path as osp
from plyfile import PlyData

from sklearn.decomposition import PCA

import sys

DIR = osp.dirname(osp.realpath(__file__))
ROOT = osp.join(DIR, "..", "..")
sys.path.insert(0, ROOT)

from torch_points3d.core.data_transform import Random3AxisRotation, SaveOriginalPosId
from torch_points3d.core.data_transform import GridSampling3D, AddSpecularity, AddFeatByKey
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch

from torch_points3d.applications.pretrained_api import PretainedRegistry
from torch_points3d.utils.registration import get_matches, fast_global_registration, teaser_pp_registration
from torch_points3d.metrics.registration_metrics import compute_transfo_error


def read_ply(path):
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
    vertex = plydata["vertex"]
    pos = np.vstack((vertex["x"], vertex["y"], vertex["z"])).T
    norm = np.vstack((vertex["nx"], vertex["ny"], vertex["nz"])).T
    color = np.vstack((vertex["red"], vertex["green"], vertex["blue"])).T
    return pos, norm, color


def compute_color_from_features(list_feat):
    feats = np.vstack(list_feat)
    pca = PCA(n_components=3)
    pca.fit(feats)
    min_col = pca.transform(feats).min(axis=0)
    max_col = pca.transform(feats).max(axis=0)
    list_color = []
    for feat in list_feat:
        color = pca.transform(feat)
        color = (color - min_col) / (max_col - min_col)
        list_color.append(color)
    return list_color


if __name__ == "__main__":

    path_file = os.path.dirname(os.path.abspath(__file__))
    path_s = osp.join(path_file, "..", "..", "notebooks", "data", "liffre", "L605D.ply")
    path_t = osp.join(path_file, "..", "..", "notebooks", "data", "liffre", "L722D.ply")

    pcd_s, norm_s, color_s = read_ply(path_s)
    pcd_t, norm_t, color_t = read_ply(path_t)

    transform = Compose(
        [
            SaveOriginalPosId(),
            AddSpecularity(gamma=40),
            Random3AxisRotation(apply_rotation=True, rot_x=2, rot_y=2, rot_z=360),
            GridSampling3D(mode="last", size=0.1, quantize_coords=True),
            AddFeatByKey(add_to_x=True, feat_name="spec"),
        ]
    )

    data_s = transform(
        Batch(
            pos=torch.from_numpy(pcd_s).float(),
            norm=torch.from_numpy(norm_s).float(),
            col=torch.from_numpy(color_s).float(),
            batch=torch.zeros(pcd_s.shape[0]).long(),
        )
    )
    data_t = transform(
        Batch(
            pos=torch.from_numpy(pcd_t).float(),
            norm=torch.from_numpy(norm_t).float(),
            col=torch.from_numpy(color_t).float(),
            batch=torch.zeros(pcd_t.shape[0]).long(),
        )
    )

    path = osp.join(path_file, "..", "..", "outputs", "2020-08-21", "18-06-48", "MinkUNet_Fragment")
    prop = {"feature_dimension": 1}
    model = PretainedRegistry.from_file(path, mock_property=prop).cuda()

    with torch.no_grad():
        model.set_input(data_s, "cuda")
        output_s = model.forward()
        model.set_input(data_t, "cuda")
        output_t = model.forward()

    print(data_s)
    rand_s = torch.randint(0, len(output_s), (5000,))
    rand_t = torch.randint(0, len(output_t), (5000,))
    matches = get_matches(output_s[rand_s], output_t[rand_t])
    T_est = fast_global_registration(data_s.pos[rand_s][matches[:, 0]], data_t.pos[rand_t][matches[:, 1]])
    T_teaser = teaser_pp_registration(
        data_s.pos[rand_s][matches[:, 0]], data_t.pos[rand_t][matches[:, 1]], noise_bound=0.1
    )
    trans_error, rot_error = compute_transfo_error(T_teaser, T_est)
    print("trans_err:{}, rot_err: {}".format(trans_error, rot_error))
    o3d_pcd_s = o3d.geometry.PointCloud()
    o3d_pcd_s.points = o3d.utility.Vector3dVector(data_s.pos.cpu().numpy())
    o3d_pcd_s.normals = o3d.utility.Vector3dVector(data_s.norm.cpu().numpy())
    # o3d_pcd_s.colors = o3d.utility.Vector3dVector(data_s.col.cpu().numpy())
    o3d_pcd_s.paint_uniform_color([0.9, 0.7, 0.1])
    print(o3d_pcd_s)

    o3d_pcd_t = o3d.geometry.PointCloud()
    o3d_pcd_t.points = o3d.utility.Vector3dVector(data_t.pos.cpu().numpy())
    o3d_pcd_t.normals = o3d.utility.Vector3dVector(data_t.norm.cpu().numpy())
    # o3d_pcd_t.colors = o3d.utility.Vector3dVector(data_t.col.cpu().numpy())
    o3d_pcd_t.paint_uniform_color([0.1, 0.7, 0.9])

    o3d.visualization.draw_geometries([o3d_pcd_s, o3d_pcd_t])
    o3d.visualization.draw_geometries([deepcopy(o3d_pcd_s).transform(T_est.cpu().numpy()), o3d_pcd_t])
    o3d.visualization.draw_geometries([deepcopy(o3d_pcd_s).transform(T_teaser.cpu().numpy()), o3d_pcd_t])

    print("Visualize features")
    list_color = compute_color_from_features([output_s.detach().cpu().numpy(), output_t.detach().cpu().numpy()])
    o3d_pcd_s.colors = o3d.utility.Vector3dVector(list_color[0])
    o3d_pcd_t.colors = o3d.utility.Vector3dVector(list_color[1])
    o3d.visualization.draw_geometries([o3d_pcd_s.transform(T_est.cpu().numpy()).translate([20, 0, 0]), o3d_pcd_t])
