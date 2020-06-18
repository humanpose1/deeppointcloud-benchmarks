import hydra
import numpy as np
import open3d as o3d
import os
import os.path as osp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import torch
from omegaconf import OmegaConf
from copy import deepcopy

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..", "..")
sys.path.insert(0, ROOT)
from scripts.test_registration_scripts.descriptor_matcher import compute_matches
from torch_points3d.metrics.registration_metrics import fast_global_registration


def prepare_pointcloud(path, norm_radius=0.1):

    data = np.load(path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data["pcd"])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=norm_radius, max_nn=30))

    return pcd, data["keypoints"]


def draw_keypoints(point_cloud, keypoints, color=[1, 0, 0]):
    """
    draw keypoints detectors.
    return a point cloud to vizualize the detected points
    """

    color_kps = np.asarray([color for p in keypoints])
    keypoints[:, 2] += 0.1
    points = np.vstack([np.asarray(point_cloud.points), keypoints])
    colors = np.vstack([np.asarray(point_cloud.colors), color_kps])

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points)
    pts.colors = o3d.utility.Vector3dVector(colors)
    return pts


def create_sphere(kp, color, radius):
    T = np.eye(4)
    T[:3, 3] = kp
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.transform(T)
    sphere.paint_uniform_color(color)
    return sphere


def create_line(kp1, kp2, colors=[1, 0, 0]):
    line_set = o3d.geometry.LineSet()
    points = [list(kp1), list(kp2)]

    lines = [[0, 1]]
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    line_set.colors = o3d.utility.Vector3dVector([colors])
    return line_set


def match_visualizer(pcd1, keypoints1, pcd2, keypoints2, t=20, radius=0.1, inliers=None):
    """
    display the match using Open3D draw_geometries.
    input :
    -pcd: open3d point cloud
    -keypoints : the keypoints (open3d point cloud)
    -t: float a translation at the x axis
    condition : keypoints1 and keypoints2 have the same size
    """

    colors = []
    keypoints1_copy = o3d.geometry.PointCloud(keypoints1)
    keypoints2_copy = o3d.geometry.PointCloud(keypoints2)
    T_trans = np.eye(4)
    T_trans[0, 3] = t
    list_displayed = [pcd1, pcd2]
    pcd2.transform(T_trans)

    keypoints2_copy.transform(T_trans)
    kp1 = np.asarray(keypoints1_copy.points)
    kp2 = np.asarray(keypoints2_copy.points)
    # print(len(kp1))
    # print(len(kp2))
    if len(kp1) != len(kp2):
        raise Exception("number of points is different")
    for i in range(len(kp1)):
        if inliers is None:
            colors.append([0, 0, 0])
        else:
            col = inliers[i] * np.asarray([0, 1, 0]) + (1 - inliers[i]) * np.asarray([1, 0, 0])
            colors.append(list(col))
        p1 = kp1[i]
        p2 = kp2[i]
        sphere1 = create_sphere(p1, colors[-1], radius)
        sphere2 = create_sphere(p2, colors[-1], radius)
        line = create_line(p1, p2, colors[-1])
        list_displayed.append(line)
        list_displayed.append(sphere1)
        list_displayed.append(sphere2)

    o3d.visualization.draw_geometries(list_displayed)


@hydra.main(config_path="conf/config_visu.yaml")
def main(cfg):

    OmegaConf.set_struct(cfg, False)
    list_path = [osp.join(cfg.path_pcd, "cloud_bin_{}_desc.npz".format(i)) for i in cfg.list_num_frag]

    data0 = np.load(list_path[0])
    data1 = np.load(list_path[1])
    pos0, feat0, k0 = data0["pcd"], data0["feat"], data0["keypoints"]
    pos1, feat1, k1 = data1["pcd"], data1["feat"], data1["keypoints"]

    kp_0, kp_1 = compute_matches(feat0[k0], feat1[k1], pos0[k0], pos1[k1], sym=True)
    pcd0 = o3d.geometry.PointCloud()

    kp0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(pos0)
    kp0.points = o3d.utility.Vector3dVector(kp_0[: cfg.num_pt])
    pcd0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    pcd0.paint_uniform_color([0.7, 0.1, 0.05])

    pcd1 = o3d.geometry.PointCloud()
    kp1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pos1)
    kp1.points = o3d.utility.Vector3dVector(kp_1[: cfg.num_pt])
    pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    pcd1.paint_uniform_color([0.1, 0.05, 0.9])

    match_visualizer(deepcopy(pcd0), kp0, deepcopy(pcd1), kp1, t=cfg.trans_x, radius=cfg.radius_sphere)

    T_est = fast_global_registration(torch.from_numpy(kp_0), torch.from_numpy(kp_1)).cpu().numpy()
    o3d.visualization.draw_geometries([pcd0, pcd1])
    pcd0.transform(T_est)
    print("estimate rotation...")
    print(T_est)
    o3d.visualization.draw_geometries([pcd0, pcd1])


if __name__ == "__main__":
    main()
