from torch_points3d.core.spatial_ops import KNNNeighbourFinder
import torch.nn as nn


class PartialChamferLoss(nn.Module):

    """
    Chamfer loss for partial dense format
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.neigh_finder = KNNNeighbourFinder(k=1)

    def forward(self, xyz, xyz_gt, batch_xyz, batch_xyz_gt):
        ind = self.neigh_finder(xyz, xyz_gt, batch_xyz, batch_xyz_gt)
        ind_ = self.neigh_finder(xyz_gt, xyz, batch_xyz_gt, batch_xyz)

        loss = (xyz[ind[1]] - xyz_gt[ind[0]]).norm(dim=1).mean()
        loss_ = (xyz[ind_[0]] - xyz_gt[ind_[1]]).norm(dim=1).mean()

        return 0.5 * (loss + loss_)
