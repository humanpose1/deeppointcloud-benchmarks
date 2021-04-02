import torch

from torch_points3d.models.registration.base import FragmentBaseModel
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq
from torch_points3d.models.registration.ms_spconv3d import UnetMSparseConv3d
from torch_points3d.models.registration.ms_spconv3d import BaseMS_SparseConv3d


class SparseConv3D(BaseMS_SparseConv3d):
    def __init__(self, option, model_type, dataset, modules):
        BaseMS_SparseConv3d.__init__(self, option, model_type, dataset, modules)
        option_unet = option.option_unet
        self.grid_size = option_unet.grid_size
        self.unet = UnetMSparseConv3d(
            option_unet.backbone,
            input_nc=option_unet.input_nc,
            pointnet_nn=option_unet.pointnet_nn,
            post_mlp_nn=option_unet.post_mlp_nn,
            pre_mlp_nn=option_unet.pre_mlp_nn,
            add_pos=option_unet.add_pos,
            add_pre_x=option_unet.add_pre_x,
            aggr=option_unet.aggr,
            backend=option.backend,
        )

    def apply_nn(self, input):
        self.unet.set_grid_size(self.grid_size)
        out_feat = self.unet(input).x
        if self.normalize_feature:
            return out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)
        else:
            return out_feat
