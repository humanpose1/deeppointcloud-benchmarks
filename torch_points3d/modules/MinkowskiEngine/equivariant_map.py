import torch
import torch.nn as nn
import MinkowskiEngine as ME
from .common import ConvType, NormType

from .modules import ResnetBlockDown

from torch_points3d.core.common_modules.base_modules import BaseModule

from torch_points3d.utils.config import is_list
from torch_points3d.core.common_modules.base_modules import FastBatchNorm1d, MLP
from torch_points3d.core.data_transform import GridSampling3D
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid
from torch_scatter import scatter_mean


class CoarseVoxelBlock(nn.Module):
    def __init__(self, channels, voxel_size=0.1, kernel_size=3, dilation=1, bn_momentum=0.05, dimension=3):
        super(CoarseVoxelBlock, self).__init__()
        self._grid_size = voxel_size
        modules = []
        for i in range(1, len(channels)):
            block = ResnetBlockDown(
                down_conv_nn=[channels[i - 1], channels[i], channels[i]],
                kernel_size=kernel_size,
                stride=1,
                dimension=dimension,
                dilation=dilation,
            )
            modules.append(block)
        self.mod = nn.Sequential(*modules)

    def _prepare_data(self, data):
        coords = torch.round((data.pos) / self._grid_size)
        cluster = voxel_grid(coords, data.batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        coords = coords[unique_pos_indices]
        coords = torch.cat([data.batch[unique_pos_indices].unsqueeze(-1).float(), coords.float()], -1)
        x = scatter_mean(data.x, cluster, dim=0)
        sparse_tensor = ME.SparseTensor(x, coords=coords)
        return sparse_tensor, cluster

    def forward(self, data):
        d = data.clone()
        sparse_tensor, cluster = self._prepare_data(d)
        sparse_tensor = self.mod(sparse_tensor)
        d.x = sparse_tensor.F[cluster]
        return d


class FineBlock(nn.Module):
    def __init__(self, channels, bn_momentum=0.02, activation=nn.LeakyReLU(0.1)):
        super(FineBlock, self).__init__()
        self.mlp = MLP(channels, bn_momentum=bn_momentum, activation=activation)

    def forward(self, data, precomputed=None, **kwargs):
        d = data.clone()
        d.x = self.mlp(d.x)
        return d


class EMHSBlock(BaseModule):
    """
    Perform a EMHSBlock which is two EMHS layers and
    """

    def __init__(
        self,
        down_conv_nn=None,
        grid_size=None,
        prev_grid_size=None,
        bn_momentum=0.02,
        activation=torch.nn.LeakyReLU(negative_slope=0.1),
        voxel_size=0.1,
        kernel_size=3,
        dilation=1,
        num_layer_mlp=2,
        num_layer_conv=2,
        **kwargs
    ):
        super(EMHSBlock, self).__init__()
        assert len(down_conv_nn) == 2

        num_inputs, num_outputs = down_conv_nn
        self.is_strided = prev_grid_size != grid_size
        channels_conv = [down_conv_nn[0]] + [down_conv_nn[1] for _ in range(num_layer_conv)]
        channels_mlp = [down_conv_nn[0]] + [down_conv_nn[1] for _ in range(num_layer_mlp)]
        if num_layer_conv >= 1:
            self.voxel = CoarseVoxelBlock(
                channels=channels_conv,
                bn_momentum=bn_momentum,
                voxel_size=voxel_size,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        else:
            self.voxel = None

        self.mlp = FineBlock(channels=channels_mlp, bn_momentum=bn_momentum, activation=activation)
        if self.is_strided:
            self.pool = GridSampling3D(grid_size)

    def forward(self, data, **kwargs):

        out_v = data.clone()
        out_mlp = data.clone()
        out_mlp = self.mlp(out_mlp)
        if self.voxel is not None:
            out_v = self.voxel(out_v)
        else:
            out_v = out_mlp

        assert out_v.x.shape == out_mlp.x.shape
        out_mlp.x = out_mlp.x + out_v.x
        if self.is_strided:
            out_mlp = self.pool(out_mlp)
        return out_mlp
