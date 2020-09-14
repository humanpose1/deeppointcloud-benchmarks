import torch
import torch.nn as nn
import MinkowskiEngine as ME
from .common import ConvType, NormType

from torch_points3d.utils.config import is_list
from torch_points3d.core.common_modules.base_modules import FastBatchNorm1d, MLP

from torch_geometric.data import Batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid
from torch_scatter import scatter_mean


class OuterEquivariantMap(nn.Module):
    def __init__(self, input_nc=3, output_nc=64, dilation=1, size_conv=3, D=3):

        super(OuterEquivariantMap, self).__init__()

        self.conv3d = ME.MinkowskiConvolution(
            input_nc, output_nc, kernel_size=size_conv, stride=1, dilation=dilation, dimension=D
        )

        self.conv = torch.nn.Linear(input_nc + output_nc, output_nc)

    def _prepare_and_pool(self, data):
        assert hasattr(data, "coords")
        assert hasattr(data, "cluster")
        assert hasattr(data, "unique_pos_indices")
        x = scatter_mean(data.x, data.cluster, dim=0)
        batch = data.batch[data.unique_pos_indices]
        coords = torch.cat([batch.unsqueeze(-1).float(), data.coords.float()], -1)
        sparse_tensor = ME.SparseTensor(x, coords=coords)
        return sparse_tensor

    def forward(self, data):

        st = self._prepare_and_pool(data)
        out = self.conv3d(st)
        u = self.conv(torch.cat([data.x, out.F[data.cluster]], -1))
        data.x = u
        return data


class InnerEquivariantMap(nn.Module):
    """
    TODO: add attention mechanism
    """

    def __init__(self, input_nc, output_nc, L=1, D=3):
        super(InnerEquivariantMap, self).__init__()
        channels = [input_nc] + [output_nc for _ in range(L)]
        # self.conv1 = nn.Linear(input_nc, output_nc)
        self.conv1 = MLP(channels)

    def forward(self, data):

        data.x = self.conv1(data.x)
        return data


class EMLayer(nn.Module):
    def __init__(
        self, input_nc, output_nc, L=2, size_conv=3, dilation=1, bn_momentum=0.02, D=3,
    ):
        super(EMLayer, self).__init__()
        self.oem = OuterEquivariantMap(input_nc, output_nc, dilation, size_conv, D)
        self.iem = InnerEquivariantMap(input_nc, output_nc, L, D)
        self.norm = FastBatchNorm1d(output_nc, momentum=bn_momentum)
        self.activation = nn.ReLU()

    def forward(self, data):
        inner = Batch(x=data.x, cluster=data.cluster)
        data.x = self.activation(self.norm(self.oem(data).x + self.iem(inner).x))
        return data


class ResEMBlock(nn.Module):
    def __init__(
        self, input_nc, output_nc, L=2, dilation=1, size_conv=3, bn_momentum=0.02, D=3,
    ):
        super(ResEMBlock, self).__init__()
        self.em1 = EMLayer(
            input_nc, output_nc, L=L, dilation=dilation, size_conv=size_conv, bn_momentum=bn_momentum, D=D
        )
        self.em2 = EMLayer(
            output_nc, output_nc, L=L, dilation=dilation, size_conv=size_conv, bn_momentum=bn_momentum, D=D
        )

    def forward(self, data):
        residual = data.x
        out = self.em2(self.em1(data))
        out.x += residual
        return out


class EquivariantMapNetwork(nn.Module):
    def __init__(self, input_nc=3, dim_feat=64, output_nc=32, grid_size=0.1, L=2, num_layer=20, dilation=1, D=3):
        super(EquivariantMapNetwork, self).__init__()
        self.layer1 = EMLayer(input_nc, dim_feat, dilation=dilation, D=D, L=1)
        self.list_res = nn.ModuleList()
        for _ in range(num_layer):
            self.list_res.append(ResEMBlock(dim_feat, dim_feat, dilation=dilation, D=D, L=L))
        self._grid_size = grid_size

    def _prepare_data(self, data):
        coords = torch.round((data.pos) / self._grid_size)
        cluster = voxel_grid(coords, data.batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)
        data.cluster = cluster
        data.coords = coords[unique_pos_indices].int()
        data.unique_pos_indices = unique_pos_indices
        return data

    def forward(self, data):
        x = self._prepare_data(data)
        x = self.layer1(x)
        for i in range(len(self.list_res)):
            x = self.list_res[i](x)
        return x


class SuperEMHS(nn.Module):
    def __init__(
        self,
        dim_feat=[3, 64, 64, 64, 32],
        num_layer=[4, 4, 4, 1],
        grid_size=[0.1, 0.2, 0.4, 0.8],
        dilation=1,
        D=3,
        L=[1, 2, 2, 2],
    ):
        super(SuperEMHS, self).__init__()
        self.list_nn = nn.ModuleList()
        for i in range(1, len(dim_feat)):
            self.list_nn.append(
                EquivariantMapNetwork(
                    dim_feat[i - 1],
                    dim_feat[i],
                    dim_feat[i],
                    grid_size=grid_size[i - 1],
                    num_layer=num_layer[i - 1],
                    L=L[i - 1],
                    dilation=dilation,
                    D=D,
                )
            )

    def forward(self, data):
        x = data
        for i in range(len(self.list_nn)):
            x = self.list_nn[i](x)
        return x
