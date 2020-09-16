import torch

from torch_points3d.core.common_modules.base_modules import *
from torch_points3d.core.common_modules.spatial_transform import BaseLinearTransformSTNkD
from torch_points3d.models.base_model import BaseInternalLossModule


class DenseMLP(torch.nn.Module):
    def __init__(self, channels, activation=torch.nn.ReLU(), bn_momentum=0.02):
        super(DenseMLP, self).__init__()
        module = []
        for i in range(1, len(channels)):
            conv = torch.nn.Conv1d(channels[i - 1], channels[i], 1)
            bn = torch.nn.BatchNorm1d(channels[i])
            act = activation
            module.append(conv)
            module.append(bn)
            module.append(act)

        self.mod = torch.nn.Sequential(*module)

    def forward(self, x):
        return self.mod(x)


class DenseMiniPointNet(torch.nn.Module):
    def __init__(self, local_nn, global_nn, aggr="max", return_local_out=False):
        super().__init__()

        self.local_nn = DenseMLP(local_nn)
        self.global_nn = MLP(global_nn) if global_nn else None
        self.aggr = aggr
        self.return_local_out = return_local_out

    def g_pool(self, x):
        # size B x Fo x N
        if self.aggr == "max":
            return torch.max(x, dim=1)[0]
        else:
            return torch.mean(x, dim=1)

    def forward(self, x):
        # size B x N x Fi
        x = x.transpose(1, 2)
        y = x = self.local_nn(x)  # [B x Fi x N] -> [B x Fo x N]
        x = self.g_pool(x)  # [B x Fo x N] -> [B, Fo]
        if self.global_nn:
            x = self.global_nn(x)  # [B x Fo] -> [B x Fof]
        if self.return_local_out:
            return x, y
        return x
