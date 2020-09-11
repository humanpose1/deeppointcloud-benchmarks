import torch
import torch.nn as nn

from torch_scatter import scatter_max, scatter_mean, scatter_softmax
from torch_points3d.core.common_modules.base_modules import FastBatchNorm1d


class CentralAttentiveModule(torch.nn.Module):
    def __init__(self, input_size, output_size, aggr="max", bn_momentum=0.1, activation=nn.LeakyReLU(0.2)):
        super(CentralAttentiveModule, self).__init__()
        self.lin_v = nn.Linear(input_size, output_size)
        self.lin_k = nn.Linear(input_size, output_size)
        self.lin_q = nn.Linear(input_size, output_size)
        self.aggr = aggr
        self.bn = FastBatchNorm1d(output_size, momentum=bn_momentum)
        self.activation = activation

    def _pool(self, x, cluster):
        if self.aggr == "mean":
            return scatter_mean(x, cluster, dim=0)
        elif self.aggr == "max":
            return scatter_max(x, cluster, dim=0)[0]
        else:
            raise NotImplementedError("the mode of pooling is not present")

    def forward(self, x, cluster):

        v = self.lin_v(x)  # N x F
        k = self.lin_k(x)  # N x F
        q = self._pool(self.lin_q(x), cluster)  # B x F
        M = (q[cluster] * k).sum(-1)  # N x 1
        M = scatter_softmax(M, cluster, dim=0)  # N x 1
        return self.activation(self.bn(M.unsqueeze(-1) * v))


class CombineModel(nn.Sequential):
    """ Class to combine multiple models. Sequential allowing multiple inputs."""

    def forward(self, x, cluster, *args, **kwargs):
        for module in self._modules.values():
            x = module(x, cluster)
        return x


def AttMLP(channels, activation=nn.LeakyReLU(0.2), bn_momentum=0.1, bias=True, aggr="max"):
    layers = [
        CentralAttentiveModule(channels[i - 1], channels[i], aggr=aggr, activation=activation, bn_momentum=bn_momentum)
        for i in range(1, len(channels))
    ]
    return CombineModel(*layers)
