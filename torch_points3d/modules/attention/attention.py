import torch
import torch.nn as nn

from torch_scatter import scatter_max, scatter_mean, scatter_softmax


class CentralAttentiveModule(torch.nn.Module):
    def __init__(self, input_size, output_size, activation=nn.ReLU(), aggr="max"):
        self.lin_v = nn.Linear(input_size, output_size)
        self.lin_k = nn.Linear(input_size, output_size)
        self.lin_q = nn.Linear(input_size, output_size)
        self.aggr = aggr

    def _pool(self, x, cluster):
        if self.aggr == "mean":
            return scatter_mean(x, cluster, dim=0)
        elif self.aggr == "max":
            return scatter_max(x, cluster, dim=0)
        else:
            raise NotImplementedError("the mode of pooling is not present")

    def forward(self, pos, x, cluster):

        v = self.lin_v(x)  # N x F
        k = self.lin_k(x)  # N x F
        q = self._pool(self.lin_q(x))  # B x F
        M = (q[cluster] * k).sum(-1)  # N x 1
        M = scatter_softmax(M, cluster, dim=0)  # N x 1
        res = self.activation(self.bn(M * v))
        return res
