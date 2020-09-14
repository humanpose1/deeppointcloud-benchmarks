import torch

from torch_points3d.modules.MinkowskiEngine.equivariant_map import SuperEMHS
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.models.registration.base import FragmentBaseModel
from torch_points3d.models.registration.minkowski import BaseMinkowski
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq
from torch_geometric.data import Data


class EMHS_Model(FragmentBaseModel):
    def __init__(self, option, model_type, dataset, modules):
        FragmentBaseModel.__init__(self, option)

        self.model = SuperEMHS(
            D=3,
            dilation=1,
            dim_feat=option.dim_feat,
            num_layer=option.num_layer,
            grid_size=option.grid_size,
            L=option.L,
        )
        self.normalize_feature = option.normalize_feature
        self.mode = option.loss_mode
        self.loss_names = ["loss_reg", "loss"]
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )

    def set_input(self, data, device):
        self.input, self.input_target = data.to_data()
        if hasattr(data, "pos_target"):
            self.match = data.pair_ind.to(torch.long).to(device)
            self.size_match = data.size_pair_ind.to(torch.long).to(device)
        else:
            self.match = None
        self.input = self.input.to(device)
        self.input_target = self.input_target.to(device)

    def get_batch(self):
        if self.match is not None:
            batch = self.input.batch
            batch_target = self.input_target.batch
            return batch, batch_target
        else:
            batch = self.input.batch
            return batch, None

    def get_input(self):

        if self.match is not None:
            input = Data(pos=self.input.pos, ind=self.match[:, 0], size=self.size_match)
            input_target = Data(pos=self.input_target.pos, ind=self.match[:, 1], size=self.size_match)
            return input, input_target
        else:
            input = Data(pos=self.xyz)
            return input

    def apply_nn(self, input):
        output = self.model(input).x
        if self.normalize_feature:
            return output / (torch.norm(output, p=2, dim=1, keepdim=True) + 1e-3)
        else:
            return output
