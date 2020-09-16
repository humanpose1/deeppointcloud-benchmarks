import torch

from torch_points3d.modules.MinkowskiEngine.equivariant_map import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.modules.PointNet.dense_modules import DenseMiniPointNet
from torch_points3d.models.base_architectures import UnwrappedUnetBasedModel
from torch_points3d.models.registration.base import FragmentBaseModel
from torch_points3d.models.registration.minkowski import BaseMinkowski
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq
from torch_geometric.data import Data


class EMHS_Model(FragmentBaseModel, UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)

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

    def _apply_nn(self, input):
        stack_down = []
        data = input
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data)
            stack_down.append(data)

        data = self.down_modules[-1](data)
        innermost = False

        if not isinstance(self.inner_modules[0], torch.nn.Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True

        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()))
        output = data.x
        if self.normalize_feature:
            return output / (torch.norm(output, p=2, dim=1, keepdim=True) + 1e-20)
        else:
            return output

    def apply_nn(self, input):
        return self._apply_nn(input)


class PPFEMHS(EMHS_Model):

    """
    perform ppfnet before using EMHS
    """

    def __init__(self, option, model_type, dataset, modules):
        super(PPFEMHS, self).__init__(option, model_type, dataset, modules)
        self.ppfnet = DenseMiniPointNet(**option.ppf_param)

    def apply_nn(self, input):
        assert hasattr(input, "ppf")
        input_feat = self.ppfnet(input.ppf)  # N x K x 4 -> N x D

        input.x = torch.cat([input.x, input_feat], 1)
        return self._apply_nn(input)
