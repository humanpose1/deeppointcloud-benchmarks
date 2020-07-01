import torch
from torch_geometric.data import Data

from torch_points3d.models.registration.base import FragmentBaseModel
from torch_points3d.models.base_model import BaseModel
from torch_points3d.applications import models
from torch_points3d.applications.minkowski import Minkowski
import MinkowskiEngine as ME

from torch_points3d.core.common_modules.base_modules import MLP
from torch.nn import Linear


class ProjectNet(torch.nn.Module):
    """
    project the features in a non linear space
    """

    def __init__(self, nn_size=[128, 256, 256], last_size=256, normalize=True):
        super().__init__()
        if nn_size is None:
            self.local_nn = torch.nn.Identity()
            self.last_layer = Linear(last_size, last_size, bias=False)
        else:
            self.local_nn = MLP(nn_size)
            self.last_layer = Linear(nn_size[-1], last_size, bias=False)
        self.normalize = normalize

    def forward(self, x):
        x = self.local_nn(x)
        norm = self.last_layer(x)
        if self.normalize:
            return torch.nn.functional.normalize(norm, dim=1)
        else:
            return norm


class BaseSSFM(BaseModel):
    __REQUIRED_DATA__ = [
        "pos",
        "x",
    ]

    __REQUIRED_LABELS__ = ["pair_ind"]

    def __init__(self, option, model_type, dataset, modules):
        BaseModel.__init__(self, option)
        self.normalize_feature = option.normalize_feature
        self.mode = option.loss_mode
        self.loss_names = ["metric_loss", "loss"]
        input_nc = dataset.feature_dimension
        backbone_option = option.backbone
        # backbone_cls = getattr(models, backbone_option.model_type)
        backbone_cls = Minkowski
        self.backbone_model = backbone_cls(architecture="unet", input_nc=input_nc, config=backbone_option)
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )
        self.gp = ME.MinkowskiGlobalAvgPooling()
        self.last = ProjectNet(
            option.normalnet_option.nn_size, option.normalnet_option.last_size, option.normalnet_option.normalize
        )

    def compute_loss_match(self):
        if hasattr(self, "xyz"):
            xyz = self.xyz
            xyz_target = self.xyz_target
        else:
            xyz = self.input.pos
            xyz_target = self.input_target.pos
        loss_reg = self.metric_loss_module(
            self.output_loss, self.output_loss_target, self.match[:, :2], xyz, xyz_target
        )
        return loss_reg

    def compute_loss_label(self):
        """
        compute the loss separating the miner and the loss
        each point correspond to a labels
        """
        output = torch.cat([self.output_loss[self.match[:, 0]], self.output_target_loss[self.match[:, 1]]], 0)
        rang = torch.arange(0, len(self.match), dtype=torch.long, device=self.match.device)
        labels = torch.cat([rang, rang], 0)
        hard_pairs = None
        if self.miner_module is not None:
            hard_pairs = self.miner_module(output, labels)
        # loss
        loss_reg = self.metric_loss_module(output, labels, hard_pairs)
        return loss_reg

    def compute_metric_loss(self):
        if self.mode == "match":
            self.metric_loss = self.compute_loss_match()
        elif self.mode == "label":
            self.metric_loss = self.compute_loss_label()
        else:
            raise NotImplementedError("The mode for the loss is incorrect")

    def compute_normal_loss(self):
        loss = torch.sum(torch.abs(self.normal * self.input.norm), dim=1).mean()
        loss_target = torch.sum(torch.abs(self.normal_target * self.input_target.norm), dim=1).mean()
        self.normal_loss = 2 - loss - loss_target

    def set_input(self, data, device):
        self.input, self.input_target = data.to_data()
        self.input = self.input.to(device)
        self.input_target = self.input_target.to(device)
        self.match = data.pair_ind.to(torch.long).to(device)
        self.size_match = data.size_pair_ind.to(torch.long).to(device)
        self.xyz = self.input.pos.to(device)
        self.xyz_target = self.input_target.pos.to(device)

    def forward(self):
        feature = self.backbone_model.forward(self.input)
        feature_target = self.backbone_model.forward(self.input_target)
        if self.normalize_feature:
            self.output = torch.nn.functional.normalize(feature.x, dim=1)
            self.output_target = torch.nn.functional.normalize(feature_target.x, dim=1)
        self.output_loss = self.last(self.output)
        self.output_loss_target = self.last(self.output_target)

        self.compute_metric_loss()
        self.loss = self.metric_loss

    def backward(self):
        if hasattr(self, "loss"):
            self.loss.backward()

    def get_output(self):
        return self.output, self.output_target

    def get_batch(self):
        return self.input.batch, self.input_target.batch

    def get_input(self):
        if self.match is not None:
            input = Data(pos=self.xyz, ind=self.match[:, 0], size=self.size_match)
            input_target = Data(pos=self.xyz_target, ind=self.match[:, 1], size=self.size_match)
            return input, input_target
        else:
            input = Data(pos=self.xyz)
            return input, None
