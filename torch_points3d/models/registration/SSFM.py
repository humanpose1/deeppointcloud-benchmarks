import torch
import torch.nn as nn
from torch_geometric.data import Data

from torch_points3d.models.registration.base import FragmentBaseModel
from torch_points3d.models.base_model import BaseModel
from torch_points3d.core.losses.chamfer_loss import PartialChamferLoss

from torch_points3d.applications import models
from torch_points3d.applications.minkowski import Minkowski
import MinkowskiEngine as ME

from torch_points3d.core.common_modules.base_modules import MLP
from torch.nn import Linear

from torch_points3d.core.data_transform.transforms import apply_mask
from torch_points3d.datasets.registration.utils import tracked_matches


class MLPNet(torch.nn.Module):
    """
    project the features in a non linear space
    """

    def __init__(self, nn_size=[128, 256, 256], last_size=256, normalize=True, first_activation=True):
        super().__init__()
        if first_activation:
            # because at the end of the network there is no activation
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.Identity()

        if nn_size is None:
            self.local_nn = torch.nn.Identity()
            self.last_layer = Linear(last_size, last_size, bias=False)
        else:
            self.local_nn = MLP(nn_size)
            self.last_layer = Linear(nn_size[-1], last_size, bias=False)
        self.normalize = normalize

    def forward(self, x):

        x = self.activation(self.local_nn(x))
        norm = self.last_layer(x)
        if self.normalize:
            return torch.nn.functional.normalize(norm, dim=1)
        else:
            return norm


class FoldingNet(nn.Module):
    def __init__(self, in_channel, size_seed=8):
        super().__init__()

        self.size_seed = size_seed
        self.in_channel = in_channel

        a = (
            torch.linspace(-1.0, 1.0, steps=size_seed, dtype=torch.float)
            .view(1, size_seed)
            .expand(size_seed, size_seed)
            .reshape(1, -1)
        )
        b = (
            torch.linspace(-1.0, 1.0, steps=size_seed, dtype=torch.float)
            .view(size_seed, 1)
            .expand(size_seed, size_seed)
            .reshape(1, -1)
        )
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),
        )

    def forward(self, x):
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, self.size_seed ** 2)
        seed = self.folding_seed.view(1, 2, self.size_seed ** 2).expand(bs, 2, self.size_seed ** 2).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


class Decoder(nn.Module):
    def __init__(self, in_channel, size_seed, scale=0.1):
        super().__init__()

        self.size_seed = size_seed
        self.scale = scale
        self.fn = FoldingNet(in_channel=in_channel, size_seed=size_seed)

    def forward(self, x, xyz):
        assert len(x) == len(xyz)
        bs = x.size(0)
        offset = xyz.view(bs, 3, 1).expand(bs, 3, self.size_seed ** 2)

        new_pc = self.scale * self.fn(x) + offset

        return new_pc.transpose(2, 1).reshape(-1, 3)


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
        self.loss_names = ["metric_loss", "loss", "normal_loss", "chamfer_loss"]
        input_nc = dataset.feature_dimension
        backbone_option = option.backbone
        # backbone_cls = getattr(models, backbone_option.model_type)
        backbone_cls = Minkowski
        self.backbone_model = backbone_cls(architecture="unet", input_nc=input_nc, config=backbone_option)
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )

        self.last = MLPNet(
            option.feat_options.nn_size,
            option.feat_options.last_size,
            option.feat_options.normalize,
            option.feat_options.first_activation,
        )
        self.normal_net = MLPNet(
            option.normal_options.nn_size, 3, option.normal_options.normalize, option.normal_options.first_activation,
        )
        self.lambda_normal = option.normal_options.const

        self.size_seed = option.reconst_options.size_seed
        self.scale = option.reconst_options.scale
        self.voxel_size = option.reconst_options.voxel_size
        self.decoder = Decoder(option.reconst_options.in_channel, self.size_seed, self.scale)
        self.lambda_chamfer = option.reconst_options.const

        self.chamfer_loss_fn = PartialChamferLoss()

    def compute_loss_match(self):
        if hasattr(self, "xyz"):
            xyz = self.xyz
            xyz_target = self.xyz_target
        else:
            xyz = self.input.pos
            xyz_target = self.input_target.pos
        loss_reg = self.metric_loss_module(self.output, self.output_target, self.match[:, :2], xyz, xyz_target)
        return loss_reg

    def compute_loss_label(self):
        """
        compute the loss separating the miner and the loss
        each point correspond to a labels
        """
        output = torch.cat([self.output[self.match[:, 0]], self.output_target[self.match[:, 1]]], 0)
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
        self.normal_loss = 1 - loss + 1 - loss_target

    def set_input(self, data, device):
        self.input, self.input_target = data.to_data()

        # For reconstruction
        self.xyz_gt = torch.stack([self.input.pos_x, self.input.pos_y, self.input.pos_z], 1).to(device)
        self.xyz_target_gt = torch.stack(
            [self.input_target.pos_x, self.input_target.pos_y, self.input_target.pos_z], 1
        ).to(device)
        num_batch = self.input.batch[-1] + 1
        num_pt = len(self.xyz_gt) // num_batch
        self.batch_gt = (
            torch.arange(0, num_batch).view(num_batch, 1).expand(num_batch, num_pt).reshape(-1).long().to(device)
        )
        self.batch_target_gt = (
            torch.arange(0, num_batch).view(num_batch, 1).expand(num_batch, num_pt).reshape(-1).long().to(device)
        )

        # to device

        self.match = data.pair_ind.to(torch.long).to(device)
        self.size_match = data.size_pair_ind.to(torch.long).to(device)
        self.input = self.input.to(device)
        self.input_target = self.input_target.to(device)

        self.xyz = self.input.pos.to(device)
        self.xyz_target = self.input_target.pos.to(device)

    def forward(self, *args, **kwargs):
        feature = self.backbone_model.forward(self.input)
        small_feature = self.backbone_model.down[-1]
        feature_target = self.backbone_model.forward(self.input_target)
        small_feature_target = self.backbone_model.down[-1]

        # compute last feature
        self.output = self.last(feature.x)
        self.output_target = self.last(feature_target.x)

        # reconstruct loss

        if self.lambda_chamfer > 0:

            small_xyz = (small_feature.C[:, 1:] * self.voxel_size).float().to(small_feature.F.device)
            small_xyz_target = (small_feature_target.C[:, 1:] * self.voxel_size).float().to(small_feature.F.device)
            xyz_rec = self.decoder(small_feature.F, small_xyz)
            xyz_rec_target = self.decoder(small_feature_target.F, small_xyz_target)

            small_batch = (
                small_feature.C[:, 0]
                .view(len(small_feature.C), 1, 1)
                .expand(len(small_feature.C), 1, self.size_seed ** 2)
                .transpose(2, 1)
                .reshape(-1)
                .long()
                .to(small_feature.F.device)
            )
            small_batch_target = (
                small_feature_target.C[:, 0]
                .view(len(small_feature_target.C), 1, 1)
                .expand(len(small_feature_target.C), 1, self.size_seed ** 2)
                .transpose(2, 1)
                .reshape(-1)
                .long()
                .to(small_feature.F.device)
            )

            _, ind = small_batch.sort()
            _, ind_target = small_batch_target.sort()
            chamfer_loss = self.chamfer_loss_fn(xyz_rec[ind], self.xyz_gt, small_batch[ind], self.batch_gt)

            chamfer_loss_t = self.chamfer_loss_fn(
                xyz_rec_target[ind_target], self.xyz_target_gt, small_batch_target[ind_target], self.batch_target_gt
            )
            self.chamfer_loss = 0.5 * (chamfer_loss + chamfer_loss_t)
        else:
            self.chamfer_loss = 0

        # normal loss
        if hasattr(self.input, "norm") and self.lambda_normal > 0:
            # compute normals
            self.normal = self.normal_net(feature.x)
            self.normal_target = self.normal_net(feature_target.x)
            self.compute_normal_loss()
        else:
            self.normal_loss = 0
        self.compute_metric_loss()
        self.loss = self.metric_loss + self.lambda_normal * self.normal_loss + self.lambda_chamfer * self.chamfer_loss

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
