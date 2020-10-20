import logging
import torch
import torchsparse as TS


from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.applications.sparseconv3d import SparseConv3d
from torch_points3d.models.registration.base import FragmentBaseModel
from torch.nn import LeakyReLU, Linear, Sequential
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq


log = logging.getLogger(__name__)


class APIModel(FragmentBaseModel):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method of UnetBasedModel
        super().__init__(option)
        self.backbone = SparseConv3d(
            "unet", dataset.feature_dimension, config=option.backbone, backend=option.get("backend", "minkowski")
        )
        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.loss_names = ["loss_reg", "loss"]
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )
        # Last Layer

        if option.mlp_cls is not None:
            last_mlp_opt = option.mlp_cls
            last_mlp_opt.nn[0]
            self.FC_layer = Seq()
            for i in range(1, len(last_mlp_opt.nn)):
                self.FC_layer.append(
                    Sequential(
                        *[
                            Linear(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                            LeakyReLU(0.2),
                        ]
                    )
                )
        else:
            self.FC_layer = torch.nn.Identity()

    def set_input(self, data, device):
        self.input, self.input_target = data.to_data()
        self.input = self.input.to(device)
        if hasattr(data, "pos_target"):
            self.input_target = self.input_target.to(device)
            self.match = data.pair_ind.to(torch.long).to(device)
            self.size_match = data.size_pair_ind.to(torch.long).to(device)
        else:
            self.match = None

    def get_batch(self):
        if self.match is not None:
            return self.input.batch, self.input_target.batch
        else:
            return self.input.batch, None

    def get_input(self):
        if self.match is not None:
            input = self.input.clone()
            input_target = self.input_target.clone()
            input.ind = self.match[:, 0]
            input_target.ind = self.match[:, 1]
            input.size = self.size_match
            input_target.size = self.size_match
            return input, input_target
        else:
            return self.input, None

    def apply_nn(self, input):
        out = self.backbone(input)
        out_feat = self.FC_layer(out.x)
        if self.normalize_feature:
            out_feat = out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)
        return out_feat
