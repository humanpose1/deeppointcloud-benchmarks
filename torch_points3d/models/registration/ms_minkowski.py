import torch

from torch.nn import LeakyReLU, Linear, Sequential
from torch_points3d.models.registration.base import FragmentBaseModel
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq


class MS_Minkowski(FragmentBaseModel):
    def __init__(self, option, dataset, modules):
        FragmentBaseModel.__init__(self, option)
        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.loss_names = ["loss_reg", "loss"]
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )
        # Last Layer

        self.unet = []
        assert option.mlp_cls is not None
        last_mlp_opt = option.mlp_cls

        self.FC_layer = Seq()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.append(
                str(i),
                Sequential(
                    *[
                        Linear(last_mlp_opt.nn[i - 1], last_mlp_opt.nn[i], bias=False),
                        FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                        LeakyReLU(0.2),
                    ]
                ),
            )

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
                return self.input, self.input_target
            else:
                return self.input, None

        def apply_nn(self, input):
            inputs = self.compute_scales(input)
            outputs = []
            for i in range(len(inputs)):
                out = self.unet[i](inputs[i])
                out.x = out.x / (torch.norm(out.x, p=2, dim=1, keepdim=True) + 1e-3)
                outputs.append(out)
            x = torch.cat([o.x for o in outputs], 1)
            out_feat = self.FC_layer(x)
            if self.normalize_feature:
                out_feat = out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)

            return out_feat, outputs
