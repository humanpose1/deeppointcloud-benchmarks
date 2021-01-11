import torch
import torch.nn as nn
from torch.nn import LeakyReLU, Linear, Sequential

from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn import voxel_grid
from torch_geometric.data import Batch

from torch_points3d.applications.minkowski import Minkowski
from torch_points3d.core.common_modules import FastBatchNorm1d, Seq
from torch_points3d.core.common_modules import MLP
from torch_points3d.models.registration.base import FragmentBaseModel


class UnetMinkowski(nn.Module):
    def __init__(self, option_unet, input_nc=1, grid_size=0.05, post_mlp_nn=[64, 64, 32], add_pos=False):
        nn.Module.__init__(self)
        self.unet = Minkowski(architecture="unet", input_nc=input_nc, config=option_unet)
        self.post_mlp = MLP(post_mlp_nn)
        self._grid_size = grid_size
        self.add_pos = add_pos

    def set_grid_size(self, grid_size):
        self._grid_size = grid_size

    def _prepare_data(self, data):
        coords = torch.round((data.pos) / self._grid_size).float()
        cluster = voxel_grid(coords, data.batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        coords = coords[unique_pos_indices]
        new_batch = data.batch[unique_pos_indices]
        new_pos = data.pos[unique_pos_indices]
        x = data.x[unique_pos_indices]
        sparse_data = Batch(x=x, pos=new_pos, coords=coords, batch=new_batch)
        return sparse_data, cluster

    def forward(self, data, **kwargs):

        d, cluster = self._prepare_data(data.clone())
        d = self.unet.forward(d)
        if self.add_pos:
            data.x = self.post_mlp(torch.cat([d.x[cluster], data.pos - data.pos.mean(0)], 1))
        else:
            data.x = self.post_mlp(d.x[cluster])
        return data


class BaseMS_Minkowski(FragmentBaseModel):
    def __init__(self, option, model_type, dataset, modules):
        FragmentBaseModel.__init__(self, option)
        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.loss_names = ["loss", "loss_reg"]
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
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
        raise NotImplementedError("It depends on the networks")


class MS_Minkowski(BaseMS_Minkowski):
    def __init__(self, option, model_type, dataset, modules):
        # Last Layer
        BaseMS_Minkowski.__init__(self, option, model_type, dataset, modules)
        option_unet = option.option_unet
        num_scales = option_unet.num_scales
        self.unet = nn.ModuleList()
        for i in range(num_scales):
            module = UnetMinkowski(
                option_unet["config_{}".format(i)],
                grid_size=option_unet.grid_size[i],
                post_mlp_nn=option_unet.post_mlp_nn,
                add_pos=option_unet.add_pos,
            )
            self.unet.add_module(name=str(i), module=module)
        # Last MLP layer
        assert option.mlp_cls is not None
        last_mlp_opt = option.mlp_cls
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

    def apply_nn(self, input):
        # inputs = self.compute_scales(input)
        outputs = []
        for i in range(len(self.unet)):
            out = self.unet[i](input.clone())
            out.x = out.x / (torch.norm(out.x, p=2, dim=1, keepdim=True) + 1e-20)
            outputs.append(out)
        x = torch.cat([o.x for o in outputs], 1)
        out_feat = self.FC_layer(x)
        if self.normalize_feature:
            out_feat = out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)
        return out_feat


class MS_Minkowski_Shared(BaseMS_Minkowski):
    def __init__(self, option, model_type, dataset, modules):
        BaseMS_Minkowski.__init__(self, option, model_type, dataset, modules)
        option_unet = option.option_unet
        self.grid_size = option_unet.grid_size
        self.unet = UnetMinkowski(option_unet.config, post_mlp_nn=option_unet.post_mlp_nn,)
        assert option.mlp_cls is not None
        last_mlp_opt = option.mlp_cls
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

    def apply_nn(self, input):
        # inputs = self.compute_scales(input)
        outputs = []
        for i in range(len(self.grid_size)):
            self.unet.set_grid_size(self.grid_size[i])
            out = self.unet(input.clone())
            out.x = out.x / (torch.norm(out.x, p=2, dim=1, keepdim=True) + 1e-20)
            outputs.append(out)
        x = torch.cat([o.x for o in outputs], 1)
        out_feat = self.FC_layer(x)
        if self.normalize_feature:
            out_feat = out_feat / (torch.norm(out_feat, p=2, dim=1, keepdim=True) + 1e-20)
        return out_feat
