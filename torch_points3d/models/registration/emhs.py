import torch
from .base import FragmentBaseModel

from torch_geometric.data import Data, Batch
from torch_scatter import scatter_max, scatter_mean
from torch_geometric.nn import voxel_grid, consecutive_cluster

from torch_points3d.applications import models
from torch_points3d.core.common_modules import MLP


class EquiModule(torch.nn.module):
    def __init__(self, pre_pointnet, unet, post_pointnet, add_pos=True, is_center=True, voxel_size=0.1, pool="mean"):
        super().__init__()
        self.pre_pointnet = pre_pointnet
        self.post_pointnet = post_pointnet
        self.unet = unet
        self.add_pos = add_pos
        self.is_center = is_center
        self.voxel_size = voxel_size
        self.pool = pool

    def pool_fn(self, x, cluster, mode="mean"):
        if mode == "mean":
            return scatter_mean(x, cluster)
        if mode == "max":
            return scatter_max(x, cluster)
        else:
            raise NotImplementedError("the mode of pooling is not present")

    def cluster_data(self, pos, batch):
        coords = torch.round(pos / self.voxel_size)
        cluster = voxel_grid(coords, batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        # pre_x = self.pool_fn(x, cluster, mode=self.pool)
        # new_pos = self.pool_fn(pos, cluster, mode="mean")
        new_batch = batch[unique_pos_indices]
        new_coords = coords[unique_pos_indices]
        return cluster, new_batch, new_coords

    def forward(self, data):
        cluster, pre_batch, pre_coords = self.cluster_data(data.pos, data.batch)
        pre_pos = self.pool_fn(data.pos, cluster, mode="mean")
        first_x = data.x
        if self.add_pos:
            if self.is_center:
                center = pre_pos[cluster]
                pos = data.pos - center
            else:
                pos = data.pos
            if data.x is not None:
                first_x = torch.cat([data.x, pos])
            else:
                first_x = pos
        x = self.pre_pointnet(first_x)
        # average pooling
        pre_x = self.pool_fn(x, cluster, mode=self.pool)
        pre_data = Batch(x=pre_x, pos=pre_pos, coords=pre_coords, batch=pre_batch)
        pre_data = self.unet(pre_data)
        post_x = torch.cat([first_x, pre_data.x[cluster]])
        post_x = self.post_pointnet(post_x)
        post_pos = data.pos
        post_batch = data.batch
        post_data = Batch(x=post_x, pos=post_pos, batch=post_batch)
        return post_data


class EquiMHS(FragmentBaseModel):
    def __init__(self, option, model_type, dataset, modules):
        FragmentBaseModel.__init__(self, option)

        self.mode = option.loss_mode
        self.normalize_feature = option.normalize_feature
        self.loss_names = ["loss_reg", "loss"]
        self.metric_loss_module, self.miner_module = FragmentBaseModel.get_metric_loss_and_miner(
            getattr(option, "metric_loss", None), getattr(option, "miner", None)
        )
        # call unet backbone

        unet_option = option.unet
        input_nc = unet_option.nn_pre[-1]
        unet_cls = getattr(models, unet_option.model_type)
        unet_extr_options = unet_option.get("extra_options", {})
        self.unet_model = unet_cls(
            architecture="unet", input_nc=input_nc, num_layers=4, config=unet_option.config, **unet_extr_options
        )

        self.pre_mini_pointnet = MLP(unet_option.nn_pre)
        self.post_mini_pointnet = MLP(unet_option.nn_post)
        self.equi_module = EquiModule(
            pre_pointnet=self.pre_mini_pointnet,
            post_pointnet=self.post_mini_pointnet,
            unet=self.unet_model,
            is_center=unet_option.is_center,
            add_pos=unet_option.add_pos,
            voxel_size=unet_option.voxel_size,
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

    def apply_nn(self, input):
        data = self.equi_module(input)
        output = data.x
        if self.normalize_feature:
            return output / (torch.norm(output, p=2, dim=1, keepdim=True) + 1e-3)
        else:
            return output

    def get_batch(self):
        if self.match is not None:
            batch = self.input.batch
            batch_target = self.input_target.batch
            return batch, batch_target
        else:
            return None

    def get_input(self):
        if self.match is not None:
            input = Data(pos=self.input.pos, ind=self.match[:, 0], size=self.size_match)
            input_target = Data(pos=self.input_target.pos, ind=self.match[:, 1], size=self.size_match)
            return input, input_target
        else:
            input = Data(pos=self.xyz)
            return input
