import os.path as osp
from .base_dataset import BaseDataset
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from datasets.utils import contains_key
from datasets.transforms import MeshToNormal

AVAILABLE_NUMBERS = ["10", "40"]


class ModelNetDataset(BaseDataset):
    def __init__(self, dataset_opt, training_opt):
        super().__init__(dataset_opt, training_opt)

        number = dataset_opt.number
        if str(number) not in AVAILABLE_NUMBERS:
            raise Exception("Only ModelNet10 and ModelNet40 are available")
        name = 'ModelNet{}'.format(number)
        self._data_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        pre_transform = T.Compose([T.NormalizeScale(),
                                   MeshToNormal()])

        train_dataset = ModelNet(
            self._data_path,
            name=str(number),
            train=True,
            transform=self.transform,
            pre_transform=pre_transform)

        test_dataset = ModelNet(
            self._data_path,
            name=str(number),
            train=False,
            transform=self.transform,
            pre_transform=pre_transform)

        self._create_dataloaders(train_dataset, test_dataset, validation=None)
