from typing import Optional, Callable

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, InMemoryDataset, download_url

from ...utils.general import node_class_split


class Cora_ml(InMemoryDataset):
    r"""Data loader for the Cora_ML data set used in the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.url = (
            'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/cora_ml.npz')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['cora_ml.npz']

    @property
    def processed_file_names(self):
        return ['cora_ml.pt']

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        with np.load(self.raw_dir+'/cora_ml.npz', allow_pickle=True) as loader:
            loader = dict(loader)
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                      loader['attr_indptr']), shape=loader['attr_shape'])
            labels = loader.get('labels')

        coo = adj.tocoo()
        values = torch.from_numpy(coo.data).float()
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        features = torch.from_numpy(features.todense()).float()
        labels = torch.from_numpy(labels).long()
        data = Data(x=features, edge_index=indices,
                    edge_weight=values, y=labels)
        data = node_class_split(data, train_size_per_class=20, val_size=500)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])


class Citeseer(InMemoryDataset):
    r"""Data loader for the CiteSeer data set used in the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.url = (
            'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/citeseer.npz')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['citeseer.npz']

    @property
    def processed_file_names(self):
        return ['citeseer.pt']

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        with np.load(self.raw_dir+'/citeseer.npz', allow_pickle=True) as loader:
            loader = dict(loader)
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                      loader['attr_indptr']), shape=loader['attr_shape'])
            labels = loader.get('labels')

        coo = adj.tocoo()
        values = torch.from_numpy(coo.data)
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        features = torch.from_numpy(features.todense()).float()
        labels = torch.from_numpy(labels).long()
        data = Data(x=features, edge_index=indices,
                    edge_weight=values, y=labels)
        data = node_class_split(data, train_size_per_class=20, val_size=500)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
