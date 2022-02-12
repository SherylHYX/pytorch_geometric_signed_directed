from typing import Optional, Callable

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, InMemoryDataset, download_url

from ...utils.general import node_class_split


class Telegram(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable]=None, pre_transform: Optional[Callable]=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    url = ('https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/telegram')
    @property
    def raw_file_names(self):
        return ['telegram_adj.npz', 'telegram_labels.npy']

    @property
    def processed_file_names(self):
        return ['telegram.pt']

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        A = sp.load_npz(self.raw_paths[0])
        label = np.load(self.raw_paths[1])
        rs = np.random.RandomState(seed=0)

        test_ratio = 0.2
        train_ratio = 0.6
        val_ratio = 1 - train_ratio - test_ratio

        label = torch.from_numpy(label).long()
        s_A = sp.csr_matrix(A)
        coo = s_A.tocoo()
        values = coo.data
        
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        features = torch.from_numpy(rs.normal(0, 1.0, (s_A.shape[0], 1))).float()

        data = Data(x=features, edge_index=indices, edge_weight=None, y=label)
        data = node_class_split(data, train_size_per_class=train_ratio, val_size_per_class=val_ratio)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
