from typing import Optional, Callable

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, InMemoryDataset, download_url


class DIGRAC_real_data(InMemoryDataset):
    def __init__(self, name: str, root: str, transform: Optional[Callable]=None, pre_transform: Optional[Callable]=None):
        self.name = name
        url = ('https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/'+name+'.npz')
        self.url = url
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return [self.name+'.npz']

    @property
    def processed_file_names(self):
        return [self.name+'.pt']

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        adj = sp.load_npz(self.raw_dir+'/'+self.name+'.npz')
        coo = adj.tocoo()
        values = torch.from_numpy(coo.data)
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        data = Data(edge_index=indices, edge_weight=values)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])