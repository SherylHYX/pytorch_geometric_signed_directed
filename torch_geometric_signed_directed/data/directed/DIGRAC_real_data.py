from typing import Optional, Callable

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, InMemoryDataset, download_url


class DIGRAC_real_data(InMemoryDataset):
    r"""Data loader for the data sets used in the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance" <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.

    Args:
        name (str): Name of the data set, choices are: 'blog', 'wikitalk', 'migration', 'lead_lag"+str(year) (year from 2001 to 2019).
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, name: str, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
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
        values = torch.from_numpy(coo.data).float()
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        data = Data(num_nodes=indices.max().item() + 1,
                    edge_index=indices, edge_weight=values)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
