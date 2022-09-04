from typing import Optional, Callable, Tuple

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, InMemoryDataset, download_url


class MSGNN_real_data(InMemoryDataset):
    r"""Data loader for the data sets used in the
    `MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian <https://arxiv.org/pdf/2209.00546.pdf>`_ paper.

    Args:
        name (str): Name of the data set, choices are: 'FiLL-pvCLCL"+str(year), 'FiLL-OPCL"+str(year) (year from 2000 to 2020).
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        sparsify_level (float, optional): the density of the graph, a value between 0 and 1. Default: 1.
    """

    def __init__(self, name: str, root: str, transform: Optional[Callable] = None, 
    pre_transform: Optional[Callable] = None, sparsify_level: float=1):
        self.name = name
        self.url = self._generate_url(name)
        self.sparsify_level = sparsify_level
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _generate_url(self, name: str) -> Tuple:
        if self.name[:11].lower() == 'fill-pvclcl':
            url = (
                'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/FiLL')
        elif self.name[:9].lower() == 'fill-opcl':
            url = (
                'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/FiLL')
        return url

    @property
    def raw_file_names(self):
        return [self.name[5:]+'.npy']

    @property
    def processed_file_names(self):
        return [self.name+'_10p_'+str(int(self.sparsify_level*10))+'.npy']

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        adj = np.load(self.raw_paths[0])
        if self.sparsify_level > 1:
            raise ValueError('Sparsify level should be greater than 0 and less than 1 but got!'.format(self.sparsify_level))
        elif self.sparsify_level <= 0:
            raise ValueError('Sparsify level should be greater than 0 and less than 1 but got!'.format(self.sparsify_level))
        else:
            flattened_abs_adj = np.abs(adj).flatten()
            sorted_abs_vals = np.sort(flattened_abs_adj)
            threshold = sorted_abs_vals[-int(len(sorted_abs_vals) * self.sparsify_level)]
            adj[np.abs(adj)<threshold] = 0

        coo = sp.csr_matrix(adj).tocoo()
        values = torch.from_numpy(coo.data).float()
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        data = Data(num_nodes=indices.max().item() + 1,
                    edge_index=indices, edge_weight=values)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
