from typing import Optional, Callable, Tuple

import torch
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, InMemoryDataset, download_url


class SSSNET_real_data(InMemoryDataset):
    r"""Data loader for the data sets used in the
    `SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.

    Args:
        name (str): Name of the data set, choices are: 'rainfall', 'PPI', 'wikirfa', 'sampson', 'SP1500', 'Fin_YNet"+str(year) (year from 2000 to 2020).
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
        self.url = self._generate_url(name)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _generate_url(self, name: str) -> Tuple:
        if name.lower() == 'sampson':
            url = (
                'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/Sampson')
        elif name.lower() == 'wikirfa':
            url = (
                'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/wikirfa')
        elif self.name[:8].lower() == 'fin_ynet':
            url = (
                'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/Fin_YNet')
        elif name.lower() == 'sp1500':
            url = (
                'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/SP1500')
        elif name.lower() == 'ppi':
            url = (
                'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/PPI')
        elif name.lower() == 'rainfall':
            url = (
                'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/rainfall')
        return url

    @property
    def raw_file_names(self):
        return [self.name.lower()+'_adj.npz', self.name.lower()+'_labels.npy']

    @property
    def processed_file_names(self):
        return [self.name.lower()+'.pt']

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        adj = sp.load_npz(self.raw_paths[0])
        labels = torch.LongTensor(np.load(self.raw_paths[1]))
        if self.name.lower() == 'sampson':
            features = np.array(
                [[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]).T
            scaler = StandardScaler().fit(features)
            features = torch.FloatTensor(scaler.transform(features))

        coo = adj.tocoo()
        values = torch.from_numpy(coo.data).float()
        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        if self.name.lower() == 'sampson':
            data = Data(num_nodes=indices.max().item(
            ) + 1, edge_index=indices, edge_weight=values, x=features, y=labels)
        elif self.name.lower() in ['sp1500', 'rainfall'] or self.name[:8].lower() == 'fin_ynet':
            data = Data(num_nodes=indices.max().item(
            ) + 1, edge_index=indices, edge_weight=values, y=torch.LongTensor(labels))
        else:
            data = Data(num_nodes=indices.max().item() + 1,
                        edge_index=indices, edge_weight=values)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
