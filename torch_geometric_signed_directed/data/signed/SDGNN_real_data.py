from typing import Optional, Callable
import os
import json

import torch
from torch_geometric.data import (InMemoryDataset, download_url, Data)


dataset_name_url_dic = {
    "bitcoin_alpha": "https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/bitcoin_alpha.csv",
    "bitcoin_otc": "https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/bitcoin_otc.csv",
    "wiki": "https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/wikirfa.csv",
    "epinions": "https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/epinions.csv",
    "slashdot": "https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/slashdot.csv"
}


class SDGNN_real_data(InMemoryDataset):
    r"""Signed Directed Graph from the `"SDGNN: Learning Node Representation 
    for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper, consising of five different
    datasets: Bitcoin-Alpha, Bitcoin-OTC, Wikirfa, Slashdot and Epinions from `snap.stanford.edu <http://snap.stanford.edu/data/#signnets>`_.

    Args:
        name (str): Name of the dataset, choices are: 'bitcoin_alpha', 'bitcoin_otc', 'wiki', 'epinions', 'slashdot'. 
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, name: str,  root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.url = dataset_name_url_dic[name]
        self.root = root

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        _, _, filename = self.url.rpartition('/')
        return filename

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = []
        edge_weight = []
        edge_index = []
        node_map = {}
        with open(self.raw_paths[0], 'r') as f:
            for line in f:
                x = line.strip().split(',')
                assert len(x) == 3
                a, b = x[0], x[1]
                if a not in node_map:
                    node_map[a] = len(node_map)
                if b not in node_map:
                    node_map[b] = len(node_map)
                a, b = node_map[a], node_map[b]
                data.append([a, b])

                edge_weight.append(float(x[2]))

            edge_index = [[i[0], int(i[1])] for i in data]
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            edge_weight = torch.FloatTensor(edge_weight)
        map_file = os.path.join(self.processed_dir, 'node_id_map.json')
        with open(map_file, 'w') as f:
            f.write(json.dumps(node_map))

        data = Data(edge_index=edge_index, edge_weight=edge_weight)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1
