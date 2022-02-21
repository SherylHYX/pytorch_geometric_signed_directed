
   
from typing import List, Optional, Callable

import os
import json

import torch
from torch_geometric.data import (InMemoryDataset, download_url, Data)

dataset_name_url_dic = {
    'bitcoin_alpha': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/bitcoin_alpha.csv',
    'bitcoin_otc': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/bitcoin_otc.csv',
    "epinions": 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/epinions.csv',
    'slashdot': 'https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/slashdot.csv'
}


class SignedDirectedGraphDataset(InMemoryDataset):
    r"""Signed Directed Graph from the `"SDGNN: Learning Node Representation 
    for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper, consising of four different
    datasets: Bitcoin-Alpha, Bitcoin-OTC, Slashdot and Epinions.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str ='bitcoin_alpha',
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        seed=2021,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
    ):
        self.dataset_name = dataset_name.lower()
        self.url = dataset_name_url_dic[dataset_name]
        assert train_ratio+test_ratio == 1.0

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data = self.get(0)

        torch.random.manual_seed(seed)
        idx = torch.randperm(self.data.edge_index.shape[1])

        train_num = int(train_ratio * len(idx))

        self.data.train_edge_index  = self.data.edge_index[:, idx[:train_num]]
        self.data.test_edge_index   = self.data.edge_index[:, idx[train_num:]]
        self.data.train_edge_weight = self.data.edge_weight[idx[:train_num]]
        self.data.test_edge_weight  = self.data.edge_weight[idx[train_num:]]
        self.data.num_nodes = self.num_nodes

        self.data, self.slices = self.collate([self.data])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.dataset_name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.dataset_name, 'processed')

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
        with open(self.raw_paths[0], 'r', encoding='utf-8-sig') as f: 
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
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1
