
   
from typing import List, Optional, Callable

import os
import json

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)

dataset_name_url_dic = {
    'bitcoin_alpha': 'https://github.com/NDS-VU/signed-network-datasets/files/6969330/bitcoinalpha.csv',
    'bitcoin_otc': 'https://github.com/NDS-VU/signed-network-datasets/files/6970437/bitcoinotc.csv',
    "epinions": 'https://github.com/NDS-VU/signed-network-datasets/files/6970960/epinions.csv',
    'slashdot': 'https://github.com/NDS-VU/signed-network-datasets/files/6971382/slashdot.csv.zip'
}

      

class SignedDirectedGraph(InMemoryDataset):
    r"""
    Signed Directed Graph from the `"SDGNN: Learning Node Representation 
    for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper, consising of five different
    datasets: Bitcoin-Alpha, Bitcoin-OTC, Wikirfa, Slashdotm and Epinions.
     
    """

    def __init__(self, root: str, dataset_name: str ='bitcoin_alpha',
                    train_ratio: float = 0.8, test_ratio: float = 0.2,
                    seed=2021, transform: Optional[Callable] = None, 
                    pre_transform: Optional[Callable] = None):
        self.dataset_name = dataset_name
        self.url = dataset_name_url_dic[dataset_name]
        assert train_ratio+test_ratio == 1.0

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data = self.get(0)
        
        torch.random.manual_seed(seed)
        idx = torch.randperm(self.data.edge_index.shape[1])

        train_num = int(train_ratio * len(idx))

        self.data.train_edge_index = self.data.edge_index[:, idx[:train_num]]
        self.data.train_edge_sign  = self.data.edge_sign[idx[:train_num]]
        self.data.test_edge_index  = self.data.edge_index[:,idx[train_num:]]
        self.data.test_edge_sign   = self.data.edge_sign[idx[train_num:]]
        self.data.num_nodes = self.num_nodes
        
        self.data, self.slices = self.collate([self.data])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'signed_directed_graph', 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'signed_directed_graph', 'processed')


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
        edge_sign = []
        edge_index = []
        node_map = {}
        with open(self.raw_paths[0], 'r', encoding='utf-8-sig') as f: 
            for line in f:
                x = line.strip().split(',')
                assert len(x)  == 3
                a, b = x[0], x[1]
                if not a in node_map:
                    node_map[a] = len(node_map)
                if not b in node_map:
                    node_map[b] = len(node_map)
                a, b = node_map[a], node_map[b]
                data.append([a, b])

                edge_sign.append(float(x[2]))

            edge_index = [[i[0], int(i[1])] for i in data]
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            if self.transform == None:
                func = lambda x: 1 if x > 0 else -1
                edge_sign = [func(i) for i in edge_sign]
            edge_sign = torch.tensor(edge_sign, dtype=torch.long)
        map_file = os.path.join(self.processed_dir, 'node_id_map.json')
        with open(map_file, 'w') as f:
            f.write(json.dumps(node_map))


        data = Data()
        data.edge_index= edge_index
        data.edge_sign = edge_sign # we use edge_sign instead of edge_attr
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1

    
    
        