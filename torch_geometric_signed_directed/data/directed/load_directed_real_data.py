from typing import Optional, Callable

import json
import os.path as osp
from itertools import chain
from torch_sparse import coalesce

import torch
import numpy as np
import torch_geometric
import scipy.sparse as sp
from .DirectedData import DirectedData
from torch_geometric.datasets import WebKB
from torch_geometric.data import Data, InMemoryDataset, download_url

class WikipediaNetwork(InMemoryDataset):
    r"""The code is modified from torch_geometric.datasets.WikipediaNetwork (v1.6.3)
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Chameleon"` :obj:`"Squirrel"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['chameleon', 'squirrel']

        super(WikipediaNetwork, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'] + [
            '{}_split_0.6_0.2_{}.npz'.format(self.name, i) for i in range(10)
        ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.float)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

class WikiCS(InMemoryDataset):
    r"""This is the copy of the torch_geometric.datasets.WikiCS (v1.6.3)
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
    url = 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset'

    def __init__(self, root, transform=None, pre_transform=None):
        super(WikiCS, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.json']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = json.load(f)

        x = torch.tensor(data['features'], dtype=torch.float)
        y = torch.tensor(data['labels'], dtype=torch.long)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        train_mask = torch.tensor(data['train_masks'], dtype=torch.bool)
        train_mask = train_mask.t().contiguous()

        val_mask = torch.tensor(data['val_masks'], dtype=torch.bool)
        val_mask = val_mask.t().contiguous()

        test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)

        stopping_mask = torch.tensor(data['stopping_masks'], dtype=torch.bool)
        stopping_mask = stopping_mask.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask,
                    stopping_mask=stopping_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def load_from_npz(file_name:str) -> torch_geometric.data: 
    """
    Load a graph from a npz file for unweighted graph cora and citeseer.
    Args:
        file_name : str name of the file to load.
    Rreturn: torch_geometric.data object
    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        edge_index = loader['adj_indices'].copy()
        adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])
        features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])
        labels = loader.get('labels')

    coo = adj.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    features = torch.from_numpy(features.todense()).float()
    labels = torch.from_numpy(labels).long()
    data = Data(x=features, edge_index=indices, edge_weight=None, y=labels)
    return data

def load_directed_real_data(dataset: str='WebKB', root:str = './', name:str = 'Texas',
transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None) -> DirectedData:
    """The function for WebKB data downloading and convert to DirectedData object

    Args:
        dataset (str, optional) data set name (default: 'WebKB').
        root (str, optional) path to save the dataset (default: './').
        name (str, required) the name of the subdataset (default: 'Texas').
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    if dataset.lower() == 'webkb':
        data = WebKB(root=root, name=name, transform=transform, pre_transform=pre_transform)[0]
    elif dataset.lower() == 'citeseer':
        data = load_from_npz('./dataset/citeseer.npz')
    elif dataset.lower() == 'cora_ml':
        data = load_from_npz('./dataset/cora_ml.npz')
    elif dataset.lower() == 'wikics':
        data = WikiCS(root=root,transform=transform, pre_transform=pre_transform)[0]
    elif dataset.lower() == 'wikipedianetwork':
    	data = WikipediaNetwork(root=root, name=name, transform=transform, pre_transform=pre_transform)[0]
    else:
        raise NameError('Please input the correct data set name instead of {}!'.format(dataset))
    directed_dataset = DirectedData(x=data.x,edge_index=data.edge_index,y=data.y)
    return directed_dataset