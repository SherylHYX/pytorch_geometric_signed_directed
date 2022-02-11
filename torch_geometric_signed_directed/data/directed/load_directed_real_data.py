from typing import Optional, Callable
from torch_geometric.datasets import WebKB

from .DirectedData import DirectedData
from ...utils.general import node_class_split
from .WikiCS import WikiCS
from .WikipediaNetwork import WikipediaNetwork
from .citation import Cora_ml, Citeseer


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
        data = Citeseer(root=root, transform=transform, pre_transform=pre_transform)[0]
        data = node_class_split(data, train_size_per_class=20, val_size=500)
    elif dataset.lower() == 'cora_ml':
        data = Cora_ml(root=root, transform=transform, pre_transform=pre_transform)[0]
        data = node_class_split(data, train_size_per_class=20, val_size=500)
    elif dataset.lower() == 'wikics':
        data = WikiCS(root=root,transform=transform, pre_transform=pre_transform)[0]
    elif dataset.lower() == 'wikipedianetwork':
        data = WikipediaNetwork(root=root, name=name, transform=transform, pre_transform=pre_transform)[0]
    else:
        raise NameError('Please input the correct data set name instead of {}!'.format(dataset))
    directed_dataset = DirectedData(x=data.x,edge_index=data.edge_index,y=data.y,
                                        train_mask=data.train_mask,val_mask=data.val_mask,test_mask=data.test_mask)
    directed_dataset.inherit_attributes(data)
    return directed_dataset