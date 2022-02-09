import sys
sys.path.append('../../pytorch_geometric_signed_directed')

import numpy as np
from torch_geometric.utils import is_undirected

from torch_geometric_signed_directed.data import load_directed_real_data, DirectedData

def test_datasets():
    """
    Testing load_dataset()
    """
    directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Texas')
    assert isinstance(directed_dataset, DirectedData)
    assert not is_undirected(directed_dataset.edge_index)
    directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Cornell')
    assert isinstance(directed_dataset, DirectedData)
    assert not is_undirected(directed_dataset.edge_index)
    directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Wisconsin')
    assert isinstance(directed_dataset, DirectedData)
    assert not is_undirected(directed_dataset.edge_index)
    directed_dataset = load_directed_real_data(dataset='cora_ml', root='./tmp_data/')
    assert isinstance(directed_dataset, DirectedData)
    assert not is_undirected(directed_dataset.edge_index)
    directed_dataset = load_directed_real_data(dataset='citeseer', root='./tmp_data/')
    assert isinstance(directed_dataset, DirectedData)
    assert not is_undirected(directed_dataset.edge_index)
    directed_dataset = load_directed_real_data(dataset='wikics', root='./tmp_data/')
    assert isinstance(directed_dataset, DirectedData)
    assert not is_undirected(directed_dataset.edge_index)
    directed_dataset = load_directed_real_data(dataset='wikipedianetwork', root='./tmp_data/', name='chameleon')
    assert isinstance(directed_dataset, DirectedData)
    assert not is_undirected(directed_dataset.edge_index)
    directed_dataset = load_directed_real_data(dataset='wikipedianetwork', root='./tmp_data/', name='squirrel')
    assert isinstance(directed_dataset, DirectedData)
    assert not is_undirected(directed_dataset.edge_index)
    return

test_datasets()