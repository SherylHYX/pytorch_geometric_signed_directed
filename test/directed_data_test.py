import torch
import numpy as np
import torch_geometric.transforms as T

from torch_geometric_signed_directed.data import load_directed_real_data, DirectedData
from torch_geometric_signed_directed.utils import directed_link_class_split, node_class_split

def test_directed_datasets():
    """
    Testing load_dataset()
    """
    directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Texas')
    assert isinstance(directed_dataset, DirectedData)
    assert directed_dataset.is_directed
    directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Cornell')
    assert isinstance(directed_dataset, DirectedData)
    assert directed_dataset.is_directed
    directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Wisconsin')
    assert isinstance(directed_dataset, DirectedData)
    assert directed_dataset.is_directed
    directed_dataset = load_directed_real_data(dataset='wikipedianetwork', root='./tmp_data/wikipedianetwork', name='chameleon')
    assert isinstance(directed_dataset, DirectedData)
    assert directed_dataset.is_directed
    directed_dataset = load_directed_real_data(dataset='wikipedianetwork', root='./tmp_data/wikipedianetwork', name='squirrel', pre_transform=T.GCNNorm(), transform=T.ToUndirected())
    assert isinstance(directed_dataset, DirectedData)
    assert not directed_dataset.is_directed
    assert directed_dataset.is_weighted
    directed_dataset.to_unweighted()
    assert not directed_dataset.is_weighted
    directed_dataset = load_directed_real_data(dataset='telegram', root='./tmp_data/')
    assert isinstance(directed_dataset, DirectedData)
    assert directed_dataset.is_weighted
    directed_dataset.to_unweighted()
    assert not directed_dataset.is_weighted
    for dataset_name in ['wikitalk', 'telegram', 'migration']:
        directed_dataset = load_directed_real_data(dataset=dataset_name, root='./tmp_data/'+dataset_name)
        assert isinstance(directed_dataset, DirectedData)
        assert directed_dataset.is_directed
        assert isinstance(directed_dataset.edge_weight, torch.Tensor)
    for year in range(2001, 2020):
        directed_dataset = load_directed_real_data(dataset='lead_lag'+str(year), root='./tmp_data/lead_lag/')
        assert isinstance(directed_dataset, DirectedData)
        assert directed_dataset.is_directed
        assert isinstance(directed_dataset.edge_weight, torch.Tensor)
    directed_dataset = load_directed_real_data(dataset='blog', root='./tmp_data/blog/', pre_transform=T.GCNNorm(), transform=T.ToUndirected())
    assert isinstance(directed_dataset, DirectedData)
    assert not directed_dataset.is_directed
    assert isinstance(directed_dataset.edge_weight, torch.Tensor)
    assert directed_dataset.is_weighted
    directed_dataset.to_unweighted()
    assert not directed_dataset.is_weighted
    directed_dataset = load_directed_real_data(dataset='cora_ml', root='./tmp_data/cora_ml', pre_transform=T.GCNNorm(), transform=T.ToUndirected(), train_size_per_class=20, val_size=500)
    assert isinstance(directed_dataset, DirectedData)
    assert not directed_dataset.is_directed
    assert isinstance(directed_dataset.edge_weight, torch.Tensor)
    directed_dataset = load_directed_real_data(dataset='citeseer', root='./tmp_data/citeseer', pre_transform=T.GCNNorm(), transform=T.ToUndirected(), train_size_per_class=20, val_size=500)
    assert isinstance(directed_dataset, DirectedData)
    assert not directed_dataset.is_directed
    directed_dataset = load_directed_real_data(dataset='wikics', root='./tmp_data/wikics', pre_transform=T.GCNNorm(), transform=T.ToUndirected())
    assert isinstance(directed_dataset, DirectedData)
    assert not directed_dataset.is_directed
    assert isinstance(directed_dataset.edge_weight, torch.Tensor)
    return

def test_link_split():
    """
    Testing link_split()
    """
    directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Texas')
    edges = directed_dataset.edge_index.T.tolist()
    datasets = directed_link_class_split(directed_dataset, prob_val = 0.15, prob_test = 0.05, task = 'direction')
    assert len(list(datasets.keys())) == 10
    for i in datasets:
        for e, l in zip(datasets[i]['train']['edges'], datasets[i]['train']['label']):
            if l == 0:
                assert ([e[0],e[1]] in edges)
            else:
                assert ([e[1],e[0]] in edges)

    datasets = directed_link_class_split(directed_dataset, prob_val = 0.15, prob_test = 0.05, task = 'existence')
    assert len(list(datasets.keys())) == 10
    for i in datasets:
        for e, l in zip(datasets[i]['val']['edges'], datasets[i]['val']['label']):
            if l == 0:
                assert ([e[0],e[1]] in edges)
            else:
                assert not ([e[0],e[1]] in edges)
    datasets = directed_link_class_split(directed_dataset, prob_val = 0.15, prob_test = 0.05, task = 'all')
    for i in datasets:
        for e, l in zip(datasets[i]['test']['edges'], datasets[i]['test']['label']):
            if l == 0:
                assert ([e[0],e[1]] in edges)
            elif l == 1:
                assert ([e[1],e[0]] in edges)
            else:
                assert  not (([e[0],e[1]] in edges) and ([e[1],e[0]] in edges))
    assert len(list(datasets.keys())) == 10
    return

def test_node_split():
    """
    Testing node_class_split()
    """    
    directed_dataset = load_directed_real_data(dataset='WebKB', root='./tmp_data/', name='Texas')
    data = node_class_split(directed_dataset, train_size = 5, val_size = 10, test_size = 15, data_split=3)
    assert isinstance(data.train_mask, torch.Tensor)
    assert data.train_mask.shape[-1] == 3
    assert torch.sum(data.train_mask) == 15
    assert torch.sum(data.val_mask) == 30
    assert torch.sum(data.test_mask) == 45
    
    directed_dataset = load_directed_real_data(dataset='cora_ml', root='./tmp_data/')
    assert directed_dataset.is_directed
    num_classes = len(np.unique(directed_dataset.y))
    data = node_class_split(directed_dataset, train_size_per_class = 20, seed_size_per_class = 0.1, val_size_per_class = 10, test_size_per_class = 20, data_split=3)
    assert data.train_mask.shape[-1] == 3
    assert torch.sum(data.train_mask) == 20*3*num_classes
    assert torch.sum(data.val_mask) == 10*3*num_classes
    assert torch.sum(data.test_mask) == 20*3*num_classes
    assert torch.sum(data.seed_mask) == 2*3*num_classes

    data = node_class_split(directed_dataset, train_size_per_class = 20, seed_size_per_class = 5, val_size_per_class = 10, test_size_per_class = 20)
    assert isinstance(data.seed_mask, torch.Tensor)
    num_classes = len(np.unique(directed_dataset.y))
    assert torch.sum(data.seed_mask) == 10*5*num_classes

    _, counts = np.unique(directed_dataset.y, return_counts=True)
    data = node_class_split(directed_dataset, train_size_per_class = 0.1, val_size_per_class = 0.2, test_size_per_class = 0.3, data_split=3)
    assert data.train_mask.shape[-1] == 3
    train_size = np.sum([int(c*0.1) for c in counts])
    assert torch.sum(data.train_mask) == 3*train_size
    val_size = np.sum([int(c*0.2) for c in counts])
    assert torch.sum(data.val_mask) == 3*val_size
    test_size = np.sum([int(c*0.3) for c in counts])
    assert torch.sum(data.test_mask) == 3*test_size

    data = node_class_split(directed_dataset, train_size_per_class = 0.1, val_size = 50, test_size = 23, data_split=3)
    assert data.train_mask.shape[-1] == 3
    train_size = np.sum([int(c*0.1) for c in counts])
    assert torch.sum(data.train_mask) == 3*train_size
    assert torch.sum(data.val_mask) == 3*50
    assert torch.sum(data.test_mask) == 3*23

    data = node_class_split(directed_dataset, train_size = 0.1, val_size = 0.1, test_size = 0.3, data_split=3)
    assert data.train_mask.shape[-1] == 3
    assert torch.sum(data.train_mask) == 3*int(0.1*len(data.y))
    assert torch.sum(data.val_mask) == 3*int(0.1*(len(data.y)-torch.sum(data.train_mask)/3))
    assert torch.sum(data.test_mask) == 3*int(0.3*(len(data.y)-torch.sum(data.train_mask)/3-torch.sum(data.val_mask)/3))
    return