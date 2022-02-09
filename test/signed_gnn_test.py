
import os.path as osp
from torch_geometric_signed_directed.datasets import SignedDirectedGraph
from torch_geometric.datasets import Planetoid

def test_cora_dataset():
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset)
    assert len(dataset)


def test_dataset():
    dataset_name = 'bitcoin_otc'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
    dataset = SignedDirectedGraph(path, dataset_name)
    data = dataset[0]
    assert len(data.edge_sign) > 0
    assert data.train_edge_index.shape[1] == len(data.train_edge_sign)
    assert data.test_edge_index.shape[1] == len(data.test_edge_sign)
    assert data.train_edge_index.shape[1] +  data.test_edge_index.shape[1] == len(data.edge_sign)
    import ipdb; ipdb.set_trace()
    print(data.train_edge_index[10])
    
