
import os.path as osp
from torch_geometric_signed_directed.data.signed import SignedDirectedGraphDataset


def test_dataset():
    dataset_name = 'bitcoin_otc'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
    dataset = SignedDirectedGraphDataset(path, dataset_name)
    data = dataset[0]
    assert len(data.edge_weight) > 0
    assert data.train_edge_index.shape[1] == len(data.train_edge_weight)
    assert data.test_edge_index.shape[1] == len(data.test_edge_weight)
    assert data.train_edge_index.shape[1] + data.test_edge_index.shape[1] == len(data.edge_weight)
    pos = (data.edge_weight > 0).sum()
    neg = (data.edge_weight < 0).sum()
    assert pos.item() == 32029
    assert neg.item() == 3563
