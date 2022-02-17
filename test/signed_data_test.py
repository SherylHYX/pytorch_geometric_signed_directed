import numpy as np
import torch

from torch_geometric_signed_directed.data import (
    SSBM, polarized_SSBM, SignedData, load_signed_real_data, SignedDirectedGraphDataset
)

def test_load_signed_real_data():
    signed_dataset = load_signed_real_data(root='./tmp_data/', dataset='epinions')
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    signed_dataset = load_signed_real_data(root='./tmp_data/', dataset='bitcoin_alpha')
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    signed_dataset = load_signed_real_data(root='./tmp_data/', dataset='bitcoin_otc')
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    signed_dataset = load_signed_real_data(root='./tmp_data/Sampson/', dataset='Sampson', train_size=15, val_size=5)
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    for dataset_name in ['PPI', 'wikirfa', 'SP1500', 'rainfall']:
        signed_dataset = load_signed_real_data(root='./tmp_data/'+dataset_name+'/', dataset=dataset_name)
        assert isinstance(signed_dataset, SignedData)
        assert signed_dataset.is_signed
    for year in range(2001, 2021):
        signed_dataset = load_signed_real_data(dataset='Fin_YNet'+str(year), root='./tmp_data/Fin_YNet/')
        assert isinstance(signed_dataset, SignedData)
        assert signed_dataset.is_signed

def test_SignedDirectedGraphDataset():
    dataset_node_edge_dict = {
        'bitcoin_alpha': (3783, 22650, 1536),
        'bitcoin_otc': (5881, 32029, 3563),
        'slashdot': (82140, 380933, 119548),
        'epinions': (131580, 589888, 121322)
    }
    for dataset_name, (node, pos_edge, neg_edge) in dataset_node_edge_dict.items():
        path = './tmp_data/' + dataset_name
        dataset = SignedDirectedGraphDataset(path, dataset_name)
        data = dataset[0]
        assert len(data.edge_weight) > 0
        assert data.train_edge_index.shape[1] == len(data.train_edge_weight)
        assert data.test_edge_index.shape[1] == len(data.test_edge_weight)
        assert data.train_edge_index.shape[1] + data.test_edge_index.shape[1] == len(data.edge_weight)
        pos = (data.edge_weight > 0).sum()
        neg = (data.edge_weight < 0).sum()
        assert pos.item() == pos_edge
        assert neg.item() == neg_edge
        assert data.num_nodes == node

        dataset2 = SignedDirectedGraphDataset(path, dataset_name)
        data2 = dataset2[0]
        assert torch.all(torch.eq(data.train_edge_index, data2.train_edge_index))

        dataset3 = SignedDirectedGraphDataset(path, dataset_name, seed=2022)
        data3 = dataset3[0]
        assert not torch.all(torch.eq(data.train_edge_index, data3.train_edge_index))


def test_SSBM():
    num_nodes = 1000
    num_classes = 3
    p = 0.1
    eta = 0.1

    (A_p, A_n), labels = SSBM(num_nodes, num_classes, p, eta, size_ratio = 2.0, values='exp')
    assert A_p.shape == (num_nodes, num_nodes)
    assert A_p.min() >= 0
    assert A_n.min() >= 0
    assert np.max(labels) == num_classes - 1

    (A_p, _), labels = SSBM(num_nodes, num_classes, p, eta, size_ratio = 1.5, values='uniform')
    assert A_p.shape == (num_nodes, num_nodes)
    assert np.max(labels) == num_classes - 1

def test_polarized():
    total_n = 1000
    N = 200
    num_com = 3
    K = 2
    (A_p, _), labels, conflict_groups = polarized_SSBM(total_n=total_n, num_com=num_com, N=N, K=K, \
        p=0.1, eta=0.1, size_ratio=1.5)
    assert A_p.shape == (total_n, total_n)
    assert np.max(labels) == num_com*K
    assert np.max(conflict_groups) == num_com

    (A_p, _), _, _ = polarized_SSBM(total_n=total_n, num_com=num_com, N=N, K=K, \
        p=0.002, eta=0.1, size_ratio=1)
    assert A_p.shape[1] <= total_n

def test_SignedData():
    num_nodes = 400
    num_classes = 3
    p = 0.1
    eta = 0.1
    (A_p, A_n), labels = SSBM(num_nodes, num_classes, p, eta, size_ratio = 1.0, values='exp')
    data = SignedData(y=labels, A=(A_p, A_n))
    assert data.is_signed
    data.separate_positive_negative()
    assert data.A.shape == (num_nodes, num_nodes)
    assert data.A_p.shape == (num_nodes, num_nodes)
    assert data.A_n.shape == (num_nodes, num_nodes)
    assert data.edge_index_p[0].shape == A_p.nonzero()[0].shape
    assert data.edge_index_n[0].shape == A_n.nonzero()[0].shape
    assert data.edge_weight_p.shape == A_p.data.shape
    assert data.edge_weight_n.shape == A_n.data.shape
    
   
    data = SignedData(y=labels, A=A_p-A_n, init_data=data)
    assert data.y.shape == labels.shape
    data.separate_positive_negative()
    assert data.edge_index_p[0].shape == A_p.nonzero()[0].shape
    assert data.edge_index_n[0].shape == A_n.nonzero()[0].shape
    assert data.edge_weight_p.shape == A_p.data.shape
    assert data.edge_weight_n.shape == A_n.data.shape
    assert data.A.shape == (num_nodes, num_nodes)
    assert data.A_p.shape == (num_nodes, num_nodes)
    assert data.A_n.shape == (num_nodes, num_nodes)
    data.node_split(train_size=0.8, val_size=0.1, test_size=0.1, seed_size=0.1)
    assert data.seed_mask.sum() == 0.1*num_nodes*10*0.8
    data.node_split(train_size=80, val_size=10, test_size=10, seed_size=8)
    assert data.seed_mask.sum() == 10*8
    data2 = SignedData(edge_index=data.edge_index, edge_weight=data.edge_weight)
    data2.set_signed_Laplacian_features(k=2*num_classes)
    assert data2.x.shape == (num_nodes, 2*num_classes)
    data2.set_spectral_adjacency_reg_features(k=num_classes,normalization='sym')
    assert data2.x.shape == (num_nodes, num_classes)
    data2.set_spectral_adjacency_reg_features(k=num_classes,normalization='sym_sep')
    assert data2.x.shape == (num_nodes, num_classes)
    data2.set_spectral_adjacency_reg_features(k=num_classes)
    assert data2.x.shape == (num_nodes, num_classes)
    data.separate_positive_negative()
    assert data.edge_index_p[0].shape == A_p.nonzero()[0].shape
