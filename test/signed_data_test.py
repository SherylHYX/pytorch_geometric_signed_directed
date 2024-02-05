import numpy as np
import torch

from torch_geometric_signed_directed.data import (
    SSBM, polarized_SSBM, SignedData, load_signed_real_data
)
from torch_geometric_signed_directed.utils import link_class_split, in_out_degree

def test_in_out_degree():
    signed_dataset = load_signed_real_data(
        root='./tmp_data/', dataset='bitcoin_alpha')
    degrees = in_out_degree(signed_dataset.edge_index,
                            size=signed_dataset.num_nodes, signed=False)
    assert degrees.shape == (signed_dataset.num_nodes, 2)
    assert degrees.min() >= 0
    degrees = in_out_degree(signed_dataset.edge_index,
                            size=signed_dataset.num_nodes,
                            edge_weight= signed_dataset.edge_weight, signed=True)
    assert degrees.shape == (signed_dataset.num_nodes, 4)
    assert degrees.min() >= 0

def test_sign_link_split():
    signed_dataset = load_signed_real_data(
        root='./tmp_data/', dataset='bitcoin_alpha')
    datasets = signed_dataset.link_split(
        splits=2, prob_val=0.01, prob_test=0.02, ratio=0.2)
    assert len(list(datasets.keys())) == 2
    assert signed_dataset.is_weighted
    assert signed_dataset.is_signed
    assert signed_dataset.is_directed
    
    datasets = link_class_split(signed_dataset, prob_val=0.01, prob_test=0.01, task='five_class_signed_digraph',
                                maintain_connect=True, ratio=1)
    A = signed_dataset.A.tocsr()
    assert len(list(datasets.keys())) == 2
    for i in datasets:
        assert torch.sum(datasets[i]['train']['label'] == 0) > 0
        assert torch.sum(datasets[i]['train']['label'] == 1) > 0
        assert torch.sum(datasets[i]['train']['label'] == 2) > 0
        assert torch.sum(datasets[i]['train']['label'] == 3) > 0
        assert torch.sum(datasets[i]['train']['label'] == 4) > 0

        assert torch.sum(datasets[i]['test']['label'] == 0) > 0
        assert torch.sum(datasets[i]['test']['label'] == 1) > 0
        assert torch.sum(datasets[i]['test']['label'] == 2) > 0
        assert torch.sum(datasets[i]['test']['label'] == 3) > 0
        assert torch.sum(datasets[i]['test']['label'] == 4) > 0
        
        assert torch.sum(datasets[i]['val']['label'] == 0) > 0
        assert torch.sum(datasets[i]['val']['label'] == 1) > 0
        assert torch.sum(datasets[i]['val']['label'] == 2) > 0
        assert torch.sum(datasets[i]['val']['label'] == 3) > 0
        assert torch.sum(datasets[i]['val']['label'] == 4) > 0
        
        for e, l in zip(datasets[i]['train']['edges'], datasets[i]['train']['label']):
            if l == 0:
                assert A[e[0], e[1]] > 0
            elif l == 1:
                assert A[e[0], e[1]] < 0
            elif l == 2:
                assert A[e[1], e[0]] > 0
            elif l == 3:
                assert A[e[1], e[0]] < 0
            elif l == 4:
                assert A[e[1], e[0]] == 0
                assert A[e[0], e[1]] == 0

        for e, l in zip(datasets[i]['test']['edges'], datasets[i]['test']['label']):
            if l == 0:
                assert A[e[0], e[1]] > 0
            elif l == 1:
                assert A[e[0], e[1]] < 0
            elif l == 2:
                assert A[e[1], e[0]] > 0
            elif l == 3:
                assert A[e[1], e[0]] < 0
            elif l == 4:
                assert A[e[1], e[0]] == 0
                assert A[e[0], e[1]] == 0

        for e, l in zip(datasets[i]['val']['edges'], datasets[i]['val']['label']):
            if l == 0:
                assert A[e[0], e[1]] > 0
            elif l == 1:
                assert A[e[0], e[1]] < 0
            elif l == 2:
                assert A[e[1], e[0]] > 0
            elif l == 3:
                assert A[e[1], e[0]] < 0
            elif l == 4:
                assert A[e[1], e[0]] == 0
                assert A[e[0], e[1]] == 0

    edges = signed_dataset.edge_index.T.tolist()
    A = signed_dataset.A.tocsr()
    for i in datasets:
        for e, l in zip(datasets[i]['graph'].T, datasets[i]['weights']):
            assert(abs(l) > 0)
            assert([e[0].item(), e[1].item()] in edges)
            assert(A[e[0].item(), e[1].item()] == l)

    datasets = link_class_split(signed_dataset, prob_val=0.01, prob_test=0.01, task='sign',
                                maintain_connect=True, ratio=1)
    A = signed_dataset.A.tocsr()
    assert len(list(datasets.keys())) == 2
    for i in datasets:
        assert torch.sum(datasets[i]['train']['label'] == 0) > 0
        assert torch.sum(datasets[i]['train']['label'] != 0) > 0
        assert torch.sum(datasets[i]['train']['label'] > 1) == 0

        assert torch.sum(datasets[i]['test']['label'] == 0) > 0
        assert torch.sum(datasets[i]['test']['label'] != 0) > 0
        assert torch.sum(datasets[i]['test']['label'] > 1) == 0

        assert torch.sum(datasets[i]['val']['label'] == 0) > 0
        assert torch.sum(datasets[i]['val']['label'] != 0) > 0
        assert torch.sum(datasets[i]['val']['label'] > 1) == 0

        for e, l in zip(datasets[i]['train']['edges'], datasets[i]['train']['label']):
            if l == 0:
                assert A[e[0], e[1]] < 0
            else:
                assert A[e[0], e[1]] > 0
        for e, l in zip(datasets[i]['test']['edges'], datasets[i]['test']['label']):
            if l == 0:
                assert A[e[0], e[1]] < 0
            else:
                assert A[e[0], e[1]] > 0
        for e, l in zip(datasets[i]['val']['edges'], datasets[i]['val']['label']):
            if l == 0:
                assert A[e[0], e[1]] < 0
            else:
                assert A[e[0], e[1]] > 0
    edges = signed_dataset.edge_index.T.tolist()
    A = signed_dataset.A.tocsr()
    for i in datasets:
        for e, l in zip(datasets[i]['graph'].T, datasets[i]['weights']):
            assert(abs(l) > 0)
            assert([e[0].item(), e[1].item()] in edges)
            assert(A[e[0].item(), e[1].item()] == l)

    datasets = link_class_split(signed_dataset, prob_val=0.01, prob_test=0.01, task='four_class_signed_digraph',
                                maintain_connect=False, ratio=0.2)
    
    A = signed_dataset.A.tocsr()
    assert len(list(datasets.keys())) == 2
    for i in datasets:
        assert torch.sum(datasets[i]['train']['label'] == 0) > 0
        assert torch.sum(datasets[i]['train']['label'] == 1) > 0
        assert torch.sum(datasets[i]['train']['label'] == 2) > 0
        assert torch.sum(datasets[i]['train']['label'] == 3) > 0
        assert torch.sum(datasets[i]['train']['label'] > 3) == 0

        assert torch.sum(datasets[i]['test']['label'] == 0) > 0
        assert torch.sum(datasets[i]['test']['label'] == 1) > 0
        assert torch.sum(datasets[i]['test']['label'] == 2) > 0
        assert torch.sum(datasets[i]['test']['label'] == 3) > 0
        assert torch.sum(datasets[i]['test']['label'] > 3) == 0
        
        assert torch.sum(datasets[i]['val']['label'] == 0) > 0
        assert torch.sum(datasets[i]['val']['label'] == 1) > 0
        assert torch.sum(datasets[i]['val']['label'] == 2) > 0
        assert torch.sum(datasets[i]['val']['label'] == 3) > 0
        assert torch.sum(datasets[i]['val']['label'] > 3) == 0
        for e, l in zip(datasets[i]['train']['edges'], datasets[i]['train']['label']):
            if l == 0:
                assert A[e[0], e[1]] > 0
            elif l == 1:
                assert A[e[0], e[1]] < 0
            elif l == 2:
                assert A[e[1], e[0]] > 0
            elif l == 3:
                assert A[e[1], e[0]] < 0
            else:
                assert A[e[1], e[0]] == 0
        for e, l in zip(datasets[i]['test']['edges'], datasets[i]['test']['label']):
            if l == 0:
                assert A[e[0], e[1]] > 0
            elif l == 1:
                assert A[e[0], e[1]] < 0
            elif l == 2:
                assert A[e[1], e[0]] > 0
            elif l == 3:
                assert A[e[1], e[0]] < 0
            else:
                assert A[e[1], e[0]] == 0
        for e, l in zip(datasets[i]['val']['edges'], datasets[i]['val']['label']):
            if l == 0:
                assert A[e[0], e[1]] > 0
            elif l == 1:
                assert A[e[0], e[1]] < 0
            elif l == 2:
                assert A[e[1], e[0]] > 0
            elif l == 3:
                assert A[e[1], e[0]] < 0
            else:
                assert A[e[1], e[0]] == 0
    edges = signed_dataset.edge_index.T.tolist()
    A = signed_dataset.A.tocsr()
    for i in datasets:
        for e, l in zip(datasets[i]['graph'].T, datasets[i]['weights']):
            assert(abs(l) > 0)
            assert([e[0].item(), e[1].item()] in edges)
            assert(A[e[0].item(), e[1].item()] == l)

    datasets = link_class_split(signed_dataset, prob_val=0.1, prob_test=0.2, task='sign',
                                maintain_connect=False, ratio=1.0)
    A = signed_dataset.A.tocsr()
    assert len(list(datasets.keys())) == 2
    for i in datasets:
        assert torch.sum(datasets[i]['train']['label'] == 0) > 0
        assert torch.sum(datasets[i]['train']['label'] != 0) > 0
        assert torch.sum(datasets[i]['test']['label'] == 0) > 0
        assert torch.sum(datasets[i]['test']['label'] != 0) > 0
        assert torch.sum(datasets[i]['val']['label'] == 0) > 0
        assert torch.sum(datasets[i]['val']['label'] != 0) > 0
        for e, l in zip(datasets[i]['train']['edges'], datasets[i]['train']['label']):
            if l == 0:
                assert A[e[0], e[1]] < 0
            else:
                assert A[e[0], e[1]] > 0
        for e, l in zip(datasets[i]['test']['edges'], datasets[i]['test']['label']):
            if l == 0:
                assert A[e[0], e[1]] < 0
            else:
                assert A[e[0], e[1]] > 0
        for e, l in zip(datasets[i]['val']['edges'], datasets[i]['val']['label']):
            if l == 0:
                assert A[e[0], e[1]] < 0
            else:
                assert A[e[0], e[1]] > 0

    edges = signed_dataset.edge_index.T.tolist()
    A = signed_dataset.A.tocsr()
    for i in datasets:
        for e, l in zip(datasets[i]['graph'].T, datasets[i]['weights']):
            assert(abs(l) > 0)
            assert([e[0].item(), e[1].item()] in edges)
            assert(A[e[0].item(), e[1].item()] == l)

def test_load_signed_real_data():
    for year in range(2000, 2021):
        signed_dataset = load_signed_real_data(
            dataset='FiLL-pvCLCL'+str(year), root='./tmp_data/FiLL/', sparsify_level=0.2)
        assert isinstance(signed_dataset, SignedData)
        assert signed_dataset.is_signed
        assert signed_dataset.is_directed
    for year in range(2000, 2021):
        signed_dataset = load_signed_real_data(
            dataset='FiLL-OPCL'+str(year), root='./tmp_data/FiLL/')
        assert isinstance(signed_dataset, SignedData)
        assert signed_dataset.is_signed
        assert signed_dataset.is_directed
    signed_dataset = load_signed_real_data(
        root='./tmp_data/', dataset='epinions')
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    assert not signed_dataset.is_weighted
    signed_dataset = load_signed_real_data(
        root='./tmp_data/', dataset='slashdot')
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    assert not signed_dataset.is_weighted
    signed_dataset = load_signed_real_data(
        root='./tmp_data/', dataset='bitcoin_alpha')
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    assert signed_dataset.is_weighted
    signed_dataset.to_unweighted()
    assert not signed_dataset.is_weighted
    signed_dataset = load_signed_real_data(
        root='./tmp_data/', dataset='bitcoin_otc')
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    assert signed_dataset.is_weighted
    signed_dataset.separate_positive_negative()
    signed_dataset.to_unweighted()
    assert not signed_dataset.is_weighted
    signed_dataset = load_signed_real_data(
        root='./tmp_data/Sampson/', dataset='Sampson', train_size=15, val_size=5)
    assert isinstance(signed_dataset, SignedData)
    assert signed_dataset.is_signed
    for dataset_name in ['PPI', 'wikirfa', 'SP1500', 'rainfall']:
        signed_dataset = load_signed_real_data(
            root='./tmp_data/'+dataset_name+'/', dataset=dataset_name)
        assert isinstance(signed_dataset, SignedData)
        assert signed_dataset.is_signed
    for year in range(2000, 2021):
        signed_dataset = load_signed_real_data(
            dataset='Fin_YNet'+str(year), root='./tmp_data/Fin_YNet/')
        assert isinstance(signed_dataset, SignedData)
        assert signed_dataset.is_signed


def test_connectivity():
    seed = 0
    dataset_name = 'bitcoin_otc'

    # Load data using torch geometric signed directed data loader
    data = load_signed_real_data(dataset=dataset_name)

    # Create several train, val, test splits

    signed_datasets = data.link_split(prob_val=0.1,
                                      prob_test=0.1,
                                      task='sign',
                                      maintain_connect=True,
                                      seed=seed,
                                      splits=1)

    # check that all nodes in validation and test have at least an edge in the training set
    for split_id in signed_datasets:
        val_nodes = torch.unique(torch.flatten(signed_datasets[split_id]['val']['edges']))
        for node_id in val_nodes:
            node_id_mask = torch.logical_or(signed_datasets[split_id]['train']['edges'][:, 0] == node_id,
                                            signed_datasets[split_id]['train']['edges'][:, 1] == node_id)
            assert node_id_mask.sum().item() > 0, f'[VAL] node id: {node_id} has no incident edges in training set'
        test_nodes = torch.unique(torch.flatten(signed_datasets[split_id]['test']['edges']))
        for node_id in test_nodes:
            node_id_mask = torch.logical_or(signed_datasets[split_id]['train']['edges'][:, 0] == node_id,
                                            signed_datasets[split_id]['train']['edges'][:, 1] == node_id)

            assert node_id_mask.sum().item() > 0, f'[TEST] node id: {node_id} has no incident edges in training set'

def test_SSBM():
    num_nodes = 1000
    num_classes = 3
    p = 0.1
    eta = 0.1

    (A_p, A_n), labels = SSBM(num_nodes, num_classes,
                              p, eta, size_ratio=2.0, values='exp')
    assert A_p.shape == (num_nodes, num_nodes)
    assert A_p.min() >= 0
    assert A_n.min() >= 0
    assert np.max(labels) == num_classes - 1

    (A_p, _), labels = SSBM(num_nodes, num_classes,
                            p, eta, size_ratio=1.5, values='uniform')
    assert A_p.shape == (num_nodes, num_nodes)
    assert np.max(labels) == num_classes - 1


def test_polarized():
    total_n = 1000
    N = 200
    num_com = 3
    K = 2
    (A_p, _), labels, conflict_groups = polarized_SSBM(total_n=total_n, num_com=num_com, N=N, K=K,
                                                       p=0.1, eta=0.1, size_ratio=1.5)
    assert A_p.shape == (total_n, total_n)
    assert np.max(labels) == num_com*K
    assert np.max(conflict_groups) == num_com

    (A_p, _), _, _ = polarized_SSBM(total_n=total_n, num_com=num_com, N=N, K=K,
                                    p=0.002, eta=0.1, size_ratio=1)
    assert A_p.shape[1] <= total_n


def test_SignedData():
    num_nodes = 400
    num_classes = 3
    p = 0.1
    eta = 0.1
    (A_p, A_n), labels = SSBM(num_nodes, num_classes,
                              p, eta, size_ratio=1.0, values='exp')
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
    assert not data.is_directed
    assert data.is_weighted
    data.separate_positive_negative()
    assert data.edge_index_p[0].shape == A_p.nonzero()[0].shape
    assert data.edge_index_n[0].shape == A_n.nonzero()[0].shape
    assert data.edge_weight_p.shape == A_p.data.shape
    assert data.edge_weight_n.shape == A_n.data.shape
    assert data.A.shape == (num_nodes, num_nodes)
    assert data.A_p.shape == (num_nodes, num_nodes)
    assert data.A_n.shape == (num_nodes, num_nodes)
    data.node_split(train_size=0.8, val_size=0.1, test_size=0.1, seed_size=0.1)
    assert data.seed_mask.sum() == 0.1*num_nodes*2*0.8
    data.node_split(train_size=80, val_size=10, test_size=10, seed_size=8)
    assert data.seed_mask.sum() == 2*8
    data2 = SignedData(edge_index=data.edge_index,
                       edge_weight=data.edge_weight)
    data2.set_signed_Laplacian_features(k=2*num_classes)
    assert data2.x.shape == (num_nodes, 2*num_classes)
    data2.set_spectral_adjacency_reg_features(
        k=num_classes, normalization='sym')
    assert data2.x.shape == (num_nodes, num_classes)
    data2.set_spectral_adjacency_reg_features(
        k=num_classes, normalization='sym_sep')
    assert data2.x.shape == (num_nodes, num_classes)
    data2.set_spectral_adjacency_reg_features(k=num_classes)
    assert data2.x.shape == (num_nodes, num_classes)
    data.separate_positive_negative()
    assert data.edge_index_p[0].shape == A_p.nonzero()[0].shape
