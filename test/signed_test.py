import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric_signed_directed.nn.signed import (
    SSSNET_node_clustering
)
from torch_geometric_signed_directed.data import (
    SSBM, polarized_SSBM, SignedData
)
from torch_geometric_signed_directed.utils import (
    Prob_Balanced_Ratio_Loss, Prob_Balanced_Normalized_Loss, Unhappy_Ratio
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_mock_data(num_nodes, num_features, num_classes=3, eta=0.1, p=0.2):
    """
    Creating a mock feature matrix, edge index and edge weight.
    """
    (A_p_scipy, A_n_scipy), _ = SSBM(num_nodes, num_classes, p, eta)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (num_nodes, num_features))).to(device)
    edge_index_p = torch.LongTensor(np.array(A_p_scipy.nonzero())).to(device)
    edge_weight_p = torch.FloatTensor(sp.csr_matrix(A_p_scipy).data).to(device)
    edge_index_n = torch.LongTensor(np.array(A_n_scipy.nonzero())).to(device)
    edge_weight_n = torch.FloatTensor(sp.csr_matrix(A_n_scipy).data).to(device)
    return X, A_p_scipy, A_n_scipy, edge_index_p, edge_index_n, edge_weight_p, edge_weight_n




def test_SSSNET():
    """
    Testing SSSNET
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    X, A_p_scipy, A_n_scipy, edge_index_p, edge_index_n, edge_weight_p, edge_weight_n = \
        create_mock_data(num_nodes, num_features, num_classes)

    loss_func_pbrc = Prob_Balanced_Ratio_Loss(A_p=A_p_scipy, A_n=A_n_scipy)
    loss_func_pbnc = Prob_Balanced_Normalized_Loss(A_p=A_p_scipy, A_n=A_n_scipy)

    model = SSSNET_node_clustering(nfeat=num_features,
                    hidden=8,
                    nclass=num_classes,
                    dropout=0.5,
                    hop=2,
                    fill_value=0.5,
                    directed=False).to(device)
    _, _, _, prob = model(edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, X) 
    loss_pbrc = loss_func_pbrc(prob=prob).item()
    loss_pbnc = loss_func_pbnc(prob=prob).item()
    unhappy_ratio = Unhappy_Ratio(A_p_scipy, A_n_scipy)(prob).item()
    assert prob.shape == (
        num_nodes, num_classes
    )
    assert unhappy_ratio < 1.1
    assert loss_pbrc >= 0
    assert loss_pbnc >= 0

    model = SSSNET_node_clustering(nfeat=num_features,
                    hidden=16,
                    nclass=num_classes,
                    dropout=0.5,
                    hop=2,
                    fill_value=0.5,
                    directed=True).to(device)
    _, _, _, prob = model(edge_index_p, None,
                edge_index_n, None, X) 
    assert prob.shape == (
        num_nodes, num_classes
    )

def test_SSBM():
    num_nodes = 1000
    num_classes = 3
    p = 0.1
    eta = 0.1
    (A_p, _), labels = SSBM(num_nodes, num_classes, p, eta, size_ratio = 1.0, values='gaussian')
    assert A_p.shape == (num_nodes, num_nodes)
    assert np.max(labels) == num_classes - 1

    (A_p, _), labels = SSBM(num_nodes, num_classes, p, eta, size_ratio = 2.0, values='exp')
    assert A_p.shape == (num_nodes, num_nodes)
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
    (A_p, A_n), labels = SSBM(num_nodes, num_classes, p, eta, size_ratio = 1.0, values='gaussian')
    data = SignedData(y=labels, A=(A_p, A_n))
    assert data.is_signed
    assert data.A.shape == (num_nodes, num_nodes)
    assert data.A_p.shape == (num_nodes, num_nodes)
    assert data.A_n.shape == (num_nodes, num_nodes)
    assert data.A_p.shape == (num_nodes, num_nodes)
    data = SignedData(y=labels, A=A_p-A_n)
    assert data.y.shape == labels.shape
    assert data.edge_index_p[0].shape == data.A_p.nonzero()[0].shape
    assert data.edge_index_n[0].shape == data.A_n.nonzero()[0].shape
    assert data.edge_weight_p.shape == data.A_p.data.shape
    assert data.edge_weight_n.shape == data.A_n.data.shape
    assert data.A.shape == (num_nodes, num_nodes)
    data.clear_separate_storage()
    assert data.edge_index_n[0].shape == data.A_n.nonzero()[0].shape
    data2 = SignedData(edge_index=data.edge_index, edge_weight=data.edge_weight)
    data2.set_signed_Laplacian_features(k=2*num_classes)
    assert data2.x.shape == (num_nodes, 2*num_classes)
    assert data2.A_n.shape == (num_nodes, num_nodes)
    data2.set_spectral_adjacency_reg_features(k=num_classes,normalization='sym')
    assert data2.x.shape == (num_nodes, num_classes)
    assert data.edge_weight_p.shape == data.A_p.data.shape
    data2.set_spectral_adjacency_reg_features(k=num_classes,normalization='sym_sep')
    assert data2.x.shape == (num_nodes, num_classes)
    data2.set_spectral_adjacency_reg_features(k=num_classes)
    assert data2.x.shape == (num_nodes, num_classes)
    assert data.edge_weight_n.shape == data.A_n.data.shape
            


