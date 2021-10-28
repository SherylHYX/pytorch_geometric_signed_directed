import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric_signed_directed.nn.directed import (
    DIGRAC, MagNet
)
from torch_geometric_signed_directed.data import (
    DSBM, meta_graph_generation, fix_network
)
from torch_geometric_signed_directed.utils import (
    Prob_Imbalance_Loss, scipy_sparse_to_torch_sparse
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_mock_data(num_nodes, num_features, num_classes=3, F_style='cyclic', eta=0.1, p=0.2):
    """
    Creating a mock feature matrix, edge index and edge weight.
    """
    F = meta_graph_generation(F_style, num_classes, eta, False, 0)
    F_data = meta_graph_generation(F_style, num_classes, eta, False, 0.5)
    A, labels = DSBM(N=num_nodes, K=num_classes, p=p, F=F_data, size_ratio=1.5)
    A, labels = fix_network(A, labels)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (num_nodes, num_features))).to(device)
    edge_index = torch.LongTensor(np.array(A.nonzero())).to(device)
    edge_weight = torch.FloatTensor(sp.csr_matrix(A).data).to(device)
    return X, A, F, F_data, edge_index, edge_weight




def test_DIGRAC():
    """
    Testing DIGRAC
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    X, A, F, _, edge_index, edge_weight = \
        create_mock_data(num_nodes, num_features, num_classes)
    A = scipy_sparse_to_torch_sparse(A).to(device)

    prob_imbalance_loss = Prob_Imbalance_Loss(F)

    model = DIGRAC(nfeat=num_features,
                    hidden=8,
                    nclass=num_classes,
                    dropout=0.5,
                    hop=2,
                    fill_value=0.5).to(device)
    _, _, _, prob = model(edge_index, edge_weight, X) 
    loss1 = prob_imbalance_loss(prob, A, num_classes, 'vol_sum', 'sort').item()
    loss2 = prob_imbalance_loss(prob, A, num_classes, 'vol_min', 'std').item()
    loss3 = prob_imbalance_loss(prob, A, num_classes, 'vol_max', 'std').item()
    loss4 = prob_imbalance_loss(prob, A, num_classes, 'plain', 'std').item()
    loss5 = prob_imbalance_loss(prob, A, num_classes, 'vol_sum', 'std').item()
    loss6 = prob_imbalance_loss(prob, A, num_classes, 'plain', 'naive').item()
    prob_imbalance_loss = Prob_Imbalance_Loss(3)
    loss7 = prob_imbalance_loss(prob, A, num_classes, 'plain', 'sort').item()
    assert prob.shape == (
        num_nodes, num_classes
    )
    assert loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 >= 0

def test_MagNet():
    """
    Testing MagNet
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    X, _, _, _, edge_index, edge_weight = \
        create_mock_data(num_nodes, num_features, num_classes)

    model = MagNet(X.shape[1], K = 1, label_dim=num_classes, layer = 2, \
                                activation = True, num_filter = 2, dropout=0.5).to(device)  
    preds = model(X, X, 0.25, edge_index, edge_weight) 
    
    assert preds.shape == (
        num_nodes, num_classes
    )
    
def test_DSBM():
    num_nodes = 200
    num_classes = 3
    p = 0.01
    eta = 0.1
    F = meta_graph_generation(F_style='cyclic', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    F = meta_graph_generation(F_style='path', K=num_classes, eta=0.0, ambient=True, fill_val=0.0)
    assert F.shape == (num_classes, num_classes)
    F = meta_graph_generation(F_style='complete', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    
    num_classes = 2
    F = meta_graph_generation(F_style='complete', K=num_classes, eta=eta, ambient=False, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    F = meta_graph_generation(F_style='cyclic', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    F = meta_graph_generation(F_style='cyclic', K=num_classes, eta=eta, ambient=False, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)

    num_classes = 6
    F = meta_graph_generation(F_style='star', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    num_classes = 5
    F = meta_graph_generation(F_style='star', K=num_classes, eta=eta, ambient=False, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    num_classes = 10
    F = meta_graph_generation(F_style='multipartite', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    num_classes = 9
    F = meta_graph_generation(F_style='multipartite', K=num_classes, eta=eta, ambient=False, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)

    A, _ = DSBM(N=num_nodes, K=num_classes, p=p, F=F, size_ratio=1)
    assert A.shape[1] <= num_nodes
