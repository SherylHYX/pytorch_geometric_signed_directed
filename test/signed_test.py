import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor
from torch_geometric_signed_directed.nn.signed import (
    SSSNET_node_clustering, SDGNN, SGCN_SNEA, SiGAT, SGCNConv
)
from torch_geometric_signed_directed.data import (
    SSBM, SignedData
)
from torch_geometric_signed_directed.utils import (
    Prob_Balanced_Ratio_Loss, Prob_Balanced_Normalized_Loss, Unhappy_Ratio, 
    link_sign_prediction_logistic_function, triplet_loss_node_classification
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_mock_data(num_nodes, num_features, num_classes=3, eta=0.1, p=0.2):
    """
    Creating a mock feature matrix, edge index and edge weight.
    """
    (A_p_scipy, A_n_scipy), labels = SSBM(num_nodes, num_classes, p, eta)
    X = torch.FloatTensor(np.random.uniform(-1, 1, (num_nodes, num_features))).to(device)
    edge_index_p = torch.LongTensor(np.array(A_p_scipy.nonzero())).to(device)
    edge_weight_p = torch.FloatTensor(sp.csr_matrix(A_p_scipy).data).to(device)
    edge_index_n = torch.LongTensor(np.array(A_n_scipy.nonzero())).to(device)
    edge_weight_n = torch.FloatTensor(sp.csr_matrix(A_n_scipy).data).to(device)
    return X, A_p_scipy, A_n_scipy, edge_index_p, edge_index_n, edge_weight_p, edge_weight_n, labels

def test_SSSNET():
    """
    Testing SSSNET
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    X, A_p_scipy, A_n_scipy, edge_index_p, edge_index_n, edge_weight_p, edge_weight_n, labels = \
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
    Z, _, _, prob = model(edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, X) 
    loss_pbrc = loss_func_pbrc(prob=prob).item()
    loss_pbnc = loss_func_pbnc(prob=prob).item()
    triplet_loss = triplet_loss_node_classification(y=labels, Z=Z, n_sample=500, thre=0.1)
    unhappy_ratio = Unhappy_Ratio(A_p_scipy, A_n_scipy)(prob).item()
    assert prob.shape == (
        num_nodes, num_classes
    )
    assert unhappy_ratio < 1.1
    assert loss_pbrc >= 0
    assert loss_pbnc >= 0
    assert triplet_loss.item() >= 0

    model = SSSNET_node_clustering(nfeat=num_features,
                    hidden=16,
                    nclass=num_classes,
                    dropout=0.5,
                    hop=2,
                    fill_value=0.5,
                    directed=True).to(device)
    Z, _, _, prob = model(edge_index_p, None,
                edge_index_n, None, X) 
    triplet_loss = triplet_loss_node_classification(y=torch.LongTensor(labels), Z=Z, n_sample=500, thre=0.1)
    unhappy_ratio = Unhappy_Ratio(A_p_scipy, A_n_scipy)(prob).item()
    assert prob.shape == (
        num_nodes, num_classes
    )
    assert unhappy_ratio < 1.1
    assert loss_pbrc >= 0
    assert loss_pbnc >= 0
    assert triplet_loss.item() >= 0

def test_SGCN_SNEA():
    """
    Testing SGCN and SNEA
    """
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv1 = SGCNConv(16, 32, first_aggr=True)
    assert conv1.__repr__() == 'SGCNConv(16, 32, first_aggr=True)'

    conv2 = SGCNConv(32, 48, first_aggr=False)
    assert conv2.__repr__() == 'SGCNConv(32, 48, first_aggr=False)'

    out1 = conv1(x, edge_index, edge_index)
    assert out1.size() == (4, 64)
    assert conv1(x, adj.t(), adj.t()).tolist() == out1.tolist()

    out2 = conv2(out1, edge_index, edge_index)
    assert out2.size() == (4, 96)
    assert conv2(out1, adj.t(), adj.t()).tolist() == out2.tolist()

    num_nodes = 100
    num_features = 3
    num_classes = 3

    _, A_p_scipy, A_n_scipy, _, _, _, _, _ = \
        create_mock_data(num_nodes, num_features, num_classes)

    data = SignedData(A=(A_p_scipy, A_n_scipy))
    train_edge_index = data.edge_index
    train_edge_weight = data.edge_weight
    test_edge_index = data.edge_index
    test_edge_weight = data.edge_weight
    nodes_num = data.num_nodes
    edge_i_list = train_edge_index.t().cpu().numpy().tolist()
    edge_s_list = train_edge_weight.long().cpu().numpy().tolist()
    edge_index_s = torch.LongTensor([[i, j, s] for (i, j), s in zip(edge_i_list, edge_s_list)]).to(device)

    model = SGCN_SNEA('SGCN', nodes_num, edge_index_s, 20, 20, layer_num=2, lamb=5).to(device)
    loss = model.loss()
    with torch.no_grad():
        z = model()

    embeddings = z.cpu().numpy()
    train_X = train_edge_index.t().cpu().numpy()
    test_X  = test_edge_index.t().cpu().numpy()
    train_y = train_edge_weight.cpu().numpy()
    test_y  = test_edge_weight.cpu().numpy()
    accuracy, f1, f1_macro, f1_micro, auc_score = link_sign_prediction_logistic_function(embeddings, train_X, train_y, test_X, test_y)
    assert auc_score >= 0
    assert loss >= 0
    assert accuracy >= 0
    assert f1 >= 0
    assert f1_macro >= 0
    assert f1_micro >= 0

    model = SGCN_SNEA('SNEA', nodes_num, edge_index_s, 20, 20, layer_num=2, lamb=5).to(device)
    loss = model.loss()
    with torch.no_grad():
        z = model()

    embeddings = z.cpu().numpy()
    train_X = train_edge_index.t().cpu().numpy()
    test_X  = test_edge_index.t().cpu().numpy()
    train_y = train_edge_weight.cpu().numpy()
    test_y  = test_edge_weight.cpu().numpy()
    accuracy, f1, f1_macro, f1_micro, auc_score = link_sign_prediction_logistic_function(embeddings, train_X, train_y, test_X, test_y)
    assert auc_score >= 0
    assert loss >= 0
    assert accuracy >= 0
    assert f1 >= 0
    assert f1_macro >= 0
    assert f1_micro >= 0

def test_SiGAT():
    """
    Testing SiGAT
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    _, A_p_scipy, A_n_scipy, _, _, _, _, _ = \
        create_mock_data(num_nodes, num_features, num_classes)

    data = SignedData(A=(A_p_scipy, A_n_scipy))
    train_edge_index = data.edge_index
    train_edge_weight = data.edge_weight
    test_edge_index = data.edge_index
    test_edge_weight = data.edge_weight
    nodes_num = data.num_nodes
    edge_i_list = train_edge_index.t().cpu().numpy().tolist()
    edge_s_list = train_edge_weight.long().cpu().numpy().tolist()
    edge_index_s = torch.LongTensor([[i, j, s] for (i, j), s in zip(edge_i_list, edge_s_list)]).to(device)

    model = SiGAT(nodes_num, edge_index_s, 20, 20).to(device)
    total_loss = []
    nodes_pku = np.random.permutation(nodes_num).tolist()
    batch_size = 50
    model.train()
    for batch in range(nodes_num // batch_size):
        b_index = batch * batch_size
        e_index = (batch + 1) * batch_size
        nodes = nodes_pku[b_index:e_index]
        loss = model.loss(np.array(nodes))
        total_loss.append(loss.data.cpu().numpy())
    with torch.no_grad():
        nodes = np.arange(0, nodes_num)
        z = model(nodes)
        z = model(torch.from_numpy(nodes).to(device))

    embeddings = z.cpu().numpy()
    train_X = train_edge_index.t().cpu().numpy()
    test_X  = test_edge_index.t().cpu().numpy()
    train_y = train_edge_weight.cpu().numpy()
    test_y  = test_edge_weight.cpu().numpy()
    accuracy, f1, f1_macro, f1_micro, auc_score = link_sign_prediction_logistic_function(embeddings, train_X, train_y, test_X, test_y)
    assert auc_score >= 0
    assert loss >= 0
    assert accuracy >= 0
    assert f1 >= 0
    assert f1_macro >= 0
    assert f1_micro >= 0

def test_SDGNN():
    """
    Testing SDGNN
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    _, A_p_scipy, A_n_scipy, _, _, _, _, _ = \
        create_mock_data(num_nodes, num_features, num_classes)

    data = SignedData(A=(A_p_scipy, A_n_scipy))
    train_edge_index = data.edge_index
    train_edge_weight = data.edge_weight
    test_edge_index = data.edge_index
    test_edge_weight = data.edge_weight
    nodes_num = data.num_nodes
    edge_i_list = train_edge_index.t().cpu().numpy().tolist()
    edge_s_list = train_edge_weight.long().cpu().numpy().tolist()
    edge_index_s = torch.LongTensor([[i, j, s] for (i, j), s in zip(edge_i_list, edge_s_list)]).to(device)

    model = SDGNN(nodes_num, edge_index_s, 20, 20).to(device)
    loss = model.loss()
    with torch.no_grad():
        z = model()

    embeddings = z.cpu().numpy()
    train_X = train_edge_index.t().cpu().numpy()
    test_X  = test_edge_index.t().cpu().numpy()
    train_y = train_edge_weight.cpu().numpy()
    test_y  = test_edge_weight.cpu().numpy()
    accuracy, f1, f1_macro, f1_micro, auc_score = link_sign_prediction_logistic_function(embeddings, train_X, train_y, test_X, test_y)
    assert auc_score >= 0
    assert loss >= 0
    assert accuracy >= 0
    assert f1 >= 0
    assert f1_macro >= 0
    assert f1_micro >= 0







            


