import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric_signed_directed.nn import (
    MSGNN_node_classification,
    MSGNN_link_prediction,
    SSSNET_link_prediction,
    SGCN
)
from torch_geometric_signed_directed.data import (
    SDSBM, SignedData
)
from torch_geometric_signed_directed.utils import (
    extract_network, meta_graph_generation,
    link_class_split, get_magnetic_signed_Laplacian,
    link_sign_direction_prediction_logistic_function
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mock_data(num_nodes, num_features, num_classes=3, F_style='cyclic', eta=0.1, p=0.2, size_ratio=1.5):
    """
    Creating a mock feature matrix, edge index and edge weight.
    """
    F_data = meta_graph_generation(F_style, num_classes, eta, False, 0.5)
    for i in range(num_classes):
        for j in range(num_classes):
            if (i + j) % 2:
                F_data[i, j] = - F_data[i, j] # flip the signs

    A, labels = SDSBM(N=num_nodes, K=num_classes, p=p, F=F_data, eta=eta, size_ratio=size_ratio)
    A, labels = extract_network(A=A, labels=labels)
    X = torch.FloatTensor(
        np.random.uniform(-1, 1, (num_nodes, num_features))).to(device)
    edge_index = torch.LongTensor(np.array(A.nonzero())).to(device)
    edge_weight = torch.FloatTensor(sp.csr_matrix(A).data).to(device)
    return X, A, F_data, edge_index, edge_weight


def test_link_sign_direction_logistic_function():
    """
    Testing link sign direction logistic function
    """
    num_nodes = 100
    num_features = 3
    num_classes = 4

    X, _, _, edge_index, edge_weight = \
        create_mock_data(num_nodes, num_features, num_classes)
    data = SignedData(x=X, edge_index=edge_index, edge_weight=edge_weight)
    link_data = link_class_split(data, splits=2, task="four_class_signed_digraph", prob_val=0.15, prob_test=0.1, seed=10, device=device)

    train_edge_index = link_data[0]['graph']
    train_edge_weight = link_data[0]['weights']
    nodes_num = num_nodes
    edge_i_list = train_edge_index.t().cpu().numpy().tolist()
    edge_s_list = train_edge_weight.long().cpu().numpy().tolist()
    edge_index_s = torch.LongTensor(
        [[i, j, s] for (i, j), s in zip(edge_i_list, edge_s_list)]).to(device)
    split = 0
    query_edges = link_data[split]['train']['edges']
    y = link_data[split]['train']['label']
    query_test_edges = link_data[split]['test']['edges']
    y_test = link_data[split]['test']['label']  

    model = SGCN(nodes_num, edge_index_s, 20, 20,
                 layer_num=2, lamb=5).to(device)
    loss = model.loss()
    with torch.no_grad():
        z = model()

    embeddings = z.cpu().numpy()
    accuracy, f1_macro, f1_micro = link_sign_direction_prediction_logistic_function(
        embeddings, query_edges.cpu(), y.cpu(), query_test_edges.cpu(), y_test.cpu())
    assert loss >= 0
    assert accuracy >= 0
    assert f1_macro >= 0
    assert f1_micro >= 0

    model.reset_parameters()

def test_SSSNET_Link():
    """
    Testing SSSNET for link prediction
    """
    num_nodes = 100
    num_features = 3
    num_classes = 4

    X, _, _, edge_index, edge_weight = \
        create_mock_data(num_nodes, num_features, num_classes)
    data = SignedData(x=X, edge_index=edge_index, edge_weight=edge_weight)
    link_data = link_class_split(data, splits=2, task="four_class_signed_digraph", prob_val=0.15, prob_test=0.1, seed=10, device=device)

    model = SSSNET_link_prediction(nfeat=num_features, hidden=4, nclass=num_classes, dropout=0.5, 
        hop=2, fill_value=0.5, directed=data.is_directed).to(device)
    data1 = SignedData(edge_index=edge_index, edge_weight=edge_weight).to(device)
    data1.separate_positive_negative()
    preds = model(data1.edge_index_p, data1.edge_weight_p, data1.edge_index_n, data1.edge_weight_n, data.x, link_data[0]['train']['edges'])

    assert preds.shape == (
        len(link_data[0]['train']['edges']), num_classes
    )

    model = SSSNET_link_prediction(nfeat=num_features, hidden=4, nclass=num_classes, dropout=0.5, 
        hop=2, fill_value=0.5, directed=False).to(device)
    data1 = SignedData(edge_index=edge_index, edge_weight=edge_weight).to(device)
    data1.separate_positive_negative()
    preds = model(data1.edge_index_p, data1.edge_weight_p, data1.edge_index_n, data1.edge_weight_n, data.x, link_data[0]['train']['edges'])

    assert preds.shape == (
        len(link_data[0]['train']['edges']), num_classes
    )


def test_MSGNN():
    """
    Testing MSGNN
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    X, _, _, edge_index, edge_weight = \
        create_mock_data(num_nodes, num_features, num_classes)

    model = MSGNN_node_classification(q=0.25, K=1, num_features=X.shape[1], hidden=2, label_dim=num_classes, 
        dropout=0.5, activation=True, trainable_q=True, cached=False, conv_bias=False).to(device)
    _, _, _, preds = model(X, X, edge_index=edge_index, 
                    edge_weight=edge_weight)

    assert preds.shape == (
        num_nodes, num_classes
    )

    model = MSGNN_node_classification(q=0.25, K=2, num_features=X.shape[1], hidden=2, label_dim=num_classes, 
        dropout=0.5, cached=True, normalization=None).to(device)
    _, _, _, preds = model(X, X, edge_index=edge_index, 
                    edge_weight=edge_weight)

    assert preds.shape == (
        num_nodes, num_classes
    )
    _, _, _, preds = model(X, X, edge_index=edge_index, 
                    edge_weight=edge_weight)

    assert preds.shape == (
        num_nodes, num_classes
    )
    assert model.Chebs[0].__repr__(
    ) == 'MSConv(3, 2, filter size=3, normalization=None)'

    model.reset_parameters()

def test_MSGNN_Link():
    """
    Testing MSGNN for link prediction
    """
    num_nodes = 100
    num_features = 3
    num_classes = 4

    X, _, _, edge_index, edge_weight = \
        create_mock_data(num_nodes, num_features, num_classes)
    data = SignedData(x=X, edge_index=edge_index, edge_weight=edge_weight)
    link_data = link_class_split(data, splits=2, task="four_class_signed_digraph", prob_val=0.15, prob_test=0.1, seed=10, device=device)

    model = MSGNN_link_prediction(q=0.25, K=2, num_features=num_features, hidden=2, label_dim=num_classes, \
            trainable_q = False, dropout=0.5, cached=True).to(device)
    preds = model(data.x, data.x, edge_index=link_data[0]['graph'], query_edges=link_data[0]['train']['edges'],
                  edge_weight=link_data[0]['weights'])

    assert preds.shape == (
        len(link_data[0]['train']['edges']), num_classes
    )

    preds = model(data.x, data.x, link_data[0]['graph'], query_edges=link_data[0]['train']['edges'],
                  edge_weight=link_data[0]['weights'])

    assert preds.shape == (
        len(link_data[0]['train']['edges']), num_classes
    )
    assert model.Chebs[0].__repr__(
    ) == 'MSConv(3, 2, filter size=3, normalization=sym)'

    num_classes = 5
    link_data = link_class_split(data, splits=2, task="five_class_signed_digraph", prob_val=0.15, prob_test=0.1, seed=10, device=device)

    model = MSGNN_link_prediction(q=0.25, K=3, num_features=num_features, hidden=2, label_dim=num_classes, \
            trainable_q = False, dropout=0.5, cached=True).to(device)
    preds = model(data.x, data.x, edge_index=link_data[0]['graph'], query_edges=link_data[0]['train']['edges'],
                  edge_weight=link_data[0]['weights'])

    assert preds.shape == (
        len(link_data[0]['train']['edges']), num_classes
    )

    preds = model(data.x, data.x, edge_index=link_data[0]['graph'], query_edges=link_data[0]['train']['edges'],
                  edge_weight=link_data[0]['weights'])

    assert preds.shape == (
        len(link_data[0]['train']['edges']), num_classes
    )

    model.reset_parameters()

def test_magnetic_signed_Laplacian():
    """
    Testing magnetic signed Laplacian function
    """
    num_nodes = 100
    num_features = 3
    num_classes = 4

    X, _, _, edge_index, edge_weight = \
        create_mock_data(num_nodes, num_features, num_classes, size_ratio=1)
    _, _, _ = get_magnetic_signed_Laplacian(edge_index, edge_weight, absolute_degree=False)
    _, _, _ = get_magnetic_signed_Laplacian(edge_index, None, absolute_degree=True)
