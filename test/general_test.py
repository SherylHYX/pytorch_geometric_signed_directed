import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric_signed_directed.nn import (
    MSGNN_node_classification,
    MSGNN_link_prediction
)
from torch_geometric_signed_directed.data import (
    SDSBM, SignedData
)
from torch_geometric_signed_directed.utils import (
    extract_network, meta_graph_generation,
    link_class_split
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mock_data(num_nodes, num_features, num_classes=3, F_style='cyclic', eta=0.1, p=0.2):
    """
    Creating a mock feature matrix, edge index and edge weight.
    """
    F_data = meta_graph_generation(F_style, num_classes, eta, False, 0.5)
    for i in range(num_classes):
        for j in range(num_classes):
            if (i + j) % 2:
                F_data[i, j] = - F_data[i, j] # flip the signs

    A, labels = SDSBM(N=num_nodes, K=num_classes, p=p, F=F_data, eta=eta, size_ratio=1.5)
    A, labels = extract_network(A=A, labels=labels)
    X = torch.FloatTensor(
        np.random.uniform(-1, 1, (num_nodes, num_features))).to(device)
    edge_index = torch.LongTensor(np.array(A.nonzero())).to(device)
    edge_weight = torch.FloatTensor(sp.csr_matrix(A).data).to(device)
    return X, A, F_data, edge_index, edge_weight

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
        dropout=0.5, cached=False, conv_bias=False, normalization=None).to(device)
    _, _, _, preds = model(X, X, edge_index=edge_index, 
                    edge_weight=edge_weight)

    assert preds.shape == (
        num_nodes, num_classes
    )

    model = MSGNN_node_classification(q=0.25, K=3, num_features=X.shape[1], hidden=2, label_dim=num_classes, 
        dropout=0.5, cached=True).to(device)
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
    ) == 'MSConv(3, 2, K=3, normalization=sym)'

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

    model = model = MSGNN_link_prediction(q=0.25, K=3, num_features=num_features, hidden=2, label_dim=num_classes, \
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
    ) == 'MSConv(3, 2, K=3, normalization=sym)'

    num_classes = 5
    link_data = link_class_split(data, splits=2, task="five_class_signed_digraph", prob_val=0.15, prob_test=0.1, seed=10, device=device)

    model = model = MSGNN_link_prediction(q=0.25, K=3, num_features=num_features, hidden=2, label_dim=num_classes, \
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
