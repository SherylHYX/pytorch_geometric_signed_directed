import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor

from torch_geometric_signed_directed.nn import (
    DiGCN_node_classification, DiGCN_Inception_Block_node_classification,
    DIGRAC_node_clustering, MagNet_node_classification,
    DGCN_node_classification, DGCNConv, DiGCL,
    DGCN_link_prediction,
    MagNet_link_prediction, DiGCN_link_prediction,
    DiGCN_Inception_Block_link_prediction
)
from torch_geometric_signed_directed.data import (
    DSBM, DirectedData
)
from torch_geometric_signed_directed.utils import (
    Prob_Imbalance_Loss, scipy_sparse_to_torch_sparse,
    get_appr_directed_adj, get_second_directed_adj,
    directed_features_in_out, meta_graph_generation, extract_network,
    cal_fast_appr, pred_digcl_node, pred_digcl_link, drop_feature, fast_appr_power,
    link_class_split
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mock_data(num_nodes, num_features, num_classes=3, F_style='cyclic', eta=0.1, p=0.2):
    """
    Creating a mock feature matrix, edge index and edge weight.
    """
    F = meta_graph_generation(F_style, num_classes, eta, False, 0)
    F_data = meta_graph_generation(F_style, num_classes, eta, False, 0.5)
    A, labels = DSBM(N=num_nodes, K=num_classes, p=p, F=F_data, size_ratio=1.5)
    A, labels = extract_network(A=A, labels=labels)
    X = torch.FloatTensor(
        np.random.uniform(-1, 1, (num_nodes, num_features))).to(device)
    edge_index = torch.LongTensor(np.array(A.nonzero())).to(device)
    edge_weight = torch.FloatTensor(sp.csr_matrix(A).data).to(device)
    return X, A, F, F_data, edge_index, edge_weight

def test_MagNet():
    """
    Testing MagNet
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    X, _, _, _, edge_index, edge_weight = \
        create_mock_data(num_nodes, num_features, num_classes)

    model = MagNet_node_classification(X.shape[1], K=1, q=0.1, label_dim=num_classes, layer=2,
                                       activation=True, hidden=2, dropout=0.5, normalization=None).to(device)
    preds = model(X, X, edge_index, edge_weight)

    assert preds.shape == (
        num_nodes, num_classes
    )

    model = MagNet_node_classification(X.shape[1], K=2, label_dim=num_classes, layer=3, trainable_q=True,
                                       activation=True, hidden=2, dropout=0.5, cached=True).to(device)
    preds = model(X, X, edge_index, edge_weight)

    assert preds.shape == (
        num_nodes, num_classes
    )
    preds = model(X, X, edge_index, edge_weight)

    assert preds.shape == (
        num_nodes, num_classes
    )
    assert model.Chebs[0].__repr__(
    ) == 'MagNetConv(3, 2, filter size=3, normalization=sym)'

    model.reset_parameters()


def test_MagNet_Link():
    """
    Testing MagNet for link prediction
    """
    num_nodes = 100
    num_features = 3
    num_classes = 2

    X, _, _, _, edge_index, edge_weight = \
        create_mock_data(num_nodes, num_features, num_classes)
    data = DirectedData(x=X, edge_index=edge_index, edge_weight=edge_weight)
    link_data = link_class_split(
        data, prob_val=0.15, prob_test=0.05, task='existence', device=device)
    model = MagNet_link_prediction(data.x.shape[1], K=1, q=0.1, label_dim=num_classes, layer=2,
                                   activation=True, hidden=2, dropout=0.5, normalization=None).to(device)
    preds = model(data.x, data.x, edge_index=link_data[0]['graph'], query_edges=link_data[0]['train']['edges'],
                  edge_weight=link_data[0]['weights'])

    assert preds.shape == (
        len(link_data[0]['train']['edges']), num_classes
    )

    model = MagNet_link_prediction(data.x.shape[1], K=2, label_dim=num_classes, layer=3, trainable_q=True,
                                   activation=True, hidden=2, dropout=0.5).to(device)
    preds = model(data.x, data.x, link_data[0]['graph'], query_edges=link_data[0]['train']['edges'],
                  edge_weight=link_data[0]['weights'])

    assert preds.shape == (
        len(link_data[0]['train']['edges']), num_classes
    )
    assert model.Chebs[0].__repr__(
    ) == 'MagNetConv(3, 2, filter size=3, normalization=sym)'

    num_classes = 3
    link_data = link_class_split(
        data, prob_val=0.15, prob_test=0.05, task='three_class_digraph', device=device)

    model = MagNet_link_prediction(data.x.shape[1], K=1, q=0.1, label_dim=num_classes, layer=2,
                                   activation=True, hidden=2, dropout=0.5, normalization=None, cached=True).to(device)
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


def test_fast_appr_power():
    """
    Testing fast_appr_power()
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    _, A, _, _, _, _ = \
        create_mock_data(num_nodes, num_features, num_classes)
    L, _ = fast_appr_power(A, alpha=0.1, max_iter=1,
                           tol=1e-06, personalize=None)
    assert L.shape[0] == num_nodes


def test_DGCN():
    """
    Testing DGCN
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    X, A, _, _, edge_index, edge_weights = \
        create_mock_data(num_nodes, num_features, num_classes)
    edge_index = edge_index.cpu()
    edge_weights = edge_weights.cpu()

    edge_index, edge_in, in_weight, edge_out, out_weight = directed_features_in_out(
        edge_index, A.shape[0], edge_weights)
    edge_index = edge_index.to(device)
    edge_in, in_weight, edge_out, out_weight = edge_in.to(device), in_weight.to(
        device), edge_out.to(device), out_weight.to(device)

    model = DGCN_node_classification(
        num_features, 4, num_classes, 0.5, cached=True).to(device)

    preds = model(X, edge_index, edge_in, edge_out, in_weight, out_weight)

    assert preds.shape == (
        num_nodes, num_classes
    )

    edge_index = edge_index.cpu()
    edge_weights = edge_weights.cpu()
    edge_index, edge_in, in_weight, edge_out, out_weight = directed_features_in_out(
        edge_index, A.shape[0], None)
    edge_index = edge_index.to(device)
    edge_in, in_weight, edge_out, out_weight = edge_in.to(device), in_weight.to(
        device), edge_out.to(device), out_weight.to(device)

    model = DGCN_node_classification(
        num_features, 4, num_classes, 0.0, True, True).to(device)

    preds = model(X, edge_index, edge_in, edge_out, in_weight, out_weight)

    assert preds.shape == (
        num_nodes, num_classes
    )

    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    row, col = edge_index
    value = torch.rand(row.size(0))
    adj2 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))
    adj1 = adj2.set_value(None)

    conv = DGCNConv()
    out1 = conv(x, edge_index)
    assert out1.size() == (4, 16)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)
    conv = DGCNConv(cached=True)
    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 16)
    assert torch.allclose(conv(x, adj2.t()), out2, atol=1e-6)
    out2 = conv(x, edge_index, value)
    assert out2.size() == (4, 16)
    assert torch.allclose(conv(x, adj2.t()), out2, atol=1e-6)
    model.reset_parameters()


def test_DGCN_link():
    """
    Testing DGCN for link prediction
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    X, A, _, _, edge_index, edge_weights = \
        create_mock_data(num_nodes, num_features, num_classes)
    edge_index = edge_index.cpu()
    edge_weights = edge_weights.cpu()

    edge_index, edge_in, in_weight, edge_out, out_weight = directed_features_in_out(
        edge_index, A.shape[0], edge_weights)
    edge_index = edge_index.to(device)
    edge_in, in_weight, edge_out, out_weight = edge_in.to(device), in_weight.to(
        device), edge_out.to(device), out_weight.to(device)

    model = DGCN_link_prediction(num_features, 4, num_classes, 0.5).to(device)
    preds = model(X, edge_index, edge_in, edge_out,
                  edge_index[:, :10].T, in_weight, out_weight)

    assert preds.shape == (
        10, num_classes
    )

    model = DGCN_link_prediction(
        num_features, 4, num_classes, 0.0, True, True).to(device)
    preds = model(X, edge_index, edge_in, edge_out,
                  edge_index[:, :10].T, in_weight, out_weight)

    assert preds.shape == (
        10, num_classes
    )

    model.reset_parameters()


def test_DiGCN():
    """
    Testing DiGCN
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    X, _, _, _, edge_index, edge_weights = \
        create_mock_data(num_nodes, num_features, num_classes)
    edge_index = edge_index.cpu()
    edge_weights = edge_weights.cpu()

    edge_index1, edge_weights1 = get_appr_directed_adj(0.1, edge_index, X.shape[0],
                                                       X.dtype, None)
    edge_index1 = edge_index1.to(device)
    edge_weights1 = edge_weights1.to(device)

    model = DiGCN_node_classification(num_features, 4, num_classes,
                                      0.5).to(device)

    preds = model(X, edge_index1, edge_weights1)

    assert preds.shape == (
        num_nodes, num_classes
    )
    assert model.conv1.__repr__() == 'DiGCNConv(3, 4)'

    edge_index = edge_index.cpu()
    edge_weights = edge_weights.cpu()
    edge_index2, edge_weights2 = get_second_directed_adj(
        edge_index, X.shape[0], X.dtype, edge_weights)
    edge_index2 = edge_index2.to(device)
    edge_weights2 = edge_weights2.to(device)
    edge_index = (edge_index1, edge_index2)
    edge_weights = (edge_weights1, edge_weights2)
    del edge_index2, edge_weights2

    model = DiGCN_Inception_Block_node_classification(num_features, 4, num_classes,
                                                      0.5).to(device)
    preds = model(X, edge_index, edge_weights)

    assert preds.shape == (
        num_nodes, num_classes
    )


def test_DiGCN_Link():
    """
    Testing DiGCN for link prediction
    """
    num_nodes = 100
    num_features = 3
    num_classes = 2

    X, _, _, _, edge_index, edge_weights = \
        create_mock_data(num_nodes, num_features, num_classes)

    data = DirectedData(x=X, edge_index=edge_index, edge_weight=edge_weights)
    link_data = link_class_split(
        data, prob_val=0.15, prob_test=0.05, task='existence', device=device)

    edge_index = link_data[0]['graph']
    edge_weights = link_data[0]['weights']
    edge_index1, edge_weights1 = get_appr_directed_adj(0.1, edge_index.cpu(), X.shape[0],
                                                       X.dtype, None)
    link_data[0]['graph'] = edge_index1.to(device)
    link_data[0]['weights'] = edge_weights1.to(device)

    model = DiGCN_link_prediction(num_features, 4, num_classes, 0.5).to(device)

    preds = model(data.x, link_data[0]['graph'], query_edges=link_data[0]['train']['edges'],
                  edge_weight=link_data[0]['weights'])

    assert preds.shape == (
        len(link_data[0]['train']['edges']), num_classes
    )
    assert model.conv1.__repr__() == 'DiGCNConv(3, 4)'

    edge_index = edge_index.cpu()
    edge_weights = edge_weights.cpu()
    edge_index2, edge_weights2 = get_second_directed_adj(
        edge_index, X.shape[0], X.dtype, None)
    edge_index2 = edge_index2.to(device)
    edge_weights2 = edge_weights2.to(device)
    edge_index = (link_data[0]['graph'], edge_index2)
    edge_weights = (link_data[0]['weights'], edge_weights2)
    del edge_index2, edge_weights2

    model = DiGCN_Inception_Block_link_prediction(num_features, 4, num_classes,
                                                  0.5).to(device)
    preds = model(X, edge_index, query_edges=link_data[0]['train']['edges'],
                  edge_weight_tuple=edge_weights)

    assert preds.shape == (
        len(link_data[0]['train']['edges']), num_classes
    )
    model.reset_parameters()


def test_DiGCL():
    """
    Testing DiGCL
    """
    num_nodes = 100
    num_features = 3
    num_classes = 3

    X, _, _, _, edge_index, edge_weights = \
        create_mock_data(num_nodes, num_features, num_classes)
    y = np.zeros((num_nodes))
    train_index = np.zeros((num_nodes), dtype=bool)
    curr_ind = 0
    step_range = int(np.floor(num_nodes/num_classes))
    for i in range(num_classes):
        y[curr_ind:curr_ind+step_range] = i
        train_index[curr_ind: curr_ind+int(step_range/2)] = True
        curr_ind += step_range
    y = torch.LongTensor(y).to(device)
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)
    alpha_1 = 0.1
    drop_feature_rate_1 = 0.3
    drop_feature_rate_2 = 0.4
    hidden = 4

    edge_index_init, edge_weight_init = cal_fast_appr(
        alpha_1, edge_index, X.shape[0], X.dtype, edge_weight=None)
    x = X.to(device)
    model = DiGCL(in_channels=X.shape[1], activation='relu',
                  num_hidden=2*hidden, num_proj_hidden=hidden,
                  tau=0.5, num_layers=2).to(device)
    a = 0.9
    b = 0.1
    epochs = 10
    alpha_2 = a - (a-b)*(1/3*np.log(epochs/(epochs+1)+np.exp(-3)))
    edge_index_1, edge_weight_1 = cal_fast_appr(
        alpha_1, edge_index, x.shape[0], x.dtype, edge_weight=edge_weights)
    edge_index_2, edge_weight_2 = cal_fast_appr(
        alpha_2, edge_index, x.shape[0], x.dtype, edge_weight=edge_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.00001)
    for _ in range(epochs):
        x_1 = drop_feature(x, drop_feature_rate_1)
        x_2 = drop_feature(x, drop_feature_rate_2)

        z1 = model(x_1, edge_index_1, edge_weight_1)
        z2 = model(x_2, edge_index_2, edge_weight_2)

        loss = model.loss(z1, z2, batch_size=0)
        loss.backward()
        optimizer.step()
    # test
    model.eval()
    z = model(x, edge_index_init, edge_weight_init)
    pred = pred_digcl_node(z, y, train_index)

    assert pred.shape == (
        num_nodes,
    )

    model = DiGCL(in_channels=X.shape[1], activation='prelu',
                  num_hidden=2*hidden, num_proj_hidden=hidden,
                  tau=0.5, num_layers=3).to(device)
    for _ in range(epochs):
        x_1 = drop_feature(x, drop_feature_rate_1)
        x_2 = drop_feature(x, drop_feature_rate_2)

        z1 = model(x_1, edge_index_1, edge_weight_1)
        z2 = model(x_2, edge_index_2, edge_weight_2)

        loss = model.loss(z1, z2, batch_size=16)
        loss.backward()
        optimizer.step()
    # test
    model.eval()
    z = model(x, edge_index_init, edge_weight_init)
    pred = pred_digcl_node(z, y, train_index, train_index)

    assert pred.shape == (
        sum(train_index),
    )
    # test (link prediction)
    pred = pred_digcl_link(z, y=torch.randint(3, size=(50, 1), device=device),
                           train_index=edge_index.T.cpu()[:50], test_index=edge_index.T.cpu()[50:60])
    assert pred.shape == (
        10,
    )
    model.reset_parameters()


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

    model = DIGRAC_node_clustering(num_features=num_features,
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


def test_DSBM():
    num_nodes = 200
    num_classes = 3
    p = 0.01
    eta = 0.1
    F = meta_graph_generation(
        F_style='cyclic', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    F = meta_graph_generation(
        F_style='path', K=num_classes, eta=0.0, ambient=True, fill_val=0.0)
    assert F.shape == (num_classes, num_classes)
    F = meta_graph_generation(
        F_style='complete', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)

    num_classes = 2
    F = meta_graph_generation(
        F_style='complete', K=num_classes, eta=eta, ambient=False, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    F = meta_graph_generation(
        F_style='cyclic', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    F = meta_graph_generation(
        F_style='cyclic', K=num_classes, eta=eta, ambient=False, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)

    num_classes = 6
    F = meta_graph_generation(
        F_style='star', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    num_classes = 5
    F = meta_graph_generation(
        F_style='star', K=num_classes, eta=eta, ambient=False, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    num_classes = 10
    F = meta_graph_generation(
        F_style='multipartite', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)
    num_classes = 9
    F = meta_graph_generation(
        F_style='multipartite', K=num_classes, eta=eta, ambient=False, fill_val=0.5)
    assert F.shape == (num_classes, num_classes)

    A, labels = DSBM(N=num_nodes, K=num_classes, p=p, F=F, size_ratio=1)
    A, labels = extract_network(A, labels, 2)
    assert A.shape[1] <= num_nodes or A is None

    A, labels = DSBM(N=num_nodes, K=num_classes, p=p, F=F, size_ratio=1)
    A, labels = extract_network(A=A, lowest_degree=10)
    assert labels is None
    assert A.shape[1] <= num_nodes


def test_DirectedData():
    num_nodes = 200
    num_classes = 3
    p = 0.1
    eta = 0.1
    F = meta_graph_generation(
        F_style='cyclic', K=num_classes, eta=eta, ambient=True, fill_val=0.5)
    A, _ = DSBM(N=num_nodes, K=num_classes, p=p, F=F, size_ratio=1)
    data = DirectedData(A=A)
    assert data.edge_index[0].max() < num_nodes
    assert data.edge_weight.max() <= 1
    data2 = DirectedData(edge_index=data.edge_index)
    assert data2.A.shape[0] == num_nodes
    data2.set_hermitian_features(k=num_classes)
    assert data2.x.shape == (num_nodes, 2*num_classes)
