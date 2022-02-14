import os.path as osp
import numpy as np
import torch

from torch_geometric_signed_directed.nn.signed import SiGAT
from torch_geometric_signed_directed.data.signed import SignedDirectedGraphDataset
from torch_geometric_signed_directed.utils.signed import link_sign_prediction_logistic_function

dataset_name = 'bitcoin_alpha'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
dataset = SignedDirectedGraphDataset(path, dataset_name)
pos_edge_indices, neg_edge_indices = [], []
for data in dataset:
    pos_edge_indices.append(data.edge_index[:, data.edge_weight > 0])
    neg_edge_indices.append(data.edge_index[:, data.edge_weight < 0])

train_pos_edge_index = data.train_edge_index[:, data.train_edge_weight > 0]
test_pos_edge_index  = data.test_edge_index[:, data.test_edge_weight > 0]
train_neg_edge_index = data.train_edge_index[:, data.train_edge_weight < 0]
test_neg_edge_index  = data.test_edge_index[:, data.test_edge_weight < 0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nodes_num = data.num_nodes
edge_i_list = data.train_edge_index.t().numpy().tolist()
edge_s_list = data.train_edge_weight.numpy().tolist()
edge_index_s = torch.LongTensor([[i, j, s] for (i, j), s in zip(edge_i_list, edge_s_list)], device=device)
model = SiGAT(nodes_num, edge_index_s, 20, 20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
print(model.train())

def test():
    model.eval()
    with torch.no_grad():
        nodes = np.arange(0, nodes_num)
        z = model(nodes)
    
    embeddings = z.cpu().numpy()
    train_X = data.train_edge_index.t().cpu().numpy()
    test_X  = data.test_edge_index.t().cpu().numpy()
    train_y = data.train_edge_weight.cpu().numpy()
    test_y  = data.test_edge_weight.cpu().numpy()
    accuracy, f1, f1_macro, f1_micro, auc_score = link_sign_prediction_logistic_function(embeddings, train_X, train_y, test_X, test_y)
    
    return auc_score, f1, f1_macro, f1_micro, accuracy


for epoch in range(101):
    total_loss = []
    nodes_pku = np.random.permutation(nodes_num).tolist()
    batch_size = 500
    model.train()
    for batch in range(nodes_num // batch_size):
        optimizer.zero_grad()
        b_index = batch * batch_size
        e_index = (batch + 1) * batch_size
        nodes = nodes_pku[b_index:e_index]
        loss = model.loss(np.array(nodes))
        total_loss.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()
    auc, f1,  f1_macro, f1_micro, accuracy = test()
    loss = np.mean(total_loss)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'AUC: {auc:.4f}, F1: {f1:.4f}, MacroF1: {f1_macro:.4f}, MicroF1: {f1_micro:.4f}')