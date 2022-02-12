import os.path as osp
import numpy as np
import torch
from torch import optim

from torch_geometric_signed_directed.nn.signed import SDGNN
from torch_geometric_signed_directed.datasets import SignedDirectedGraph
from torch_geometric_signed_directed.utils.signed import link_sign_prediction_logistic_function

dataset_name = 'bitcoin_alpha'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
dataset = SignedDirectedGraph(path, dataset_name)
pos_edge_indices, neg_edge_indices = [], []
for data in dataset:
    pos_edge_indices.append(data.edge_index[:, data.edge_sign > 0])
    neg_edge_indices.append(data.edge_index[:, data.edge_sign < 0])

train_pos_edge_index = data.train_edge_index[:, data.train_edge_sign > 0]
test_pos_edge_index  = data.test_edge_index[:, data.test_edge_sign > 0]
train_neg_edge_index = data.train_edge_index[:, data.train_edge_sign < 0]
test_neg_edge_index  = data.test_edge_index[:, data.test_edge_sign < 0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nodes_num = data.num_nodes
edge_i_list = data.train_edge_index.t().numpy().tolist()
edge_s_list = data.train_edge_sign.numpy().tolist()
edge_index_s = torch.LongTensor([[i, j, s] for (i, j), s in zip(edge_i_list, edge_s_list)], device=device)

model = SDGNN(nodes_num, edge_index_s, 20, 20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
print(model)

def test():
    model.eval()
    with torch.no_grad():
        z = model()

    embeddings = z.cpu().numpy()
    train_X = data.train_edge_index.t().cpu().numpy()
    test_X  = data.test_edge_index.t().cpu().numpy()
    train_y = data.train_edge_sign.cpu().numpy()
    test_y  = data.test_edge_sign.cpu().numpy()
    accuracy, f1, f1_macro, f1_micro, auc_score = link_sign_prediction_logistic_function(embeddings, train_X, train_y, test_X, test_y)
    
    return auc_score, f1, f1_macro, f1_micro, accuracy


def train():
    model.train()
    optimizer.zero_grad()
    loss = model.loss()
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(201):
    loss = train()
    auc, f1,  f1_macro, f1_micro, accuracy = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'AUC: {auc:.4f}, F1: {f1:.4f}, MacroF1: {f1_macro:.4f}, MicroF1: {f1_micro:.4f}')