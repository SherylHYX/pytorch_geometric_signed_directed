import os.path as osp
import argparse
from sklearn.metrics import adjusted_rand_score

import torch
from torch_geometric_signed_directed.nn.directed import DIGRAC_node_clustering
from torch_geometric_signed_directed.data import DirectedData, DSBM
from torch_geometric_signed_directed.utils import (meta_graph_generation, 
extract_network, scipy_sparse_to_torch_sparse, Prob_Imbalance_Loss)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=0.0005)
args = parser.parse_args()

num_classes = 5
eta = 0.1
F_style = 'complete'
num_nodes = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

F = meta_graph_generation(F_style, num_classes, eta, False, 0)
F_data = meta_graph_generation(F_style, num_classes, eta, False, 0.5)
A, labels = DSBM(N=num_nodes, K=num_classes, p=0.1, F=F_data, size_ratio=1.5)
A, labels = extract_network(A=A, labels=labels)
data = DirectedData(A=A, y=torch.LongTensor(labels))
data.set_hermitian_features(num_classes)
data.node_split(train_size_per_class=0.8, val_size_per_class=0.1, test_size_per_class=0.1)
imbalance_loss = Prob_Imbalance_Loss(F)


model = DIGRAC_node_clustering(num_features=data.x.shape[1], dropout=0.5, hop=2, fill_value=0.5, 
                        hidden=32, nclass=num_classes).to(device)


def train(features, edge_index, edge_weight, mask, train_A, y):
    model.train()
    _, _, _, prob = model(edge_index, edge_weight, features)
    loss = imbalance_loss(prob[mask], train_A, num_classes, 'vol_sum', 'sort')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return loss.detach().item(), train_ari

def test(features, edge_index, edge_weight, mask, y):
    model.eval()
    with torch.no_grad():
        _, _, _, prob = model(edge_index, edge_weight, features)
    test_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return test_ari

data.x = torch.FloatTensor(data.x).to(device)
data.edge_index = data.edge_index.to(device)
data.edge_weight = data.edge_weight.to(device)

for split in range(data.train_mask.shape[1]):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_index = data.train_mask[:, split]
    val_index = data.val_mask[:, split]
    test_index = data.test_mask[:, split]
    train_A = scipy_sparse_to_torch_sparse(data.A[train_index][:, train_index]).to(device)
    for epoch in range(args.epochs):
        train_loss, train_ari = train(data.x, data.edge_index, 
                                        data.edge_weight, train_index, train_A, data.y)
        Val_ari = test(data.x, data.edge_index, 
                    data.edge_weight, val_index, data.y)
        print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_ARI: {train_ari:.4f}, Val_ARI: {Val_ari:.4f}')
    
    test_ari = test(data.x, data.edge_index, 
                    data.edge_weight, test_index, data.y)
    print(f'Split: {split:02d}, Test_ARI: {test_ari:.4f}')
    model._reset_parameters()