import os.path as osp
import numpy as np
import argparse
from sklearn import metrics

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torch_geometric_signed_directed.utils import (
    directed_features_in_out, get_second_directed_adj,
    get_appr_directed_adj)
from torch_geometric_signed_directed.nn.directed import DiGCN_Inception_Block_node_classification
from torch_geometric_signed_directed.data import load_directed_real_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='webkb/cornell')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.0005)
args = parser.parse_args()

def train(X, y, edge_index, edge_weight, mask):
    model.train()
    out = model(X, edge_index_tuple=edge_index,  
                    edge_weight_tuple=edge_weight)
    loss = criterion(out[mask], y[mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = metrics.accuracy_score(y[mask].cpu(), out.max(dim=1)[1].cpu()[mask])
    return loss.detach().item(), train_acc

def test(X, y, edge_index, edge_weight, mask):
    model.eval()
    with torch.no_grad():
        out = model(X, edge_index_tuple=edge_index, 
                    edge_weight_tuple=edge_weight)
    test_acc = metrics.accuracy_score(y[mask].cpu(), out.max(dim=1)[1].cpu()[mask])
    return test_acc

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = args.dataset.split('/')
data = load_directed_real_data(dataset=dataset_name[0], root=path, name=dataset_name[1]).to(device)

num_classes = (data.y.max() - data.y.min() + 1).cpu().numpy()
model = DiGCN_Inception_Block_node_classification(num_features=data.x.shape[1], hidden=16, label_dim=num_classes, dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.NLLLoss()

for split in range(data.train_mask.shape[1]):
    X = data.x
    train_mask = data.train_mask 
    val_mask = data.val_mask
    test_mask = data.test_mask
    edge_index = data.edge_index
    edge_weight = data.edge_weight

    edge_index1, edge_weight1 = get_appr_directed_adj(args.alpha, edge_index, data.x.shape[0], 
                                                        data.x.dtype, edge_weight)
    edge_index2, edge_weight2 = get_second_directed_adj(edge_index, data.x.shape[0], 
                                                            data.x.dtype, edge_weight)
    edge_index = (edge_index1, edge_index2)
    edge_weight = (edge_weight1, edge_weight2)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(X, data.y, edge_index, edge_weight, train_mask[:,split])
        val_acc = test(X, data.y, edge_index, edge_weight, val_mask[:,split])
        print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}, Val_Acc: {val_acc:.4f}')
    test_acc = test(X, data.y, edge_index, edge_weight, test_mask[:,split])
    print(f'Split: {split:02d}, Test_Acc: {test_acc:.4f}')
    model.reset_parameters()