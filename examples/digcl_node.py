import os.path as osp
import numpy as np
import argparse

import torch
from sklearn import metrics

from torch_geometric_signed_directed.utils import (
    cal_fast_appr, drop_feature, pred_digcl_node)
from torch_geometric_signed_directed.nn.directed import DiGCL
from torch_geometric_signed_directed.data import load_directed_real_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora_ml')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--drop_feature_rate_1', type=float, default=0.3)
parser.add_argument('--drop_feature_rate_2', type=float, default=0.4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--curr-type', type=str, default='log')
args = parser.parse_args()

def train(X, edge_index, 
            alpha_1, alpha_2, 
            drop_feature1, drop_feature2):
    model.train()
    optimizer.zero_grad()

    edge_index_1, edge_weight_1 = cal_fast_appr(
        alpha_1, edge_index, X.shape[0], X.dtype, edge_weight=edge_weight)
    edge_index_2, edge_weight_2 = cal_fast_appr(
        alpha_2, edge_index, X.shape[0], X.dtype, edge_weight=edge_weight)

    x_1 = drop_feature(X, drop_feature1)
    x_2 = drop_feature(X, drop_feature2)
    z1 = model(x_1, edge_index_1, edge_weight_1)
    z2 = model(x_2, edge_index_2, edge_weight_2)
    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = args.dataset
data = load_directed_real_data(dataset=dataset_name, root=path).to(device)

model = DiGCL(in_channels=data.x.shape[1], activation='relu',
                 num_hidden=64, num_proj_hidden=32,
                 tau=0.4, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

alpha_1 = 0.1
for split in range(data.train_mask.shape[-1]):
    edge_index = data.edge_index
    edge_weight = data.edge_weight
    X = data.x

    edge_index_init, edge_weight_init = cal_fast_appr(
        alpha_1, edge_index, X.shape[0], X.dtype, edge_weight=edge_weight)

    num_epochs = args.epochs
    for epoch in range(num_epochs):
        a = 0.9
        b = 0.1

        if args.curr_type == 'linear':
            alpha_2 = a-(a-b)/(num_epochs+1)*epoch
        elif args.curr_type == 'exp':
            alpha_2 = a - (a-b)/(np.exp(3)-1) * \
                (np.exp(3*epoch/(num_epochs+1))-1)
        elif args.curr_type == 'log':
            alpha_2 = a - (a-b)*(1/3*np.log(epoch/(num_epochs+1)+np.exp(-3)))
        elif args.curr_type == 'fixed':
            alpha_2 = 0.9
        else:
            print('wrong curr type')
            exit()

        loss = train(X, edge_index, 
                        alpha_1, alpha_2, 
                        args.drop_feature_rate_1, args.drop_feature_rate_2)
        print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {loss:.4f}')

    model.eval()
    z = model(X, edge_index_init, edge_weight_init)
    pred = pred_digcl_node(z, y=data.y, 
                            train_index=data.train_mask[:,split].cpu(), 
                            test_index=data.test_mask[:,split].cpu())
    print(f'Split: {split:02d}, Test_Acc: {metrics.accuracy_score(data.y[data.test_mask[:,split]].cpu(), pred):.4f}')
    model.reset_parameters()
