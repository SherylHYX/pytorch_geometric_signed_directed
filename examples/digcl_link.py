import os.path as osp
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn import metrics

from torch_geometric_signed_directed.utils import (
    directed_link_class_split, in_out_degree, cal_fast_appr, drop_feature, pred_digcl_link)
from torch_geometric_signed_directed.nn.directed import DiGCL
from torch_geometric_signed_directed.data import load_directed_real_data, DirectedData

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='webkb/cornell')
parser.add_argument('--random_splits', type=bool, default=False)
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

dataset_name = args.dataset.split('/')
data = load_directed_real_data(dataset=dataset_name[0], root=path, name=dataset_name[1]).to(device)
link_data = directed_link_class_split(data, prob_val=0.15, prob_test=0.05, task = 'direction', device=device)

model = DiGCL(in_channels=2, activation='relu',
                 num_hidden=32, num_proj_hidden=16,
                 tau=0.5, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

alpha_1 = 0.1
for split in list(link_data.keys()):
    edge_index = link_data[split]['graph']
    edge_weight = link_data[split]['weights']
    X = in_out_degree(edge_index, size=len(data.x)).to(device)

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
    query_train = link_data[split]['train']['edges'].cpu()
    query_test = link_data[split]['test']['edges'].cpu()
    y = link_data[split]['train']['label'].cpu()
    test_y = link_data[split]['test']['label'].cpu()
    pred = pred_digcl_link(z, y=y, train_index=query_train, test_index=query_test)
    print(f'Split: {split:02d}, Test_Acc: {metrics.accuracy_score(test_y, pred):.4f}')
    model.reset_parameters()
