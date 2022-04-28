import os.path as osp
import argparse


import torch
import torch.nn as nn
from sklearn import metrics

from torch_geometric_signed_directed.utils import link_class_split, in_out_degree
from torch_geometric_signed_directed.nn.directed import MagNet_link_prediction
from torch_geometric_signed_directed.data import load_directed_real_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='webkb/cornell')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--q', type=float, default=0.25)
parser.add_argument('--weight_decay', type=float, default=0.0005)
args = parser.parse_args()


def train(X_real, X_img, y, edge_index, edge_weight, query_edges):
    model.train()
    out = model(X_real, X_img, edge_index=edge_index,
                query_edges=query_edges,
                edge_weight=edge_weight)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
    return loss.detach().item(), train_acc


def test(X_real, X_img, y, edge_index, edge_weight, query_edges):
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img, edge_index=edge_index,
                    query_edges=query_edges,
                    edge_weight=edge_weight)
    test_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
    return test_acc


path = osp.join(osp.dirname(osp.realpath(__file__)),
                '..', 'data', args.dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = args.dataset.split('/')
data = load_directed_real_data(
    dataset=dataset_name[0], root=path, name=dataset_name[1]).to(device)
link_data = link_class_split(
    data, prob_val=0.15, prob_test=0.05, task='direction', device=device)

model = MagNet_link_prediction(
    q=args.q, K=args.K, num_features=2, hidden=16, label_dim=2).to(device)
criterion = nn.NLLLoss()

for split in list(link_data.keys()):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    edge_index = link_data[split]['graph']
    edge_weight = link_data[split]['weights']
    query_edges = link_data[split]['train']['edges']
    y = link_data[split]['train']['label']
    X_real = in_out_degree(edge_index, size=len(data.x)).to(device)
    X_img = X_real.clone()

    query_val_edges = link_data[split]['val']['edges']
    y_val = link_data[split]['val']['label']
    for epoch in range(args.epochs):
        train_loss, train_acc = train(
            X_real, X_img, y, edge_index, edge_weight, query_edges)
        val_acc = test(X_real, X_img, y_val, edge_index,
                       edge_weight, query_val_edges)
        print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}, Val_Acc: {val_acc:.4f}')

    query_test_edges = link_data[split]['test']['edges']
    y_test = link_data[split]['test']['label']
    test_acc = test(X_real, X_img, y_test, edge_index,
                    edge_weight, query_test_edges)
    print(f'Split: {split:02d}, Test_Acc: {test_acc:.4f}')
    model.reset_parameters()
