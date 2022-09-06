import os
import argparse

from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
from torch_geometric_signed_directed.utils import link_class_split, in_out_degree
from torch_geometric_signed_directed.data import load_signed_real_data, SignedData
from torch_geometric_signed_directed.nn import MSGNN_link_prediction

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bitcoin_alpha')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--year', type=int, default=2000)
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes.')
    parser.add_argument('--direction_only_task', action='store_true', help='Whether to degrade the task to consider direction only.')
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--q', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalization', type=str, default='sym')
    parser.add_argument('--trainable_q', action='store_true')
    parser.add_argument('--emb_loss_coeff', type=float, default=0, help='Coefficient for the embedding loss term.')
    parser.add_argument('--method', type=str, default='MSGNN')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0)
    parser.add_argument('--sparsify_level', type=float, default=1)
    parser.add_argument('--weighted_input_feat', action='store_true', help='Whether to use edge weights to calculate degree features as input.')
    parser.add_argument('--weighted_nonnegative_input_feat', action='store_true', help='Whether to use absolute values of edge weights to calculate degree features as input.')
    parser.add_argument('--absolute_degree', action='store_true', help='Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix.')
    parser.add_argument('--sd_input_feat', action='store_true', help='Whether to use both signed and directed features as input.')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--cpu', action='store_true',
                            help='use cpu')
    return parser.parse_args()

args = parameter_parser()

def train_MSGNN(X_real, X_img, y, edge_index, edge_weight, query_edges):
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

def test_MSGNN(X_real, X_img, y, edge_index, edge_weight, query_edges):
    model.eval()
    with torch.no_grad():
        out = model(X_real, X_img, edge_index=edge_index, 
                    query_edges=query_edges, 
                    edge_weight=edge_weight)
    test_y = y.cpu()
    pred = out.max(dim=1)[1].detach().cpu().numpy()
    test_acc  = metrics.accuracy_score(test_y, pred)
    f1_macro  =  metrics.f1_score(test_y, pred, average='macro')
    f1_micro  =  metrics.f1_score(test_y, pred, average='micro')
    return test_acc, f1_macro, f1_micro

device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', args.dataset)
data = load_signed_real_data(dataset=args.dataset, root=path, sparsify_level=args.sparsify_level).to(device)

sub_dir_name = 'runs' + str(args.runs) + 'epochs' + str(args.epochs) + \
    '100train_ratio' + str(int(100*args.train_ratio)) + '100val_ratio' + str(int(100*args.val_ratio)) + \
        '1000lr' + str(int(1000*args.lr)) + '1000weight_decay' + str(int(1000*args.weight_decay)) + '100dropout' + str(int(100*args.dropout)) 
suffix = 'K' + str(args.K) + '100q' + str(int(100*args.q)) + 'trainable_q' + str(args.trainable_q) + \
    'hidden' + str(args.hidden) + '10sparsify_level' + str(int(args.sparsify_level*10))
num_input_feat = 2
if args.sd_input_feat:
    suffix += 'SdInput'
    num_input_feat = 4
if args.weighted_input_feat:
    suffix += 'WeightedInput'
    if args.weighted_nonnegative_input_feat:
        suffix += 'nonnegative'


if args.num_classes == 4:
    task = "four_class_signed_digraph"
elif args.num_classes == 5:
    task = "five_class_signed_digraph"

link_data = link_class_split(data, splits=args.runs, task=task, prob_val=args.val_ratio, prob_test=1-args.train_ratio-args.val_ratio, seed=args.seed, device=device)
if args.direction_only_task:
    task += '_direction_only'
    args.num_classes -= 2

nodes_num = data.num_nodes
in_dim = args.in_dim
out_dim = args.out_dim



criterion = nn.NLLLoss()
res_array = np.zeros((args.runs, 3))
for split in list(link_data.keys()):
    edge_index = link_data[split]['graph']
    edge_weight = link_data[split]['weights']
    edge_i_list = edge_index.t().cpu().numpy().tolist()
    edge_s_list = edge_weight.long().cpu().numpy().tolist()
    edge_index_s = torch.LongTensor([[i, j, s] for (i, j), s in zip(edge_i_list, edge_s_list)]).to(device)
    query_edges = link_data[split]['train']['edges']
    y = link_data[split]['train']['label']
    if args.direction_only_task:
        y = torch.div(y, 2, rounding_mode='floor').to(device)
    if args.weighted_input_feat:
        if args.weighted_nonnegative_input_feat:
            X_real = in_out_degree(edge_index, size=int(edge_index.max()-edge_index.min())+1, signed=args.sd_input_feat, \
                edge_weight=torch.abs(edge_weight)).to(device)
        else:
            X_real = in_out_degree(edge_index, size=int(edge_index.max()-edge_index.min())+1, signed=args.sd_input_feat, \
                edge_weight=edge_weight).to(device)
    else:
        if args.sd_input_feat:
            data1 = SignedData(edge_index=edge_index, edge_weight=edge_weight).to(device)
            data1.separate_positive_negative()
            x1 = in_out_degree(data1.edge_index_p, size=int(data1.edge_index.max()-data1.edge_index.min())+1).to(device)
            x2 = in_out_degree(data1.edge_index_n, size=int(data1.edge_index.max()-data1.edge_index.min())+1).to(device)
            X_real = torch.concat((x1, x2), 1)
        else:
            X_real = in_out_degree(edge_index, size=int(edge_index.max()-edge_index.min())+1, signed=args.sd_input_feat).to(device)

    X_img = X_real.clone()
    model = MSGNN_link_prediction(q=args.q, K=args.K, num_features=num_input_feat, hidden=args.hidden, label_dim=args.num_classes, \
        trainable_q = args.trainable_q, dropout=args.dropout, normalization=args.normalization, cached=(not args.trainable_q)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    

    query_test_edges = link_data[split]['test']['edges']
    y_test = link_data[split]['test']['label']  
    if args.direction_only_task:
        y_test = torch.div(y_test, 2, rounding_mode='floor').to(device)
    for epoch in range(args.epochs):
        train_loss, train_acc = train_MSGNN(X_real, X_img, y, edge_index, edge_weight, query_edges)
        print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f}')


    accuracy, f1_macro, f1_micro = test_MSGNN(X_real, X_img, y_test, edge_index, edge_weight, query_test_edges)
    print(f'Split: {split:02d}, Test_Acc: {accuracy:.4f}, F1 macro: {f1_macro:.4f}, \
        F1 micro: {f1_micro:.4f}.')
    res_array[split] = [accuracy, f1_macro, f1_micro]

print("{}'s average accuracy, MacroF1 and MicroF1: {}".format(args.method, res_array.mean(0)))