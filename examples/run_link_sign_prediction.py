import argparse
import os.path as osp

import torch

from torch_geometric_signed_directed.nn.signed import SGCN, SDGNN, SiGAT, SNEA
from torch_geometric_signed_directed.data.signed import load_signed_real_data
from torch_geometric_signed_directed.utils.general.link_split import link_class_split
from torch_geometric_signed_directed.utils.signed import link_sign_prediction_logistic_function

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bitcoin_alpha')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model', type=str, default='SGCN')
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--in_dim', type=int, default=20)
parser.add_argument('--out_dim', type=int, default=20)
args = parser.parse_args()


dataset_name = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)),
                '..', 'data', dataset_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = load_signed_real_data(
    dataset=dataset_name, root=path).to(device)
data.to_unweighted()
link_data = link_class_split(data, prob_val=0.0, prob_test=0.2, splits=1, seed=args.seed, task='sign', device=device)
splited_data = link_data[0]

nodes_num = data.num_nodes
edge_index = splited_data['train']['edges']
edge_sign = splited_data['train']['label']
edge_index_s = torch.cat([edge_index, edge_sign.unsqueeze(-1)], dim=-1)

in_dim = args.in_dim
out_dim = args.out_dim

if args.model == 'SGCN':
    model = SGCN(nodes_num, edge_index_s, in_dim,
                 out_dim, layer_num=2, lamb=5).to(device)
elif args.model == 'SNEA':
    model = SNEA(nodes_num, edge_index_s, in_dim,
                 out_dim, layer_num=2, lamb=5).to(device)
elif args.model == 'SiGAT':
    model = SiGAT(nodes_num, edge_index_s, in_dim, out_dim).to(device)
elif args.model == 'SDGNN':
    model = SDGNN(nodes_num, edge_index_s, in_dim, out_dim).to(device)


print(model)
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def test():
    model.eval()
    with torch.no_grad():
        z = model()

    embeddings = z.cpu().numpy()
    train_X = splited_data['train']['edges'].cpu().numpy()
    test_X = splited_data['test']['edges'].cpu().numpy()
    train_y = splited_data['train']['label'].cpu().numpy()
    test_y = splited_data['test']['label'].cpu().numpy()
    accuracy, f1, f1_macro, f1_micro, auc_score = link_sign_prediction_logistic_function(
        embeddings, train_X, train_y, test_X, test_y)
    return auc_score, f1, f1_macro, f1_micro, accuracy


def train():
    model.train()
    optimizer.zero_grad()
    loss = model.loss()
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(args.epochs):
    loss = train()
    auc, f1,  f1_macro, f1_micro, accuracy = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'AUC: {auc:.4f}, F1: {f1:.4f}, MacroF1: {f1_macro:.4f}, MicroF1: {f1_micro:.4f}')
