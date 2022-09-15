import argparse
import os.path as osp
import time
import torch
import numpy as np

from torch_geometric.seed import seed_everything
from torch_geometric_signed_directed.nn.signed import SGCN, SDGNN, SiGAT, SNEA
from torch_geometric_signed_directed.data.signed import load_signed_real_data
from torch_geometric_signed_directed.utils.general.link_split import link_class_split
from torch_geometric_signed_directed.utils.signed import link_sign_prediction_logistic_function

def parameter_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bitcoin_alpha')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='SGCN')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--in_dim', type=int, default=20)
    parser.add_argument('--out_dim', type=int, default=20)
    parser.add_argument('--eval_step', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    return parser.parse_args()

args = parameter_parser()
seed_everything(args.seed)

dataset_name = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)),
                '..', 'tmp_data')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = load_signed_real_data(
    dataset=dataset_name, root=path).to(device)
data.to_unweighted()
link_data = link_class_split(data, prob_val=0.1, prob_test=0.1,
                             splits=args.runs, seed=args.seed,
                             task='sign', maintain_connect=False, device=device)




def test(model, splited_data):
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


def evaluate(model, splited_data, eval_flag='test'):
    model.eval()
    with torch.no_grad():
        z = model()
    embeddings = z.cpu().numpy()
    train_X = splited_data['train']['edges'].cpu().numpy()
    test_X = splited_data[eval_flag]['edges'].cpu().numpy()
    train_y = splited_data['train']['label'].cpu().numpy() 
    test_y = splited_data[eval_flag]['label'].cpu().numpy()
    accuracy, f1, f1_macro, f1_micro, auc_score = link_sign_prediction_logistic_function(
        embeddings, train_X, train_y, test_X, test_y)
    eval_info = {}
    eval_info['acc'] = accuracy
    eval_info['f1'] = f1
    eval_info['f1_macro'] = f1_macro
    eval_info['f1_micro'] = f1_micro
    eval_info['auc'] = auc_score
    return eval_info


def train(model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = model.loss()
    loss.backward()
    optimizer.step()
    return loss.item()



res = []
for index in list(link_data.keys()):
    splited_data = link_data[index]
    nodes_num = data.num_nodes
    edge_index = splited_data['train']['edges']
    edge_sign = splited_data['train']['label'] * 2 - 1
    edge_index_s = torch.cat([edge_index, edge_sign.unsqueeze(-1)], dim=-1)
    in_dim = args.in_dim
    out_dim = args.out_dim
    if args.model == 'SGCN':
        model = SGCN(nodes_num, edge_index_s, in_dim,
                     out_dim, layer_num=2, lamb=5).to(device)
    elif args.model == 'SNEA':
        model = SNEA(nodes_num, edge_index_s, in_dim,
                     out_dim, layer_num=2, lamb=4).to(device)
    elif args.model == 'SiGAT':
        model = SiGAT(nodes_num, edge_index_s, in_dim,
                     out_dim).to(device)
    elif args.model == 'SDGNN':
        model = SDGNN(nodes_num, edge_index_s, in_dim,
                     out_dim).to(device)

    print(args)
    print(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = 0
    test_info = {}
    patience = args.patience
    for epoch in range(args.epochs):
        t = time.time()
        loss = train(model, optimizer)
        if (epoch + 1) % args.eval_step == 0:
            eval_info = evaluate(model, splited_data, eval_flag='val')
            t = time.time() - t
            print(f'Val Time: {t:.3f}s, Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                f'AUC: {eval_info["auc"]:.4f}, F1: {eval_info["f1"]:.4f}, MacroF1: {eval_info["f1_macro"]:.4f}, MicroF1: {eval_info["f1_micro"]:.4f}')
            if eval_info['auc'] > best_auc:
                best_auc = eval_info['auc']
                test_info = evaluate(model, splited_data, eval_flag='test')
                test_info['epoch'] = epoch
                patience = args.patience
            patience -= 1
            if patience <=0:
                break
    print(f'Test Result: Epoch: {test_info["epoch"]:03d}, Loss: {loss:.4f}, '
        f'AUC: {test_info["auc"]:.4f}, F1: {test_info["f1"]:.4f}, MacroF1: {test_info["f1_macro"]:.4f}, MicroF1: {test_info["f1_micro"]:.4f}')
    res.append(test_info)

average_metrics = []
for i in res:
    average_metrics.append(
        (i['auc'], i['f1'], i['f1_macro'], i['f1_micro'])
    )
v = np.array(average_metrics)
print(f"{args.model}: AUC: {v[:, 0].mean():.4f} ({v[:, 0].std():.4f}) ")
print(f"{args.model}: F1: {v[:, 1].mean():.4f} ({v[:, 1].std():.4f}) ")
print(f"{args.model}: MacroF1: {v[:, 2].mean():.4f} ({v[:, 2].std():.4f}) ")
print(f"{args.model}: MicroF1: {v[:, 3].mean():.4f} ({v[:, 3].std():.4f}) ")