import os
import pickle as pk
import argparse

from sklearn.metrics import adjusted_rand_score
import scipy.sparse as sp
import numpy as np
import torch
from torch_geometric_signed_directed.nn import MSGNN_node_classification
from torch_geometric_signed_directed.data import SignedData, SDSBM
from torch_geometric_signed_directed.utils import \
    (Prob_Balanced_Normalized_Loss, Prob_Imbalance_Loss, scipy_sparse_to_torch_sparse, 
extract_network, triplet_loss_node_classification, in_out_degree)
import numpy.random as rnd

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SDSBM')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--K_model', type=int, default=1)
    parser.add_argument('--q', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalization', type=str, default='sym')
    parser.add_argument('--trainable_q', action='store_true')
    parser.add_argument('--method', type=str, default='MSGNN')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--year', type=int, default=2000)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed_ratio', type=float, default=0.1)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--cpu', action='store_true',
                            help='use cpu')
    parser.add_argument('--p', type=float, default=0.02,
                        help='probability of the existence of a link within communities, with probability (1-p), we have 0.')
    parser.add_argument('--N', type=int, default=1000,
                        help='number of nodes in the signed stochastic block model.')
    parser.add_argument('--K', type=int, default=2,
                        help=' number of blocks in each SSBM.')
    parser.add_argument('--imbalance_loss_ratio', type=float, default=1,
                        help='Ratio of imbalance loss to signed loss. Default 1.')
    parser.add_argument("--imb_normalization", type=str, default='vol_sum',
                        help="Normalization method to choose from: vol_min, vol_sum, vol_max and plain.")
    parser.add_argument("--imb_threshold", type=str, default='sort',
                        help="Thresholding method to choose from: sort, std and naive.")
    parser.add_argument('--triplet_loss_ratio', type=float, default=0.1,
                        help='Ratio of triplet loss to cross entropy loss in supervised loss part. Default 0.1.')
    parser.add_argument('--pbnc_loss_ratio', type=float, default=1,
                        help='Ratio of probablistic balanced normlized cut loss in the self-supervised loss part. Default 1.')
    parser.add_argument('--supervised_loss_ratio', type=float, default=50,
                        help='Ratio of factor of supervised loss part to self-supervised loss part.')
    parser.add_argument('--absolute_degree', action='store_true', help='Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix.')
    parser.add_argument('--sd_input_feat', action='store_true', help='Whether to use both signed and directed features as input.')
    parser.add_argument('--weighted_input_feat', action='store_true', help='Whether to use edge weights to calculate degree features as input.')
    parser.add_argument('--weighted_nonnegative_input_feat', action='store_true', help='Whether to use absolute values of edge weights to calculate degree features as input.')
    parser.add_argument('--size_ratio', type=float, default=1.5,
                        help='The size ratio of the largest to the smallest block. 1 means uniform sizes. should be at least 1.')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='sign flip probability in the meta-graph adjacency matrix, less than 0.5.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='direction noise level in the meta-graph adjacency matrix, less than 0.5.')
    parser.add_argument('--early_stopping', type=int, default=200, help='Number of iterations to consider for early stopping.')
    return parser.parse_args()

args = parameter_parser()

device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

assert args.supervised_loss_ratio + args.imbalance_loss_ratio + args.pbnc_loss_ratio > 0, 'No loss in use!'
sub_dir_name = 'runs' + str(args.runs) + 'epochs' + str(args.epochs) + 'early' + str(args.early_stopping) + \
    '100train_ratio' + str(int(100*args.train_ratio)) + '100val_ratio' + str(int(100*args.val_ratio)) + '100seed_ratio' + str(int(100*args.seed_ratio)) + \
        '1000lr' + str(int(1000*args.lr)) + '1000weight_decay' + str(int(1000*args.weight_decay)) + '100dropout' + str(int(100*args.dropout)) 
if args.seed != 0:
    sub_dir_name += 'seed' + str(args.seed)
suffix = 'hidden' + str(args.hidden) + 'trip_ratio' + str(int(100*args.triplet_loss_ratio)) 
suffix += 'sup_ratio' + str(int(100*args.supervised_loss_ratio)) + 'imb_ratio' + str(int(100*args.imbalance_loss_ratio))
if args.imbalance_loss_ratio > 0:
    suffix += 'imb_norm' + str(args.imb_normalization) + 'imb_thre' + str(args.imb_threshold) 
if args.supervised_loss_ratio > 0:
    suffix += '10sup_loss_rat' + str(int(10*args.supervised_loss_ratio))
    if args.triplet_loss_ratio > 0:
        suffix += 'trip_ratio' + str(int(100*args.triplet_loss_ratio))
if args.pbnc_loss_ratio != 1:
    suffix += '10pbnc_loss_rat' + str(int(10*args.pbnc_loss_ratio))

suffix += 'hidden' + str(args.hidden) + 'K' + str(args.K_model) + '100q' + str(int(100*args.q)) + 'trainable_q' + str(args.trainable_q)
num_input_feat = 2
if args.sd_input_feat:
    suffix += 'SdInput'
    num_input_feat = 4
if args.weighted_input_feat:
    suffix += 'WeightedInput'
    if args.weighted_nonnegative_input_feat:
            suffix += 'nonnegative'

rnd.seed(args.seed)
if args.dataset.lower() == '3c':
    args.K = 3
    F = np.array([[0.5, args.gamma, -args.gamma], [1-args.gamma, 0.5, -0.5], [-1+args.gamma, -0.5, 0.5]])
    F_plain = 2
elif args.dataset.lower() == '4c':
    args.K = 4
    F = np.array([[0.5, args.gamma, -args.gamma, args.gamma], [1-args.gamma, 0.5, -args.gamma, -0.5], [-1+args.gamma, -1+args.gamma, 0.5, -1+args.gamma], [-1+args.gamma, -0.5, -args.gamma, 0.5]])
    F_plain = 4
syn_data_str = '1000p' + str(int(1000*args.p)) + 'K' + str(args.K) + '100eta' + str(int(100*args.eta))  + '100gamma' + str(int(100*args.gamma)) + '10rho' + str(int(10*args.size_ratio)) + 'N' + str(args.N)
args.dataset = 'SDSBM/' + args.dataset.lower() + '/' + syn_data_str
save_path = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data', args.dataset+'.pk')
if os.path.exists(save_path):
    print('Loading existing data!')
    data = pk.load(open(save_path, 'rb'))
    A = sp.csr_matrix(data.A)
else:
    print('Generating new data!')
    A, labels = SDSBM(args.N, args.K, args.p, F, args.size_ratio, args.eta)
    A, labels = extract_network(A=A, labels=labels)
    data = SignedData(A=A, y=torch.LongTensor(labels))
    dir_name = os.path.dirname(save_path)
    if os.path.isdir(dir_name) == False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')
    pk.dump(data, open(save_path, 'wb'))
A_abs = A.copy()
A_abs.data = np.abs(A_abs.data)
num_classes = int(data.y.max() - data.y.min() + 1)
if args.weighted_input_feat:
    if args.weighted_nonnegative_input_feat:
        data.x = in_out_degree(data.edge_index, size=int(data.edge_index.max()-data.edge_index.min())+1, signed=args.sd_input_feat, \
            edge_weight=torch.abs(data.edge_weight)).to(device)
    else:
        data.x = in_out_degree(data.edge_index, size=int(data.edge_index.max()-data.edge_index.min())+1, signed=args.sd_input_feat, \
            edge_weight=data.edge_weight).to(device)
else:
    if args.sd_input_feat:
        data.separate_positive_negative()
        x1 = in_out_degree(data.edge_index_p, size=int(data.edge_index.max()-data.edge_index.min())+1).to(device)
        x2 = in_out_degree(data.edge_index_n, size=int(data.edge_index.max()-data.edge_index.min())+1).to(device)
        data.x = torch.concat((x1, x2), 1)
    else:
        data.x = in_out_degree(data.edge_index, size=int(data.edge_index.max()-data.edge_index.min())+1, signed=args.sd_input_feat).to(device)

data.node_split(data_split=args.runs, train_size_per_class=args.train_ratio, val_size_per_class=args.val_ratio, 
    test_size_per_class=1-args.train_ratio-args.val_ratio, seed_size_per_class=args.seed_ratio)
data.separate_positive_negative()
data = data.to(device)
loss_func_ce = torch.nn.NLLLoss()
loss_func_imbalance = Prob_Imbalance_Loss(F_plain)

def train(A_abs_torch_train, features, edge_index, edge_weight, mask, seed_mask, loss_func_pbnc, y):
    model.train()
    Z, log_prob, _, prob = model(features, features, edge_index=edge_index, 
                    edge_weight=edge_weight)
    loss_pbnc = loss_func_pbnc(prob[mask])
    loss_triplet = 0
    if args.triplet_loss_ratio > 0:
        loss_triplet = triplet_loss_node_classification(y=y[seed_mask], Z=Z[seed_mask], n_sample=500, thre=0.1)
    loss_imbalance = 0
    if args.imbalance_loss_ratio:
        loss_imbalance = loss_func_imbalance(prob[mask], A_abs_torch_train,
                num_classes, args.imb_normalization, args.imb_threshold)
    loss_ce = loss_func_ce(log_prob[seed_mask], y[seed_mask])
    loss = args.supervised_loss_ratio*(loss_ce + args.triplet_loss_ratio*loss_triplet) + \
        args.pbnc_loss_ratio * loss_pbnc + args.imbalance_loss_ratio * loss_imbalance
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return loss.detach().item(), train_ari

def val(A_abs_torch_val, features, edge_index, edge_weight, mask, loss_func_pbnc):
    model.eval()
    _, _, _, prob = model(features, features, edge_index=edge_index, 
                    edge_weight=edge_weight)
    loss_pbnc = loss_func_pbnc(prob[mask])
    loss_imbalance = 0
    if args.imbalance_loss_ratio:
        loss_imbalance = loss_func_imbalance(prob[mask], A_abs_torch_val,
                num_classes, args.imb_normalization, args.imb_threshold)
    loss = args.pbnc_loss_ratio * loss_pbnc + args.imbalance_loss_ratio * loss_imbalance
    return loss.detach().item()

def test(features, edge_index, edge_weight, mask, y):
    model.eval()
    with torch.no_grad():
        _, _, _, prob = model(features, features, edge_index=edge_index, 
                    edge_weight=edge_weight)
    test_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return test_ari

res_array = np.zeros((args.runs, 3))
for split in range(data.train_mask.shape[1]):
    best_model_path = os.path.join(dir_name, sub_dir_name, args.method, suffix, str(split) + '.pkl')
    model = MSGNN_node_classification(q=args.q, K=args.K_model, num_features=data.x.shape[1], hidden=args.hidden, label_dim=num_classes, 
    dropout=args.dropout, cached=(not args.trainable_q)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_index = data.train_mask[:, split].cpu().numpy()
    val_index = data.val_mask[:, split].cpu().numpy()
    test_index = data.test_mask[:, split]
    seed_index = data.seed_mask[:, split]
    A_abs_torch_train = scipy_sparse_to_torch_sparse(A_abs[train_index][:, train_index]).to(device)
    A_abs_torch_val = scipy_sparse_to_torch_sparse(A_abs[val_index][:, val_index]).to(device)
    loss_func_pbnc_train = Prob_Balanced_Normalized_Loss(A_p=sp.csr_matrix(data.A_p)[train_index][:, train_index], 
    A_n=sp.csr_matrix(data.A_n)[train_index][:, train_index])
    loss_func_pbnc_val = Prob_Balanced_Normalized_Loss(A_p=sp.csr_matrix(data.A_p)[val_index][:, val_index], 
    A_n=sp.csr_matrix(data.A_n)[val_index][:, val_index])
    best_val_loss = np.inf
    no_improve_epoch = 0
    for epoch in range(args.epochs):
        train_loss, train_ari = train(A_abs_torch_train, data.x, data.edge_index, data.edge_weight, train_index, seed_index, loss_func_pbnc_train, data.y)
        Val_loss = val(A_abs_torch_val, data.x, data.edge_index, data.edge_weight, val_index, loss_func_pbnc_val)
        print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_ARI: {train_ari:.4f}, Val_loss: {Val_loss:.4f}')
        if Val_loss < best_val_loss:
            best_val_loss = Val_loss
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= args.early_stopping:
                break
    print('With {} out of {} epochs:'.format(epoch + 1, args.epochs))
    model.load_state_dict(torch.load(best_model_path))
    train_ari = test(data.x, data.edge_index, data.edge_weight, train_index, data.y)
    val_ari = test(data.x, data.edge_index, data.edge_weight, val_index, data.y)
    test_ari = test(data.x, data.edge_index, data.edge_weight, test_index, data.y)
    print(f'Split: {split:02d}, Train_ARI: {train_ari:.4f}, Val_ARI: {val_ari:.4f}, Test_ARI: {test_ari:.4f}')
    res_array[split] = [train_ari, val_ari, test_ari]
    

print("For dataset{}, {} obtains average train, val and test ARIs: {}".format(args.dataset, args.method, res_array.mean(0)))