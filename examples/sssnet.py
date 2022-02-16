import argparse
from sklearn.metrics import adjusted_rand_score
import scipy.sparse as sp
import torch
from torch_geometric_signed_directed.nn import SSSNET_node_clustering
from torch_geometric_signed_directed.data import SignedData, SSBM
from torch_geometric_signed_directed.utils import (Prob_Balanced_Normalized_Loss, 
extract_network, triplet_loss_node_classification)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--triplet_loss_ratio', type=float, default=0.1,
                    help='Ratio of triplet loss to cross entropy loss in supervised loss part. Default 0.1.')
parser.add_argument('--supervised_loss_ratio', type=float, default=50,
                    help='Ratio of factor of supervised loss part to self-supervised loss part.')
args = parser.parse_args()

num_classes = 5
eta = 0.1
num_nodes = 100
p = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


(A_p_scipy, A_n_scipy), labels = SSBM(num_nodes, num_classes, p, eta)
A = A_p_scipy - A_n_scipy
A, labels = extract_network(A=A, labels=labels)
data = SignedData(A=A, y=torch.LongTensor(labels))
data.set_signed_Laplacian_features(num_classes)
data.node_split(train_size_per_class=0.8, val_size_per_class=0.1, test_size_per_class=0.1, seed_size_per_class=0.1)
data.separate_positive_negative()
loss_func_ce = torch.nn.NLLLoss()

model = SSSNET_node_clustering(nfeat=data.x.shape[1], dropout=0.5, hop=2, fill_value=0.5, 
                        hidden=32, nclass=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
data = data.to(device)

def train(features, edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, mask, seed_mask, loss_func_pbnc, y):
    model.train()
    Z, log_prob, _, prob = model(edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, features)
    loss_pbnc = loss_func_pbnc(prob[mask])
    loss_triplet = triplet_loss_node_classification(y=y[seed_mask], Z=Z[seed_mask], n_sample=500, thre=0.1)
    loss_ce = loss_func_ce(log_prob[seed_mask], y[seed_mask])
    loss = args.supervised_loss_ratio*(loss_ce +
                                    args.triplet_loss_ratio*loss_triplet) + loss_pbnc
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return loss.detach().item(), train_ari

def test(features, edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, mask, y):
    model.eval()
    with torch.no_grad():
        _, _, _, prob = model(edge_index_p, edge_weight_p,
                edge_index_n, edge_weight_n, features)
    test_ari = adjusted_rand_score(y[mask].cpu(), (torch.argmax(prob, dim=1)).cpu()[mask])
    return test_ari

data.x = torch.FloatTensor(data.x).to(device)
data.edge_index = data.edge_index.to(device)
data.edge_weight = data.edge_weight.to(device)

for split in range(data.train_mask.shape[1]):
    train_index = data.train_mask[:, split].cpu().numpy()
    val_index = data.val_mask[:, split]
    test_index = data.test_mask[:, split]
    seed_index = data.seed_mask[:, split]
    loss_func_pbnc = Prob_Balanced_Normalized_Loss(A_p=sp.csr_matrix(data.A_p)[train_index][:, train_index], 
    A_n=sp.csr_matrix(data.A_n)[train_index][:, train_index])
    for epoch in range(args.epochs):
        train_loss, train_ari = train(data.x, data.edge_index_p, data.edge_weight_p,
                                        data.edge_index_n, data.edge_weight_n, train_index, seed_index, loss_func_pbnc, data.y)
        Val_ari = test(data.x, data.edge_index_p, data.edge_weight_p,
                        data.edge_index_n, data.edge_weight_n, val_index, data.y)
        print(f'Split: {split:02d}, Epoch: {epoch:03d}, Train_Loss: {train_loss:.4f}, Train_ARI: {train_ari:.4f}, Val_ARI: {Val_ari:.4f}')
    
    test_ari = test(data.x, data.edge_index_p, data.edge_weight_p,
                    data.edge_index_n, data.edge_weight_n, test_index, data.y)
    print(f'Split: {split:02d}, Test_ARI: {test_ari:.4f}')
    model._reset_parameters_undirected()