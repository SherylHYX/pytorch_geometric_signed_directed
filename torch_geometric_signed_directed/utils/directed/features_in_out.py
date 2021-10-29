import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_undirected

def directed_features_in_out(edge_index, size, edge_weight=None):
    if edge_weight is not None:
        a = sp.coo_matrix((edge_weight, edge_index), shape=(size, size)).tocsc()
    else:
        a = sp.coo_matrix((np.ones(len(edge_index[0])), edge_index), shape=(size, size)).tocsc()
    
    out_degree = np.array(a.sum(axis=0))[0]
    out_degree[out_degree == 0] = 1

    in_degree = np.array(a.sum(axis=1))[:, 0]
    in_degree[in_degree == 0] = 1

    # sparse implementation
    a = sp.csr_matrix(a)
    A_in = sp.csr_matrix(np.zeros((size, size)))
    A_out = sp.csr_matrix(np.zeros((size, size)))
    for k in range(size):
        A_in += np.dot(a[k, :].T, a[k, :])/out_degree[k]
        A_out += np.dot(a[:,k], a[:,k].T)/in_degree[k]

    A_in = A_in.tocoo()
    A_out = A_out.tocoo()

    edge_in  = torch.from_numpy(np.vstack((A_in.row,  A_in.col))).long()
    edge_out = torch.from_numpy(np.vstack((A_out.row, A_out.col))).long()
    
    in_weight  = torch.from_numpy(A_in.data).float()
    out_weight = torch.from_numpy(A_out.data).float()
    return to_undirected(edge_index), edge_in, in_weight, edge_out, out_weight