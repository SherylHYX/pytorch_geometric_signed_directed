
import torch
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from torch_sparse import coalesce


def create_spectral_features(
    pos_edge_index: torch.LongTensor,
    neg_edge_index: torch.LongTensor,
    node_num: int,
    dim: int
) -> torch.FloatTensor:

    edge_index = torch.cat(
        [pos_edge_index, neg_edge_index], dim=1)
    N = node_num
    edge_index = edge_index.to(torch.device('cpu'))
    
    pos_val = torch.full(
        (pos_edge_index.size(1), ), 2, dtype=torch.float)
    neg_val = torch.full(
        (neg_edge_index.size(1), ), 0, dtype=torch.float)
    val = torch.cat([pos_val, neg_val], dim=0)

    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    val = torch.cat([val, val], dim=0)

    edge_index, val = coalesce(edge_index, val, N, N)
    val = val - 1

    # Borrowed from:
    # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
    edge_index = edge_index.detach().numpy()
    val = val.detach().numpy()
    A = sp.coo_matrix((val, edge_index), shape=(N, N))
    svd = TruncatedSVD(n_components=dim, n_iter=128)
    svd.fit(A)
    x = svd.components_.T
    return torch.from_numpy(x).to(torch.float)
