from typing import Optional, Tuple

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_undirected


def directed_features_in_out(edge_index: torch.LongTensor, size: int,
                             edge_weight: Optional[torch.FloatTensor] = None, device: str = 'cpu') -> Tuple[torch.LongTensor, torch.LongTensor,
                                                                                                            torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
    r""" Computes directed in-degree and out-degree features.

    Arg types:
        * **edge_index** (PyTorch LongTensor) - The edge indices.
        * **size** (int or None) - The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)
        * **device** (str, optional) - The device to store the returned values. (default: :str:`cpu`)

    Return types:
        * **index_undirected** (PyTorch LongTensor) - Undirected edge_index.
        * **edge_in** (PyTorch LongTensor) - Inwards edge indices.
        * **in_weight** (PyTorch Tensor) - Inwards edge weights.
        * **edge_out** (PyTorch LongTensor) - Outwards edge indices.
        * **out_weight** (PyTorch Tensor) - Outwards edge weights.
    """
    if edge_weight is not None:
        a = sp.coo_matrix((edge_weight.cpu(), edge_index.cpu()),
                          shape=(size, size)).tocsc()
    else:
        a = sp.coo_matrix(
            (np.ones(len(edge_index[0])), edge_index.cpu()), shape=(size, size)).tocsc()

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
        A_out += np.dot(a[:, k], a[:, k].T)/in_degree[k]

    A_in = A_in.tocoo()
    A_out = A_out.tocoo()

    edge_in = torch.from_numpy(np.vstack((A_in.row,  A_in.col))).long()
    edge_out = torch.from_numpy(np.vstack((A_out.row, A_out.col))).long()

    in_weight = torch.from_numpy(A_in.data).float()
    out_weight = torch.from_numpy(A_out.data).float()
    index_undirected = to_undirected(edge_index)

    device = edge_index.device
    return index_undirected.to(device), edge_in.to(device), in_weight.to(device), edge_out.to(device), out_weight.to(device)
