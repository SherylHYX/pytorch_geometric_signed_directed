from typing import Optional

import torch
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes


def in_out_degree(edge_index: torch.LongTensor, size: Optional[int]=None, signed: bool=False, 
    edge_weight: Optional[torch.FloatTensor]=None) -> torch.Tensor:
    r"""
    Get the in degrees and out degrees of nodes

    Arg types:
        * **edge_index** (torch.LongTensor) The edge index from a torch geometric data / DirectedData object . 
        * **size** (int) - The node number.
        * **signed** (bool, optional) - Whether to take into account signed edge weights and to return signed 4D features. Defualt is False and to only account for absolute degrees.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)

    Return types:
        * **degree** (Torch.Tensor) - The degree matrix (|V|*2) when signed=False, otherwise the degree matrix (|V|*4) with in-pos, in-neg, out-pos, out-neg degrees.
    """
    
    cpu_edge_index = edge_index.cpu()
    size = maybe_num_nodes(edge_index, size)
    if signed:
        if edge_weight is None:
            raise ValueError('Edge weight input should not be None when generating features based on edge signs!')
        else:
            edge_weight = edge_weight.cpu().numpy()
        A = coo_matrix((edge_weight, (cpu_edge_index[0], cpu_edge_index[1])),
                    shape=(size, size), dtype=np.float32).tocsr()
        A_abs = A.copy()
        A_abs.data = np.abs(A_abs.data)
        A_p = (A_abs + A)/2
        A_n = (A_abs - A)/2
        out_pos_degree = np.sum(A_p, axis=0).T
        out_neg_degree = np.sum(A_n, axis=0).T
        in_pos_degree = np.sum(A_p, axis=1)
        in_neg_degree = np.sum(A_n, axis=1)
        degree = torch.from_numpy(np.c_[in_pos_degree, in_neg_degree, out_pos_degree, out_neg_degree]).float()
    else:
        if edge_weight is None:
            edge_weight = np.ones(len(cpu_edge_index.T))
        else:
            edge_weight = np.abs(edge_weight.cpu().numpy())
        A = coo_matrix((edge_weight, (cpu_edge_index[0], cpu_edge_index[1])),
                    shape=(size, size), dtype=np.float32).tocsr()
        out_degree = np.sum(A, axis=0).T
        in_degree = np.sum(A, axis=1)
        degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree
