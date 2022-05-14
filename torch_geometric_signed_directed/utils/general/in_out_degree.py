from typing import Optional

import torch
import numpy as np
from scipy.sparse import coo_matrix


def in_out_degree(edge_index: torch.LongTensor, size: int, edge_weight: Optional[torch.FloatTensor]=None) -> torch.Tensor:
    r"""
    Get the in degrees and out degrees of nodes

    Arg types:
        * **edge_index** (torch.LongTensor) The edge index from a torch geometric data / DirectedData object . 
        * **size** (int) - The node number.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)

    Return types:
        * **degree** (Torch.Tensor) - The degree matrix (|V|*2).
    """
    
    cpu_edge_index = edge_index.cpu()
    if edge_weight is None:
        edge_weight = np.ones(len(cpu_edge_index.T))
    else:
        edge_weight = edge_weight.cpu().numpy()
    A = coo_matrix((edge_weight, (cpu_edge_index[0], cpu_edge_index[1])),
                   shape=(size, size), dtype=np.float32).tocsr()
    out_degree = np.sum(A, axis=0).T
    in_degree = np.sum(A, axis=1)
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree
