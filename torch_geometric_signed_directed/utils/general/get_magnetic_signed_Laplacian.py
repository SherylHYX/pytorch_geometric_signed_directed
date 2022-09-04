from typing import Optional

import torch
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import add_self_loops, remove_self_loops, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
from scipy.sparse.linalg import eigsh


def get_magnetic_signed_Laplacian(edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None,
                  normalization: Optional[str] = 'sym',
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None,
                  q: Optional[float] = 0.25,
                  return_lambda_max: bool = False, 
                  absolute_degree: bool = True):
    r""" Computes the magnetic signed Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight` from the 
    `MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian <https://arxiv.org/pdf/2209.00546.pdf>`_ paper.
    
    Arg types:
        * **edge_index** (PyTorch LongTensor) - The edge indices.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)
        * **normalization** (str, optional) - The normalization scheme for the magnetic Laplacian (default: :obj:`sym`) -
            1. :obj:`None`: No normalization :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            
            2. :obj:`"sym"`: Symmetric normalization :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
        
        * **dtype** (torch.dtype, optional) - The desired data type of returned tensor in case :obj:`edge_weight=None`. (default: :obj:`None`)
        * **num_nodes** (int, optional) - The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        * **q** (float, optional) - The value q in the paper for phase.
        * **return_lambda_max** (bool, optional) - Whether to return the maximum eigenvalue. (default: :obj:`False`)
        * **absolute_degree** (bool, optional) - Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
    
    Return types:
        * **edge_index** (PyTorch LongTensor) - The edge indices of the magnetic signed Laplacian.
        * **edge_weight.real, edge_weight.imag** (PyTorch Tensor) - Real and imaginary parts of the one-dimensional edge weights for the magnetic signed Laplacian.
        * **lambda_max** (float, optional) - The maximum eigenvalue of the magnetic signed Laplacian, only returns this when required by setting return_lambda_max as True.
    """

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
    sym_abs_attr = torch.cat([torch.abs(edge_weight), torch.abs(edge_weight)], dim=0)
    edge_attr = torch.stack([sym_attr, theta_attr, sym_abs_attr], dim=1)

    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes, "add")

    row, col = edge_index_sym[0], edge_index_sym[1]
    edge_weight_sym = edge_attr[:, 0]
    edge_weight_sym = edge_weight_sym/2

    if absolute_degree:
        edge_weight_abs_sym = edge_attr[:, 2]
        edge_weight_abs_sym = edge_weight_abs_sym/2
        deg = scatter_add(edge_weight_abs_sym, row, dim=0, dim_size=num_nodes)
    else:
        deg = scatter_add(torch.abs(edge_weight_sym), row, dim=0, dim_size=num_nodes) # absolute values for edge weights

    edge_weight_q = torch.exp(1j * 2* np.pi * q * edge_attr[:, 1])

    if normalization is None:
        # L = D_bar_sym - A_sym Hadamard \exp(i \Theta^{(q)}).
        edge_index, _ = add_self_loops(edge_index_sym, num_nodes=num_nodes)
        edge_weight = torch.cat([-edge_weight_sym * edge_weight_q, deg], dim=0)
    elif normalization == 'sym':
        # Compute A_norm = D_bar_sym^{-1/2} A_sym D_bar_sym^{-1/2} Hadamard \exp(i \Theta^{(q)}).
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight_sym * deg_inv_sqrt[col] * edge_weight_q

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index_sym, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp
    if not return_lambda_max:
        return edge_index, edge_weight.real, edge_weight.imag
    else:
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)
        lambda_max = float(lambda_max.real)
        return edge_index, edge_weight.real, edge_weight.imag, lambda_max