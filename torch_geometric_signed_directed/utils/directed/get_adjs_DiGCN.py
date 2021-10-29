# modified from https://github.com/flyingtango/DiGCN/blob/main/code/get_adj.py
from typing import Union, Optional, Tuple

import torch
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
import scipy

def get_appr_directed_adj(alpha: float, edge_index: torch.LongTensor, \
    num_nodes: Union[int, None], dtype: torch.dtype, \
    edge_weight: Optional[torch.FloatTensor]=None) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    r""" Computes the approximate pagerank adjacency matrix of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight` from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        alpha (float): alpha used in approximate personalized page rank.
        edge_index (PyTorch LongTensor): The edge indices.
        num_nodes (int or None): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`.
        dtype (torch.dtype): The desired data type of returned tensor
            in case :obj:`edge_weight=None`.
        edge_weight (PyTorch Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
    Return types:
        edge_index (PyTorch LongTensor): The edge indices of the approximate page-rank matrix.
        edge_weight (PyTorch Tensor): One-dimensional edge weights of the approximate page-rank matrix.
    """
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)  
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) 
    deg_inv = deg.pow(-1) 
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1]))
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v 

    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(),left=True,right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:,ind[0]] # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_second_directed_adj(edge_index: torch.LongTensor, \
    num_nodes: Union[int, None], dtype: torch.dtype, \
    edge_weight: Optional[torch.FloatTensor]=None) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    r""" Computes the second-order proximity matrix of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight` from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        edge_index (PyTorch LongTensor): The edge indices.
        num_nodes (int or None): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`.
        dtype (torch.dtype): The desired data type of returned tensor
            in case :obj:`edge_weight=None`.
        edge_weight (PyTorch Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
    Return types:
        edge_index (PyTorch LongTensor): The edge indices of the approximate page-rank matrix.
        edge_weight (PyTorch Tensor): One-dimensional edge weights of the approximate page-rank matrix.
    """
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight 
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    
    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())
    
    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values
    
    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]