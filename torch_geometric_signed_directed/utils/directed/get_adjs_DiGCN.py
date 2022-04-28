from typing import Union, Optional, Tuple

import torch
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
import scipy
import numpy as np
import scipy.sparse as sp


def fast_appr_power(A, alpha=0.1, max_iter=100,
                    tol=1e-06, personalize=None):
    r""" Computes the fast pagerank adjacency matrix of the graph from the
    `Directed Graph Contrastive Learning 
    <https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf>`_ paper.

    Arg types:
        * **A** (sp.csr_matrix) - Sparse adjacency matrix.
        * **alpha** (float, optional) -alpha used in page rank. Default 0.1.
        * **max_iter** (int -Maximum number of iterations. Default 100.
        * **tol** (flot, optional) -Tolerance. Default 1e-6.
        * **personalize** (array, optional) -if not None, should be an array with the size of the nodes containing probability distributions. It will be normalized automatically. Default None.

    Return types:
        PageRank Scores for the nodes.
    """
    # In Moler's algorithm, $G_{ij}$ represents the existences of an edge
    # from node $j$ to $i$, while we have assumed the opposite!

    n, _ = A.shape
    r = np.asarray(A.sum(axis=1)).reshape(-1)
    k = r.nonzero()[0]
    D_1 = sp.csr_matrix((1 / r[k], (k, k)), shape=(n, n))
    if personalize is None:
        personalize = np.ones(n)
    personalize = personalize.reshape(n, 1)
    s = 1/(1+alpha)/n * personalize
    z_T = ((alpha*(1+alpha)) * (r != 0) + ((1-alpha)/(1+alpha)+alpha*(1+alpha))
           * (r == 0))[scipy.newaxis, :]
    W = (1-alpha) * A.T @ D_1
    x = s
    oldx = np.zeros((n, 1))
    iteration = 0
    while scipy.linalg.norm(x - oldx) > tol:
        oldx = x
        x = W @ x + s @ (z_T @ x)
        iteration += 1
        if iteration >= max_iter:
            break
    x = x / sum(x)
    x = x.reshape(-1)
    p = D_1 * A
    pi_sqrt = sp.diags(np.power(x, 0.5))
    pi_inv_sqrt = sp.diags(np.power(x, -0.5))
    L = (pi_sqrt * p * pi_inv_sqrt + pi_inv_sqrt * p.T * pi_sqrt)/2.0
    L.data[np.isnan(L.data)] = 0.0
    return L, x


def cal_fast_appr(alpha: float, edge_index: torch.LongTensor,
                  num_nodes: Union[int, None], dtype: torch.dtype,
                  edge_weight: Optional[torch.FloatTensor] = None) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    r""" Computes the fast approximate pagerank adjacency matrix of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight` from the
    `Directed Graph Contrastive Learning
    <https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf>`_ paper.

    Arg types:
        * **alpha** (float) -alpha used in approximate personalized page rank.
        * **edge_index** (PyTorch LongTensor) -The edge indices.
        * **num_nodes** (int or None) -The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        * **dtype** (torch.dtype) -The desired data type of returned tensor in case :obj:`edge_weight=None`.
        * **edge_weight** (PyTorch Tensor, optional) -One-dimensional edge weights. (default: :obj:`None`)

    Return types:
        * **edge_index** (PyTorch LongTensor) -The edge indices of the approximate page-rank matrix.
        * **edge_weight** (PyTorch Tensor) -One-dimensional edge weights of the approximate page-rank matrix.
    """
    if edge_weight == None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index

    # from tensor to csr matrix
    sparse_adj = sp.csr_matrix(
        (edge_weight.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())), shape=(num_nodes, num_nodes))

    tol = 1e-6
    L, _ = fast_appr_power(
        sparse_adj, alpha=alpha, tol=tol)

    L = L.tocoo()
    values = L.data
    indices = np.vstack((L.row, L.col))

    L_indices = torch.LongTensor(indices).to(edge_index.device)
    L_values = torch.FloatTensor(values).to(edge_index.device)

    edge_index = L_indices
    edge_weight = L_values

    # sys normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def get_appr_directed_adj(alpha: float, edge_index: torch.LongTensor,
                          num_nodes: Union[int, None], dtype: torch.dtype,
                          edge_weight: Optional[torch.FloatTensor] = None) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    r""" Computes the approximate pagerank adjacency matrix of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight` from the
    `Digraph Inception Convolutional Networks
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.

    Arg types:
        * **alpha** (float) -alpha used in approximate personalized page rank.
        * **edge_index** (PyTorch LongTensor) -The edge indices.
        * **num_nodes** (int or None) -The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        * **dtype** (torch.dtype) -The desired data type of returned tensor in case :obj:`edge_weight=None`.
        * **edge_weight** (PyTorch Tensor, optional) -One-dimensional edge weights. (default: :obj:`None`)

    Return types:
        * **edge_index** (PyTorch LongTensor) -The edge indices of the approximate page-rank matrix.
        * **edge_weight** (PyTorch Tensor) -One-dimensional edge weights of the approximate page-rank matrix.
    """
    if edge_weight == None:
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
    p_dense = torch.sparse.FloatTensor(
        edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1, num_nodes+1]))
    p_v[0:num_nodes, 0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes, 0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes, num_nodes] = alpha
    p_v[num_nodes, num_nodes] = 0.0
    p_ppr = p_v

    eig_value, left_vector = scipy.linalg.eig(
        p_ppr.numpy(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    _, ind = eig_value.sort(descending=True)

    pi = left_vector[:, ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi
    # print(pi)
    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi < 0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    pi_sqrt = pi_sqrt.to(p_ppr.device)
    pi_inv_sqrt = pi_inv_sqrt.to(p_ppr.device)
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) +
         torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0
    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()

    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def get_second_directed_adj(edge_index: torch.LongTensor,
                            num_nodes: Union[int, None], dtype: torch.dtype,
                            edge_weight: Optional[torch.FloatTensor] = None) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    r""" Computes the second-order proximity matrix of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight` from the
    `Digraph Inception Convolutional Networks 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.

    Arg types:
        * **edge_index** (PyTorch LongTensor) -The edge indices.
        * **num_nodes** (int or None) -The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        * **dtype** (torch.dtype) -The desired data type of returned tensor in case :obj:`edge_weight=None`.
        * **edge_weight** (PyTorch Tensor, optional) -One-dimensional edge weights. (default: :obj:`None`)

    Return types:
        * **edge_index** (PyTorch LongTensor) -The edge indices of the approximate page-rank matrix.
        * **dge_weight** (PyTorch Tensor) -One-dimensional edge weights of the approximate page-rank matrix.
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
    p_dense = torch.sparse.FloatTensor(
        edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())

    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
