from typing import Tuple, Union

import numpy as np
import scipy.sparse as sp
import networkx as nx
from torch import LongTensor


def extract_network(A: sp.spmatrix, labels: Union[np.array, LongTensor, None] = None, lowest_degree: int = 2, max_iter=10) -> Tuple[sp.spmatrix, np.array]:
    """Find the largest connected component and iteratively only include nodes with degree at least lowest_degree, 
    for at most max_iter iterations, from the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.

    Arg types:
        * **A** (scipy sparse matrix) - Adjacency matrix.
        * **labels** (numpy array or torch.LongTensor, optional) - Node labels, default None.
        * **lowest_degree** (int, optional) - The lowest degree for the output network, default 2.
        * **max_iter** (int, optional) - The maximum number of iterations.

    Return types:
        * **A** (scipy sparse matrix) - Adjacency matrix after fixing degrees and obtaining a connected netework.
        * **labels** (numpy array) - Node labels after fixing degrees and obtaining a connected netework.
    """
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph)
    largest_cc = max(nx.weakly_connected_components(G))
    A_new = A[list(largest_cc)][:, list(largest_cc)]
    labels_new = None
    if labels is not None:
        labels_new = labels[list(largest_cc)]
    G0 = nx.from_scipy_sparse_matrix(A_new, create_using=nx.DiGraph)
    flag = True
    iter_num = 0
    keep = []
    while flag and iter_num < max_iter:
        while flag and iter_num < max_iter:
            iter_num += 1
            remove = [node for node, degree in dict(
                G0.degree()).items() if degree < lowest_degree]
            keep = np.array([node for node, degree in dict(
                G0.degree()).items() if degree >= lowest_degree])
            if len(keep):
                if len(remove):
                    G0.remove_nodes_from(remove)
                else:
                    flag = False
            else:
                lowest_degree -= 1
                print('Nothing to keep, reducing lowest_degree by one to be {}!'.format(
                    lowest_degree))
                G0 = nx.from_scipy_sparse_matrix(
                    A_new, create_using=nx.DiGraph)
                break

    A_new = A[keep][:, keep]
    if labels is not None:
        labels_new = labels[keep]
    return A_new, labels_new
