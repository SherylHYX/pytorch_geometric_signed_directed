from typing import Tuple
import math

import numpy as np
import scipy.sparse as sp
import networkx as nx
import numpy.random as rnd


def SDSBM(N: int, K: int, p: float, F: np.array, size_ratio: float=1, eta: float=0.1) -> Tuple[sp.spmatrix, np.array]:
    """A signed directed stochastic block model graph generator from the 
    `MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian <https://arxiv.org/pdf/2209.00546.pdf>`_ paper.
    
    Arg types:
        * **N** (int) - Number of nodes.
        * **K** (int) - Number of clusters.
        * **p** (float) - Sparsity value, edge probability.
        * **F** (np.array) - The meta-graph adjacency matrix to generate edges.
        * **size_ratio** (float) - The communities have number of nodes multiples of each other, \
            with the largest size_ratio times the number of nodes of the smallest. \
            A geometric sequence is generated to denote the node size of each cluster based on the size_ratio. 
        * **eta** (float) - Sign flip probability.
    
    Return types:
        * **a** (sp.csr_matrix) - a is a sparse N by N matrix of the edges.
        * **c** (np.array) - c is an array of cluster membership.
    """

    assign = np.zeros(N, dtype=int)

    size = [0] * K

    perm = rnd.permutation(N)
    if size_ratio > 1:
        ratio_each = np.power(size_ratio, 1/(K-1))
        smallest_size = math.floor(
            N*(1-ratio_each)/(1-np.power(ratio_each, K)))
        size[0] = smallest_size
        if K > 2:
            for i in range(1, K-1):
                size[i] = math.floor(size[i-1] * ratio_each)
        size[K-1] = N - np.sum(size)
    else:  # degenerate case, equaivalent to 'uniform' sizes
        size = [math.floor((i + 1) * N / K) -
                math.floor((i) * N / K) for i in range(K)]
    labels = []
    for i, s in enumerate(size):
        labels.extend([i]*s)
    labels = np.array(labels)
    # permutation
    assign = labels[perm]

    g = nx.stochastic_block_model(sizes=size, p=p*np.abs(F), directed=True)
    A = nx.adjacency_matrix(g)[perm][:, perm]
    
    # match expected signs
    for i in range(K):
        for j in range(K):
            if F[i, j] < 0:
                ind_bool_row = np.where(assign==i)[0]
                ind_bool_col = np.where(assign==j)[0]
                A[tuple(np.meshgrid(ind_bool_row, ind_bool_col))] *= -1
                
    # flip the signs
    flip_ind = np.random.choice(np.arange(len(A.data)), size=int(len(A.data)*eta), replace=False)
    A.data[flip_ind] *= -1

    return A, assign