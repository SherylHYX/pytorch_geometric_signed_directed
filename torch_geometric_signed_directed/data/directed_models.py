import math

import numpy as np
import networkx as nx
import numpy.random as rnd


def DSBM(N, K, p, F, size_ratio=1):
    """A directed stochastic block model graph generator.
    Args:
        N: (int) Number of nodes.
        K: (int) Number of clusters.
        p: (float) Sparsity value, edge probability.
        F : meta-graph adjacency matrix to generate edges
        size_ratio: The communities have number of nodes multiples of each other, with the largest size_ratio times the number of nodes of the smallest.
    Returns:
        a,c where a is a sparse N by N matrix of the edges, c is an array of cluster membership.
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

    g = nx.stochastic_block_model(sizes=size, p=p*F, directed=True)
    A = nx.adjacency_matrix(g)[perm][:, perm]

    return A, assign


def fix_network(A, labels):
    '''find the largest connected component and then increase the degree of nodes with low degrees
    Parameters
    ----------
    A : scipy sparse adjacency matrix
    labels : an array of labels of the nodes in the original network
    Returns
    -------
    fixed-degree and connected network, submatrices of A, and subarray of labels
    '''
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph)
    largest_cc = max(nx.weakly_connected_components(G))
    A_new = A[list(largest_cc)][:, list(largest_cc)]
    labels_new = labels[list(largest_cc)]
    G0 = nx.from_scipy_sparse_matrix(A_new, create_using=nx.DiGraph)
    flag = True
    iter_num = 0
    while flag and iter_num < 10:
        iter_num += 1
        remove = [node for node, degree in dict(
            G0.degree()).items() if degree <= 1]
        keep = np.array([node for node, degree in dict(
            G0.degree()).items() if degree > 1])
        if len(remove):
            print(len(G0.nodes()), len(remove))
            G0.remove_nodes_from(remove)
        else:
            flag = False
    print('After {} iteration(s), we extract lcc with degree at least 2 for each node to have network with {} nodes, compared to {} nodes before.'.format(
        iter_num, len(keep), A.shape[0]))
    A_new = A[keep][:, keep]
    labels_new = labels[keep]
    return A_new, labels_new

def meta_graph_generation(F_style='cyclic', K=4, eta=0.05, ambient=False, fill_val=0.5):
    if eta == 0:
        eta = -1
    F = np.eye(K) * 0.5
    # path
    if F_style == 'path':
        for i in range(K-1):
            j = i + 1
            F[i, j] = 1 - eta
            F[j, i] = 1 - F[i, j]
    # cyclic structure
    elif F_style == 'cyclic':
        if K > 2:
            if ambient:
                for i in range(K-1):
                    j = (i + 1) % (K-1)
                    F[i, j] = 1 - eta
                    F[j, i] = 1 - F[i, j]
            else:
                for i in range(K):
                    j = (i + 1) % K
                    F[i, j] = 1 - eta
                    F[j, i] = 1 - F[i, j]
        else:
            if ambient:
                F = np.array([[0.5, 0.5], [0.5, 0.5]])
            else:
                F = np.array([[0.5, 1-eta], [eta, 0.5]])
    # complete meta-graph structure
    elif F_style == 'complete':
        if K > 2:
            for i in range(K-1):
                for j in range(i+1, K):
                    direction = np.random.randint(
                        2, size=1)  # random direction
                    F[i, j] = direction * (1 - eta) + (1-direction) * eta
                    F[j, i] = 1 - F[i, j]
        else:
            F = np.array([[0.5, 1-eta], [eta, 0.5]])
    elif F_style == 'star':
        if K < 3:
            raise Exception("Sorry, star shape requires K at least 3!")
        if ambient and K < 4:
            raise Exception(
                "Sorry, star shape with ambient nodes requires K at least 4!")
        center_ind = math.floor((K-1)/2)
        F[center_ind, ::2] = eta  # only even indices
        F[center_ind, 1::2] = 1-eta  # only odd indices
        F[::2, center_ind] = 1-eta
        F[1::2, center_ind] = eta
    elif F_style == 'multipartite':
        if K < 3:
            raise Exception("Sorry, multipartite shape requires K at least 3!")
        if ambient:
            if K < 4:
                raise Exception(
                    "Sorry, multipartite shape with ambient nodes requires K at least 4!")
            G1_ind = math.ceil((K-1)/9)
            G2_ind = math.ceil((K-1)*3/9)+G1_ind
        else:
            G1_ind = math.ceil(K/9)
            G2_ind = math.ceil(K*3/9)+G1_ind
        F[:G1_ind, G1_ind:G2_ind] = eta
        F[G1_ind:G2_ind, G2_ind:] = eta
        F[G2_ind:, G1_ind:G2_ind] = 1-eta
        F[G1_ind:G2_ind, :G1_ind] = 1-eta
    else:
        raise Exception("Sorry, please give correct F style string!")
    if ambient:
        F[-1, :] = 0
        F[:, -1] = 0
    F[F == 0] = fill_val
    F[F == -1] = 0
    F[F == 2] = 1
    return F