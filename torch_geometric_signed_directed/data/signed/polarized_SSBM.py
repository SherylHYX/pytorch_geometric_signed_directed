from typing import Tuple
import random
import math

import numpy as np
import networkx as nx
import scipy.sparse as sp
import numpy.random as rnd

from .SSBM import SSBM


def polarized_SSBM(total_n: int = 100, num_com: int = 3, N: int = 30, K: int = 2, p: float = 0.1, eta: float = 0.1,
                   size_ratio: float = 1) -> Tuple[Tuple[sp.spmatrix, sp.spmatrix], np.array, np.array]:
    """A polarized signed stochastic block model graph generator from the
    `SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.

    Arg types:
        * **total_n** (int) - Total number of nodes in the polarized network.
        * **num_com** (int) - Number of conflicting communities.
        * **N** (int) - Default size of an SSBM community.
        * **K** (int) - Number of blocks(clusters) within a conflicting community.
        * **p** (int) - Probability of existence of an edge.
        * **eta** (float) - Sign flip probability, 0 <= eta <= 0.5.
        * **size_ratio** (float) - The communities have number of nodes multiples of each other, with the largest size_ratio times the number of nodes of the smallest.

    Return types:
        * **A_p_new, A_n_new** (sp.spmatrix) - Positive and negative parts of the polarized network.
        * **labels_new** (np.array) - Ordered labels of the nodes, with conflicting communities labeled together, cluster 0 is the ambient cluster.
        * **conflict_groups** (np.array) - An array indicating which conflicting group the node is in, 0 is ambient.

    """
    select_num = math.floor(
        total_n*p/4*total_n)  # number of links in large_A_p and large_A_n respectively
    # note that we need to add each link twice for the undirected graph
    tuples_full = []
    for x in range(total_n):
        for y in range(total_n):
            tuples_full.append((x, y))
    full_idx = random.sample(tuples_full, select_num*2)
    full_idx = list(set([(x[1], x[0]) for x in full_idx]) -
                    set([(x[0], x[1]) for x in full_idx]))
    select_num = math.floor(len(full_idx)/2)
    p_row_idx = []
    p_col_idx = []
    p_dat = []
    for p_idx in full_idx[:select_num]:
        p_row_idx.append(p_idx[0])
        p_col_idx.append(p_idx[1])
        p_dat.append(1)
        p_row_idx.append(p_idx[1])
        p_col_idx.append(p_idx[0])
        p_dat.append(1)
    n_row_idx = []
    n_col_idx = []
    n_dat = []
    for n_idx in full_idx[select_num:2*select_num]:
        n_row_idx.append(n_idx[0])
        n_col_idx.append(n_idx[1])
        n_dat.append(1)
        n_row_idx.append(n_idx[1])
        n_col_idx.append(n_idx[0])
        n_dat.append(1)
    large_A_p = sp.coo_matrix(
        (p_dat, (p_row_idx, p_col_idx)), shape=(total_n, total_n)).tolil()
    large_A_n = sp.coo_matrix(
        (n_dat, (n_row_idx, n_col_idx)), shape=(total_n, total_n)).tolil()
    large_labels = np.zeros(total_n)
    conflict_groups = np.zeros(total_n)
    total_n_com = num_com * N  # the total number of nodes in communities
    size = [0] * num_com
    if size_ratio > 1:
        ratio_each = np.power(size_ratio, 1/(num_com - 1))
        smallest_size = math.floor(
            total_n_com*(1-ratio_each)/(1-np.power(ratio_each, num_com)))
        size[0] = smallest_size
        if num_com > 2:
            for i in range(1, num_com - 1):
                size[i] = math.floor(size[i-1] * ratio_each)
        size[num_com-1] = total_n_com - np.sum(size)
    else:  # degenerate case, equaivalent to 'uniform' sizes
        size = [math.floor((i + 1) * total_n_com / num_com) -
                math.floor((i) * total_n_com / num_com) for i in range(num_com)]
    counter = 0  # a counter of how many nodes have already been counted
    for com in range(num_com):
        com_size = size[com]  # the size of this conflicting group, a SSBM
        (A_p, A_n), labels = SSBM(n=com_size, k=K,
                                  pin=p, etain=eta, size_ratio=size_ratio)
        large_A_p[counter:counter+com_size, counter:counter+com_size] = A_p
        large_A_n[counter:counter+com_size, counter:counter+com_size] = A_n
        large_labels[counter:counter+com_size] = labels + (2*com + 1)
        conflict_groups[counter:counter+com_size] = com + 1  # start from 1
        counter += com_size
    # do permutation
    # perm[i] is the new index for node i (i is the old index)
    # label of perm[i] should therefore be the current label of node i, similar for conflict group number
    np.random.seed(2020)
    perm = rnd.permutation(total_n)
    p_row_idx, p_col_idx = large_A_p.nonzero()
    large_A_p_values = sp.csc_matrix(large_A_p).data
    p_row_idx = perm[p_row_idx]
    p_col_idx = perm[p_col_idx]
    large_A_p = sp.coo_matrix(
        (large_A_p_values, (p_row_idx, p_col_idx)), shape=(total_n, total_n)).tocsc()
    n_row_idx, n_col_idx = large_A_n.nonzero()
    large_A_n_values = sp.csc_matrix(large_A_n).data
    n_row_idx = perm[n_row_idx]
    n_col_idx = perm[n_col_idx]
    large_A_n = sp.coo_matrix(
        (large_A_n_values, (n_row_idx, n_col_idx)), shape=(total_n, total_n)).tocsc()
    large_labels_old = large_labels.copy()
    conflict_groups_old = conflict_groups.copy()
    for i in range(total_n):
        large_labels[perm[i]] = large_labels_old[i]
        conflict_groups[perm[i]] = conflict_groups_old[i]
    # now fix the network connectedness and degree
    # first we fix connectedness
    G = nx.from_scipy_sparse_matrix(large_A_p-large_A_n)
    largest_cc = max(nx.connected_components(G), key=len)
    A_p_new = sp.lil_matrix(large_A_p[list(largest_cc)][:, list(largest_cc)])
    A_n_new = sp.lil_matrix(large_A_n[list(largest_cc)][:, list(largest_cc)])
    labels_new = large_labels[list(largest_cc)]
    conflict_groups = conflict_groups[list(largest_cc)]
    A_bar = sp.lil_matrix(A_p_new+A_n_new)
    # sum over columns to get row sums
    A_bar_row_sum = np.array(sp.lil_matrix.sum(A_bar, axis=1))
    if np.sum(A_bar_row_sum <= 2):  # only do this fix if few degree node exists
        for i in np.arange(len(labels_new))[(A_bar_row_sum <= 2).flatten()]:
            row_to_fix = A_bar[i].toarray().flatten()
            # only do this fix if it is (still) a degree one node, as we may fix the nodes on the way
            if sum(row_to_fix != 0) == 1:
                # add two more edges, only add to locations currently without edges
                node_idx = (np.arange(len(labels_new))[row_to_fix == 0])[
                    random.sample(range(len(labels_new)-sum(row_to_fix != 0)), 2)]
                flip_flag = np.random.binomial(
                    size=2, n=1, p=eta)  # whether to do sign flip
                for j, flip in zip(node_idx, flip_flag):
                    # fix A_bar and then adjancency matrix
                    A_bar[i, j] = 1  # += 1
                    A_bar[j, i] = 1  # += 1
                    # only apply to conflicting groups
                    if conflict_groups[i] == conflict_groups[j]:
                        if labels_new[j] == labels_new[i]:
                            if flip:
                                A_n_new[i, j] = 1  # += 1
                                A_n_new[j, i] = 1  # += 1
                            else:
                                A_p_new[i, j] = 1  # += 1
                                A_p_new[j, i] = 1  # += 1
                        else:
                            if not flip:
                                A_n_new[i, j] = 1  # += 1
                                A_n_new[j, i] = 1  # += 1
                            else:
                                A_p_new[i, j] = 1  # += 1
                                A_p_new[j, i] = 1  # += 1
                    else:
                        negative = np.random.binomial(size=1, n=1, p=0.5)[0]
                        if negative:
                            A_n_new[i, j] = 1  # += 1
                            A_n_new[j, i] = 1  # += 1
                        else:
                            A_p_new[i, j] = 1  # += 1
                            A_p_new[j, i] = 1  # += 1
            # only do this fix if it is (still) a degree two node, as we may fix the nodes on the way
            if sum(row_to_fix != 0) == 2:
                # add one more edge, only add to locations currently without edges
                node_idx = (np.arange(len(labels_new))[row_to_fix == 0])[
                    np.random.randint(len(labels_new)-sum(row_to_fix != 0), size=1)]
                flip_flag = np.random.binomial(
                    size=1, n=1, p=eta)  # whether to do sign flip
                for j, flip in zip(node_idx, flip_flag):
                    # fix A_bar and then adjancency matrix
                    A_bar[i, j] = 1  # += 1
                    A_bar[j, i] = 1  # += 1
                    # only apply to conflicting groups
                    if conflict_groups[i] == conflict_groups[j]:
                        if labels_new[j] == labels_new[i]:
                            if flip:
                                A_n_new[i, j] = 1  # += 1
                                A_n_new[j, i] = 1  # += 1
                            else:
                                A_p_new[i, j] = 1  # += 1
                                A_p_new[j, i] = 1  # += 1
                        else:
                            if not flip:
                                A_n_new[i, j] = 1  # += 1
                                A_n_new[j, i] = 1  # += 1
                            else:
                                A_p_new[i, j] = 1  # += 1
                                A_p_new[j, i] = 1  # += 1
                    else:
                        negative = np.random.binomial(size=1, n=1, p=0.5)[0]
                        if negative:
                            A_n_new[i, j] = 1  # += 1
                            A_n_new[j, i] = 1  # += 1
                        else:
                            A_p_new[i, j] = 1  # += 1
                            A_p_new[j, i] = 1  # += 1
    return (A_p_new, A_n_new), labels_new, conflict_groups
