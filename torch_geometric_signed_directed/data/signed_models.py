import random
import math

import numpy as np
import networkx as nx
import scipy.sparse as sp
import numpy.random as rnd

def SSBM(n, k, pin, etain, pout=None, size_ratio = 2, etaout=None, values='ones'):
    """A signed stochastic block model graph generator.
    Args:
        n: (int) Number of nodes.
        k: (int) Number of communities.
        pin: (float) Sparsity value within communities.
        etain: (float) Noise value within communities.
        pout: (float) Sparsity value between communities.
        etaout: (float) Noise value between communities.
        size_ratio: The communities have number of nodes multiples of each other, with the largest size_ratio times the number of nodes of the smallest.
        values: (string) Edge weight distribution (within community and without sign flip; otherwise weight is negated):
            'ones': Weights are 1.
            'gaussian': Weights are Gaussian, with variance 1 and expectation of 1.#
            'exp': Weights are exponentially distributed, with parameter 1.
            'uniform: Weights are uniformly distributed between 0 and 1.
        Returns:
        (a,b),c where a is a sparse n by n matrix of positive edges, b is a sparse n by n matrix of negative edges c is an array of cluster membership.
    """

    if pout == None:
        pout = pin
    if etaout == None:
        etaout = etain

    rndinrange = math.floor(n * n * pin / 2 + n)
    rndin = rnd.geometric(pin, size=rndinrange)
    flipinrange = math.floor(n * n / 2 * pin + n)
    flipin = rnd.binomial(1, etain, size=flipinrange)
    rndoutrange = math.floor(n * n / 2 * pout + n)
    rndout = rnd.geometric(pout, size=rndoutrange)
    flipoutrange = math.floor(n * n / 2 * pout + n)
    flipout = rnd.binomial(1, etaout, size=flipoutrange)
    assign = np.zeros(n, dtype=int)
    ricount = 0
    rocount = 0
    ficount = 0
    focount = 0

    size = [0] * k


    perm = rnd.permutation(n)
    if size_ratio > 1:
        ratio_each = np.power(size_ratio,1/(k-1))
        smallest_size = math.floor(n*(1-ratio_each)/(1-np.power(ratio_each,k)))
        size[0] = smallest_size
        if k>2:
            for i in range(1,k-1):
                size[i] = math.floor(size[i-1] * ratio_each)
        size[k-1] = n - np.sum(size)
    else: # degenerate case, equaivalent to 'uniform' sizes
        size = [math.floor((i + 1) * n / k) - math.floor((i) * n / k) for i in range(k)]
    tot = size[0]
    cluster = 0
    i = 0
    while i < n:
        if tot == 0:
            cluster += 1
            tot += size[cluster]
        else:
            tot -= 1
            assign[perm[i]] = cluster
            i += 1


    index = -1
    last = [0] * k
    for i in range(k):
        index += size[i]
        last[i] = index

    pdat = []
    prow = []
    pcol = []
    ndat = []
    nrow = []
    ncol = []
    for x in range(n):
        me = perm[x]
        y = x + rndin[ricount]
        ricount += 1
        while y <= last[assign[me]]:
            val = fill(values)
            if flipin[ficount] == 1:
                ndat.append(val)
                ndat.append(val)
                ncol.append(me)
                ncol.append(perm[y])
                nrow.append(perm[y])
                nrow.append(me)
            else:
                pdat.append(val)
                pdat.append(val)
                pcol.append(me)
                pcol.append(perm[y])
                prow.append(perm[y])
                prow.append(me)
            ficount += 1
            y += rndin[ricount]
            ricount += 1
        y = last[assign[me]] + rndout[rocount]
        rocount += 1
        while y < n:
            val = fill(values)
            if flipout[focount] != 1:
                ndat.append(val)
                ndat.append(val)
                ncol.append(me)
                ncol.append(perm[y])
                nrow.append(perm[y])
                nrow.append(me)
            else:
                pdat.append(val)
                pdat.append(val)
                pcol.append(me)
                pcol.append(perm[y])
                prow.append(perm[y])
                prow.append(me)
            focount += 1
            y += rndout[rocount]
            rocount += 1
    return (sp.coo_matrix((pdat, (prow, pcol)), shape=(n, n)).tocsc(),
            sp.coo_matrix((ndat, (nrow, ncol)), shape=(n, n)).tocsc()), assign



def fill(values='ones'):
    if values == 'ones':
        return float(1)
    elif values == 'gaussian':
        return np.random.normal(1)
    elif values == 'exp':
        return np.random.exponential()
    elif values == 'uniform':
        return np.random.uniform()

def polarized_ssbm(total_n=100, num_com=3, N=30, K=2, p=0.1, eta=0.1, size_ratio=1):
    ''' function to generate polarized ssbm models
    Parameters
    ----------
    total_n : total number of nodes in the polarized network
    num_com : number of conflicting communities
    N : an array of labels of the nodes in the original network
    K : number of sub-communities within a conflicting community
    p : probability of existence of an edge
    eta: sign flip probability
    size_ratio : the size ratio of the largest to the smallest block in SSBM and community size. 1 means uniform sizes. should be at least 1.
    Returns
    -------
    large_A_p and large_A_n : positive and negative parts of the polarized network
    large_labels : ordered labels of the nodes, with conflicting communities labeled together, cluster 0 is the background
    conflict_groups: an array indicating which conflicting group the node is in, 0 is background
    '''
    select_num = math.floor(total_n*p/4*total_n) # number of links in large_A_p and large_A_n respectively
    # note that we need to add each link twice for the undirected graph
    tuples_full = []
    for x in range(total_n):
        for y in range(total_n):
            tuples_full.append((x,y))  
    full_idx = random.sample(tuples_full,select_num*2)
    full_idx = list(set([(x[1],x[0]) for x in full_idx])-set([(x[0],x[1]) for x in full_idx]))
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
    large_A_p = sp.coo_matrix((p_dat, (p_row_idx, p_col_idx)), shape=(total_n, total_n)).tolil()
    large_A_n = sp.coo_matrix((n_dat, (n_row_idx, n_col_idx)), shape=(total_n, total_n)).tolil()
    large_labels = np.zeros(total_n)
    conflict_groups = np.zeros(total_n)
    total_n_com = num_com * N # the total number of nodes in communities
    size = [0] * num_com 
    if size_ratio > 1:
        ratio_each = np.power(size_ratio,1/(num_com -1))
        smallest_size = math.floor(total_n_com*(1-ratio_each)/(1-np.power(ratio_each,num_com)))
        size[0] = smallest_size
        if num_com>2:
            for i in range(1,num_com -1):
                size[i] = math.floor(size[i-1] * ratio_each)
        size[num_com-1] = total_n_com - np.sum(size)
    else: # degenerate case, equaivalent to 'uniform' sizes
        size = [math.floor((i + 1) * total_n_com / num_com) - math.floor((i) * total_n_com / num_com) for i in range(num_com)]
    counter = 0 # a counter of how many nodes have already been counted
    for com in range(num_com): 
        com_size = size[com] # the size of this conflicting group, a SSBM
        (A_p, A_n), labels = SSBM(n=com_size, k=K, pin=p, etain=eta, size_ratio=size_ratio)
        large_A_p[counter:counter+com_size,counter:counter+com_size] = A_p
        large_A_n[counter:counter+com_size,counter:counter+com_size] = A_n
        large_labels[counter:counter+com_size] = labels + (2*com + 1)
        conflict_groups[counter:counter+com_size] = com + 1 # start from 1
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
    large_A_p = sp.coo_matrix((large_A_p_values, (p_row_idx, p_col_idx)), shape=(total_n, total_n)).tocsc()
    n_row_idx, n_col_idx = large_A_n.nonzero()
    large_A_n_values = sp.csc_matrix(large_A_n).data
    n_row_idx = perm[n_row_idx]
    n_col_idx = perm[n_col_idx]
    large_A_n = sp.coo_matrix((large_A_n_values, (n_row_idx, n_col_idx)), shape=(total_n, total_n)).tocsc()
    large_labels_old = large_labels.copy()
    conflict_groups_old = conflict_groups.copy()
    for i in range(total_n):
        large_labels[perm[i]] = large_labels_old[i]
        conflict_groups[perm[i]] = conflict_groups_old[i]
    # now fix the network connectedness and degree
    # first we fix connectedness
    G = nx.from_scipy_sparse_matrix(large_A_p-large_A_n)
    largest_cc = max(nx.connected_components(G),key=len)
    A_p_new = sp.lil_matrix(large_A_p[list(largest_cc)][:,list(largest_cc)])
    A_n_new = sp.lil_matrix(large_A_n[list(largest_cc)][:,list(largest_cc)])
    labels_new = large_labels[list(largest_cc)]
    conflict_groups = conflict_groups[list(largest_cc)]
    A_bar=sp.lil_matrix(A_p_new+A_n_new)
    A_bar_row_sum = np.array(sp.lil_matrix.sum(A_bar,axis=1)) # sum over columns to get row sums
    if np.sum(A_bar_row_sum<=2): # only do this fix if few degree node exists
        for i in np.arange(len(labels_new))[(A_bar_row_sum<=2).flatten()]: 
            row_to_fix = A_bar[i].toarray().flatten()
            if sum(row_to_fix!=0)==1: # only do this fix if it is (still) a degree one node, as we may fix the nodes on the way
                # add two more edges, only add to locations currently without edges
                node_idx = (np.arange(len(labels_new))[row_to_fix==0])[random.sample(range(len(labels_new)-sum(row_to_fix!=0)),2)]
                flip_flag = np.random.binomial(size=2,n=1,p=eta) # whether to do sign flip
                for j, flip in zip(node_idx,flip_flag):
                    # fix A_bar and then adjancency matrix
                    A_bar[i,j] = 1 # += 1
                    A_bar[j,i] = 1 # += 1
                    if conflict_groups[i] == conflict_groups[j]: # only apply to conflicting groups
                        if labels_new[j] == labels_new[i]:
                            if flip:
                                A_n_new[i,j] = 1 # += 1
                                A_n_new[j,i] = 1 # += 1
                            else:
                                A_p_new[i,j] = 1 # += 1
                                A_p_new[j,i] = 1 # += 1
                        else:
                            if not flip:
                                A_n_new[i,j] = 1 # += 1
                                A_n_new[j,i] = 1 # += 1
                            else:
                                A_p_new[i,j] = 1 # += 1
                                A_p_new[j,i] = 1 # += 1
                    else:
                        negative = np.random.binomial(size=1, n=1, p= 0.5)[0]
                        if negative:
                            A_n_new[i,j] = 1 # += 1
                            A_n_new[j,i] = 1 # += 1
                        else:
                            A_p_new[i,j] = 1 # += 1
                            A_p_new[j,i] = 1 # += 1
            if sum(row_to_fix!=0)==2: # only do this fix if it is (still) a degree two node, as we may fix the nodes on the way
                # add one more edge, only add to locations currently without edges
                node_idx = (np.arange(len(labels_new))[row_to_fix==0])[np.random.randint(len(labels_new)-sum(row_to_fix!=0),size=1)]
                flip_flag = np.random.binomial(size=1,n=1,p=eta) # whether to do sign flip
                for j, flip in zip(node_idx,flip_flag):
                    # fix A_bar and then adjancency matrix
                    A_bar[i,j] = 1 # += 1
                    A_bar[j,i] = 1 # += 1
                    if conflict_groups[i] == conflict_groups[j]: # only apply to conflicting groups
                        if labels_new[j] == labels_new[i]:
                            if flip:
                                A_n_new[i,j] = 1 # += 1
                                A_n_new[j,i] = 1 # += 1
                            else:
                                A_p_new[i,j] = 1 # += 1
                                A_p_new[j,i] = 1 # += 1
                        else:
                            if not flip:
                                A_n_new[i,j] = 1 # += 1
                                A_n_new[j,i] = 1 # += 1
                            else:
                                A_p_new[i,j] = 1 # += 1
                                A_p_new[j,i] = 1 # += 1
                    else:
                        negative = np.random.binomial(size=1, n=1, p= 0.5)[0]
                        if negative:
                            A_n_new[i,j] = 1 # += 1
                            A_n_new[j,i] = 1 # += 1
                        else:
                            A_p_new[i,j] = 1 # += 1
                            A_p_new[j,i] = 1 # += 1
    return (A_p_new, A_n_new), labels_new, conflict_groups