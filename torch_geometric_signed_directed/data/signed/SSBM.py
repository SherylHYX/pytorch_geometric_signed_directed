from typing import Optional, Tuple
import math

import numpy as np
import scipy.sparse as sp
import numpy.random as rnd


def SSBM(n: int, k: int, pin: float, etain: float, pout: Optional[float] = None, size_ratio: float = 2,
         etaout: Optional[float] = None, values: str = 'ones') -> Tuple[Tuple[sp.spmatrix, sp.spmatrix], np.array]:
    """A signed stochastic block model graph generator from the
    `SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.

    Arg types:
        * **n** (int) - Number of nodes.
        * **k** (int) - Number of communities.
        * **pin** (float) - Sparsity value within communities.
        * **etain** (float) - Noise value within communities.
        * **pout** (float) - Sparsity value between communities.
        * **etaout** (float) - Noise value between communities.
        * **size_ratio** (float) - The communities have number of nodes multiples of each other, with the largest size_ratio times the number of nodes of the smallest.
        * **values** (string) - Edge weight distribution (within community and without sign flip; otherwise weight is negated):

            1. :obj:`ones`: Weights are 1.

            2. :obj:`"exp"`: Weights are exponentially distributed, with parameter 1.

            3. :obj:`"uniform"`: Weights are uniformly distributed between 0 and 1.

    Return types:
        * **A_p** (sp.spmatrix) - A sparse adjacency matrix for the positive part.
        * **A_n** (sp.spmatrix) - A sparse adjacency matrix for the negative part.
        * **labels** (np.array) - Labels.

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
        ratio_each = np.power(size_ratio, 1/(k-1))
        smallest_size = math.floor(
            n*(1-ratio_each)/(1-np.power(ratio_each, k)))
        size[0] = smallest_size
        if k > 2:
            for i in range(1, k-1):
                size[i] = math.floor(size[i-1] * ratio_each)
        size[k-1] = n - np.sum(size)
    else:  # degenerate case, equaivalent to 'uniform' sizes
        size = [math.floor((i + 1) * n / k) - math.floor((i) * n / k)
                for i in range(k)]
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


def fill(values: str = 'ones') -> float:
    """A filling method for the signed stochastic block model graph generator from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.
    Arg types:
        * **values** (string): Edge weight:

            'ones': Weights are 1.

            'exp': Weights are exponentially distributed, with parameter 1.

            'uniform: Weights are uniformly distributed between 0 and 1.
    Return types:
        * **value** (float): A filled value.
    """
    if values == 'ones':
        return float(1)
    elif values == 'exp':
        return np.random.exponential()
    elif values == 'uniform':
        return np.random.uniform()
