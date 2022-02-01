"""Two fast implementations of PageRank:
    An exact solution using a sparse linear system solver,
    and an a power method approximation.
    Both solutions are taking full advantage of sparse matrix calculations.

    [Reference]:
    Cleve Moler. 2011. Experiments with MATLAB (Electronic ed.).
    MathWorks, Inc.
    Code borrows from:
    __author__ = "Armin Sajadi"
    __copyright__ = "Copyright 2015, The Wikisim Project"
    __email__ = "asajadi@gmail.com"
"""
# uncomment
from __future__ import division

import scipy
import scipy.sparse as sp
import scipy.spatial
import scipy.sparse.linalg
import numpy as np


def fast_appr_power(A, alpha=0.1, max_iter=100,
                    tol=1e-06, personalize=None, reverse=False):
    """ Calculates PageRank given a csr graph

    Inputs:
    -------
    A: a csr graph.
    p: damping factor
    max_iter: maximum number of iterations
    personlize: if not None, should be an array with the size of the nodes
                containing probability distributions.
                It will be normalized automatically.
    reverse: If true, returns the reversed-PageRank

    Returns:
    --------
    PageRank Scores for the nodes

    """
    # In Moler's algorithm, $G_{ij}$ represents the existences of an edge
    # from node $j$ to $i$, while we have assumed the opposite!

    if reverse:
        A = A.T
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


def fast_appr(A, alpha=0.1,
              personalize=None, reverse=False):
    """ Calculates PageRank given a csr graph

    Inputs:
    -------

    G: a csr graph.
    p: damping factor
    personlize: if not None, should be an array with the size of the nodes
                containing probability distributions.
                It will be normalized automatically
    reverse: If true, returns the reversed-PageRank

    outputs
    -------

    PageRank Scores for the nodes

    """
    # In Moler's algorithm, $A_{ij}$ represents the existences of an edge
    # from node $j$ to $i$, while we have assumed the opposite!
    if reverse:
        A = A.T
    n, _ = A.shape
    r = np.asarray(A.sum(axis=1)).reshape(-1)
    k = r.nonzero()[0]
    D_1 = sp.csr_matrix((1 / r[k], (k, k)), shape=(n, n))
    if personalize is None:
        personalize = np.ones(n)
    personalize = personalize.reshape(n, 1)
    s = alpha/(1+alpha)/n * personalize
    I = sp.eye(n)
    x = sp.linalg.spsolve((I - (1-alpha) * A.T @ D_1), s)
    x = x / x.sum()
    p = D_1 * A
    pi_sqrt = sp.diags(np.power(x, 0.5))
    pi_inv_sqrt = sp.diags(np.power(x, -0.5))
    L = (pi_sqrt * p * pi_inv_sqrt + pi_inv_sqrt * p.T * pi_sqrt)/2.0
    return L, x
