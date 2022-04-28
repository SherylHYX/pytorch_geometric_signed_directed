import math

import numpy as np


def meta_graph_generation(F_style: str = 'cyclic', K: int = 4, eta: float = 0.05,
                          ambient: bool = False, fill_val: float = 0.5) -> np.array:
    """The meta-graph generation function from the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.

    Arg types:
        * **F_style** (str) - Style of the meta-graph: 'cyclic', 'path', 'complete', 'star' or 'multipartite'.
        * **K** (int) - Number of clusters.
        * **eta** (float) - Noise parameter, 0 <= eta <= 0.5.
        * **ambient** (bool) - Whether there are ambient nodes.
        * **fill_val** (float) - Value to fill in the ambient locations.

    Return types:
        * **F** (NumPy array) - The resulting meta-graph adjacency matrix.
    """
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
