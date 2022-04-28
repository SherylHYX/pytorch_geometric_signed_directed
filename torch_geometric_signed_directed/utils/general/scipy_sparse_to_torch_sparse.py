import numpy as np
import scipy.sparse as sp
import torch


def scipy_sparse_to_torch_sparse(A: sp.spmatrix) -> torch.Tensor:
    r""" Convert scipy.sparse.spmatrix to torch sparse coo tensor

    Arg types:
        * **A** (sparse.spmatrix): The scipy sparse matrix to be converted.

    Return types:
        * **obj** (torch.sparse_coo_tensor): The returned torch.sparse_coo_tensor.
    """
    A = sp.csr_matrix(A)
    return torch.sparse_coo_tensor(torch.LongTensor(np.array(A.nonzero())), torch.FloatTensor(A.data), A.shape)
