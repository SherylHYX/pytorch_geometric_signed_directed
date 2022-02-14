import numpy as np
import scipy.sparse as sp
import torch

def scipy_sparse_to_torch_sparse(A: sp.spmatrix) -> torch.Tensor:
    A = sp.csr_matrix(A)
    return torch.sparse_coo_tensor(torch.LongTensor(np.array(A.nonzero())), torch.FloatTensor(A.data), A.shape)