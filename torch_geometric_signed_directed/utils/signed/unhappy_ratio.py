import torch
import scipy.sparse as sp

from ..general.scipy_sparse_to_torch_sparse import scipy_sparse_to_torch_sparse


class Unhappy_Ratio(torch.nn.Module):
    r"""A calculation of the ratio of unhappy edges among all edges from the
    `SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.

    Args:
        A_p (scipy sparse matrices): Positive part of adjacency matrix A.
        A_n (scipy sparse matrices): Negative part of adjacency matrix A.
    """

    def __init__(self, A_p: sp.spmatrix, A_n: sp.spmatrix):
        super(Unhappy_Ratio, self).__init__()
        D_p = sp.diags(A_p.transpose().sum(
            axis=0).tolist(), [0]).tocsc()
        mat = D_p - (A_p - A_n)
        self.mat = scipy_sparse_to_torch_sparse(mat)
        self.num_edges = len((A_p - A_n).nonzero()[0])

    def forward(self, prob: torch.FloatTensor) -> torch.Tensor:
        """Making a forward pass of the calculation of the ratio of unhappy edges among all edges.
        Arg types:
            * prob (PyTorch FloatTensor) - Prediction probability matrix made by the model

        Return types:
            * loss value (torch.Tensor).
        """
        device = prob.device
        mat = self.mat.to(device)
        result = torch.zeros(1).to(device)
        for k in range(prob.shape[-1]):
            prob_vector_mat = prob[:, k, None]
            numerator = (torch.matmul(torch.transpose(
                prob_vector_mat, 0, 1), torch.matmul(mat, prob_vector_mat)))[0, 0]
            result += numerator
        return result/self.num_edges
