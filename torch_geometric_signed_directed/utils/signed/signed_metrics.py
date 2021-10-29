import torch
import scipy.sparse as sp

from ..general.data_utils import scipy_sparse_to_torch_sparse

class Prob_Balanced_Ratio_Loss(torch.nn.Module):
    r"""An implementation of the probablistic balanced ratio cut loss function from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.
    Args:
        A_p, A_n (scipy sparse matrices): positive and negative parts of adjacency matrix A.
    """
    def __init__(self, A_p: sp.spmatrix, A_n: sp.spmatrix):
        super(Prob_Balanced_Ratio_Loss, self).__init__()
        D_p = sp.diags(A_p.transpose().sum(
            axis=0).tolist(), [0]).tocsc()
        mat = D_p - (A_p - A_n)
        self.mat = scipy_sparse_to_torch_sparse(mat)

    def forward(self, prob: torch.FloatTensor) -> torch.Tensor:
        """Making a forward pass of the probablistic balanced ratio cut loss function from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.
        Args:
            prob: (PyTorch FloatTensor) Prediction probability matrix made by the model
        
        Returns:
            loss value.
        """
        device = prob.device
        mat = self.mat.to(device)
        result = torch.zeros(1).to(device)
        for k in range(prob.shape[-1]):       
            prob_vector_mat = prob[:, k, None]
            denominator = (torch.matmul(torch.transpose(prob_vector_mat, 0, 1),prob_vector_mat) + 1)[0,0]    # avoid dividing by zero
            numerator = (torch.matmul(torch.transpose(prob_vector_mat, 0, 1),torch.matmul(mat,prob_vector_mat)))[0,0]

            result += numerator/denominator
        return result

class Prob_Balanced_Normalized_Loss(torch.nn.Module):
    r"""An implementation of the probablistic balanced normalized cut loss function from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.
    Args:
        A_p, A_n (scipy sparse matrices): positive and negative parts of adjacency matrix A.
    """
    def __init__(self, A_p: sp.spmatrix, A_n: sp.spmatrix):
        super(Prob_Balanced_Normalized_Loss, self).__init__()
        D_p = sp.diags(A_p.transpose().sum(
            axis=0).tolist(), [0]).tocsc()
        D_n = sp.diags(A_n.transpose().sum(
            axis=0).tolist(), [0]).tocsc()
        self.D_bar = scipy_sparse_to_torch_sparse(D_p + D_n)
        mat = D_p - (A_p - A_n)
        self.mat = scipy_sparse_to_torch_sparse(mat)

    def forward(self, prob: torch.FloatTensor) -> torch.Tensor:
        """Making a forward pass of the probablistic balanced normalized cut loss function from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.
        Args:
            prob: (PyTorch FloatTensor) Prediction probability matrix made by the model
        
        Returns:
            loss value.
        """
        device = prob.device
        epsilon = torch.FloatTensor([1e-6]).to(device)
        mat = self.mat.to(device)
        D_bar = self.D_bar.to(device)
        result = torch.zeros(1).to(device)
        for k in range(prob.shape[-1]):
            prob_vector_mat = prob[:, k, None]
            denominator = torch.matmul(torch.transpose(prob_vector_mat, 0, 1),torch.matmul(D_bar,prob_vector_mat))[0,0] + epsilon    # avoid dividing by zero
            numerator = (torch.matmul(torch.transpose(prob_vector_mat, 0, 1),torch.matmul(mat,prob_vector_mat)))[0,0]

            result += numerator/denominator
        return result

class Unhappy_ratio(torch.nn.Module):
    r"""A calculation of the ratio of unhappy edges among all edges from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.
    Args:
        A_p, A_n (scipy sparse matrices): positive and negative parts of adjacency matrix A.
    """
    def __init__(self, A_p: sp.spmatrix, A_n: sp.spmatrix):
        super(Unhappy_ratio, self).__init__()
        D_p = sp.diags(A_p.transpose().sum(
            axis=0).tolist(), [0]).tocsc()
        mat = D_p - (A_p - A_n)
        self.mat = scipy_sparse_to_torch_sparse(mat)
        self.num_edges = len((A_p - A_n).nonzero()[0])

    def forward(self, prob: torch.FloatTensor) -> torch.Tensor:
        """Making a forward pass of the calculation of the ratio of unhappy edges among all edges from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.
        Args:
            prob: (PyTorch FloatTensor) Prediction probability matrix made by the model
        
        Returns:
            loss value.
        """
        device = prob.device
        mat = self.mat.to(device)
        result = torch.zeros(1).to(device)
        for k in range(prob.shape[-1]):
            prob_vector_mat = prob[:, k, None]
            numerator = (torch.matmul(torch.transpose(prob_vector_mat, 0, 1),torch.matmul(mat,prob_vector_mat)))[0,0]
            result += numerator
        return result/self.num_edges

