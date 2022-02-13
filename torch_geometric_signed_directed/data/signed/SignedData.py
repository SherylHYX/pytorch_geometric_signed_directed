from typing import Any, Optional

from torch_geometric.typing import OptTensor, Tuple, Union
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data
from torch import FloatTensor, LongTensor
import numpy as np

def sqrtinvdiag(M: sp.spmatrix) -> sp.csc_matrix:
    """Inverts and square-roots a positive diagonal matrix.
    Args:
        M (scipy sparse matrix): matrix to invert
    Returns:
        scipy sparse matrix of inverted square-root of diagonal
    """

    d = M.diagonal()
    dd = [1 / max(np.sqrt(x), 1 / 999999999) for x in d]

    return sp.dia_matrix((dd, [0]), shape=(len(d), len(d))).tocsc()

class SignedData(Data):
    r"""A data object describing a homogeneous signed graph.

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        edge_weight (Tensor, optional): Edge weights with shape
            :obj:`[num_edges,]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth labels
            with arbitrary shape. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        A (sp.spmatrix or a tuple of sp.spmatrix, optional): SciPy sparse adjacency matrix,
            or a tuple of the positive and negative parts. (default: :obj:`None`)
        init_data (Data, optional): Initial data object, whose attributes will be inherited. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None, y: OptTensor = None,
                pos: OptTensor = None, 
                A: Union[Tuple[sp.spmatrix, sp.spmatrix], sp.spmatrix, None] = None, 
                init_data: Optional[Data] = None, **kwargs):
        super().__init__(x=x, edge_index=edge_index,
                 edge_attr=edge_attr, y=y,
                 pos=pos, **kwargs)
        if A is None:
            A = to_scipy_sparse_matrix(edge_index, edge_weight)
        elif isinstance(A, tuple):
            A_p_scipy = A[0]
            A_n_scipy = A[1]
            A = A_p_scipy - A_n_scipy

        self.A = sp.csr_matrix(A)
        self.A.eliminate_zeros()
        self.edge_weight = FloatTensor(self.A.data)
        self.edge_index = LongTensor(np.array(A.nonzero()))
        if init_data is not None:
            self.inherit_attributes(init_data)

    def separate_positive_negative(self):
        ind = self.edge_weight > 0
        self.edge_index_p = self.edge_index[:,ind]
        self.edge_weight_p = self.edge_weight[ind]
        ind = self.edge_weight < 0
        self.edge_index_n = self.edge_index[:,ind]
        self.edge_weight_n = - self.edge_weight[ind]
        self.A_p = to_scipy_sparse_matrix(self.edge_index_p, self.edge_weight_p)
        self.A_n = to_scipy_sparse_matrix(self.edge_index_n, self.edge_weight_n)

    @property
    def is_signed(self) -> bool:
        return bool(self.edge_weight.max()*self.edge_weight.min() < 0)

    def set_signed_Laplacian_features(self, k: int=2):
        """generate the graph features using eigenvectors of the signed Laplacian matrix.
        Args:
            k (int): The dimension of the features. Default is 2.
        """
        self.separate_positive_negative()
        A_p = self.A_p
        A_n = self.A_n
        A = (A_p - A_n).tocsc()
        D_p = sp.diags(A_p.sum(axis=0).tolist(), [0]).tocsc()
        D_n = sp.diags(A_n.sum(axis=0).tolist(), [0]).tocsc()
        Dbar = (D_p + D_n)
        d = sqrtinvdiag(Dbar)
        normA = d * A * d
        L = sp.eye(A_p.shape[0], format="csc") - normA # normalized symmetric signed Laplacian
        (vals, vecs) = sp.linalg.eigsh(L, int(k), maxiter=A_p.shape[0], which='SA')
        vecs = vecs / vals  # weight eigenvalues by eigenvectors, since smaller eigenvectors are more likely to be informative
        self.x = vecs


    def set_spectral_adjacency_reg_features(self, k: int=2, normalization: Optional[int]=None, tau_p=None, tau_n=None, \
        eigens=None, mi=None):
        """generate the graph features using eigenvectors of the regularised adjacency matrix.
        Args:
            k (int): The dimension of the features. Default is 2.
            normalization (string): How to normalise for cluster size:
                'none' - do not normalise.
                'sym' - symmetric normalization.
                'rw' - random walk normalization.
                'sym_sep' - separate symmetric normalization of positive and negative parts.
                'rw_sep' - separate random walk normalization of positive and negative parts.
            tau_p (int): Regularisation coefficient for positive adjacency matrix.
            tau_n (int): Regularisation coefficient for negative adjacency matrix.

        Other parameters:
            eigens (int): The number of eigenvectors to take. Defaults to k.
            mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.
        """
        self.separate_positive_negative()
        A = (self.A_p - self.A_n).tocsc()
        A_p = sp.csc_matrix(self.A_p)
        A_n = sp.csc_matrix(self.A_n)
        D_p = sp.diags(A_p.sum(axis=0).tolist(), [0]).tocsc()
        D_n = sp.diags(A_n.sum(axis=0).tolist(), [0]).tocsc()
        Dbar = (D_p + D_n)
        d = sqrtinvdiag(Dbar)
        size = A_p.shape[0]
        if eigens == None:
            eigens = k

        if mi == None:
            mi = size

        if tau_p == None or tau_n == None:
            tau_p = 0.25 * np.mean(Dbar.data) / size
            tau_n = 0.25 * np.mean(Dbar.data) / size

        symmetric = True

        p_tau = A_p.copy().astype(np.float32)
        n_tau = A_n.copy().astype(np.float32)
        p_tau.data += tau_p
        n_tau.data += tau_n

        Dbar_c = size - Dbar.diagonal()

        Dbar_tau_s = (p_tau + n_tau).sum(axis=0) + (Dbar_c * abs(tau_p - tau_n))[None, :]

        Dbar_tau = sp.diags(Dbar_tau_s.tolist(), [0])

        if normalization is None:
            matrix = A
            delta_tau = tau_p - tau_n

            def mv(v):
                return matrix.dot(v) + delta_tau * v.sum()


        elif normalization == 'sym':
            d = sqrtinvdiag(Dbar_tau)
            matrix = d * A * d
            dd = d.diagonal()
            tau_dd = (tau_p - tau_n) * dd

            def mv(v):
                return matrix.dot(v) + tau_dd * dd.dot(v)

        elif normalization == 'sym_sep':

            diag_corr = sp.diags([size * tau_p] * size).tocsc()
            dp = sqrtinvdiag(D_p + diag_corr)

            matrix = dp * A_p * dp

            diag_corr = sp.diags([size * tau_n] * size).tocsc()
            dn = sqrtinvdiag(D_n + diag_corr)

            matrix = matrix - (dn * A_n * dn)

            dpd = dp.diagonal()
            dnd = dn.diagonal()
            tau_dp = tau_p * dpd
            tau_dn = tau_n * dnd

            def mv(v):
                return matrix.dot(v) + tau_dp * dpd.dot(v) - tau_dn * dnd.dot(v)

        else:
            raise NameError('Error in choosing normalization!')

        matrix_o = sp.linalg.LinearOperator(matrix.shape, matvec=mv)

        if symmetric:
            (w, v) = sp.linalg.eigsh(matrix_o, int(eigens), maxiter=mi, which='LA')
        else:
            (w, v) = sp.linalg.eigs(matrix_o, int(eigens), maxiter=mi, which='LR')

        v = v * w  # weight eigenvalues by eigenvectors, since larger eigenvectors are more likely to be informative
        self.x = v
    
    def inherit_attributes(self, data:Data): 
        for k in data.to_dict().keys():
            if k not in self.to_dict().keys():
                setattr(self, k, getattr(data, k))
