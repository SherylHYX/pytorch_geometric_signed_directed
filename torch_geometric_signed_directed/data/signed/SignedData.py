from typing import Optional, List

from torch_geometric.typing import OptTensor, Tuple, Union
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix, is_undirected
from torch_geometric.data import Data
from torch import FloatTensor, LongTensor
import numpy as np

from ...utils.general import node_class_split, link_class_split


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

        self.A = A.tocoo()
        self.edge_weight = FloatTensor(self.A.data)
        self.edge_index = LongTensor(np.array(self.A.nonzero()))
        self.num_nodes = self.A.shape[0]
        if init_data is not None:
            self.inherit_attributes(init_data)

    def separate_positive_negative(self):
        ind = self.edge_weight > 0
        self.edge_index_p = self.edge_index[:, ind]
        self.edge_weight_p = self.edge_weight[ind]
        ind = self.edge_weight < 0
        self.edge_index_n = self.edge_index[:, ind]
        self.edge_weight_n = - self.edge_weight[ind]
        self.A_p = to_scipy_sparse_matrix(
            self.edge_index_p, self.edge_weight_p, num_nodes=self.num_nodes)
        self.A_n = to_scipy_sparse_matrix(
            self.edge_index_n, self.edge_weight_n, num_nodes=self.num_nodes)

    def clear_separate_attributes(self):
        for name in ['edge_index_p', 'edge_index_n', 'edge_weight_p', 'edge_weight_n', 'A_p', 'A_n']:
            delattr(self, name)

    @property
    def is_signed(self) -> bool:
        return bool(self.edge_weight.max()*self.edge_weight.min() < 0)

    @property
    def is_directed(self) -> bool:
        return not is_undirected(self.edge_index, self.edge_weight)

    @property
    def is_weighted(self) -> bool:
        self.separate_positive_negative()
        res = self.edge_weight_p.max() != self.edge_weight_p.min(
        ) or self.edge_weight_n.max() != self.edge_weight_n.min()
        self.clear_separate_attributes()
        return bool(res)

    def to_unweighted(self):
        if hasattr(self, 'edge_weight'):
            self.edge_weight = self.edge_weight.sign()
            self.A = to_scipy_sparse_matrix(self.edge_index, self.edge_weight)
        if hasattr(self, 'edge_weight_p'):
            self.separate_positive_negative()

    def set_signed_Laplacian_features(self, k: int = 2):
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
        # normalized symmetric signed Laplacian
        L = sp.eye(A_p.shape[0], format="csc") - normA
        (vals, vecs) = sp.linalg.eigs(
            L, int(k), maxiter=A_p.shape[0], which='LR')
        vecs = vecs / vals  # weight eigenvalues by eigenvectors, since smaller eigenvectors are more likely to be informative
        self.x = FloatTensor(vecs)
        self.clear_separate_attributes()

    def set_spectral_adjacency_reg_features(self, k: int = 2, normalization: Optional[int] = None, tau_p=None, tau_n=None,
                                            eigens=None, mi=None):
        """generate the graph features using eigenvectors of the regularised adjacency matrix.

        Args:
            k (int): The dimension of the features. Default is 2.
            normalization (string): How to normalise for cluster size:

                1. :obj:`none`: No normalization.

                2. :obj:`"sym"`: Symmetric normalization
                :math:`\mathbf{A} <- \mathbf{D}^{-1/2} \mathbf{A}
                \mathbf{D}^{-1/2}`

                3. :obj:`"rw"`: Random-walk normalization
                :math:`\mathbf{A} <- \mathbf{D}^{-1} \mathbf{A}`

                4. :obj:`"sym_sep"`: Symmetric normalization for the positive and negative parts separately.

                5. :obj:`"rw_sep"`: Random-walk normalization for the positive and negative parts separately.

            tau_p (int): Regularisation coefficient for positive adjacency matrix.
            tau_n (int): Regularisation coefficient for negative adjacency matrix.
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

        p_tau = A_p.copy().astype(np.float32)
        n_tau = A_n.copy().astype(np.float32)
        p_tau.data += tau_p
        n_tau.data += tau_n

        Dbar_c = size - Dbar.diagonal()

        Dbar_tau_s = (p_tau + n_tau).sum(axis=0) + \
            (Dbar_c * abs(tau_p - tau_n))[None, :]

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

        (w, v) = sp.linalg.eigs(matrix_o, int(eigens), maxiter=mi, which='LR')

        v = v * w  # weight eigenvalues by eigenvectors, since larger eigenvectors are more likely to be informative
        self.x = FloatTensor(v)
        self.clear_separate_attributes()

    def inherit_attributes(self, data: Data):
        for k in data.to_dict().keys():
            if k not in self.to_dict().keys():
                setattr(self, k, getattr(data, k))

    def node_split(self, train_size: Union[int, float] = None, val_size: Union[int, float] = None,
                   test_size: Union[int, float] = None, seed_size: Union[int, float] = None,
                   train_size_per_class: Union[int, float] = None, val_size_per_class: Union[int, float] = None,
                   test_size_per_class: Union[int, float] = None, seed_size_per_class: Union[int, float] = None,
                   seed: List[int] = [], data_split: int = 2):
        r""" Train/Val/Test/Seed split for node classification tasks. 
        The size parameters can either be int or float.
        If a size parameter is int, then this means the actual number, if it is float, then this means a ratio.
        ``train_size`` or ``train_size_per_class`` is mandatory, with the former regardless of class labels.
        Validation and seed masks are optional. Seed masks here masks nodes within the training set, e.g., in a semi-supervised setting as described in the
        `SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper. 
        If test_size and test_size_per_class are both None, all the remaining nodes after selecting training (and validation) nodes will be included.

        Args:
            data (torch_geometric.data.Data or DirectedData, required): The data object for data split.
            train_size (int or float, optional): The size of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
            val_size (int or float, optional): The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
            test_size (int or float, optional): The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled. 
                        (Default: None. All nodes not selected for training/validation are used for testing)
            seed_size (int or float, optional): The size of random splits for the seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
            train_size_per_class (int or float, optional): The size per class of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.  
            val_size_per_class (int or float, optional): The size per class of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
            test_size_per_class (int or float, optional): The size per class of random splits for the testing dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
                        (Default: None. All nodes not selected for training/validation are used for testing)
            seed_size_per_class (int or float, optional): The size per class of random splits for seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.  
            seed (An empty list or a list with the length of data_split, optional): The random seed list for each data split.
            data_split (int, optional): number of splits (Default : 2)

        """
        self = node_class_split(self, train_size=train_size, val_size=val_size,
                                test_size=test_size, seed_size=seed_size, train_size_per_class=train_size_per_class,
                                val_size_per_class=val_size_per_class, test_size_per_class=test_size_per_class,
                                seed_size_per_class=seed_size_per_class, seed=seed, data_split=data_split)

    def link_split(self, size: int = None, splits: int = 2, prob_test: float = 0.15,
                   prob_val: float = 0.05, task: str = 'sign', seed: int = 0, ratio: float = 1.0, maintain_connect: bool = False, device: str = 'cpu') -> dict:
        r"""Get train/val/test dataset for the link sign prediction task. 

        Arg types:
            * **data** (torch_geometric.data.Data or DirectedData object) - The input dataset.
            * **prob_val** (float, optional) - The proportion of edges selected for validation (Default: 0.05).
            * **prob_test** (float, optional) - The proportion of edges selected for testing (Default: 0.15).
            * **splits** (int, optional) - The split size (Default: 10).
            * **size** (int, optional) - The size of the input graph. If none, the graph size is the maximum index of nodes plus 1 (Default: None).
            * **task** (str, optional) - The evaluation task: four_class_signed_digraph (four-class sign and direction prediction); five_class_signed_digraph (five-class sign, direction and existence prediction); sign (link sign prediction). (Default: 'sign')
            * **seed** (int, optional) - The random seed for positve edge selection (Default: 0). Negative edges are selected by pytorch geometric negative_sampling.
            * **maintain_connect** (bool, optional) - If maintaining connectivity when removing edges for validation and testing. The connectivity is maintained by obtaining edges in the minimum spanning tree/forest first. These edges will not be removed for validation and testing. (Default: False).
            * **ratio** (float, optional) - The maximum ratio of edges used for dataset generation. (Default: 1.0)
            * **device** (int, optional) - The device to hold the return value (Default: 'cpu').

        Return types:
            * **datasets** - A dict include training/validation/testing splits of edges and labels. For split index i:

                1. datasets[i]['graph'] (torch.LongTensor): the observed edge list after removing edges for validation and testing.

                2. datasets[i]['train'/'val'/'testing']['edges'] (List): the edge list for training/validation/testing.

                3. datasets[i]['train'/'val'/'testing']['label'] (List): the labels of edges:

                    * If task == "four_class_signed_digraph": 0 (the positive directed edge exists in the graph), 
                        1 (the negative directed edge exists in the graph), 2 (the positive edge of the reversed direction exists),
                        3 (the edge of the reversed direction exists). 
                        The undirected edges in the directed input graph are removed to avoid ambiguity.
                    
                    * If task == "five_class_signed_digraph": 0 (the positive directed edge exists in the graph), 
                        1 (the negative directed edge exists in the graph), 2 (the positive edge of the reversed direction exists),
                        3 (the edge of the reversed direction exists), 4 (the edge doesn't exist in both directions). 
                        The undirected edges in the directed input graph are removed to avoid ambiguity.

                    * If task == "sign": 0 (negative edge), 1 (positive edge). 
        """
        return link_class_split(self, size, splits, prob_test, prob_val, task, seed, maintain_connect, ratio, device)
