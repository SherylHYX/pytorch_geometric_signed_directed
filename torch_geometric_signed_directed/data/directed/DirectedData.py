from typing import Any

from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_scipy_sparse_matrix, is_undirected
from torch_geometric.data import Data
import scipy.sparse as sp
import numpy as np
from torch import FloatTensor, LongTensor
from sklearn.preprocessing import StandardScaler

class DirectedData(Data):
    r"""A data object describing a homogeneous directed graph.

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
        A (sp.spmatrix, optional): SciPy sparse adjacency matrix. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None, y: OptTensor = None,
                pos: OptTensor = None, A: sp.spmatrix = None, **kwargs):
        super().__init__(x=x, edge_index=edge_index,
                 edge_attr=edge_attr, y=y,
                 pos=pos, **kwargs)
        if A is None:
            A = to_scipy_sparse_matrix(edge_index, edge_attr)
        else:
            edge_weight = FloatTensor(A.data)
            edge_index = LongTensor(np.array(A.nonzero()))
        self.A = A
        self.edge_weight = edge_weight
        self.edge_index = edge_index
        

    @property
    def edge_weight(self) -> Any:
        return self['edge_weight'] if 'edge_weight' in self._store else None

    @property
    def A(self) -> sp.spmatrix:
        return self['A'] if 'A' in self._store else None
    
    @property
    def is_directed(self) -> bool:
        return not is_undirected(self.edge_index)

    def set_hermitian_features(self, k:int=2):
        """ create Hermitian feature  (rw normalized)
        inputs:
        k : (int) Half of the dimension of features. Default is 2.
        """
        A = self.A
        H = (A-A.transpose()) * 1j
        H_abs = np.abs(H)  # (np.real(H).power(2) + np.imag(H).power(2)).power(0.5)
        D_abs_inv = sp.diags(1/np.array(H_abs.sum(1))[:, 0])
        H_rw = D_abs_inv.dot(H)
        u, _, _ = sp.linalg.svds(H_rw, k=k)
        features_SVD = np.concatenate((np.real(u), np.imag(u)), axis=1)
        scaler = StandardScaler().fit(features_SVD)
        features_SVD = scaler.transform(features_SVD)
        self.x = features_SVD

