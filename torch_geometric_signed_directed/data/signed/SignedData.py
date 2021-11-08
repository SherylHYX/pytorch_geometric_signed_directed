from typing import Any

from torch_geometric.typing import OptTensor, Tuple, Union
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data
from torch import FloatTensor, LongTensor
from numpy import array

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
        **kwargs (optional): Additional attributes.
    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None, y: OptTensor = None,
                pos: OptTensor = None, 
                A: Union[Tuple[sp.spmatrix, sp.spmatrix], sp.spmatrix, None] = None, **kwargs):
        super().__init__(x=x, edge_index=edge_index,
                 edge_attr=edge_attr, y=y,
                 pos=pos, **kwargs)
        if A is None:
            if edge_weight is not None:
                edge_attr = edge_weight
            A = to_scipy_sparse_matrix(edge_index, edge_attr)
            A = sp.lil_matrix(A)
            A_abs = sp.lil_matrix(abs(A))
            A_p_scipy = (A_abs + A)/2
            A_n_scipy = (A_abs - A)/2
        elif isinstance(A, tuple):
            A_p_scipy = A[0]
            A_n_scipy = A[1]
            A = A_p_scipy - A_n_scipy
            edge_weight = FloatTensor(A.data)
            edge_index = LongTensor(array(A.nonzero()))
        else:
            edge_weight = FloatTensor(A.data)
            edge_index = LongTensor(array(A.nonzero()))
            A_abs = sp.lil_matrix(abs(A))
            A = sp.lil_matrix(A)
            A_p_scipy = (A_abs + A)/2
            A_n_scipy = (A_abs - A)/2
        self.A = sp.csr_matrix(A)
        self.edge_index_p = LongTensor(array(A_p_scipy.nonzero()))
        self.edge_weight_p = FloatTensor(sp.csr_matrix(A_p_scipy).data)
        self.edge_index_n = LongTensor(array(A_n_scipy.nonzero()))
        self.edge_weight_n = FloatTensor(sp.csr_matrix(A_n_scipy).data)
        self.edge_weight = edge_weight
        self.edge_index = edge_index
        self.A_p = A_p_scipy
        self.A_n = A_n_scipy
        

    @property
    def edge_weight(self) -> Any:
        return self['edge_weight'] if 'edge_weight' in self._store else None

    @property
    def A(self) -> sp.spmatrix:
        return self['A'] if 'A' in self._store else None


    @property
    def edge_weight_p(self) -> Any:
        return self['edge_weight_p'] if 'edge_weight_p' in self._store else None

    @property
    def edge_index_p(self) -> Any:
        return self['edge_index_p'] if 'edge_index_p' in self._store else None

    @property
    def A_p(self) -> sp.spmatrix:
        return self['A_p'] if 'A_p' in self._store else None

    @property
    def edge_weight_n(self) -> Any:
        return self['edge_weight_n'] if 'edge_weight_n' in self._store else None

    @property
    def edge_index_n(self) -> Any:
        return self['edge_index_n'] if 'edge_index_n' in self._store else None

    @property
    def A_n(self) -> sp.spmatrix:
        return self['A_n'] if 'A_n' in self._store else None