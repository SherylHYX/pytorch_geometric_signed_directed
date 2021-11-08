from typing import Any

from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data
from scipy.sparse import spmatrix
from torch import FloatTensor, LongTensor
from numpy import array

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
        A (spmatrix, optional): SciPy sparse adjacency matrix. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None, y: OptTensor = None,
                pos: OptTensor = None, A: spmatrix = None, **kwargs):
        super().__init__(x=x, edge_index=edge_index,
                 edge_attr=edge_attr, y=y,
                 pos=pos, **kwargs)
        if A is None:
            A = to_scipy_sparse_matrix(edge_index, edge_attr)
        else:
            edge_weight = FloatTensor(A.data)
            edge_index = LongTensor(array(A.nonzero()))
        self.A = A
        self.edge_weight = edge_weight
        self.edge_index = edge_index
        

    @property
    def edge_weight(self) -> Any:
        return self['edge_weight'] if 'edge_weight' in self._store else None

    @property
    def A(self) -> spmatrix:
        return self['A'] if 'A' in self._store else None

