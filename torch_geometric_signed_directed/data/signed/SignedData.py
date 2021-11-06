from torch_geometric.typing import OptTensor, Tuple, Union

import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data
from torch import FloatTensor, LongTensor
from numpy import array

def SignedData(x: OptTensor = None, edge_index: OptTensor = None,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None, y: OptTensor = None,
                pos: OptTensor = None, 
                A: Union[Tuple[sp.spmatrix, sp.spmatrix], sp.spmatrix, None] = None, **kwargs):
    r"""A function to generate a data object describing a homogeneous signed graph.

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
    A = sp.csr_matrix(A)
    edge_index_p = LongTensor(array(A_p_scipy.nonzero()))
    edge_weight_p = FloatTensor(sp.csr_matrix(A_p_scipy).data)
    edge_index_n = LongTensor(array(A_n_scipy.nonzero()))
    edge_weight_n = FloatTensor(sp.csr_matrix(A_n_scipy).data)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, A=A, \
        A_p=A_p_scipy, A_n=A_n_scipy, edge_weight=edge_weight, \
            edge_index_p=edge_index_p, edge_weight_p=edge_weight_p,\
                edge_index_n=edge_index_n, edge_weight_n=edge_weight_n,**kwargs)
