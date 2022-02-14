from typing import Union, List, Optional

from torch_geometric.typing import OptTensor
from torch_geometric.utils import to_scipy_sparse_matrix, is_undirected
from torch_geometric.data import Data
import scipy.sparse as sp
import numpy as np
from torch import FloatTensor, LongTensor
from sklearn.preprocessing import StandardScaler

from ...utils.general.node_split import node_class_split

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
        init_data (Data, optional): Initial data object, whose attributes will be inherited. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None, y: OptTensor = None,
                pos: OptTensor = None, A: sp.spmatrix = None, init_data: Optional[Data]=None, **kwargs):
        super().__init__(x=x, edge_index=edge_index,
                 edge_attr=edge_attr, y=y,
                 pos=pos, **kwargs)
        if A is None:
            A = to_scipy_sparse_matrix(edge_index, edge_weight)
        else:
            edge_index = LongTensor(np.array(A.nonzero()))
        self.A = A
        self.edge_weight = FloatTensor(A.data)
        self.edge_index = edge_index
        if init_data is not None:
            self.inherit_attributes(init_data)
        
    @property
    def is_directed(self) -> bool:
        return not is_undirected(self.edge_index)

    @property
    def is_weighted(self) -> bool:
        return self.edge_weight.max() != self.edge_weight.min()

    def to_unweighted(self):
        self.A = to_scipy_sparse_matrix(self.edge_index, None)
        self.edge_weight = FloatTensor(self.A.data)

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

    def inherit_attributes(self, data:Data): 
        for k in data.to_dict().keys():
            if k not in self.to_dict().keys():
                setattr(self, k, getattr(data, k))

    def node_split(self, train_size: Union[int,float]=None, val_size: Union[int,float]=None, 
                test_size: Union[int,float]=None, seed_size: Union[int,float]=None,
                train_size_per_class: Union[int,float]=None, val_size_per_class: Union[int,float]=None,
                test_size_per_class: Union[int,float]=None, seed_size_per_class: Union[int,float]=None, 
                seed: List[int]=[], data_split: int=10):
        r""" Train/Val/Test/Seed split for node classification tasks.

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
            data_split (int, optional): number of splits (Default : 10)

        """
        self = node_class_split(self, train_size=train_size, val_size=val_size, 
        test_size=test_size, seed_size=seed_size, train_size_per_class=train_size_per_class,
        val_size_per_class=val_size_per_class, test_size_per_class=test_size_per_class,
        seed_size_per_class=seed_size_per_class, seed=seed, data_split=data_split)

