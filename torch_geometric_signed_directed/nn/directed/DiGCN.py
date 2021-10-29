from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class DIGCNConv(MessagePassing):
    r"""The graph convolutional operator from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    The spectral operation is the same with Kipf's GCN.
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the adj matrix on first execution, and will use the
            cached version for further executions.
            Please note that, all the normalized adj matrices (including undirected)
            are calculated in the dataset preprocessing to reduce time comsume.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int, improved: bool=False, cached: bool=True,
                 bias: bool=True, **kwargs):
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
    
    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor, \
        edge_weight: torch.FloatTensor=None) -> torch.FloatTensor:
        """
        Making a forward pass of the DiGCN Convolution layer from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (PyTorch LongTensor) - Edge indices.
            * edge_weight (PyTorch FloatTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * x (PyTorch FloatTensor) - Hidden state tensor for all nodes.
        """
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_weight is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please '
                    'obtain the adj matrix in preprocessing.')
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class DiGCN(torch.nn.Module):
    r"""An implementation of the DiGCN model without inception blocks for node classification from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        num_features (int): Dimention of input features.
        hidden (int): Hidden dimention.
        num_clusters (int): Number of clusters.
        dropout (float): Dropout value.
    """
    def __init__(self, num_features: int, hidden: int, num_clusters: int, dropout: float):
        super(DiGCN, self).__init__()
        self.conv1 = DIGCNConv(num_features, hidden)
        self.conv2 = DIGCNConv(hidden, num_clusters)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor, \
        edge_weight: torch.FloatTensor=None) -> torch.FloatTensor:
        """
        Making a forward pass of the DiGCN node classification model without inception blocks from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (PyTorch LongTensor) - Edge indices.
            * edge_weight (PyTorch FloatTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * x (PyTorch FloatTensor) - Logarithmic class probabilities for all nodes, 
                with shape (num_nodes, num_classes).
        """
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        
        return F.log_softmax(x,dim=1)

class InceptionBlock(torch.nn.Module):
    r"""An implementation of the inception block model from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        in_dim (int): Dimention of input.
        out_dim (int): Dimention of output.
    """
    def __init__(self, in_dim, out_dim):
        super(InceptionBlock, self).__init__()
        self.ln = Linear(in_dim, out_dim)
        self.conv1 = DIGCNConv(in_dim, out_dim)
        self.conv2 = DIGCNConv(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor, \
        edge_weight: torch.FloatTensor, edge_index2: torch.LongTensor, \
        edge_weight2: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Making a forward pass of the DiGCN inception block model from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index, edge_index2 (PyTorch LongTensor) - Edge indices.
            * edge_weight, edge_weight2 (PyTorch FloatTensor) - Edge weights corresponding to edge indices.
        Return types:
            * x0, x1, x2 (PyTorch FloatTensor) - Hidden representations.
        """
        x0 = self.ln(x)
        x1 = self.conv1(x, edge_index, edge_weight)
        x2 = self.conv2(x, edge_index2, edge_weight2)
        return x0, x1, x2

class DiGCN_IB(torch.nn.Module):
    r"""An implementation of the DiGCN model with inception blocks for node classification from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Args:
        num_features (int): Dimention of input features.
        hidden (int): Hidden dimention.
        num_clusters (int): Number of clusters.
        dropout (float): Dropout value.
    """
    def __init__(self, num_features, hidden, num_classes, dropout=0.5):
        super(DiGCN_IB, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(hidden, hidden)
        self.ib3 = InceptionBlock(hidden, num_classes)
        self._dropout = dropout

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()

    def forward(self, features: torch.FloatTensor, \
        edge_index_tuple: Tuple[torch.LongTensor, torch.LongTensor], \
        edge_weight_tuple: Tuple[torch.FloatTensor, torch.FloatTensor]) -> torch.FloatTensor:
        """
        Making a forward pass of the DiGCN node classification model with inception blocks from the
    `Digraph Inception Convolutional Networks" 
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index_tuple (PyTorch LongTensor) - Tuple of edge indices.
            * edge_weight_tuple (PyTorch FloatTensor, optional) - Tuple of edge weights corresponding to edge indices.
        Return types:
            * x (PyTorch FloatTensor) - Logarithmic class probabilities for all nodes, 
                with shape (num_nodes, num_classes).
        """
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0,x1,x2 = self.ib3(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2

        return F.log_softmax(x, dim=1)