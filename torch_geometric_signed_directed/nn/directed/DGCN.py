from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class DGCNConv(MessagePassing):
    r"""An implementatino of the graph convolutional operator from the
    `Directed Graph Convolutional Network" 
    <https://arxiv.org/pdf/2004.13970.pdf>`_ paper.
    The same as Kipf's GCN but remove trainable weights.
    Args:
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, 
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(DGCNConv, self).__init__(**kwargs)

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """
        Making a forward pass of the graph convolutional operator from the
    `Directed Graph Convolutional Network" 
    <https://arxiv.org/pdf/2004.13970.pdf>`_ paper.
        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (Adj) - Edge indices.
            * edge_weight (OptTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * out (PyTorch FloatTensor) - Hidden state tensor for all nodes.
        """
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class DGCN(torch.nn.Module):
    r"""An implementation of the DGCN node classification model from `Directed Graph Convolutional Network" 
    <https://arxiv.org/pdf/2004.13970.pdf>`_ paper.
    Args:
        input_dim (int): Dimention of input features.
        filter_num (int): Hidden dimention.
        out_dim (int): Output dimension.
        dropout (float, optional): Dropout value. Default: None.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
    """
    def __init__(self, input_dim: int, filter_num: int, out_dim: int, dropout: Optional[float]=None, \
        improved: bool = False, cached: bool = False):
        super(DGCN, self).__init__()
        self.dropout = dropout
        self.dgconv = DGCNConv(improved=improved, cached=cached)
        self.Conv = nn.Conv1d(filter_num*3, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim,    filter_num,   bias=False)
        self.lin2 = torch.nn.Linear(filter_num*3, filter_num, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, filter_num))
        self.bias2 = nn.Parameter(torch.Tensor(1, filter_num))
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor, \
        edge_in: torch.LongTensor, edge_out: torch.LongTensor, \
        in_w: Optional[torch.FloatTensor]=None, out_w: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        """
        Making a forward pass of the DGCN node classification model from `Directed Graph Convolutional Network" 
    <https://arxiv.org/pdf/2004.13970.pdf>`_ paper.
        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (PyTorch LongTensor) - Edge indices.
            * edge_in, edge_out (PyTorch LongTensor) - Edge indices for input and output directions, respectively.
            * in_w, out_w (PyTorch FloatTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * x (PyTorch FloatTensor) - Logarithmic class probabilities for all nodes, 
                with shape (num_nodes, num_classes).
        """
        x = self.lin1(x)
        x1 = self.dgconv(x, edge_index)
        x2 = self.dgconv(x, edge_in, in_w)
        x3 = self.dgconv(x, edge_out, out_w)
        
        x1 += self.bias1
        x2 += self.bias1
        x3 += self.bias1

        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        x = self.lin2(x)
        x1 = self.dgconv(x, edge_index)
        x2 = self.dgconv(x, edge_in, in_w)
        x3 = self.dgconv(x, edge_out, out_w)

        x1 += self.bias2
        x2 += self.bias2
        x3 += self.bias2

        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1)
