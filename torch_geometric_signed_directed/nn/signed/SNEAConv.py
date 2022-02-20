from typing import Optional, Tuple, Union
from torch_geometric.typing import (PairTensor, OptTensor)
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import LongTensor, Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import coalesce
from torch_geometric.utils import (add_self_loops,
                                   softmax,
                                   remove_self_loops,
                                   negative_sampling,
                                   structured_negative_sampling)


class SNEAConv(MessagePassing):
    r"""The signed graph attentional layers operator from the `"Learning Signed
    Network Embedding via Graph Attention" <https://ojs.aaai.org/index.php/AAAI/article/view/5911>`_ paper

    .. math::
       \mathbf{h}_{i}^{\mathcal{B}(l)}=\tanh \left(\sum_{j \in \hat{\mathcal{N}}_{i}^{+}, k \in \mathcal{N}_{i}^{-}} \alpha_{i j}^{\mathcal{B}(l)} \mathbf{h}_{j}^{\mathcal{B}(l-1)} \mathbf{W}^{\mathcal{B}(l)}
       +\alpha_{i k}^{\mathcal{B}(l)} \mathbf{h}_{k}^{\mathcal{U}(l-1)} \mathbf{W}^{\mathcal{B}(l)}\right)

       \mathbf{h}_{i}^{\mathcal{U}(l)}=\tanh \left(\sum_{j \in \hat{\mathcal{N}}_{i}^{+}, k \in \mathcal{N}_{i}^{-}} \alpha_{i j}^{\mathcal{U}(l)} \mathbf{h}_{j}^{\mathcal{U}(l-1)} \mathbf{W}^{\mathcal{U}(l)}
        +\alpha_{i k}^{\mathcal{U}(l)} \mathbf{h}_{k}^{\mathcal{B}(l-1)} \mathbf{W}^{\mathcal{U}(l)}\right)



    In case :obj:`first_aggr` is :obj:`False`, the layer expects :obj:`x` to be
    a tensor where :obj:`x[:, :in_dim]` denotes the positive node features
    :math:`\mathbf{X}^{(\textrm{pos})}` and :obj:`x[:, in_dim:]` denotes
    the negative node features :math:`\mathbf{X}^{(\textrm{neg})}`.

    Args:
        in_dim (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_dim (int): Size of each output sample.
        first_aggr (bool): Denotes which aggregation formula to use.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool,
        bias: bool = True,
        add_self_loops=True,
        **kwargs
    ):

        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.add_self_loops = add_self_loops

        self.lin_b = Linear(in_dim, out_dim, bias)
        self.lin_u = Linear(in_dim, out_dim, bias)

        self.alpha_u = Linear(self.out_dim * 2, 1)
        self.alpha_b = Linear(self.out_dim * 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()
        self.alpha_b.reset_parameters()
        self.alpha_u.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: LongTensor,
                neg_edge_index: LongTensor):
        """"""
        # propagate_type    e: (x: PairTensor)
        if self.first_aggr:
            h_b = self.lin_b(x)
            h_u = self.lin_u(x)

            edge1, _ = remove_self_loops(pos_edge_index)
            edge, _ = add_self_loops(edge1)
            edge_p = torch.zeros(edge.size(-1), dtype=torch.long)
            # x = torch.stack((h_b, h_b), dim=-1)
            x1 = h_b
            x2 = h_b
            out_b = self.propagate(edge, x1=x1, x2=x2, edge_p=edge_p)

            edge1, _ = remove_self_loops(neg_edge_index)
            edge, _ = add_self_loops(edge1)
            edge_p = torch.zeros(edge.size(-1), dtype=torch.long)
            x1 = h_u
            x2 = h_u
            out_u = self.propagate(edge, x1=x1, x2=x2, edge_p=edge_p)

            return torch.cat([out_b, out_u], dim=-1)

        else:
            F_in = self.in_dim
            x_b = x[..., :F_in]
            x_u = x[..., F_in:]

            edge1, _ = remove_self_loops(pos_edge_index)
            edge1, _ = add_self_loops(edge1)
            edge2, _ = remove_self_loops(neg_edge_index)
            edge = torch.cat([edge1, edge2], dim=-1)
            edge_p1 = torch.zeros(edge1.size(-1), dtype=torch.long)
            edge_p2 = torch.ones(edge2.size(-1), dtype=torch.long)
            edge_p = torch.cat([edge_p1, edge_p2], dim=-1)
            x1 = self.lin_b(x_b)
            x2 = self.lin_b(x_u)
            out_b = self.propagate(edge, x1=x1, x2=x2, edge_p=edge_p)

            edge1, _ = remove_self_loops(neg_edge_index)
            edge1, _ = add_self_loops(edge1)
            edge2, _ = remove_self_loops(pos_edge_index)
            edge = torch.cat([edge1, edge2], dim=-1)
            edge_p1 = torch.zeros(edge1.size(-1), dtype=torch.long)
            edge_p2 = torch.ones(edge2.size(-1), dtype=torch.long)
            edge_p = torch.cat([edge_p1, edge_p2], dim=-1)
            x1 = self.lin_u(x_u)
            x2 = self.lin_u(x_b)
            out_u = self.propagate(edge, x1=x1, x2=x2, edge_p=edge_p)

            return torch.cat([out_b, out_u], dim=-1)

    def message(self, x1_j: Tensor, x2_j: Tensor, x1_i: Tensor, x2_i: Tensor, edge_p: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha1_j = -1 * self.alpha_b(torch.cat([x1_i, x1_j], dim=-1))
        alpha2_j = -1 * self.alpha_u(torch.cat([x2_i, x2_j], dim=-1))
        alpha = torch.stack([alpha1_j, alpha2_j], dim=-1)
        x = torch.stack([x1_j, x2_j], dim=-1)
        alpha = alpha[torch.arange(alpha.size(0)), :, edge_p]
        x = x[torch.arange(x.size(0)), :, edge_p]
        alpha = torch.tanh(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        return x * alpha

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')
