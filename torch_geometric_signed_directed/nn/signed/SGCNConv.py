from typing import Union

from torch_geometric.typing import PairTensor, Adj
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class SGCNConv(MessagePassing):
    r"""The signed graph convolutional operator from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper

    .. math::
        \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w , \mathbf{x}_v \right]

        \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{neg})}
        \left[ \frac{1}{|\mathcal{N}^{-}(v)|} \sum_{w \in \mathcal{N}^{-}(v)}
        \mathbf{x}_w , \mathbf{x}_v \right]

    if :obj:`first_aggr` is set to :obj:`True`, and

    .. math::
        \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w^{(\textrm{pos})}, \frac{1}{|\mathcal{N}^{-}(v)|}
        \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{neg})},
        \mathbf{x}_v^{(\textrm{pos})} \right]

        \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w^{(\textrm{neg})}, \frac{1}{|\mathcal{N}^{-}(v)|}
        \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{pos})},
        \mathbf{x}_v^{(\textrm{neg})} \right]

    otherwise.
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
        norm_emb (bool, optional): Whether to normalize embeddings. (default: :obj:`False`)
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
        norm_emb: bool = False,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.norm_emb = norm_emb

        if first_aggr:
            self.lin_b = Linear(2 * in_dim, out_dim, bias)
            self.lin_u = Linear(2 * in_dim, out_dim, bias)
        else:
            self.lin_b = Linear(3 * in_dim, out_dim, bias)
            self.lin_u = Linear(3 * in_dim, out_dim, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj) -> Tensor:

        # propagate_type    e: (x: PairTensor)
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if self.first_aggr:
            out_b = self.propagate(pos_edge_index, x=x)
            out_b = self.lin_b(torch.cat([out_b, x[1]], dim=-1))

            out_u = self.propagate(neg_edge_index, x=x)
            out_u = self.lin_u(torch.cat([out_u, x[1]], dim=-1))
            out = torch.cat([out_b, out_u], dim=-1)
        else:
            F_in = self.in_dim
            out_b1 = self.propagate(pos_edge_index, x=(
                x[0][..., :F_in], x[1][..., :F_in]))
            out_b2 = self.propagate(neg_edge_index, x=(
                x[0][..., F_in:], x[1][..., F_in:]))
            out_b = torch.cat([out_b1, out_b2, x[1][..., :F_in]], dim=-1)
            out_b = self.lin_b(out_b)

            out_u1 = self.propagate(pos_edge_index, x=(
                x[0][..., F_in:], x[1][..., F_in:]))
            out_u2 = self.propagate(neg_edge_index, x=(
                x[0][..., :F_in], x[1][..., :F_in]))
            out_u = torch.cat([out_u1, out_u2, x[1][..., F_in:]], dim=-1)
            out_u = self.lin_u(out_u)

            out = torch.cat([out_b, out_u], dim=-1)
        if self.norm_emb:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: PairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')
