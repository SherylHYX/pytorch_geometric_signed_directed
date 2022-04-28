from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DGCNConv import DGCNConv


class DGCN_node_classification(torch.nn.Module):
    r"""An implementation of the DGCN node classification model from `Directed Graph Convolutional Network
    <https://arxiv.org/pdf/2004.13970.pdf>`_ paper.

    Args:
        num_features (int): Dimention of input features.
        hidden (int): Hidden dimention.
        label_dim (int): Output dimension.
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

    def __init__(self, num_features: int, hidden: int, label_dim: int, dropout: Optional[float] = 0.5,
                 improved: bool = False, cached: bool = False):
        super(DGCN_node_classification, self).__init__()
        self.dropout = dropout
        self.dgconv = DGCNConv(improved=improved, cached=cached)
        self.Conv = nn.Conv1d(hidden*3, label_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(num_features,    hidden,   bias=False)
        self.lin2 = torch.nn.Linear(hidden*3, hidden, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, hidden))
        self.bias2 = nn.Parameter(torch.Tensor(1, hidden))
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)
        self.Conv.reset_parameters()

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_in: torch.LongTensor, edge_out: torch.LongTensor,
                in_w: Optional[torch.FloatTensor] = None, out_w: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        """
        Making a forward pass of the DGCN node classification model.

        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (PyTorch LongTensor) - Edge indices.
            * edge_in, edge_out (PyTorch LongTensor) - Edge indices for input and output directions, respectively.
            * in_w, out_w (PyTorch FloatTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * x (PyTorch FloatTensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        x = self.lin1(x)
        x1 = self.dgconv(x, edge_index)
        x2 = self.dgconv(x, edge_in, in_w)
        x3 = self.dgconv(x, edge_out, out_w)

        x1 += self.bias1
        x2 += self.bias1
        x3 += self.bias1

        x = torch.cat((x1, x2, x3), axis=-1)
        x = F.relu(x)

        x = self.lin2(x)
        x1 = self.dgconv(x, edge_index)
        x2 = self.dgconv(x, edge_in, in_w)
        x3 = self.dgconv(x, edge_out, out_w)

        x1 += self.bias2
        x2 += self.bias2
        x3 += self.bias2

        x = torch.cat((x1, x2, x3), axis=-1)
        x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        x = x.permute((0, 2, 1)).squeeze()

        return F.log_softmax(x, dim=1)
