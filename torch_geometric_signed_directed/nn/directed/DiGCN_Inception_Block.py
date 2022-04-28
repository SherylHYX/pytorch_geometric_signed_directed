from typing import Tuple

import torch
from torch.nn import Linear

from .DiGCNConv import DiGCNConv


class DiGCN_InceptionBlock(torch.nn.Module):
    r"""An implementation of the inception block model from the
    `Digraph Inception Convolutional Networks
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.

    Args:
        in_dim (int): Dimention of input.
        out_dim (int): Dimention of output.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super(DiGCN_InceptionBlock, self).__init__()
        self.ln = Linear(in_dim, out_dim)
        self.conv1 = DiGCNConv(in_dim, out_dim)
        self.conv2 = DiGCNConv(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor, edge_index2: torch.LongTensor,
                edge_weight2: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Making a forward pass of the DiGCN inception block model.

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
