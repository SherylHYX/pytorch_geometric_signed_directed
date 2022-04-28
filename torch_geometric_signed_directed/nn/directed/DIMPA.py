import torch
from torch.nn.parameter import Parameter

from ..general.conv_base import Conv_Base


class DIMPA(torch.nn.Module):
    r"""The directed mixed-path aggregation model from the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.

    Args:
        hop (int): Number of hops to consider.
        fill_value (float, optional): The layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + fill_value*\mathbf{I}`.
            (default: :obj:`0.5`)
    """

    def __init__(self, hop: int,
                 fill_value: float = 0.5):
        super(DIMPA, self).__init__()
        self._hop = hop
        self._w_s = Parameter(torch.FloatTensor(hop + 1, 1))
        self._w_t = Parameter(torch.FloatTensor(hop + 1, 1))
        self.conv_layer = Conv_Base(fill_value)

        self._reset_parameters()

    def _reset_parameters(self):
        self._w_s.data.fill_(1.0)
        self._w_t.data.fill_(1.0)

    def forward(self, x_s: torch.FloatTensor, x_t: torch.FloatTensor,
                edge_index: torch.FloatTensor,
                edge_weight: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of DIMPA.

        Arg types:
            * **x_s** (PyTorch FloatTensor) - Souce hidden representations.
            * **x_t** (PyTorch FloatTensor) - Target hidden representations.
            * **edge_index** (PyTorch FloatTensor) - Edge indices.
            * **edge_weight** (PyTorch FloatTensor) - Edge weights.
        Return types:
            * **feat** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*input_dim).
        """
        feat_s = self._w_s[0]*x_s
        feat_t = self._w_t[0]*x_t
        curr_s = x_s.clone()
        curr_t = x_t.clone()
        edge_index_t = edge_index[[1, 0]]
        for h in range(1, 1+self._hop):
            curr_s = self.conv_layer(curr_s, edge_index, edge_weight)
            curr_t = self.conv_layer(curr_t, edge_index_t, edge_weight)
            feat_s += self._w_s[h]*curr_s
            feat_t += self._w_t[h]*curr_t

        feat = torch.cat([feat_s, feat_t], dim=1)  # concatenate results

        return feat
