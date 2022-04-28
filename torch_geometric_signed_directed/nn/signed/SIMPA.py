from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from ..general.conv_base import Conv_Base


class SIMPA(nn.Module):
    r"""The signed mixed-path aggregation model from the
    `SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.

    Args:
        hop (int): Number of hops to consider.
        fill_value (float): Value for added self-loops for the positive part of the adjacency matrix.
        directed (bool, optional): Whether the input network is directed or not. (default: :obj:`False`)
    """

    def __init__(self, hop: int, fill_value: float, directed: bool = False):
        super(SIMPA, self).__init__()
        self._hop_p = hop + 1
        self._hop_n = int((1+hop)*hop/2)  # the number of enemy representations
        self._undirected = not directed
        self.conv_layer_p = Conv_Base(fill_value)
        self.conv_layer_n = Conv_Base(0.0)

        if self._undirected:
            self._w_p = Parameter(torch.FloatTensor(self._hop_p, 1))
            self._w_n = Parameter(torch.FloatTensor(self._hop_n, 1))

            self._reset_parameters_undirected()
        else:
            # different weights for different neighbours
            self._w_sp = Parameter(torch.FloatTensor(self._hop_p, 1))
            self._w_sn = Parameter(torch.FloatTensor(self._hop_n, 1))
            self._w_tp = Parameter(torch.FloatTensor(self._hop_p, 1))
            self._w_tn = Parameter(torch.FloatTensor(self._hop_n, 1))

            self._reset_parameters_directed()

    def _reset_parameters_undirected(self):
        self._w_p.data.fill_(1.0)
        self._w_n.data.fill_(1.0)

    def _reset_parameters_directed(self):
        self._w_sp.data.fill_(1.0)
        self._w_sn.data.fill_(1.0)
        self._w_tp.data.fill_(1.0)
        self._w_tn.data.fill_(1.0)

    def forward(self, edge_index_p: torch.LongTensor, edge_weight_p: torch.FloatTensor,
                edge_index_n: torch.LongTensor, edge_weight_n: torch.FloatTensor,
                x_p: torch.FloatTensor, x_n: torch.FloatTensor,
                x_pt: Optional[torch.FloatTensor] = None, x_nt: Optional[torch.FloatTensor] = None) -> Tuple[torch.FloatTensor,
                                                                                                             torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Making a forward pass of SIMPA.

        Arg types:
            * **edge_index_p, edge_index_n** (PyTorch FloatTensor) - Edge indices for positive and negative parts.
            * **edge_weight_p, edge_weight_n** (PyTorch FloatTensor) - Edge weights for positive and nagative parts.
            * **x_p** (PyTorch FloatTensor) - Souce positive hidden representations.
            * **x_n** (PyTorch FloatTensor) - Souce negative hidden representations.
            * **x_pt** (PyTorch FloatTensor, optional) - Target positive hidden representations. Default: None.
            * **x_nt** (PyTorch FloatTensor, optional) - Target negative hidden representations. Default: None.
        Return types:
            * **feat** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*input_dim) for undirected graphs and (num_nodes, 4*input_dim) for directed graphs.
        """

        if self._undirected:
            feat_p = self._w_p[0] * x_p
            feat_n = torch.zeros_like(feat_p)
            curr_p = x_p.clone()
            curr_n_aux = x_n.clone()  # auxilliary values
            j = 0
            for h in range(0, self._hop_p):
                if h > 0:
                    curr_p = self.conv_layer_p(
                        curr_p, edge_index_p, edge_weight_p)
                    curr_n_aux = self.conv_layer_p(
                        curr_n_aux, edge_index_p, edge_weight_p)
                    feat_p += self._w_p[h] * curr_p
                if h != (self._hop_p-1):
                    curr_n = self.conv_layer_n(
                        curr_n_aux, edge_index_n, edge_weight_n)  # A_n*A_P^h*x_n
                    feat_n += self._w_n[j] * curr_n
                    j += 1
                    for _ in range(self._hop_p-2-h):
                        curr_n = self.conv_layer_p(
                            curr_n, edge_index_p, edge_weight_p)  # A_p^(_)*A_n*A_P^h*x_n
                        feat_n += self._w_n[j] * curr_n
                        j += 1

            feat = torch.cat([feat_p, feat_n], dim=1)  # concatenate results
        else:
            edge_index_sp = edge_index_p
            edge_index_sn = edge_index_n
            edge_index_tp = edge_index_p[[1, 0]]
            edge_index_tn = edge_index_n[[1, 0]]
            x_sp = x_p
            x_sn = x_n
            feat_sp = self._w_sp[0] * x_sp
            feat_sn = torch.zeros_like(feat_sp)
            feat_tp = self._w_tp[0] * x_pt
            feat_tn = torch.zeros_like(feat_tp)
            curr_sp = x_sp.clone()
            curr_sn_aux = x_sn.clone()
            curr_tp = x_pt.clone()
            curr_tn_aux = x_nt.clone()
            j = 0
            for h in range(0, self._hop_p):
                if h > 0:
                    curr_sp = self.conv_layer_p(
                        curr_sp, edge_index_sp, edge_weight_p)
                    curr_sn_aux = self.conv_layer_p(
                        curr_sn_aux, edge_index_sp, edge_weight_p)
                    curr_tp = self.conv_layer_p(
                        curr_tp, edge_index_tp, edge_weight_p)
                    curr_tn_aux = self.conv_layer_p(
                        curr_tn_aux, edge_index_tp, edge_weight_p)
                    feat_sp += self._w_sp[h] * curr_sp
                    feat_tp += self._w_tp[h] * curr_tp
                if h != (self._hop_p-1):
                    curr_sn = self.conv_layer_n(
                        curr_sn_aux, edge_index_sn, edge_weight_n)
                    curr_tn = self.conv_layer_n(
                        curr_tn_aux, edge_index_tn, edge_weight_n)
                    feat_sn += self._w_sn[j] * curr_sn
                    feat_tn += self._w_tn[j] * curr_tn
                    j += 1
                    for _ in range(self._hop_p-2-h):
                        curr_sn = self.conv_layer_p(
                            curr_sn, edge_index_sp, edge_weight_p)
                        curr_tn = self.conv_layer_p(
                            curr_tn, edge_index_tp, edge_weight_p)
                        feat_sn += self._w_sn[j] * curr_sn
                        feat_tn += self._w_tn[j] * curr_tn
                        j += 1

            # concatenate results
            feat = torch.cat([feat_sp, feat_sn, feat_tp, feat_tn], dim=1)

        return feat
