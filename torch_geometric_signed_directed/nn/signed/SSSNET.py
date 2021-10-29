from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ..general.conv_base import Conv_Base


class SIMPA(nn.Module):
    r"""The signed mixed-path aggregation model from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.
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
        Making a forward pass of SIMPA from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.

        Arg types:
            * **edge_index_p, edge_index_n** (PyTorch FloatTensor) - Edge indices for positive and negative parts.
            * **edge_weight_p, edge_weight_n** (PyTorch FloatTensor) - Edge weights for positive and nagative parts.
            * **x_p** (PyTorch FloatTensor) - Souce positive hidden representations.
            * **x_n** (PyTorch FloatTensor) - Souce negative hidden representations.
            * **x_pt** (PyTorch FloatTensor, optional) - Target positive hidden representations. Default: None.
            * **x_nt** (PyTorch FloatTensor, optional) - Target negative hidden representations. Default: None.
        Return types:
            * **feat** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*input_dim) for undirected graphs 
                and (num_nodes, 4*input_dim) for directed graphs.
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


class SSSNET(nn.Module):
    r"""The signed graph clustering model from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.
    Args:
        nfeat (int): Number of features.
        hidden (int): Hidden dimensions of the initial MLP.
        nclass (int): Number of clusters.
        dropout (float): Dropout probability.
        hop (int): Number of hops to consider.
        fill_value (float): Value for added self-loops for the positive part of the adjacency matrix.
        directed (bool, optional): Whether the input network is directed or not. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float, hop: int, fill_value: float,
                 directed: bool = False, bias: bool = True):
        super(SSSNET, self).__init__()
        nh1 = hidden
        nh2 = hidden
        self._num_clusters = int(nclass)
        self._simpa = SIMPA(hop, fill_value, directed)
        if bias:
            self._bias = Parameter(torch.FloatTensor(self._num_clusters))
        else:
            self.register_parameter('_bias', None)
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(p=dropout)
        self._undirected = not directed

        if self._undirected:
            self._w_p0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_p1 = Parameter(torch.FloatTensor(nh1, nh2))
            self._w_n0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_n1 = Parameter(torch.FloatTensor(nh1, nh2))

            self._W_prob = Parameter(
                torch.FloatTensor(2*nh2, self._num_clusters))

            self._reset_parameters_undirected()
        else:
            self._w_sp0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_sp1 = Parameter(torch.FloatTensor(nh1, nh2))
            self._w_sn0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_sn1 = Parameter(torch.FloatTensor(nh1, nh2))
            self._w_tp0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_tp1 = Parameter(torch.FloatTensor(nh1, nh2))
            self._w_tn0 = Parameter(torch.FloatTensor(nfeat, nh1))
            self._w_tn1 = Parameter(torch.FloatTensor(nh1, nh2))

            self._W_prob = Parameter(
                torch.FloatTensor(4*nh2, self._num_clusters))

            self._reset_parameters_directed()

    def _reset_parameters_undirected(self):
        nn.init.xavier_uniform_(self._w_p0, gain=1.414)
        nn.init.xavier_uniform_(self._w_p1, gain=1.414)
        nn.init.xavier_uniform_(self._w_n0, gain=1.414)
        nn.init.xavier_uniform_(self._w_n1, gain=1.414)

        if self._bias is not None:
            self._bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self._W_prob, gain=1.414)

    def _reset_parameters_directed(self):
        nn.init.xavier_uniform_(self._w_sp0, gain=1.414)
        nn.init.xavier_uniform_(self._w_sp1, gain=1.414)
        nn.init.xavier_uniform_(self._w_sn0, gain=1.414)
        nn.init.xavier_uniform_(self._w_sn1, gain=1.414)
        nn.init.xavier_uniform_(self._w_tp0, gain=1.414)
        nn.init.xavier_uniform_(self._w_tp1, gain=1.414)
        nn.init.xavier_uniform_(self._w_tn0, gain=1.414)
        nn.init.xavier_uniform_(self._w_tn1, gain=1.414)

        if self._bias is not None:
            self._bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self._W_prob, gain=1.414)

    def forward(self, edge_index_p: torch.LongTensor, edge_weight_p: torch.FloatTensor,
                edge_index_n: torch.LongTensor, edge_weight_n: torch.FloatTensor,
                features: torch.FloatTensor) -> Tuple[torch.FloatTensor,
                                                      torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Making a forward pass of the SSSNET from the
    `SSSNET: Semi-Supervised Signed Network Clustering" <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.

        Arg types:
            * **edge_index_p, edge_index_n** (PyTorch FloatTensor) - Edge indices for positive and negative parts.
            * **edge_weight_p, edge_weight_n** (PyTorch FloatTensor) - Edge weights for positive and nagative parts.
            * **features** (PyTorch FloatTensor) - Input node features, with shape (num_nodes, num_features).
        Return types:
            * **z** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*hidden) for undirected graphs 
                and (num_nodes, 4*hidden) for directed graphs.
            * **output** (PyTorch FloatTensor) - Log of prob, with shape (num_nodes, num_clusters).
            * **predictions_cluster** (PyTorch LongTensor) - Predicted labels.
            * **prob** (PyTorch FloatTensor) - Probability assignment matrix of different clusters, with shape (num_nodes, num_clusters).
        """
        if self._undirected:
            # MLP
            x_p = torch.mm(features, self._w_p0)
            x_p = self._relu(x_p)
            x_p = self._dropout(x_p)
            x_p = torch.mm(x_p, self._w_p1)

            x_n = torch.mm(features, self._w_n0)
            x_n = self._relu(x_n)
            x_n = self._dropout(x_n)
            x_n = torch.mm(x_n, self._w_n1)

            z = self._simpa(edge_index_p, edge_weight_p,
                            edge_index_n, edge_weight_n, x_p, x_n)
        else:
            # MLP
            # source positive embedding
            x_sp = torch.mm(features, self._w_sp0)
            x_sp = self._relu(x_sp)
            x_sp = self._dropout(x_sp)
            x_sp = torch.mm(x_sp, self._w_sp1)

            # source negative embedding
            x_sn = torch.mm(features, self._w_sn0)
            x_sn = self._relu(x_sn)
            x_sn = self._dropout(x_sn)
            x_sn = torch.mm(x_sn, self._w_sn1)

            # target positive embedding
            x_tp = torch.mm(features, self._w_tp0)
            x_tp = self._relu(x_tp)
            x_tp = self._dropout(x_tp)
            x_tp = torch.mm(x_tp, self._w_tp1)

            # target negative embedding
            x_tn = torch.mm(features, self._w_tn0)
            x_tn = self._relu(x_tn)
            x_tn = self._dropout(x_tn)
            x_tn = torch.mm(x_tn, self._w_tn1)

            z = self._simpa(edge_index_p, edge_weight_p,
                            edge_index_n, edge_weight_n, x_sp, x_sn, x_tp, x_tn)

        output = torch.mm(z, self._W_prob)
        if self._bias is not None:
            output = output + self._bias  # to balance the difference in cluster probabilities

        predictions_cluster = torch.argmax(output, dim=1)

        prob = F.softmax(output, dim=1)

        output = F.log_softmax(output, dim=1)

        return F.normalize(z), output, predictions_cluster, prob
