from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .SIMPA import SIMPA


class SSSNET_link_prediction(nn.Module):
    r"""The signed graph link prediction model adapted from the
    `SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.

    Args:
        nfeat (int): Number of features.
        hidden (int): Hidden dimensions of the initial MLP.
        nclass (int): Number of link classes.
        dropout (float): Dropout probability.
        hop (int): Number of hops to consider.
        fill_value (float): Value for added self-loops for the positive part of the adjacency matrix.
        directed (bool, optional): Whether the input network is directed or not. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn an additive bias. (default: :obj:`True`)
    """

    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float, hop: int, fill_value: float,
                 directed: bool = False, bias: bool = True):
        super(SSSNET_link_prediction, self).__init__()
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
                torch.FloatTensor(4*nh2, self._num_clusters))

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
                torch.FloatTensor(8*nh2, self._num_clusters))

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
                features: torch.FloatTensor, query_edges: torch.LongTensor) -> Tuple[torch.FloatTensor,
                                                      torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Making a forward pass of the SSSNET.

        Arg types:
            * **edge_index_p, edge_index_n** (PyTorch FloatTensor) - Edge indices for positive and negative parts.
            * **edge_weight_p, edge_weight_n** (PyTorch FloatTensor) - Edge weights for positive and nagative parts.
            * **features** (PyTorch FloatTensor) - Input node features, with shape (num_nodes, num_features).
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
        
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
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

        x = torch.cat((z[query_edges[:,0]], z[query_edges[:,1]]), dim = -1)
        output = torch.mm(x, self._W_prob)
        if self._bias is not None:
            output = output + self._bias  # to balance the difference in cluster probabilities

        log_prob = F.log_softmax(output, dim=1)

        return log_prob

