from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .DIMPA import DIMPA


class DIGRAC_node_clustering(torch.nn.Module):
    r"""The directed graph clustering model from the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.

    Args:
        num_features (int): Number of features.
        hidden (int): Hidden dimensions of the initial MLP.
        nclass (int): Number of clusters.
        dropout (float): Dropout probability.
        hop (int): Number of hops to consider.
        fill_value (float): Value for added self-loops.
    """

    def __init__(self, num_features: int, hidden: int, nclass: int, fill_value: float, dropout: float, hop: int):
        super(DIGRAC_node_clustering, self).__init__()
        nh1 = hidden
        nh2 = hidden
        self._num_clusters = int(nclass)
        self._w_s0 = Parameter(torch.FloatTensor(num_features, nh1))
        self._w_s1 = Parameter(torch.FloatTensor(nh1, nh2))
        self._w_t0 = Parameter(torch.FloatTensor(num_features, nh1))
        self._w_t1 = Parameter(torch.FloatTensor(nh1, nh2))

        self._dimpa = DIMPA(hop, fill_value)
        self._relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

        self._bias = Parameter(torch.FloatTensor(self._num_clusters))

        self._W_prob = Parameter(torch.FloatTensor(2*nh2, self._num_clusters))

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_s0, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_s1, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_t0, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_t1, gain=1.414)

        self._bias.data.fill_(0.0)
        torch.nn.init.xavier_uniform_(self._W_prob, gain=1.414)

    def forward(self, edge_index: torch.FloatTensor, edge_weight: torch.FloatTensor,
                features: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Making a forward pass of the DIGRAC node clustering model.

        Arg types:
            * **edge_index** (PyTorch FloatTensor) - Edge indices.
            * **edge_weight** (PyTorch FloatTensor) - Edge weights.
            * **features** (PyTorch FloatTensor) - Input node features, with shape (num_nodes, num_features).
        Return types:
            * **z** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*hidden).
            * **output** (PyTorch FloatTensor) - Log of prob, with shape (num_nodes, num_clusters).
            * **predictions_cluster** (PyTorch LongTensor) - Predicted labels.
            * **prob** (PyTorch FloatTensor) - Probability assignment matrix of different clusters, with shape (num_nodes, num_clusters).
        """
        # MLP
        x_s = torch.mm(features, self._w_s0)
        x_s = self._relu(x_s)
        x_s = self.dropout(x_s)
        x_s = torch.mm(x_s, self._w_s1)

        x_t = torch.mm(features, self._w_t0)
        x_t = self._relu(x_t)
        x_t = self.dropout(x_t)
        x_t = torch.mm(x_t, self._w_t1)

        z = self._dimpa(x_s, x_t, edge_index, edge_weight)

        output = torch.mm(z, self._W_prob)
        output = output + self._bias  # to balance the difference in cluster probabilities

        predictions_cluster = torch.argmax(output, dim=1)

        prob = F.softmax(output, dim=1)

        output = F.log_softmax(output, dim=1)

        return F.normalize(z), output, predictions_cluster, prob
