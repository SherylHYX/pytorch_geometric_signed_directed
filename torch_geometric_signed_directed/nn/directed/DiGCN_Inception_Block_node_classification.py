from typing import Tuple

import torch
import torch.nn.functional as F

from .DiGCN_Inception_Block import DiGCN_InceptionBlock as InceptionBlock


class DiGCN_Inception_Block_node_classification(torch.nn.Module):
    r"""An implementation of the DiGCN model with inception blocks for node classification from the
    `Digraph Inception Convolutional Networks
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.

    Args:
        num_features (int): Dimention of input features.
        hidden (int): Hidden dimention.
        label_dim (int): Number of clusters.
        dropout (float): Dropout value.
    """

    def __init__(self, num_features: int, hidden: int, label_dim: int, dropout: float = 0.5):
        super(DiGCN_Inception_Block_node_classification, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(hidden, hidden)
        self.ib3 = InceptionBlock(hidden, label_dim)
        self._dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.ib1.reset_parameters()
        self.ib2.reset_parameters()
        self.ib3.reset_parameters()

    def forward(self, features: torch.FloatTensor,
                edge_index_tuple: Tuple[torch.LongTensor, torch.LongTensor],
                edge_weight_tuple: Tuple[torch.FloatTensor, torch.FloatTensor]) -> torch.FloatTensor:
        """
        Making a forward pass of the DiGCN node classification model.

        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index_tuple (PyTorch LongTensor) - Tuple of edge indices.
            * edge_weight_tuple (PyTorch FloatTensor, optional) - Tuple of edge weights corresponding to edge indices.
        Return types:
            * x (PyTorch FloatTensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0, x1, x2 = self.ib1(x, edge_index, edge_weight,
                              edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0, x1, x2 = self.ib2(x, edge_index, edge_weight,
                              edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2
        x = F.dropout(x, p=self._dropout, training=self.training)

        x0, x1, x2 = self.ib3(x, edge_index, edge_weight,
                              edge_index2, edge_weight2)
        x0 = F.dropout(x0, p=self._dropout, training=self.training)
        x1 = F.dropout(x1, p=self._dropout, training=self.training)
        x2 = F.dropout(x2, p=self._dropout, training=self.training)
        x = x0+x1+x2

        return F.log_softmax(x, dim=1)
