import torch
import torch.nn as nn
import torch.nn.functional as F

from .DiGCNConv import DiGCNConv


class DiGCN_link_prediction(torch.nn.Module):
    r"""An implementation of the DiGCN model without inception blocks for link prediction from the
    `Digraph Inception Convolutional Networks
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.

    Args:
        num_features (int): Dimension of input features.
        hidden (int): Hidden dimension.
        label_dim (int): The dimension of labels.
        dropout (float): Dropout value. (Default: 0.5)
    """

    def __init__(self, num_features: int, hidden: int, label_dim: int, dropout: float = 0.5):
        super(DiGCN_link_prediction, self).__init__()
        self.conv1 = DiGCNConv(num_features, hidden)
        self.conv2 = DiGCNConv(hidden, hidden)
        self.dropout = dropout
        self.linear = nn.Linear(hidden*2, label_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor,
                query_edges: torch.LongTensor, edge_weight: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Making a forward pass of the DiGCN node classification model without inception blocks.

        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (PyTorch LongTensor) - Edge indices.
            * edge_weight (PyTorch FloatTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * x (PyTorch FloatTensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat((x[query_edges[:, 0]], x[query_edges[:, 1]]), dim=-1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
