import torch
import torch.nn.functional as F

from .DiGCNConv import DiGCNConv


class DiGCN_node_classification(torch.nn.Module):
    r"""An implementation of the DiGCN model without inception blocks for node classification from the
    `Digraph Inception Convolutional Networks
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.

    Args:
        num_features (int): Dimension of input features.
        hidden (int): Hidden dimension.
        label_dim (int): Number of clusters.
        dropout (float): Dropout value. (Default: 0.5)
    """

    def __init__(self, num_features: int, hidden: int, label_dim: int, dropout: float = 0.5):
        super(DiGCN_node_classification, self).__init__()
        self.conv1 = DiGCNConv(num_features, hidden)
        self.conv2 = DiGCNConv(hidden, label_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Making a forward pass of the DiGCN node classification model without inception blocks.

        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (PyTorch LongTensor) - Edge indices.
            * edge_weight (PyTorch FloatTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * x (PyTorch FloatTensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)
