import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DiGCL_Encoder(torch.nn.Module):
    r"""An implementation of the DiGCL encoder model from the
    `Directed Graph Contrastive Learning
    <https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf>`_ paper.

    Args:
        in_channels (int): Dimension of input features.
        out_channels (int): Dimension of output representations.
        activation (str): Activation funciton to use.
        num_layers (int, Optional): Number of layers for encoder. (Default: 2)
    """

    def __init__(self, in_channels: int, out_channels: int, activation: str,
                 num_layers: int = 2):
        super(DiGCL_Encoder, self).__init__()

        assert num_layers >= 2
        self._num_layers = num_layers
        self.conv = [GCNConv(in_channels, 2 * out_channels)]
        for _ in range(1, num_layers-1):
            self.conv.append(GCNConv(2 * out_channels, 2 * out_channels))
        self.conv.append(GCNConv(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU()})[
            activation]

    def reset_parameters(self):
        for layer in self.conv:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        """
        Making a forward pass of the DiGCL encoder model.

        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (PyTorch LongTensor) - Edge indices.
            * edge_weight (PyTorch FloatTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * x (PyTorch FloatTensor) - Embeddings for all nodes, with shape (num_nodes, out_channels).
        """
        for i in range(self._num_layers):
            x = self.activation(self.conv[i](x, edge_index, edge_weight))
        return x


class DiGCL(torch.nn.Module):
    r"""An implementation of the DiGCL model from the
    `Directed Graph Contrastive Learning 
    <https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf>`_ paper.

    Args:
        in_channels (int): Dimension of input features.
        activation (str): Activation funciton to use.
        num_hidden (int): Hidden dimension.
        num_proj_hidden (int): Hidden dimension for projection.
        tau (float): Tau value in the loss.
        num_layers (int): Number of layers for encoder.
    """

    def __init__(self, in_channels: int, activation: str,
                 num_hidden: int, num_proj_hidden: int,
                 tau: float, num_layers: int):
        super(DiGCL, self).__init__()
        self.encoder = DiGCL_Encoder(
            in_channels, num_hidden, activation, num_layers)
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.encoder.reset_parameters()
        return

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Making a forward pass of the DiGCL model.

        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (PyTorch LongTensor) - Edge indices.
            * edge_weight (PyTorch FloatTensor, optional) - Edge weights corresponding to edge indices.

        Return types:
            * x (PyTorch FloatTensor) - Embeddings for all nodes, with shape (num_nodes, out_channels).
        """
        return self.encoder(x, edge_index, edge_weight)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        """
        Nonlinear transformation of the input hidden feature.

        Args types::
            * z (PyTorch FloatTensor) - Node features.

        Return types:
            * z (PyTorch FloatTensor) - Projected node features.
        """
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Normalized similarity calculation.

        Args types::
            * z1 (PyTorch FloatTensor) - Node features.
            * z2 (PyTorch FloatTensor) - Node features.

        Return types:
            * z (PyTorch FloatTensor) - Node-wise similarity.
        """
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Semi-supervised loss function.

        Arg types:
            * z1 (PyTorch FloatTensor) - Node features.
            * z2 (PyTorch FloatTensor) - Node features.

        Return types:
            * loss (PyTorch FloatTensor) - Loss.
        """
        def f(x): return torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.diag() - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        """
        Semi-supervised loss function. Space complexity: O(BN) (semi_loss: O(N^2))

        Args types::
            * z1 (PyTorch FloatTensor) - Node features.
            * z2 (PyTorch FloatTensor) - Node features.

        Return types:
            * loss (PyTorch FloatTensor) - Loss.
        """
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        def f(x): return torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        """
        The DiGCL contrastive loss.

        Arg types:
            * z1, z2 (PyTorch FloatTensor) - Node hidden representations.
            * mean (bool, optional) - Whether to return the mean of loss values, default True, otherwise return sum.
            * batch_size (int, optional) - Batch size, if 0 this means full-batch. Default 0.
        Return types:
            * ret (PyTorch FloatTensor) - Loss.
        """
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
