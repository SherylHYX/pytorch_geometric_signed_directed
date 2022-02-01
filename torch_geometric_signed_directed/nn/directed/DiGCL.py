import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DiGCL_Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str,
                 k: int = 2):
        super(DiGCL_Encoder, self).__init__()

        assert k >= 2
        self.k = k
        self.conv = [GCNConv(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(GCNConv(2 * out_channels, 2 * out_channels))
        self.conv.append(GCNConv(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU()})[activation]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index, edge_weight))
        return x


class DiGCL(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str,
                 num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5, k: int = 2):
        super(DiGCL, self).__init__()
        self.encoder = DiGCL_Encoder(in_channels, out_channels, activation, k)
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.fc3 = torch.nn.Linear(num_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weight)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def projection2(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc3(z))
        return z

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.diag() - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
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


