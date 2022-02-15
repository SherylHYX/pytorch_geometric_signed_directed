from typing import Union
from torch_geometric.typing import PairTensor, Adj
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import coalesce

from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)

from .SGCNConv import SGCNConv
from .SNEAConv import SNEAConv


class SGCN_SNEA(nn.Module):
    r"""The signed graph convolutional network model from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper, and 
    the signed graph attentional layers operator from the `"Learning Signed
    Network Embedding via Graph Attention" <https://ojs.aaai.org/index.php/AAAI/article/view/5911>`_ paper
    Internally, the first part of this module uses the
    :class:`torch_geometric.nn.conv.SignedConv` operator. 
    We have made some modifications to the original model :class:`torch_geometric.nn.SignedGCN` for the uniformity of model inputs.

    Args:
        model_name (str): The name of the model, "SGCN" or "SNEA".
        node_num (int): The number of nodes.
        edge_index_s (LongTensor): The edgelist with sign. (e.g., torch.LongTensor([[0, 1, -1], [0, 2, 1]]) )
        in_dim (int, optional): Size of each input sample features. Defaults to 64.
        out_dim (int, optional): Size of each hidden embeddings. Defaults to 64.
        layer_num (int, optional): Number of layers. Defaults to 2.
        lamb (float, optional): Balances the contributions of the overall
            objective. (default: :obj:`5`)
    """

    def __init__(
        self,
        model_name: str,
        node_num: int,
        edge_index_s: torch.LongTensor,
        in_dim: int = 64,
        out_dim: int = 64,
        layer_num: int = 2,
        lamb: float = 5
    ):
        super().__init__()

        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lamb = lamb

        self.pos_edge_index = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        self.neg_edge_index = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()
        self.x = self.create_spectral_features()

        if model_name.lower() == 'sgcn':
            self.conv1 = SGCNConv(in_dim, out_dim // 2,
                                first_aggr=True)
            self.convs = torch.nn.ModuleList()
            for _ in range(layer_num - 1):
                self.convs.append(
                    SGCNConv(out_dim // 2, out_dim // 2,
                            first_aggr=False))
        elif model_name.lower() == 'snea':
            self.conv1 = SNEAConv(in_dim, out_dim // 2,
                                first_aggr=True)
            self.convs = torch.nn.ModuleList()
            for _ in range(layer_num - 1):
                self.convs.append(
                    SNEAConv(out_dim // 2, out_dim // 2,
                            first_aggr=False))

        self.lin = torch.nn.Linear(2 * out_dim, 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def nll_loss(self, z: Tensor, pos_edge_index: torch.LongTensor, neg_edge_index: torch.LongTensor) -> Tensor:
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative nedges
        :obj:`neg_edge_index`.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1), ), 0))
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1), ), 1))
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1), ), 2))
        return nll_loss / 3.0

    def pos_embedding_loss(self, z: Tensor, pos_edge_index: torch.LongTensor) -> Tensor:
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z: Tensor, neg_edge_index: torch.LongTensor) -> Tensor:
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def discriminate(self, z: Tensor, edge_index: torch.LongTensor) -> torch.FloatTensor:
        """Given node embeddings :obj:`z`, classifies the link relation
        between node pairs :obj:`edge_index` to be either positive,
        negative or non-existent.

        Args:
            x (Tensor): The input node features.
            edge_index (LongTensor): The edge indices.
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def loss(self) -> torch.FloatTensor:
        z = self.forward()
        nll_loss = self.nll_loss(z, self.pos_edge_index, self.neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, self.pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, self.neg_edge_index)
        return nll_loss + self.lamb * (loss_1 + loss_2)

    def forward(self) -> Tensor:
        z = torch.tanh(self.conv1(
            self.x, self.pos_edge_index, self.neg_edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, self.pos_edge_index, self.neg_edge_index))
        return z

    def create_spectral_features(self) -> torch.FloatTensor:

        from sklearn.decomposition import TruncatedSVD

        edge_index = torch.cat(
            [self.pos_edge_index, self.neg_edge_index], dim=1)
        N = self.node_num
        edge_index = edge_index.to(torch.device('cpu'))

        pos_val = torch.full(
            (self.pos_edge_index.size(1), ), 2, dtype=torch.float)
        neg_val = torch.full(
            (self.neg_edge_index.size(1), ), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)

        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        val = torch.cat([val, val], dim=0)

        edge_index, val = coalesce(edge_index, val, N, N)
        val = val - 1

        # Borrowed from:
        # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = sp.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.in_dim, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return torch.from_numpy(x).to(torch.float).to(self.pos_edge_index.device)
