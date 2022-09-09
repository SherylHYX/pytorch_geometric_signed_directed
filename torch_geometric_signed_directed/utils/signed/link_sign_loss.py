import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)


class Link_Sign_Entropy_Loss(nn.Module):
    r"""An implementation of the Entropy Loss used in 
    the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper. 
    and  `"Learning Signed
    Network Embedding via Graph Attention" <https://ojs.aaai.org/index.php/AAAI/article/view/5911>`_ paper

    Args:
        emb_dim (int): The embedding size.
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2 * emb_dim, 3)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def discriminate(
        self,
        z: torch.Tensor,
        edge_index: torch.LongTensor
    ) -> torch.FloatTensor:
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

    def forward(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.LongTensor,
        neg_edge_index: torch.LongTensor
    ) -> torch.Tensor:
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


class Sign_Structure_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.LongTensor,
        neg_edge_index: torch.LongTensor
    ) -> torch.Tensor:
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return loss_1 + loss_2

    def pos_embedding_loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.LongTensor
    ) -> torch.Tensor:
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(
        self,
        z: torch.Tensor,
        neg_edge_index: torch.LongTensor
    ) -> torch.Tensor:
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()
