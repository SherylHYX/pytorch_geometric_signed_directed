import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric_signed_directed.utils.signed import (create_spectral_features,
                                                          Link_Sign_Entropy_Loss,
                                                          Sign_Structure_Loss)
from .SNEAConv import SNEAConv

class SNEA(nn.Module):
    r"""The signed graph attentional layers operator from the `"Learning Signed
    Network Embedding via Graph Attention" <https://ojs.aaai.org/index.php/AAAI/article/view/5911>`_ paper
    Args:
        node_num (int): The number of nodes.
        edge_index_s (LongTensor): The edgelist with sign. (e.g., torch.LongTensor([[0, 1, -1], [0, 2, 1]]) )
        in_dim (int, optional): Size of each input sample features. Defaults to 64.
        out_dim (int, optional): Size of each output embeddings. Defaults to 64.
        layer_num (int, optional): Number of layers. Defaults to 2.
        lamb (float, optional): Balances the contributions of the overall
            objective. (default: :obj:`4`)
    """

    def __init__(
        self,
        node_num: int,
        edge_index_s: torch.LongTensor,
        in_dim: int = 64,
        out_dim: int = 64,
        layer_num: int = 2,
        lamb: float = 4,
        init_emb: torch.FloatTensor = None,
        init_emb_grad: bool = True
    ):
        super().__init__()

        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lamb = lamb
        self.device = edge_index_s.device

        self.pos_edge_index = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        self.neg_edge_index = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()

        if init_emb is None:
            init_emb = create_spectral_features(
                pos_edge_index=self.pos_edge_index,
                neg_edge_index=self.neg_edge_index,
                node_num=self.node_num,
                dim=self.in_dim
            ).to(self.device)
        else:
            init_emb = init_emb

        self.x = nn.Parameter(init_emb, requires_grad=init_emb_grad)

        self.conv1 = SNEAConv(in_dim, out_dim // 2,
                              first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(
                SNEAConv(out_dim // 2, out_dim // 2,
                         first_aggr=False))
        
        self.weight = torch.nn.Linear(self.out_dim, self.out_dim)

        self.lsp_loss = Link_Sign_Entropy_Loss(out_dim)
        self.structure_loss = Sign_Structure_Loss()

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()


    def loss(self) -> torch.FloatTensor:
        z = self.forward()
        nll_loss = self.lsp_loss(z, self.pos_edge_index, self.neg_edge_index)
        structure_loss = self.structure_loss(
            z, self.pos_edge_index, self.neg_edge_index)
        return nll_loss + self.lamb * structure_loss

    def forward(self) -> Tensor:
        z = torch.tanh(self.conv1(
            self.x, self.pos_edge_index, self.neg_edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, self.pos_edge_index, self.neg_edge_index))
        z = torch.tanh(self.weight(z))
        return z
