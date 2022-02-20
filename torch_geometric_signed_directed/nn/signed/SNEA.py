from collections import defaultdict
from typing import Union
from torch_geometric.typing import PairTensor, Adj
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor, device
import numpy as np
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import coalesce
from random import randint
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)

from .SNEAConv import SNEAConv


class SNEA(nn.Module):
    r"""The signed graph attentional layers operator from the `"Learning Signed
    Network Embedding via Graph Attention" <https://ojs.aaai.org/index.php/AAAI/article/view/5911>`_ paper
    Args:
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
        node_num: int,
        edge_index_s: torch.LongTensor,
        in_dim: int = 64,
        out_dim: int = 64,
        layer_num: int = 2,
        lamb: float = 5,
        lambda_structure: float = 4
    ):
        super().__init__()

        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lamb = lamb
        self.lambda_structure = lambda_structure
        self.device = edge_index_s.device

        self.pos_edge_index = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        self.neg_edge_index = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()
        self.x = self.create_spectral_features()

        self.conv1 = SNEAConv(in_dim, out_dim // 2,
                            first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(
                SNEAConv(out_dim // 2, out_dim // 2,
                        first_aggr=False))

        self.structural_distance = nn.PairwiseDistance(p=2)
        self.param_src = nn.Parameter(torch.FloatTensor(2 * out_dim, 3))

        edge_index_s_list = edge_index_s.cpu().numpy().tolist()
        self.adj_pos, self.adj_neg = self.build_adj_lists(edge_index_s_list)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.param_src)

    def build_adj_lists(self, edge_index_s):
        adj_list_pos = defaultdict(set)
        adj_list_neg = defaultdict(set)
        for node_i, node_j, s in edge_index_s:

            if s == 1:
                adj_list_pos[node_i].add(node_j)
            else:
                adj_list_neg[node_i].add(node_j)
        return adj_list_pos, adj_list_neg

    def loss(self) -> torch.FloatTensor:
        max_node_index = self.node_num - 1
        i_loss2 = []
        pos_no_loss2 = []
        no_neg_loss2 = []

        i_indices = []
        j_indices = []
        ys = []
        all_nodes_set = set()
        skipped_nodes = []
        adj_lists_pos = self.adj_pos
        adj_lists_neg = self.adj_neg

        nodes = np.arange(0, self.node_num)
        nodes_embs = self.forward()
        for i in nodes:
            if (len(adj_lists_pos[i]) + len(adj_lists_neg[i])) == 0:
                skipped_nodes.append(i)
                continue
            all_nodes_set.add(i)
            for j_pos in adj_lists_pos[i]:
                i_loss2.append(i)
                pos_no_loss2.append(j_pos)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                no_neg_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_pos)
                ys.append(0)
                all_nodes_set.add(j_pos)
            for j_neg in adj_lists_neg[i]:
                i_loss2.append(i)
                no_neg_loss2.append(j_neg)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                pos_no_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_neg)
                ys.append(1)
                all_nodes_set.add(j_neg)

            need_samples = 2  # number of sampling of the no links pairs
            cur_samples = 0
            while cur_samples < need_samples:
                temp_samp = randint(0, max_node_index)
                if (temp_samp not in adj_lists_pos[i]) and (temp_samp not in adj_lists_neg[i]):
                    # got one we can use
                    i_indices.append(i)
                    j_indices.append(temp_samp)
                    ys.append(2)
                    all_nodes_set.add(temp_samp)
                cur_samples += 1

        all_nodes_list = list(all_nodes_set)
        all_nodes_map = {node: i for i, node in enumerate(all_nodes_list)}

        i_indices_mapped = [all_nodes_map[i] for i in i_indices]
        j_indices_mapped = [all_nodes_map[j] for j in j_indices]
        ys = torch.LongTensor(ys).to(self.device)
        # now that we have the mapped indices and final embeddings we can get the loss
        loss_entropy = F.cross_entropy(
            torch.mm(torch.cat((nodes_embs[i_indices_mapped],
                                nodes_embs[j_indices_mapped]), 1),
                     self.param_src),
            ys)

        i_loss2 = [all_nodes_map[i] for i in i_loss2]
        pos_no_loss2 = [all_nodes_map[i] for i in pos_no_loss2]
        no_neg_loss2 = [all_nodes_map[i] for i in no_neg_loss2]

        tensor_zeros = torch.zeros(len(i_loss2)).to(self.device)

        loss_structure = torch.mean(
            torch.max(
                tensor_zeros,
                self.structural_distance(nodes_embs[i_loss2], nodes_embs[pos_no_loss2]) ** 2
                - self.structural_distance(nodes_embs[i_loss2], nodes_embs[no_neg_loss2]) ** 2
            )
        )

        return loss_entropy + self.lambda_structure * loss_structure


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
