from collections import defaultdict
from typing import Tuple, List

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

from torch_geometric_signed_directed.utils.signed import (create_spectral_features,
                                                          Link_Sign_Product_Loss)
class SiGAT(nn.Module):
    r"""The signed graph attention network model (SiGAT) from the `"Signed Graph
    Attention Networks" <https://arxiv.org/abs/1906.10958>`_ paper.

    Args:
        node_num ([type]): Number of node.
        edge_index_s (list): The edgelist with sign. (e.g., [[0, 1, -1]] )
        in_dim (int, optional): Size of each input sample features. Defaults to 20.
        out_dim (int): Size of each output embeddings. Defaults to 20.
        batch_size (int, optional): Mini-batch size of training. Defaults to 500.
        x_require_grad (bool, optional): Modify Input Feature or Not. Defaults to True.

    """

    def __init__(
        self,
        node_num: int,
        edge_index_s,
        in_dim: int = 20,
        out_dim: int = 20,
        init_emb: torch.FloatTensor = None,
        init_emb_grad: bool = True
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_num = node_num
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

        edge_index_s_list = edge_index_s.cpu().numpy().tolist()
        self.adj_lists = self.build_adj_lists(edge_index_s_list)
        self.edge_lists = [self.map_adj_to_edges(i) for i in self.adj_lists]

        self.aggs = []
        for i in range(len(self.adj_lists)):
            self.aggs.append(
                GATConv(in_channels=in_dim, out_channels=out_dim)
            )
            self.add_module('agg_{}'.format(i), self.aggs[-1])

        self.mlp_layer = nn.Sequential(
            nn.Linear(out_dim *
                      (len(self.adj_lists) + 1), out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
            nn.Tanh()
        )
        
        self.lsp_loss = Link_Sign_Product_Loss()

        self.reset_parameters()

    def reset_parameters(self):
        for agg in self.aggs:
            agg.reset_parameters()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)
        self.mlp_layer.apply(init_weights)

    def map_adj_to_edges(self, adj_list: List) -> torch.LongTensor:
        edges = []
        for a in adj_list:
            for b in adj_list[a]:
                edges.append((a, b))
        edges = torch.LongTensor(edges).to(self.device)
        return edges.t()

    def get_tri_features(self, u: int, v: int, r_edgelists: List) -> Tuple[int, int, int, int, int,
                                                                           int, int, int, int, int, int, int, int, int, int, int]:
        pos_in_edgelists, pos_out_edgelists, neg_in_edgelists, neg_out_edgelists = r_edgelists

        d1_1 = len(set(pos_out_edgelists[u]).intersection(
            set(pos_in_edgelists[v])))
        d1_2 = len(set(pos_out_edgelists[u]).intersection(
            set(neg_in_edgelists[v])))
        d1_3 = len(set(neg_out_edgelists[u]).intersection(
            set(pos_in_edgelists[v])))
        d1_4 = len(set(neg_out_edgelists[u]).intersection(
            set(neg_in_edgelists[v])))

        d2_1 = len(set(pos_out_edgelists[u]).intersection(
            set(pos_out_edgelists[v])))
        d2_2 = len(set(pos_out_edgelists[u]).intersection(
            set(neg_out_edgelists[v])))
        d2_3 = len(set(neg_out_edgelists[u]).intersection(
            set(pos_out_edgelists[v])))
        d2_4 = len(set(neg_out_edgelists[u]).intersection(
            set(neg_out_edgelists[v])))

        d3_1 = len(set(pos_in_edgelists[u]).intersection(
            set(pos_out_edgelists[v])))
        d3_2 = len(set(pos_in_edgelists[u]).intersection(
            set(neg_out_edgelists[v])))
        d3_3 = len(set(neg_in_edgelists[u]).intersection(
            set(pos_out_edgelists[v])))
        d3_4 = len(set(neg_in_edgelists[u]).intersection(
            set(neg_out_edgelists[v])))

        d4_1 = len(set(pos_in_edgelists[u]).intersection(
            set(pos_in_edgelists[v])))
        d4_2 = len(set(pos_in_edgelists[u]).intersection(
            set(neg_in_edgelists[v])))
        d4_3 = len(set(neg_in_edgelists[u]).intersection(
            set(pos_in_edgelists[v])))
        d4_4 = len(set(neg_in_edgelists[u]).intersection(
            set(neg_in_edgelists[v])))

        return d1_1, d1_2, d1_3, d1_4, d2_1, d2_2, d2_3, d2_4, d3_1, d3_2, d3_3, d3_4, d4_1, d4_2, d4_3, d4_4

    def build_adj_lists(self, edge_index_s: torch.LongTensor) -> List:

        adj_list_pos = defaultdict(set)
        adj_list_pos_out = defaultdict(set)
        adj_list_pos_in = defaultdict(set)
        adj_list_neg = defaultdict(set)
        adj_list_neg_out = defaultdict(set)
        adj_list_neg_in = defaultdict(set)

        for node_i, node_j, s in edge_index_s:

            if s > 0 :
                adj_list_pos[node_i].add(node_j)
                adj_list_pos[node_j].add(node_i)

                adj_list_pos_out[node_i].add(node_j)
                adj_list_pos_in[node_j].add(node_i)
            if s < 0:
                adj_list_neg[node_i].add(node_j)
                adj_list_neg[node_j].add(node_i)

                adj_list_neg_out[node_i].add(node_j)
                adj_list_neg_in[node_j].add(node_i)

        adj_additions1 = [defaultdict(set) for _ in range(16)]
        adj_additions2 = [defaultdict(set) for _ in range(16)]

        pos_in_edgelists, pos_out_edgelists, neg_in_edgelists, neg_out_edgelists = adj_list_pos_in, adj_list_pos_out, adj_list_neg_in, adj_list_neg_out
        r_edgelists = (pos_in_edgelists, pos_out_edgelists,
                       neg_in_edgelists, neg_out_edgelists)

        adj1 = adj_list_pos_out.copy()
        adj2 = adj_list_pos_out.copy()

        for i in adj1:
            for j in adj1[i]:
                v_list = self.get_tri_features(i, j, r_edgelists)
                for index, v in enumerate(v_list):
                    if v > 0:
                        adj_additions1[index][i].add(j)

        for i in adj2:
            for j in adj2[i]:
                v_list = self.get_tri_features(i, j, r_edgelists)
                for index, v in enumerate(v_list):
                    if v > 0:
                        adj_additions2[index][i].add(j)

        self.adj_pos = adj_list_pos
        self.adj_neg = adj_list_neg

        return [adj_list_pos, adj_list_pos_out, adj_list_pos_in, adj_list_neg, adj_list_neg_out, adj_list_neg_in] + adj_additions1 + adj_additions2

    def forward(self) -> torch.FloatTensor:
        neigh_feats = []

        for edges, agg in zip(self.edge_lists, self.aggs):
            edges, _ = add_self_loops(edges)
            x1 = self.x
            x2 = agg(x1, edges)
            neigh_feats.append(x2)

        x0 = self.x
        combined = torch.cat([x0] + neigh_feats, 1)
        combined = self.mlp_layer(combined)
        return combined

    def loss(self) -> torch.FloatTensor:
        z = self.forward()
        nll_loss = self.lsp_loss(z, self.pos_edge_index, self.neg_edge_index)
        return nll_loss 
