from typing import List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from torch_geometric.nn import GATConv

from torch_geometric_signed_directed.utils.signed import create_spectral_features
from torch_geometric_signed_directed.utils.signed import Sign_Product_Entropy_Loss, Sign_Direction_Loss, Sign_Triangle_Loss

class SDRLayer(nn.Module):
    r"""The signed directed relationship layer from `"SDGNN: Learning Node Representation for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper.

        Args:
            in_dim (int): Dimenson of input features. Defaults to 20.
            out_dim (int): Dimenson of output features. Defaults to 20.
            edge_lists (list): Edgelist for current motifs.
    """

    def __init__(
        self,
        in_dim: int = 20,
        out_dim: int = 20,
        edge_lists: list = []
    ):
        super().__init__()

        self.edge_lists = edge_lists
        self.aggs = []

        agg = GATConv

        for i in range(len(edge_lists)):
            self.aggs.append(
                agg(in_dim, out_dim)
            )
            self.add_module('agg_{}'.format(i), self.aggs[-1])

        self.mlp_layer = nn.Sequential(
            nn.Linear(in_dim * (len(edge_lists) + 1), out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim)
        )

    def reset_parameters(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)
        self.mlp_layer.apply(init_weights)
        for agg in self.aggs:
            agg.reset_parameters()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # full batchs
        neigh_feats = []
        for edges, agg in zip(self.edge_lists, self.aggs):
            # https://github.com/pyg-team/pytorch_geometric/issues/1076
            x2 = agg(x, edges)
            neigh_feats.append(x2)
        combined = torch.cat([x] + neigh_feats, 1)
        combined = self.mlp_layer(combined)
        return combined


class SDGNN(nn.Module):
    r"""The SDGNN model from  `"SDGNN: Learning Node Representation for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper.

    Args:
        node_num (int, optional): The number of nodes.
        edge_index_s (LongTensor): The edgelist with sign. (e.g., :obj:`torch.LongTensor([[0, 1, -1], [0, 2, 1]])` )
        in_dim (int, optional): Size of each input sample features. Defaults to 20.
        out_dim (int): Size of each hidden embeddings. Defaults to 20.
        layer_num (int, optional): Number of layers. Defaults to 2.
    """

    def __init__(
        self,
        node_num: int,
        edge_index_s,
        in_dim: int = 20,
        out_dim: int = 20,
        layer_num: int = 2
    ):

        super().__init__()

        self.in_dim = in_dim
        self.node_num = node_num
        self.layer_num = layer_num
        self.device = edge_index_s.device

        self.pos_edge_index = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        self.neg_edge_index = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()

        x = create_spectral_features(
            pos_edge_index=self.pos_edge_index,
            neg_edge_index=self.neg_edge_index,
            node_num=self.node_num,
            dim=self.in_dim
        ).to(self.device)
        self.x = nn.Parameter(x, requires_grad=True)

        self.device = edge_index_s.device

        self.adj_lists = self.build_adj_lists(edge_index_s)
        self.edge_lists = [self.map_adj_to_edges(i) for i in self.adj_lists]

        self.layers = []
        for i in range(layer_num):
            if i == 0:
                layer = SDRLayer(in_dim, out_dim,
                                 edge_lists=self.edge_lists)
            else:
                layer = SDRLayer(out_dim, out_dim,
                                 edge_lists=self.edge_lists)
            self.add_module(f'SDRLayer_{i}', layer)
            self.layers.append(layer)

        self.loss_sign = Sign_Product_Entropy_Loss()
        self.loss_direction = Sign_Direction_Loss(emb_dim=out_dim)
        self.loss_tri = Sign_Triangle_Loss(emb_dim=out_dim, edge_weight=self.tri_weight)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def map_adj_to_edges(self, adj_list: List) -> torch.LongTensor:
        edges = []
        for a in adj_list:
            for b in adj_list[a]:
                edges.append((a, b))
        edges = torch.LongTensor(edges).to(self.device)
        return edges.t()

    def get_features(self, u: int, v: int, r_edgelists: List) -> Tuple[int, int, int, int, int,
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

    def build_adj_lists(self, edge_index: torch.LongTensor) -> List:
        edge_index = edge_index.cpu().numpy().tolist()
        self.weight_dict = defaultdict(dict)

        adj_list1 = defaultdict(set)
        adj_list1_1 = defaultdict(set)
        adj_list1_2 = defaultdict(set)
        adj_list2 = defaultdict(set)
        adj_list2_1 = defaultdict(set)
        adj_list2_2 = defaultdict(set)

        for node_i, node_j, s in edge_index:

            if s > 0:
                adj_list1[node_i].add(node_j)
                adj_list1[node_j].add(node_i)

                adj_list1_1[node_i].add(node_j)
                adj_list1_2[node_j].add(node_i)
            if s < 0:
                adj_list2[node_i].add(node_j)
                adj_list2[node_j].add(node_i)

                adj_list2_1[node_i].add(node_j)
                adj_list2_2[node_j].add(node_i)

        pos_in_edgelists, pos_out_edgelists, neg_in_edgelists, neg_out_edgelists = adj_list1_2, adj_list1_1, adj_list2_2, adj_list2_1

        r_edgelists = (pos_in_edgelists, pos_out_edgelists,
                       neg_in_edgelists, neg_out_edgelists)

        adj1 = adj_list1_1.copy()
        adj2 = adj_list2_1.copy()
        for i in adj1:
            for j in adj1[i]:
                v_list = self.get_features(i, j, r_edgelists)
                mask = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]
                counts = np.dot(v_list, mask)
                self.weight_dict[i][j] = counts

        for i in adj2:
            for j in adj2[i]:
                v_list = self.get_features(i, j, r_edgelists)
                mask = [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
                counts = np.dot(v_list, mask)
                self.weight_dict[i][j] = counts
        
        row = []
        col = []
        value = []
        for i in self.weight_dict:
            for j in self.weight_dict[i]:
                row.append(i)
                col.append(j)
                value.append(self.weight_dict[i][j])
        self.tri_weight = sp.csc_matrix((value, (row, col)), shape=(self.node_num, self.node_num))
        self.adj1 = adj_list1
        self.adj2 = adj_list2

        return [adj_list1_1, adj_list1_2, adj_list2_1, adj_list2_2]

    def forward(self) -> torch.FloatTensor:
        x = self.x
        for layer_m in self.layers:
            x = layer_m(x)
        return x

    def loss(self):
        z = self.forward()
        loss_sign = self.loss_sign(z, self.pos_edge_index, self.neg_edge_index)
        loss_direction = self.loss_direction(z, self.pos_edge_index, self.neg_edge_index)
        loss_triangle = self.loss_tri(z, self.pos_edge_index, self.neg_edge_index)
        return loss_sign +  loss_direction + loss_triangle
