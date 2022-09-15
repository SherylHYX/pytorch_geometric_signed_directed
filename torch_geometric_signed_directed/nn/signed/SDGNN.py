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
    r"""The signed directed relationship layer from 
    `"SDGNN: Learning Node Representation for Signed Directed Networks" <https://arxiv.org/abs/2101.02390>`_ paper.

        Args:
            in_dim (int): Dimenson of input features. Defaults to 20.
            out_dim (int): Dimenson of output features. Defaults to 20.
            edge_lists (list): Edgelist for current motifs.
    """

    def __init__(
        self,
        in_dim: int = 20,
        out_dim: int = 20,
        edge_lists: list = [],
        **kwargs
    ):
        super().__init__(**kwargs)

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
        self.mlp_layer.apply(init_weights)
        for agg in self.aggs:
            agg.reset_parameters()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        neigh_feats = []
        for edges, agg in zip(self.edge_lists, self.aggs):
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
        init_emb: (FloatTensor, optional): The initial embeddings. Defaults to :obj:`None`, which will use TSVD as initial embeddings. 
        init_emb_grad (bool optional): Whether to set the initial embeddings to be trainable. (default: :obj:`False`)
        lamb_d (float, optional): Balances the direction loss contributions of the overall objective. (default: :obj:`1.0`)
        lamb_t (float, optional): Balances the triangle loss contributions of the overall objective. (default: :obj:`1.0`)
    """

    def __init__(
        self,
        node_num: int,
        edge_index_s,
        in_dim: int = 20,
        out_dim: int = 20,
        layer_num: int = 2,
        init_emb: torch.FloatTensor = None,
        init_emb_grad: bool = True,
        lamb_d: float = 5.0,
        lamb_t: float = 1.0,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.device = edge_index_s.device
        self.lamb_d = lamb_d
        self.lamb_t = lamb_t

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
        pos_in_edgelist, pos_out_edgelist, neg_in_edgelist, neg_out_edgelist = r_edgelists

        d1_1 = len(set(pos_out_edgelist[u]).intersection(
            set(pos_in_edgelist[v])))
        d1_2 = len(set(pos_out_edgelist[u]).intersection(
            set(neg_in_edgelist[v])))
        d1_3 = len(set(neg_out_edgelist[u]).intersection(
            set(pos_in_edgelist[v])))
        d1_4 = len(set(neg_out_edgelist[u]).intersection(
            set(neg_in_edgelist[v])))

        d2_1 = len(set(pos_out_edgelist[u]).intersection(
            set(pos_out_edgelist[v])))
        d2_2 = len(set(pos_out_edgelist[u]).intersection(
            set(neg_out_edgelist[v])))
        d2_3 = len(set(neg_out_edgelist[u]).intersection(
            set(pos_out_edgelist[v])))
        d2_4 = len(set(neg_out_edgelist[u]).intersection(
            set(neg_out_edgelist[v])))

        d3_1 = len(set(pos_in_edgelist[u]).intersection(
            set(pos_out_edgelist[v])))
        d3_2 = len(set(pos_in_edgelist[u]).intersection(
            set(neg_out_edgelist[v])))
        d3_3 = len(set(neg_in_edgelist[u]).intersection(
            set(pos_out_edgelist[v])))
        d3_4 = len(set(neg_in_edgelist[u]).intersection(
            set(neg_out_edgelist[v])))

        d4_1 = len(set(pos_in_edgelist[u]).intersection(
            set(pos_in_edgelist[v])))
        d4_2 = len(set(pos_in_edgelist[u]).intersection(
            set(neg_in_edgelist[v])))
        d4_3 = len(set(neg_in_edgelist[u]).intersection(
            set(pos_in_edgelist[v])))
        d4_4 = len(set(neg_in_edgelist[u]).intersection(
            set(neg_in_edgelist[v])))

        return (d1_1, d1_2, d1_3, d1_4,\
                d2_1, d2_2, d2_3, d2_4,\
                d3_1, d3_2, d3_3, d3_4,\
                d4_1, d4_2, d4_3, d4_4)

    def build_adj_lists(self, edge_index_s: torch.LongTensor) -> List:
        edge_index_s_list = edge_index_s.cpu().numpy().tolist()
        self.weight_dict = defaultdict(dict)

        pos_edgelist = defaultdict(set)
        pos_out_edgelist = defaultdict(set)
        pos_in_edgelist = defaultdict(set)
        neg_edgelist = defaultdict(set)
        neg_out_edgelist = defaultdict(set)
        neg_in_edgelist = defaultdict(set)

        for node_i, node_j, s in edge_index_s_list:

            if s > 0:
                pos_edgelist[node_i].add(node_j)
                pos_edgelist[node_j].add(node_i)

                pos_out_edgelist[node_i].add(node_j)
                pos_in_edgelist[node_j].add(node_i)
            if s < 0:
                neg_edgelist[node_i].add(node_j)
                neg_edgelist[node_j].add(node_i)

                neg_out_edgelist[node_i].add(node_j)
                neg_in_edgelist[node_j].add(node_i)

        r_edgelists = (pos_in_edgelist, pos_out_edgelist,
                       neg_in_edgelist, neg_out_edgelist)

        adj1 = pos_out_edgelist.copy()
        adj2 = neg_out_edgelist.copy()
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
        self.tri_weight = sp.csc_matrix((value, (row, col)),
                                        shape=(self.node_num, self.node_num))

        return [pos_out_edgelist, pos_in_edgelist, neg_out_edgelist, neg_in_edgelist]

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
        return loss_sign +  self.lamb_d * loss_direction + self.lamb_t * loss_triangle
