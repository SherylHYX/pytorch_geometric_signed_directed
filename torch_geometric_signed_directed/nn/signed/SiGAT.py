
from collections import defaultdict
from typing import Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  GATConv
from torch_geometric.utils import k_hop_subgraph, add_self_loops


class SiGAT(nn.Module):
    r"""The signed graph attention network model (SiGAT) from the `"Signed Graph
    Attention Networks" <https://arxiv.org/abs/1906.10958>`_ paper.

    Args:
        node_num ([type]): Number of node.
        edge_index_s (list): The edgelist with sign. (e.g., [[0, 1, -1]] )
        in_emb_dim (int, optional): Size of each input sample features. Defaults to 20.
        hidden_emb_dim (int): Size of each hidden embeddings. Defaults to 20.
        batch_size (int, optional): Mini-batch size of training. Defaults to 500.
    """

    def __init__(
        self, 
        node_num: int, 
        edge_index_s,
        in_emb_dim: int = 20, 
        hidden_emb_dim: int = 20, 
        batch_size=500
    ):
        super().__init__()

        self.in_emb_dim = in_emb_dim
        self.hidden_emb_dim = hidden_emb_dim
        self.node_num = node_num
        self.batch_size = batch_size
        self.device = edge_index_s.device

        self.embeddings = nn.Embedding(node_num, in_emb_dim)

        edge_index_s_list = edge_index_s.cpu().numpy().tolist()
        self.adj_lists = self.build_adj_lists(edge_index_s_list)
        self.edge_lists = [self.map_adj_to_edges(i) for i in self.adj_lists]

        self.aggs = []
        for i in range(len(self.adj_lists)):
            self.aggs.append(
                GATConv(in_channels=in_emb_dim, out_channels=hidden_emb_dim)
            )
            self.add_module('agg_{}'.format(i), self.aggs[-1])

        self.mlp_layer = nn.Sequential(
            nn.Linear(hidden_emb_dim *
                      (len(self.adj_lists) + 1), hidden_emb_dim),
            nn.Tanh(),
            nn.Linear(hidden_emb_dim, hidden_emb_dim)
        )

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

            if s == 1:
                adj_list_pos[node_i].add(node_j)
                adj_list_pos[node_j].add(node_i)

                adj_list_pos_out[node_i].add(node_j)
                adj_list_pos_in[node_j].add(node_i)
            else:
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

    def forward(self, nodes: Union[np.array, torch.Tensor]) -> torch.FloatTensor:

        if isinstance(nodes, torch.Tensor):
            nodes_t = nodes
        else:
            nodes_t = torch.from_numpy(nodes).to(self.device)
        
        neigh_feats = []

        for edges, agg in zip(self.edge_lists, self.aggs):
            # https://github.com/pyg-team/pytorch_geometric/issues/1076
            # Consider flow="source_to_target", then we want a subgraph where messages flow to node_idx
            # Here, we need the nodes which can flow center node i to nodes j (neighbors)
            edges, _ = add_self_loops(edges)
            nodes_1, edges_1, inv, _ = k_hop_subgraph(
                nodes_t, 1, edges, num_nodes=self.node_num, flow='target_to_source', relabel_nodes=True)
            x1 = self.embeddings(nodes_1)
            x2 = agg(x1, edges_1)[inv]
            neigh_feats.append(x2)

        x0 = self.embeddings(nodes_t)
        combined = torch.cat([x0] + neigh_feats, 1)
        combined = self.mlp_layer(combined)
        return combined

    def loss(self, nodes: np.array) -> torch.Tensor:
        pos_neighbors, neg_neighbors = self.adj_pos, self.adj_neg
        pos_neighbors_list = [set.union(pos_neighbors[i]) for i in nodes]
        neg_neighbors_list = [set.union(neg_neighbors[i]) for i in nodes]
        unique_nodes_list = list(
            set.union(*pos_neighbors_list).union(*neg_neighbors_list).union(nodes))
        unique_nodes_list = np.array(unique_nodes_list)
        unique_nodes_dict = {n: i for i, n in enumerate(unique_nodes_list)}
        assert unique_nodes_list.shape == unique_nodes_list.shape
        nodes_embs = self.forward(unique_nodes_list)

        loss_total = 0
        for node in nodes:
            z1 = nodes_embs[unique_nodes_dict[node], :]
            pos_neigs = list([unique_nodes_dict[i]
                             for i in pos_neighbors[node]])
            neg_neigs = list([unique_nodes_dict[i]
                             for i in neg_neighbors[node]])
            pos_num = len(pos_neigs)
            neg_num = len(neg_neigs)

            if pos_num > 0:
                pos_neig_embs = nodes_embs[pos_neigs, :]
                loss_pku = -1 * \
                    torch.sum(F.logsigmoid(torch.einsum(
                        "nj,j->n", [pos_neig_embs, z1])))
                loss_total += loss_pku

            if neg_num > 0:
                neg_neig_embs = nodes_embs[neg_neigs, :]
                loss_pku = -1 * \
                    torch.sum(
                        F.logsigmoid(-1 * torch.einsum("nj,j->n", [neg_neig_embs, z1])))
                C = pos_num // neg_num
                loss_total += C * loss_pku

        return loss_total
