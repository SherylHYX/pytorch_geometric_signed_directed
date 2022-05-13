from typing import Union, List, Tuple

import torch
import scipy
import numpy as np
import networkx as nx
from networkx.algorithms import tree
import torch_geometric
from torch_geometric.utils import negative_sampling, to_undirected
from scipy.sparse import coo_matrix


def undirected_label2directed_label(adj: scipy.sparse.csr_matrix, edge_pairs: List[Tuple],
                                    task: str, directed_graph: bool = True, signed_directed: bool = False) -> Union[List, List]:
    r"""Generate edge labels based on the task.

    Arg types:
        * **adj** (scipy.sparse.csr_matrix) - Scipy sparse undirected adjacency matrix. 
        * **edge_pairs** (List[Tuple]) - The edge list for the link dataset querying. Each element 
            in the list is an edge tuple.
        * **edge_weight** (List[Tuple]) - The edge weights list for sign graphs.
        * **task** (str): The evaluation task - all (three-class link prediction); direction (direction prediction); 
            existence (existence prediction) 

    Return types:
        * **new_edge_pairs** (List) - A list of edges.
        * **labels** (List) - The labels for new_edge_pairs. 
            * If task == "existence": 0 (the directed edge exists in the graph), 1 (the edge doesn't exist).
                The undirected edges in the directed input graph are removed to avoid ambiguity.
            * If task == "direction": 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists).
                The undirected edges in the directed input graph are removed to avoid ambiguity.
            * If task == "three_class_digraph": 0 (the directed edge exists in the graph), 
                1 (the edge of the reversed direction exists), 2 (the edge doesn't exist in both directions). 
                The undirected edges in the directed input graph are removed to avoid ambiguity.
            * If task == "four_class_signed_digraph": 0 (the positive directed edge exists in the graph), 
                1 (the negative directed edge exists in the graph), 2 (the positive edge of the reversed direction exists),
                3 (the edge of the reversed direction exists). 
                The undirected edges in the directed input graph are removed to avoid ambiguity.
            * If task == "five_class_signed_digraph": 0 (the positive directed edge exists in the graph), 
                1 (the negative directed edge exists in the graph), 2 (the positive edge of the reversed direction exists),
                3 (the edge of the reversed direction exists), 4 (the edge doesn't exist in both directions). 
                The undirected edges in the directed input graph are removed to avoid ambiguity.
            * If task == "sign": 0 (negative edge), 1 (positive edge). 
        * **label_weight** (List) - The weight list of the query edges. The weight is zero if the directed edge 
            doesn't exist in both directions.
        * **undirected** (List) - The undirected edges list within the input graph.
    """
    if len(edge_pairs) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    labels = -np.ones(len(edge_pairs), dtype=np.int32)
    new_edge_pairs = np.array(list(map(list, edge_pairs)))
    counter = 0

    # get directed edges
    edge_pairs = np.array(list(map(list, edge_pairs)))

    
    if signed_directed:
        directed_pos = (
            np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten() > 0).tolist()
        directed_neg = (
            np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten() < 0).tolist()
        inversed_pos = (
            np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten() > 0).tolist()
        inversed_neg = (
            np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten() < 0).tolist()
        undirected_pos = np.logical_and(directed_pos, inversed_pos)
        undirected_neg = np.logical_and(directed_neg, inversed_neg)

        directed_pos = list(map(tuple, edge_pairs[directed_pos].tolist()))
        directed_neg = list(map(tuple, edge_pairs[directed_neg].tolist()))
        inversed_pos = list(map(tuple, edge_pairs[inversed_pos].tolist()))
        inversed_neg = list(map(tuple, edge_pairs[inversed_neg].tolist()))
        undirected_pos = list(map(tuple, edge_pairs[undirected_pos].tolist()))
        undirected_neg = list(map(tuple, edge_pairs[undirected_neg].tolist()))

        edge_pairs = list(map(tuple, edge_pairs.tolist()))
        negative = np.array(
            list(set(edge_pairs) - set(directed_pos) - set(inversed_pos) - set(directed_neg) - set(inversed_neg)))
        directed_pos = np.array(list(set(directed_pos) - set(undirected_pos)))
        inversed_pos = np.array(list(set(inversed_pos) - set(undirected_pos)))
        directed_neg = np.array(list(set(directed_neg) - set(undirected_neg)))
        inversed_neg = np.array(list(set(inversed_neg) - set(undirected_neg)))

        directed = np.vstack([directed_pos, directed_neg])
        if not inversed_pos.size:
            inversed = inversed_neg
            inversed_pos = []
        elif not inversed_neg.size:
            inversed = inversed_pos
            inversed_neg = []
        else:
            inversed = np.vstack([inversed_pos, inversed_neg])
        undirected_pos, undirected_neg = np.array(undirected_pos), np.array(undirected_neg)
        if not undirected_pos.size:
            undirected = undirected_neg
        elif not undirected_neg.size:
            undirected = undirected_pos
        else:
            undirected = np.vstack([undirected_pos, undirected_neg])
        new_edge_pairs = np.vstack(
            [directed, inversed]) if inversed.size else directed
        new_edge_pairs = np.vstack([new_edge_pairs, new_edge_pairs[:, [1, 0]]])
        new_edge_pairs = np.vstack([new_edge_pairs, negative])

        labels = np.vstack([np.zeros((len(directed_pos), 1), dtype=np.int32),
                            np.ones((len(directed_neg), 1), dtype=np.int32),
                            2 * np.ones((len(inversed_pos), 1), dtype=np.int32),
                            3 * np.ones((len(inversed_neg), 1), dtype=np.int32)])

        labels = np.vstack([labels, 2 * np.ones((len(directed_pos), 1), dtype=np.int32),
                            3 * np.ones((len(directed_neg), 1), dtype=np.int32),
                            np.zeros((len(inversed_pos), 1), dtype=np.int32),
                            np.ones((len(inversed_neg), 1), dtype=np.int32)])

        labels = np.vstack(
            [labels, 4*np.ones((len(negative), 1), dtype=np.int32)])

        if len(inversed_pos) and len(inversed_neg):
            label_weight = np.vstack([np.array(adj[directed_pos[:, 0], directed_pos[:, 1]]).flatten()[:, None],
                                    np.array(adj[directed_neg[:, 0], directed_neg[:, 1]]).flatten()[:, None], 
                                    np.array(adj[inversed_pos[:, 1], inversed_pos[:, 0]]).flatten()[:, None],
                                    np.array(adj[inversed_neg[:, 1], inversed_neg[:, 0]]).flatten()[:, None]])
        elif len(inversed_pos):
            label_weight = np.vstack([np.array(adj[directed_pos[:, 0], directed_pos[:, 1]]).flatten()[:, None],
                                    np.array(adj[directed_neg[:, 0], directed_neg[:, 1]]).flatten()[:, None], 
                                    np.array(adj[inversed_pos[:, 1], inversed_pos[:, 0]]).flatten()[:, None]])
        elif len(inversed_neg):
            label_weight = np.vstack([np.array(adj[directed_pos[:, 0], directed_pos[:, 1]]).flatten()[:, None],
                                    np.array(adj[directed_neg[:, 0], directed_neg[:, 1]]).flatten()[:, None],
                                    np.array(adj[inversed_neg[:, 1], inversed_neg[:, 0]]).flatten()[:, None]])
        else:
            label_weight = np.vstack([np.array(adj[directed_pos[:, 0], directed_pos[:, 1]]).flatten()[:, None],
                                    np.array(adj[directed_neg[:, 0], directed_neg[:, 1]]).flatten()[:, None]])
        label_weight = np.vstack([label_weight, label_weight])
        label_weight = np.vstack(
            [label_weight, np.zeros((len(negative), 1), dtype=np.int32)])
        assert label_weight[labels==0].min() > 0
        assert label_weight[labels==1].max() < 0
        assert label_weight[labels==2].min() > 0
        assert label_weight[labels==3].max() < 0
        assert label_weight[labels==4].mean() == 0
    elif directed_graph:
        directed = (np.abs(
            np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()) > 0).tolist()
        inversed = (np.abs(
            np.array(adj[edge_pairs[:, 1], edge_pairs[:, 0]]).flatten()) > 0).tolist()
        undirected = np.logical_and(directed, inversed)

        directed = list(map(tuple, edge_pairs[directed].tolist()))
        inversed = list(map(tuple, edge_pairs[inversed].tolist()))
        undirected = list(map(tuple, edge_pairs[undirected].tolist()))

        edge_pairs = list(map(tuple, edge_pairs.tolist()))
        negative = np.array(
            list(set(edge_pairs) - set(directed) - set(inversed)))
        directed = np.array(list(set(directed) - set(undirected)))
        inversed = np.array(list(set(inversed) - set(undirected)))

        new_edge_pairs = np.vstack(
            [directed, inversed]) if inversed.size else directed
        new_edge_pairs = np.vstack([new_edge_pairs, new_edge_pairs[:, [1, 0]]])
        new_edge_pairs = np.vstack([new_edge_pairs, negative])

        labels = np.vstack([np.zeros((len(directed), 1), dtype=np.int32),
                            np.ones((len(inversed), 1), dtype=np.int32)]) if len(inversed) else \
            np.zeros((len(directed), 1), dtype=np.int32)
        labels = np.vstack([labels, np.ones((len(directed), 1), dtype=np.int32),
                            np.zeros((len(inversed), 1), dtype=np.int32)]) if len(inversed) else \
            np.vstack([labels, np.ones((len(directed), 1), dtype=np.int32)])
        labels = np.vstack(
            [labels, 2*np.ones((len(negative), 1), dtype=np.int32)])

        label_weight = np.vstack([np.array(adj[directed[:, 0], directed[:, 1]]).flatten()[:, None],
                                  np.array(adj[inversed[:, 1], inversed[:, 0]]).flatten()[:, None]]) if len(inversed) else \
            np.array(adj[directed[:, 0], directed[:, 1]]).flatten()[:, None]
        label_weight = np.vstack([label_weight, label_weight])
        label_weight = np.vstack(
            [label_weight, np.zeros((len(negative), 1), dtype=np.int32)])
    else:
        undirected = list(map(tuple, edge_pairs.tolist()))
        neg_edges = (
            np.abs(np.array(adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()) == 0)
        labels = np.zeros(len(edge_pairs), dtype=np.int32)
        labels[neg_edges] = 2
        new_edge_pairs = edge_pairs
        label_weight = np.array(
            adj[edge_pairs[:, 0], edge_pairs[:, 1]]).flatten()

    if task == 'existence':
        # existence prediction
        label_weight[labels == 1] = 0  # set reversed edges as 0
        labels[labels == 2] = 1
        

    return new_edge_pairs, labels.flatten(), label_weight.flatten(), undirected


def link_class_split(data: torch_geometric.data.Data, size: int = None, splits: int = 10, prob_test: float = 0.15,
                     prob_val: float = 0.05, task: str = 'direction', seed: int = 0, maintain_connect: bool = True,
                     ratio: float = 1.0, device: str = 'cpu') -> dict:
    r"""Get train/val/test dataset for the link prediction task. 

    Arg types:
        * **data** (torch_geometric.data.Data or DirectedData object) - The input dataset.
        * **prob_val** (float, optional) - The proportion of edges selected for validation (Default: 0.05).
        * **prob_test** (float, optional) - The proportion of edges selected for testing (Default: 0.15).
        * **splits** (int, optional) - The split size (Default: 10).
        * **size** (int, optional) - The size of the input graph. If none, the graph size is the maximum index of nodes plus 1 (Default: None).
        * **task** (str, optional) - The evaluation task: all (three-class link prediction); direction (direction prediction); existence (existence prediction); sign (sign prediction). (Default: 'direction')
        * **seed** (int, optional) - The random seed for positve edge selection (Default: 0). Negative edges are selected by pytorch geometric negative_sampling.
        * **maintain_connect** (bool, optional) - If maintaining connectivity when removing edges for validation and testing. The connectivity is maintained by obtaining edges in the minimum spanning tree/forest first. These edges will not be removed for validation and testing (Default: True). 
        * **ratio** (float, optional) - The maximum ratio of edges used for dataset generation. (Default: 1.0)
        * **device** (int, optional) - The device to hold the return value (Default: 'cpu').

    Return types:
        * **datasets** - A dict include training/validation/testing splits of edges and labels. For split index i:

            * datasets[i]['graph'] (torch.LongTensor): the observed edge list after removing edges for validation and testing.

            * datasets[i]['train'/'val'/'testing']['edges'] (List): the edge list for training/validation/testing.

            * datasets[i]['train'/'val'/'testing']['label'] (List): the labels of edges:

                * If task == "existence": 0 (the directed edge exists in the graph), 1 (the edge doesn't exist). The undirected edges in the directed input graph are removed to avoid ambiguity.

                * If task == "direction": 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists). The undirected edges in the directed input graph are removed to avoid ambiguity.

                * If task == "three_class_digraph": 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists), 2 (the edge doesn't exist in both directions). The undirected edges in the directed input graph are removed to avoid ambiguity.

                * If task == "four_class_signed_digraph": 0 (the positive directed edge exists in the graph), 
                    1 (the negative directed edge exists in the graph), 2 (the positive edge of the reversed direction exists),
                    3 (the edge of the reversed direction exists). 
                    The undirected edges in the directed input graph are removed to avoid ambiguity.
                
                * If task == "five_class_signed_digraph": 0 (the positive directed edge exists in the graph), 
                    1 (the negative directed edge exists in the graph), 2 (the positive edge of the reversed direction exists),
                    3 (the edge of the reversed direction exists), 4 (the edge doesn't exist in both directions). 
                    The undirected edges in the directed input graph are removed to avoid ambiguity.
                
                * If task == "sign": 0 (negative edge), 1 (positive edge). This is the link sign prediction task for signed networks.
    """
    assert task in ["existence", "direction", "three_class_digraph", "four_class_signed_digraph", "five_class_signed_digraph", 
                    "sign"], "Please select a valid task from 'existence', 'direction', 'three_class_digraph', 'four_class_signed_digraph', 'five_class_digraph', and 'sign'!"
    edge_index = data.edge_index.cpu()
    row, col = edge_index[0], edge_index[1]
    if size is None:
        size = int(max(torch.max(row), torch.max(col))+1)
    if not hasattr(data, "edge_weight"):
        data.edge_weight = torch.ones(len(row))
    if data.edge_weight is None:
        data.edge_weight = torch.ones(len(row))


    if hasattr(data, "A"):
        A = data.A.tocsr()
    else:
        A = coo_matrix((data.edge_weight.cpu(), (row, col)),
                       shape=(size, size), dtype=np.float32).tocsr()

    
    len_val = int(prob_val*len(row))
    len_test = int(prob_test*len(row))
    if task not in ["existence", "direction", 'three_class_digraph']:
        pos_ratio = (A.toarray()>0).sum()/(A.toarray()!=0).sum()
        neg_ratio = 1 - pos_ratio
        len_val_pos = int(prob_val*len(row)*pos_ratio)
        len_val_neg = int(prob_val*len(row)*neg_ratio)
        len_test_pos = int(prob_test*len(row)*pos_ratio)
        len_test_neg = int(prob_test*len(row)*neg_ratio)

    undirect_edge_index = to_undirected(edge_index)
    neg_edges = negative_sampling(undirect_edge_index, num_neg_samples=len(
        edge_index.T), force_undirected=False).numpy().T
    neg_edges = map(tuple, neg_edges)
    neg_edges = list(neg_edges)

    undirect_edge_index = undirect_edge_index.T.tolist()
    if maintain_connect:
        assert ratio == 1, "ratio should be 1.0 if maintain_connect=True"
        G = nx.from_scipy_sparse_matrix(
            A, create_using=nx.Graph, edge_attribute='weight')
        mst = list(tree.minimum_spanning_edges(
            G, algorithm="kruskal", data=False))
        all_edges = list(map(tuple, undirect_edge_index))
        nmst = list(set(all_edges) - set(mst))
        if len(nmst) < (len_val+len_test):
            raise ValueError(
                "There are no enough edges to be removed for validation/testing. Please use a smaller prob_test or prob_val.")
    else:
        mst = []
        nmst = edge_index.T.tolist()

    rs = np.random.RandomState(seed)
    datasets = {}

    is_directed = not data.is_undirected()
    max_samples = int(ratio*len(edge_index.T))+1
    assert ratio <= 1.0 and ratio > 0, "ratio should be smaller than 1.0 and larger than 0"
    assert ratio > prob_val + prob_test, "ratio should be larger than prob_val + prob_test"
    for ind in range(splits):
        rs.shuffle(nmst)
        rs.shuffle(neg_edges)

        if task == 'sign':
            nmst = np.array(nmst)
            exist = np.array(np.abs(A[nmst[:, 0], nmst[:, 1]]) > 0).flatten()
            if np.sum(exist) < len(nmst):
                nmst = nmst[exist]

            pos_val_edges = nmst[np.array(A[nmst[:, 0], nmst[:, 1]] > 0).squeeze()].tolist()
            neg_val_edges = nmst[np.array(A[nmst[:, 0], nmst[:, 1]] < 0).squeeze()].tolist()

            ids_test = np.array(pos_val_edges[:len_test_pos].copy() + neg_val_edges[:len_test_neg].copy())
            ids_val = np.array(pos_val_edges[len_test_pos:len_test_pos+len_val_pos].copy() + \
                neg_val_edges[len_test_neg:len_test_neg+len_val_neg].copy())
            ids_train = np.array(pos_val_edges[len_test_pos+len_val_pos:max_samples] + \
                neg_val_edges[len_test_neg+len_val_neg:max_samples] + mst)

            labels_test = 1.0 * \
                np.array(A[ids_test[:, 0], ids_test[:, 1]] > 0).flatten()
            labels_val = 1.0 * \
                np.array(A[ids_val[:, 0], ids_val[:, 1]] > 0).flatten()
            labels_train = 1.0 * \
                np.array(A[ids_train[:, 0], ids_train[:, 1]] > 0).flatten()
            undirected_train = np.array([])
        elif task in ["existence", "direction", 'three_class_digraph']:
            ids_test = nmst[:len_test]+neg_edges[:len_test]
            ids_val = nmst[len_test:len_test+len_val] + \
                neg_edges[len_test:len_test+len_val]
            if len_test+len_val < len(nmst):
                ids_train = nmst[len_test+len_val:max_samples] + \
                    mst+neg_edges[len_test+len_val:max_samples]
            else:
                ids_train = mst+neg_edges[len_test+len_val:max_samples]

            ids_test, labels_test, _, _ = undirected_label2directed_label(
                A, ids_test, task, is_directed)
            ids_val, labels_val, _, _ = undirected_label2directed_label(
                A, ids_val, task, is_directed)
            ids_train, labels_train, _, undirected_train = undirected_label2directed_label(
                A, ids_train, task, is_directed)
        else:
            nmst = np.array(nmst)
            pos_val_edges = nmst[np.array(A[nmst[:, 0], nmst[:, 1]] > 0).squeeze()].tolist()
            neg_val_edges = nmst[np.array(A[nmst[:, 0], nmst[:, 1]] < 0).squeeze()].tolist()

            ids_test = np.array(pos_val_edges[:len_test_pos].copy() + neg_val_edges[:len_test_neg].copy() + \
                neg_edges[:len_test])
            ids_val = np.array(pos_val_edges[len_test_pos:len_test_pos+len_val_pos].copy() + \
                neg_val_edges[len_test_neg:len_test_neg+len_val_neg].copy() + \
                neg_edges[len_test:len_test+len_val])
            
            if len_test+len_val < len(nmst):
                ids_train = np.array(pos_val_edges[len_test_pos+len_val_pos:max_samples] + \
                    neg_val_edges[len_test_neg+len_val_neg:max_samples] + mst + neg_edges[len_test+len_val:max_samples])
            else:
                ids_train = mst+neg_edges[len_test+len_val:max_samples]

            ids_test, labels_test, _, _ = undirected_label2directed_label(
                A, ids_test, task, is_directed, True)
            ids_val, labels_val, _, _ = undirected_label2directed_label(
                A, ids_val, task, is_directed, True)
            ids_train, labels_train, _, undirected_train = undirected_label2directed_label(
                A, ids_train, task, is_directed, True)

        # convert back to directed graph
        if task == 'direction':
            ids_train = ids_train[labels_train < 2]
            #label_train_w = label_train_w[labels_train <2]
            labels_train = labels_train[labels_train < 2]

            ids_test = ids_test[labels_test < 2]
            #label_test_w = label_test_w[labels_test <2]
            labels_test = labels_test[labels_test < 2]

            ids_val = ids_val[labels_val < 2]
            #label_val_w = label_val_w[labels_val <2]
            labels_val = labels_val[labels_val < 2]
        elif task == 'four_class_signed_digraph':
            ids_train = ids_train[labels_train < 4]
            labels_train = labels_train[labels_train < 4]

            ids_test = ids_test[labels_test < 4]
            labels_test = labels_test[labels_test < 4]

            ids_val = ids_val[labels_val < 4]
            labels_val = labels_val[labels_val < 4]

        # set up the observed graph and weights after splitting
        oberved_edges = -np.ones((len(ids_train), 2), dtype=np.int32)
        oberved_weight = np.zeros((len(ids_train), 1), dtype=np.float32)

        direct = (
            np.abs(A[ids_train[:, 0], ids_train[:, 1]].data) > 0).flatten()
        oberved_edges[direct, 0] = ids_train[direct, 0]
        oberved_edges[direct, 1] = ids_train[direct, 1]
        oberved_weight[direct, 0] = np.array(
            A[ids_train[direct, 0], ids_train[direct, 1]]).flatten()

        direct = (np.abs(A[ids_train[:, 1], ids_train[:, 0]].data) > 0)[0]
        oberved_edges[direct, 0] = ids_train[direct, 1]
        oberved_edges[direct, 1] = ids_train[direct, 0]
        oberved_weight[direct, 0] = np.array(
            A[ids_train[direct, 1], ids_train[direct, 0]]).flatten()

        valid = (np.sum(oberved_edges, axis=-1) >= 0)
        oberved_edges = oberved_edges[valid]
        oberved_weight = oberved_weight[valid]

        # add undirected edges back
        if len(undirected_train) > 0:
            undirected_train = np.array(undirected_train)
            oberved_edges = np.vstack(
                (oberved_edges, undirected_train, undirected_train[:, [1, 0]]))
            oberved_weight = np.vstack((oberved_weight, np.array(A[undirected_train[:, 0],
                                                                   undirected_train[:, 1]]).flatten()[:, None],
                                        np.array(A[undirected_train[:, 1],
                                                   undirected_train[:, 0]]).flatten()[:, None]))

        datasets[ind] = {}
        datasets[ind]['graph'] = torch.from_numpy(
            oberved_edges.T).long().to(device)
        datasets[ind]['weights'] = torch.from_numpy(
            oberved_weight.flatten()).float().to(device)

        datasets[ind]['train'] = {}
        datasets[ind]['train']['edges'] = torch.from_numpy(
            ids_train).long().to(device)
        datasets[ind]['train']['label'] = torch.from_numpy(
            labels_train).long().to(device)
        #datasets[ind]['train']['weight'] = torch.from_numpy(label_train_w).float().to(device)

        datasets[ind]['val'] = {}
        datasets[ind]['val']['edges'] = torch.from_numpy(
            ids_val).long().to(device)
        datasets[ind]['val']['label'] = torch.from_numpy(
            labels_val).long().to(device)
        #datasets[ind]['val']['weight'] = torch.from_numpy(label_val_w).float().to(device)

        datasets[ind]['test'] = {}
        datasets[ind]['test']['edges'] = torch.from_numpy(
            ids_test).long().to(device)
        datasets[ind]['test']['label'] = torch.from_numpy(
            labels_test).long().to(device)
        #datasets[ind]['test']['weight'] = torch.from_numpy(label_test_w).float().to(device)
    return datasets
