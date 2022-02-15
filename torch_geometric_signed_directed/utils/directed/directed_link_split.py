from typing import Union, List, Tuple

import torch
import scipy
import numpy as np
import networkx as nx
from networkx.algorithms import tree
import torch_geometric
from torch_geometric.utils import negative_sampling, to_undirected
from scipy.sparse import coo_matrix

def undirected_label2directed_label(adj:scipy.sparse.csr_matrix, edge_pairs:List[Tuple], 
                                    task:str, rs:np.random.RandomState) -> Union[List,List]:
    r"""Generate edge labels based on the task.

    Arg types:
        * **adj** (scipy.sparse.csr_matrix) - Scipy sparse undirected adjacency matrix. 
        * **edge_pairs** (List[Tuple]) - The edge list. each element in the list is an edge tuple.
        * **task** (str): The evaluation task - all (three-class link prediction); direction (direction prediction); existence (existence prediction) 
        * **rs** (np.random.RandomState) - The randomstate for edge selection.

    Return types:
        * **new_edge_pairs** (List) - A list of edges.
        * **labels** (List) - The labels for new_edge_pairs. 

                       If task == "existence": 0 (the edge exists in the graph), 1 (the edge doesn't exist).

                       If task == "direction": 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists).
                       
                       If task == 'all': 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists), 2 (the undirected version of the edge doesn't exist).
    """
    labels = np.zeros(len(edge_pairs), dtype=np.int32)
    new_edge_pairs = np.array(edge_pairs)
    counter = 0
    for i, e in enumerate(edge_pairs): # directed edges
        if adj[e[0], e[1]] + adj[e[1], e[0]]  > 0: # exists an edge
            if adj[e[0], e[1]] > 0:
                if adj[e[1], e[0]] == 0: # rule out undirected edges
                    if counter%2 == 0:
                        labels[i] = 0
                        new_edge_pairs[i] = [e[0], e[1]]
                        counter += 1
                    else:
                        labels[i] = 1
                        new_edge_pairs[i] = [e[1], e[0]]
                        counter += 1
                else:
                    new_edge_pairs[i] = [e[0], e[1]]
                    labels[i] = -1
            else: # the other direction, and not an undirected edge
                if counter%2 == 0:
                    labels[i] = 0
                    new_edge_pairs[i] = [e[1], e[0]]
                    counter += 1
                else:
                    labels[i] = 1
                    new_edge_pairs[i] = [e[0], e[1]]
                    counter += 1
        else: # negative edges
            labels[i] = 2
            new_edge_pairs[i] = [e[0], e[1]]

    if task == 'existence':
        # existence prediction
        labels[labels == 2] = 1
        neg = np.where(labels == 1)[0]
        neg_half = rs.choice(neg, size=len(neg)-np.sum(labels==0), replace=False)
        labels[neg_half] = -1
    return new_edge_pairs[labels >= 0], labels[labels >= 0]

def directed_link_class_split(data:torch_geometric.data.Data, size:int=None, splits:int=10, prob_test:float= 0.15, 
                     prob_val:float= 0.05, task:str= 'direction', seed:int= 0, device:str= 'cpu') -> dict:
    r"""Get train/val/test dataset for the link prediction task.

    Arg types:
        * **data** (torch_geometric.data.Data or DirectedData object) - The input dataset.
        * **prob_val** (float, optional) - The proportion of edges selected for validation (Default: 0.05).
        * **prob_test** (float, optional) - The proportion of edges selected for testing (Default: 0.15).
        * **splits** (int, optional) - The split size (Default: 10).
        * **size** (int, optional) - The size of the input graph. If none, the graph size is the maximum index of nodes plus 1 (Default: None).
        * **task** (str, optional) - The evaluation task: all (three-class link prediction); direction (direction prediction); existence (existence prediction). (Default: 'direction')
        * **seed** (int, optional) - The random seed for dataset generation (Default: 0).
        * **device** (int, optional) - The device to hold the return value (Default: 'cpu').

    Return types:
        * **datasets** - A dict include training/validation/testing splits of edges and labels. For split index i:

                      datasets[i]['graph'] (torch.LongTensor): the observed edge list after removing edges for validation and testing.

                      datasets[i]['train'/'val'/'testing']['edges'] (List): the edge list for training/validation/testing.

                      datasets[i]['train'/'val'/'testing']['label'] (List): the labels of edges:

                          If task == "existence": 0 (the edge exists in the graph), 1 (the edge doesn't exist).

                          If task == "direction": 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists).

                          If task == 'all': 0 (the directed edge exists in the graph), 1 (the edge of the reversed direction exists), 2 (the undirected version of the edge doesn't exist).
    """
    edge_index = data.edge_index.cpu()
    row, col = edge_index[0], edge_index[1]
    if size is None:
        size = int(max(torch.max(row), torch.max(col))+1)
    if data.edge_weight is None:
        data.edge_weight = torch.ones(len(row))
    A = coo_matrix((data.edge_weight.cpu(), (row, col)), shape=(size, size), dtype=np.float32).tocsr()
    # create an undirected graph based on the adjacency
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.Graph, edge_attribute='weight') 
    
    # get the minimum spanning tree based on the undirected graph
    mst = list(tree.minimum_spanning_edges(G, algorithm="kruskal", data=False))
    nmst = sorted(list(set(G.edges) - set(mst)))

    undirect_edge_index = to_undirected(edge_index)
    neg_edges = negative_sampling(undirect_edge_index, force_undirected=False).numpy().T
    neg_edges = map(tuple, neg_edges)
    neg_edges = list(neg_edges)
    
    len_val = int(prob_val*len(row))
    len_test = int(prob_test*len(row))

    if len(nmst) < (len_val+len_test):
        raise ValueError("There are no enough edges to be removed for validation/testing. Please use a smaller prob_test or prob_val.")

    rs = np.random.RandomState(seed)
    datasets = {}
    for ind in range(splits):
        rs.shuffle(nmst)

        ids_test = nmst[:len_test]+neg_edges[:len_test]
        ids_val = nmst[len_test:len_test+len_val]+neg_edges[len_test:len_test+len_val]
        if len_test+len_val < len(nmst):
            ids_train = nmst[len_test+len_val:]+mst+neg_edges[len_test+len_val:]
        else:
            ids_train = mst+neg_edges[len_test+len_val:]

        ids_test, labels_test = undirected_label2directed_label(A, ids_test, task, rs)  
        ids_val, labels_val = undirected_label2directed_label(A, ids_val, task, rs)
        ids_train, labels_train = undirected_label2directed_label(A, ids_train, task, rs)

        # convert back to directed graph
        oberved_edges = np.zeros((len(ids_train),2), dtype=np.int32)
        oberved_weight = np.zeros((len(ids_train),), dtype=np.float32)
        for i, e in enumerate(ids_train):
            if A[e[0], e[1]] > 0:
                oberved_edges[i,0] = int(e[0])
                oberved_edges[i,1] = int(e[1])
                oberved_weight[i] = A[e[0], e[1]]
            if A[e[1], e[0]] > 0:
                oberved_edges[i,0] = int(e[1])
                oberved_edges[i,1] = int(e[0])
                oberved_weight[i] = A[e[1], e[0]]
        
        if task == 'direction':
            ids_train = ids_train[labels_train < 2]
            labels_train = labels_train[labels_train <2]
            ids_test = ids_test[labels_test < 2]
            labels_test = labels_test[labels_test <2]
            ids_val = ids_val[labels_val < 2]
            labels_val = labels_val[labels_val <2]

        datasets[ind] = {}
        datasets[ind]['graph'] = torch.from_numpy(oberved_edges.T).long().to(device)
        datasets[ind]['weights'] = torch.from_numpy(oberved_weight).float().to(device)

        datasets[ind]['train'] = {}
        datasets[ind]['train']['edges'] = torch.from_numpy(ids_train).long().to(device)
        datasets[ind]['train']['label'] = torch.from_numpy(labels_train).long().to(device)

        datasets[ind]['val'] = {}
        datasets[ind]['val']['edges'] = torch.from_numpy(ids_val).long().to(device)
        datasets[ind]['val']['label'] = torch.from_numpy(labels_val).long().to(device)

        datasets[ind]['test'] = {}
        datasets[ind]['test']['edges'] = torch.from_numpy(ids_test).long().to(device)
        datasets[ind]['test']['label'] = torch.from_numpy(labels_test).long().to(device)
    return datasets