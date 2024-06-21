Introduction
=======================

PyTorch Geometric Signed Directed is a signed/directed graph neural network extension library for `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric/>`_.  
It builds on open-source deep-learning and graph processing libraries. 
*PyTorch Geometric Signed Directed* consists of various signed and directed geometric deep learning, embedding, and clustering methods from a variety of published research papers and selected preprints.

Citing
=======================
If you find *PyTorch Geometric Signed Directed* useful in your research, please consider adding the following citation:

.. code-block:: latex

    @inproceedings{he2024pytorch,
        title={Pytorch Geometric Signed Directed: A software package on graph neural networks for signed and directed graphs},
        author={He, Yixuan and Zhang, Xitong and Huang, Junjie and Rozemberczki, Benedek and Cucuringu, Mihai and Reinert, Gesine},
        booktitle={Learning on Graphs Conference},
        pages={12--1},
        year={2024},
        organization={PMLR}
        }

We briefly overview the fundamental concepts and features of PyTorch Geometric Signed Directed through simple examples.

Data Structures
=============================
PyTorch Geometric Signed Directed is designed to provide easy to use data loaders and data generators. 


Data Classes
--------------------------

PyTorch Geometric Temporal offers data classes for signed and directed datasets.

- ``SignedData`` - Is designed for **signed networks** (possibly directed and weighted) defined on a static graph.
- ``DirectedData`` - Is designed for **directed networks** (possibly weighted) defined on a static graph.

Directed Unsigned Data Class
^^^^^^^^^^^^^^^^^^^^^^^

A directed data object is a PyTorch Geometric ``Data`` object. Please take a look at this `readme <https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs>`_ for the details. The returned data object has the following major attributes:

- ``edge_index`` - A PyTorch ``LongTensor`` of edge indices stored in COO format (optional).
- ``edge_weight`` - A PyTorch ``FloatTensor`` of edge weights stored in COO format (optional).
- ``edge_attr`` - A PyTorch ``FloatTensor`` of edge features stored in COO format (optional).
- ``x`` - A PyTorch ``FloatTensor`` of vertex features (optional).
- ``y`` - A PyTorch ``LongTensor`` of node labels (optional).
- ``A`` - An Scipy.sparse ``spmatrix`` of the adjacency matrix (optional).

Signed Data Class (compatible with signed undirected and signed directed graphs)
^^^^^^^^^^^^^^^^^^^^^^^

A signed data object is a PyTorch Geometric ``Data`` object. Please take a look at this `readme <https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs>`_ for the details. The returned data object has the following major attributes:

- ``edge_index`` - A PyTorch ``LongTensor`` of edge indices stored in COO format (optional).
- ``edge_weight`` - A PyTorch ``FloatTensor`` of edge weights stored in COO format (optional).
- ``edge_attr`` - A PyTorch ``FloatTensor`` of edge features stored in COO format (optional).
- ``x`` - A PyTorch ``FloatTensor`` of vertex features (optional).
- ``y`` - A PyTorch ``LongTensor`` of node labels (optional).
- ``A`` - An Scipy.sparse ``spmatrix`` of the adjacency matrix (optional).
- ``edge_index_p`` - A PyTorch ``LongTensor`` of edge indices for the positive part of the adjacency matrix stored in COO format (optional).
- ``edge_weight_p`` - A PyTorch ``FloatTensor`` of edge weights for the positive part of the adjacency matrix stored in COO format (optional).
- ``A_p`` - An Scipy.sparse ``spmatrix`` of the positive part of the adjacency matrix (optional).
- ``edge_index_n`` - A PyTorch ``LongTensor`` of edge indices for the negative part of the adjacency matrix stored in COO format (optional).
- ``edge_weight_n`` - A PyTorch ``FloatTensor`` of edge weights for the negative part of the adjacency matrix stored in COO format (optional).
- ``A_n`` - An Scipy.sparse ``spmatrix`` of the negative part of the adjacency matrix (optional).


Benchmark Datasets
-------------------

We released and included a number of datasets which can be used for comparing the performance of signed/directed graph neural networks algorithms. The related machine learning tasks are node and edge level learning.
We also provide synthetic data generators for both signed and directed networks.

Synthetic Data Generators
^^^^^^^^^^^^^^^^^^^^^^

- `Signed Stochastic Block Models (SSBMs). <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#module-torch_geometric_signed_directed.data.signed.SSBM>`_
- `Polarized SSBMs. <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#module-torch_geometric_signed_directed.data.signed.polarized_SSBM>`_
- `Directed Stochastic Block Models (DSBMs). <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#module-torch_geometric_signed_directed.data.directed.DSBM>`_
- `Signed Directed Stochastic Block Models (SDSBMs). <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#module-torch_geometric_signed_directed.data.general.SDSBM>`_

Real-World Data Loaders
^^^^^^^^^^^^^^^^^^^^^^

- `signed real-world data loader. <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#module-torch_geometric_signed_directed.data.signed.load_signed_real_data>`_
- `directed real-world data loader. <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#module-torch_geometric_signed_directed.data.directed.load_directed_real_data>`_


For example, the Telegram Dataset can be loaded by the following code snippet. The ``dataset`` returned is a ``DirectedData`` object. 

.. code-block:: python

    from torch_geometric_signed_directed.data import load_directed_real_data

    dataset = load_directed_real_data(dataset='telegram', root='./tmp_data/')


Node Splitting
-------------------------------
We provide a function to create node splits of the data objects. 
The size parameters can either be int or float.
If a size parameter is int, then this means the actual number, if it is float, then this means a ratio.
``train_size`` or ``train_size_per_class`` is mandatory, with the former regardless of class labels.
Validation and seed masks are optional. Seed masks here masks nodes within the training set, e.g., in a semi-supervised setting as described in the
`SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper. 
If test_size and test_size_per_class are both None, all the remaining nodes after selecting training (and validation) nodes will be included.
This function returns the new data object with train, validation, test and possibly also seed (some parts within the training set) masks.
The splitting can either be done via data loading or separately. 

.. code-block:: python

    from torch_geometric_signed_directed.data import load_directed_real_data

    dataset = load_directed_real_data(dataset='telegram', root='./tmp_data/', train_size_per_class=0.8, val_size_per_class=0.1, test_size_per_class=0.1)

    dataset.node_split(train_size_per_class=0.8, val_size_per_class=0.1, test_size_per_class=0.1, seed_size_per_class=0.1)

Edge Splitting
-------------------------------

We provide a function to create edge splits. The splitting can either be done via data loading or separately. 

Directed Unsigned Edge Splitting
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from torch_geometric_signed_directed.data import load_directed_real_data
    from torch_geometric_signed_directed.utils import link_split

    directed_dataset = load_directed_real_data(dataset='telegram', root='./tmp_data/')
    datasets = link_class_split(directed_dataset, prob_val = 0.15, prob_test = 0.05, task = 'direction')

.. code-block:: python

    from torch_geometric_signed_directed.data import load_directed_real_data

    directed_dataset = load_directed_real_data(dataset='telegram', root='./tmp_data/')
    datasets = directed_dataset.link_split(prob_val = 0.15, prob_test = 0.05, task = 'direction')

Signed (Directed) Edge Splitting (for link sign prediction)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from torch_geometric_signed_directed.data import load_signed_real_data
    from torch_geometric_signed_directed.utils import link_split

    signed_dataset = load_directed_real_data(dataset='bitcoin_alpha', root='./tmp_data/')
    datasets = link_class_split(signed_dataset, prob_val = 0.15, prob_test = 0.05, task = 'sign')

.. code-block:: python

    from torch_geometric_signed_directed.data import load_signed_real_data

    signed_dataset = load_directed_real_data(dataset='bitcoin_alpha', root='./tmp_data/')
    datasets = signed_dataset.link_split(prob_val = 0.15, prob_test = 0.05, task = 'sign')

Signed Directed Edge Splitting (for four/five-class link classification problem)
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from torch_geometric_signed_directed.data import load_signed_real_data
    from torch_geometric_signed_directed.utils import link_split

    signed_dataset = load_directed_real_data(dataset='bitcoin_alpha', root='./tmp_data/')
    datasets = link_class_split(signed_dataset, prob_val = 0.15, prob_test = 0.05, task = 'four_class_signed_digraph')

.. code-block:: python

    from torch_geometric_signed_directed.data import load_signed_real_data

    signed_dataset = load_directed_real_data(dataset='bitcoin_alpha', root='./tmp_data/')
    datasets = signed_dataset.link_split(prob_val = 0.15, prob_test = 0.05, task = 'five_class_signed_digraph')