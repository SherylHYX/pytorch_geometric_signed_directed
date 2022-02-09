[![CI](https://github.com/SherylHYX/pytorch_geometric_signed_directed/actions/workflows/main.yml/badge.svg)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/SherylHYX/pytorch_geometric_signed_directed/branch/main/graph/badge.svg?token=441OFDGWRB)](https://codecov.io/gh/SherylHYX/pytorch_geometric_signed_directed)
[![Documentation Status](https://readthedocs.org/projects/pytorch-geometric-signed-directed/badge/?version=latest)](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/?badge=latest)


<p align="center">
  <img width="90%" src="https://raw.githubusercontent.com/SherylHYX/pytorch_geometric_signed_directed/master/docs/source/_static/img/text_logo.jpg?sanitize=true" />
</p>

-----------------------------------------------------

*PyTorch Geometric Signed Directed* is a signed and directed extension library for [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).

<p align="justify">The library consists of various signed and directed geometric deep learning, embedding, and clustering methods from a variety of published research papers and selected preprints. It is currently under development and we welcome your contribution!


--------------------------------------------------------------------------------

**Methods Included**

In detail, the following signed or directed graph neural networks, as well as related methods designed for signed or directed netwroks, were implemented.


**Signed Network Models**

* **[SSSNET_node_clustering](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SSSNET_node_clustering.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)


<details>
<summary><b>Expand to see all methods implemented for signed networks...</b></summary>

more to come...

</details>
  
**Directed Network Models**

* **[MagNet_node_classification](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/MagNet_node_classification.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[DiGCL](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCL.py)** from Tong *et al.*: [Directed Graph Contrastive Learning.](https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf) (NeurIPS 2021)

* **[DiGCN_Inception_Block_node_classification](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCN_Inception_Block_node_classification.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DIGRAC_node_clustering](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DIGRAC_node_clustering.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)

* **[DGCN_node_classification](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DGCN_node_classification.py)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)


<details>
<summary><b>Expand to see all methods implemented for directed networks...</b></summary>


* **[DiGCN_node_classification](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCN_node_classification.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

</details>
  
**Auxiliary Network Embedding Methods**

* **[SIMPA](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SIMPA.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[MagNetConv](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/MagNetConv.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[DiGCNConv](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCNConv.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DIMPA](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DIMPA.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)


<details>
<summary><b>Expand to see all auxiliary network embedding methods...</b></summary>

* **[complex_relu_layer](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/complex_relu.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)
  
* **[DiGCN_Inception_Block](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCN_Inception_Block.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DGCNConv](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DGCNConv.py)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)
  
</details>

**Network Generation Methods**

* **[Signed Stochastic Block Model(SSBM)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/signed/SSBM.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Polarized Signed Stochastic Block Model(POL-SSBM)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/signed/polarized_SSBM.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Directed Stochastic Block Model(DSBM)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/DSBM.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)


<details>
<summary><b>Expand to see all auxiliary network generation methods...</b></summary>
  
more to come...

  
</details>

**Data Loaders and Classes**

* **[SignedData](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/signed/SignedData.py)** Signed Data Class.

* **[load_snap_signed_real_data](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/signed/snap_signed_real_data.py)** Data loader for SNAP signed real data.

* **[DirectedData](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/DirectedData.py)** Directed Data Class.

* **[load_directed_real_data](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/load_directed_real_data.py)** Directed real data loader.


<details>
<summary><b>Expand to see all data loaders and related methods...</b></summary>
  
more to come...

  
</details>

**Task-Specific Objectives and Evaluation Methods**

* **[Probablistic Balanced Normalized Loss](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/signed/prob_balanced_normalized_loss.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Probablistic Balanced Ratio Loss](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/signed/prob_balanced_ratio_loss.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Unhappy Ratio](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/signed/unhappy_ratio.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Probablistic Imbalance Objective](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/prob_imbalance_loss.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)


<details>
<summary><b>Expand to see all task-specific objectives and evaluation methods...</b></summary>
  
more to come...

  
</details>

**Utilities and Preprocessing Methods**

* **[get_magnetic_Laplacian](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/get_magnetic_Laplacian.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[get_appr_directed_adj](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/get_adjs_DiGCN.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[get_second_directed_adj](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/get_adjs_DiGCN.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[scipy_sparse_to_torch_sparse](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/general/scipy_sparse_to_torch_sparse.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)

<details>
<summary><b>Expand to see all utilities and preprocessing methods...</b></summary>
  
* **[meta_graph_generation](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/meta_graph_generation.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)

* **[extract_network](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/general/extract_network.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)

* **[directed_features_in_out](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/features_in_out.py)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)

  
</details>

--------------------------------------------------------------------------------

If you notice anything unexpected, please open an [issue](https://github.com/SherylHYX/pytorch_geometric_signed_directed/issues). If you are missing a specific method, feel free to open a [feature request](https://github.com/SherylHYX/pytorch_geometric_signed_directed/issues).


--------------------------------------------------------------------------------

**Running tests**

```
$ python setup.py test
```
--------------------------------------------------------------------------------

**License**

- [MIT License](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/master/LICENSE)