[![CI](https://github.com/SherylHYX/pytorch_geometric_signed_directed/actions/workflows/main.yml/badge.svg)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/SherylHYX/pytorch_geometric_signed_directed/branch/main/graph/badge.svg?token=441OFDGWRB)](https://codecov.io/gh/SherylHYX/pytorch_geometric_signed_directed)
[![Documentation Status](https://readthedocs.org/projects/pytorch-geometric-signed-directed/badge/?version=latest)](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://badge.fury.io/py/torch-geometric-signed-directed.svg)](https://pypi.python.org/pypi/torch-geometric-signed-directed)


<p align="center">
  <img width="90%" src="https://raw.githubusercontent.com/SherylHYX/pytorch_geometric_signed_directed/master/docs/source/_static/img/text_logo.jpg?sanitize=true" />
</p>

-----------------------------------------------------

*PyTorch Geometric Signed Directed* is a signed and directed extension library for [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric). It follows the package structure in [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal).

<p align="justify">The library consists of various signed and directed geometric deep learning, embedding, and clustering methods from a variety of published research papers and selected preprints. 

We also provide detailed examples in the [examples](https://github.com/SherylHYX/pytorch_geometric_signed_directed/tree/main/examples) folder.


--------------------------------------------------------------------------------

**Methods Included**

In detail, the following signed or directed graph neural networks, as well as related methods designed for signed or directed netwroks, were implemented.


**Signed Network Models and Layers**

* **[SSSNET_node_clustering](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SSSNET_node_clustering.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[SDGNN](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SDGNN.py)** from Huang *et al.*: [SDGNN: Learning Node Representation for Signed Directed Networks](https://arxiv.org/pdf/2101.02390.pdf) (AAAI 2021)

* **[SiGAT](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SiGAT.py)** from Huang *et al.*: [Signed Graph Attention Networks](https://arxiv.org/pdf/1906.10958.pdf) (ICANN 2019)

<details>
<summary><b>Expand to see all methods implemented for signed networks...</b></summary>

* **[SNEA](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SGCN_SNEA.py)** from Li *et al.*: [Learning Signed Network Embedding via Graph Attention](https://ojs.aaai.org/index.php/AAAI/article/view/5911) (AAAI 2021)

* **[SGCN](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SGCN_SNEA.py)** from Derr *et al.*: [Signed Graph Attention Networks](https://arxiv.org/pdf/1808.06354.pdf) (ICDM 2018)

* **[SNEAConv](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SNEAConv.py)** from Li *et al.*: [Learning Signed Network Embedding via Graph Attention](https://ojs.aaai.org/index.php/AAAI/article/view/5911) (AAAI 2021)

* **[SGCNConv](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SGCNConv.py)** from Derr *et al.*: [Signed Graph Attention Networks](https://arxiv.org/pdf/1808.06354.pdf) (ICDM 2018)


* **[SIMPA](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SIMPA.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)


</details>
  
**Directed Network Models and Layers**

* **[MagNet_node_classification](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/MagNet_node_classification.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[DiGCL](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCL.py)** from Tong *et al.*: [Directed Graph Contrastive Learning.](https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf) (NeurIPS 2021)

* **[DiGCN_Inception_Block_node_classification](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCN_Inception_Block_node_classification.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DIGRAC_node_clustering](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DIGRAC_node_clustering.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)

* **[DGCN_node_classification](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DGCN_node_classification.py)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)


<details>
<summary><b>Expand to see all methods implemented for directed networks...</b></summary>


* **[DiGCN_node_classification](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCN_node_classification.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[MagNet_link_prediction](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/MagNet_link_prediction.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[DiGCN_link_prediction](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCN_link_prediction.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DiGCN_Inception_Block_link_prediction](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCN_Inception_Block_link_prediction.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DGCN_link_prediction](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DGCN_link_prediction.py)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)


* **[DiGCN_Inception_Block](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCN_Inception_Block.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DGCNConv](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DGCNConv.py)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)

* **[MagNetConv](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/MagNetConv.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[DiGCNConv](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DiGCNConv.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DIMPA](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DIMPA.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)
  

</details>
  
**Auxiliary Methods and Layers**

* **[complex_relu_layer](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/complex_relu.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)


**Network Generation Methods**

* **[Signed Stochastic Block Model(SSBM)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/signed/SSBM.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Polarized Signed Stochastic Block Model(POL-SSBM)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/signed/polarized_SSBM.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Directed Stochastic Block Model(DSBM)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/DSBM.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)


**Data Loaders and Classes**


* **[load_directed_real_data](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/load_directed_real_data.py)** to load directed real-world data sets.

* **[SignedData](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/signed/SignedData.py)** Signed Data Class.

* **[DirectedData](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/DirectedData.py)** Directed Data Class.


<details>
<summary><b>Expand to see all data loaders and related methods...</b></summary>
  
* **[DIGRAC_directed_real_data](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/DIGRAC_directed_real_data.py)** to load directed real-world data sets from the DIGRAC paper.

* **[Telegram](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/Telegram.py)** to load the Telegram data set.

* **[Cora_ml](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/citation.py)** to load the Cora_ML data set.

* **[Citeseer](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/citation.py)** to load the CiteSeer data set.

* **[WikiCS](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/WikiCS.py)** to load the WikiCS data set.

* **[WikipediaNetwork](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed/citation.py)** to load the WikipediaNetwork data set.
  
</details>

**Task-Specific Objectives and Evaluation Methods**

* **[Probablistic Balanced Normalized Loss](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/signed/prob_balanced_normalized_loss.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Probablistic Balanced Ratio Loss](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/signed/prob_balanced_ratio_loss.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Unhappy Ratio](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/signed/unhappy_ratio.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Probablistic Imbalance Objective](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/prob_imbalance_loss.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)


<details>
<summary><b>Expand to see all task-specific objectives and evaluation methods...</b></summary>
  
* **[link_sign_prediction_logistic_function](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/signed/link_sign_prediction_logistic_function.py)** for signed networks' link sign prediction task.

</details>

**Utilities and Preprocessing Methods**

* **[node_split](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/general/node_split.py)** to split nodes into training set etc..

* **[directed_link_split](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/directed_link_split.py)** to split directed edges into training set etc..


* **[get_magnetic_Laplacian](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/get_magnetic_Laplacian.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[get_appr_directed_adj](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/get_adjs_DiGCN.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)


* **[scipy_sparse_to_torch_sparse](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/general/scipy_sparse_to_torch_sparse.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)

<details>
<summary><b>Expand to see all utilities and preprocessing methods...</b></summary>
  
* **[meta_graph_generation](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/meta_graph_generation.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)

* **[extract_network](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/general/extract_network.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)

* **[directed_features_in_out](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/features_in_out.py)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)

* **[get_second_directed_adj](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/get_adjs_DiGCN.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[cal_fast_appr](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed/get_adjs_DiGCN.py)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

  
</details>

--------------------------------------------------------------------------------

Head over to our [documentation](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/) to find out more!
If you notice anything unexpected, please open an [issue](https://github.com/SherylHYX/pytorch_geometric_signed_directed/issues). If you are missing a specific method, feel free to open a [feature request](https://github.com/SherylHYX/pytorch_geometric_signed_directed/issues).


--------------------------------------------------------------------------------

**Installation**

Binaries are provided for Python version >= 3.6.

**PyTorch 1.10.0**

To install the binaries for PyTorch 1.10.0, simply run

```sh
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+${CUDA}.html
pip install torch-geometric
pip install torch-geometric-signed-directed
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu113` depending on your PyTorch installation.

|             | `cpu` | `cu102` | `cu113` |
|-------------|-------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      |
| **Windows** | ✅    | ✅      | ✅      |
| **macOS**   | ✅    |         |         |

--------------------------------------------------------------------------------

**Running tests**

```
$ python setup.py test
```
--------------------------------------------------------------------------------

**License**

- [MIT License](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/master/LICENSE)

