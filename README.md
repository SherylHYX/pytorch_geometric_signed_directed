[![CI](https://github.com/SherylHYX/pytorch_geometric_signed_directed/actions/workflows/main.yml/badge.svg)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/SherylHYX/pytorch_geometric_signed_directed/branch/main/graph/badge.svg?token=441OFDGWRB)](https://codecov.io/gh/SherylHYX/pytorch_geometric_signed_directed)
[![Documentation Status](https://readthedocs.org/projects/pytorch-geometric-signed-directed/badge/?version=latest)](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://badge.fury.io/py/torch-geometric-signed-directed.svg)](https://pypi.org/project/torch-geometric-signed-directed/)
[![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/master/CONTRIBUTING.md)




<p align="center">
  <img width="90%" src="https://raw.githubusercontent.com/SherylHYX/pytorch_geometric_signed_directed/master/docs/source/_static/img/text_logo.jpg?sanitize=true" />
</p>

**[Documentation](https://pytorch-geometric-signed-directed.readthedocs.io)** | **[Case Study](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/notes/case_study.html)** | **[Data Set Descriptions](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/notes/datasets.html)** | **[Installation](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/notes/installation.html)** | **[Data Structures](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/notes/introduction.html#data-structures)** | **[External Resources](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/notes/resources.html)** | **[Paper](https://arxiv.org/pdf/2202.10793.pdf)**

-----------------------------------------------------

*PyTorch Geometric Signed Directed* is a signed and directed extension library for [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric). It follows the package structure in [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal).

<p align="justify">The library consists of various signed and directed geometric deep learning, embedding, and clustering methods from a variety of published research papers and selected preprints. 

We also provide detailed examples in the [examples](https://github.com/SherylHYX/pytorch_geometric_signed_directed/tree/main/examples) folder.


--------------------------------------------------------------------------------

**Citing**


If you find *PyTorch Geometric Signed Directed* useful in your research, please consider adding the following citation:

```bibtex
@article{he2022pytorch,
        title={{PyTorch Geometric Signed Directed: A Software Package on Graph Neural Networks for Signed and Directed Graphs}},
        author={He, Yixuan and Zhang, Xitong and Huang, Junjie and Rozemberczki, Benedek and Cucuringu, Mihai and Reinert, Gesine},
        journal={arXiv preprint arXiv:2202.10793},
        year={2022}
        }
```

--------------------------------------------------------------------------------

**Methods Included**

In detail, the following signed or directed graph neural networks, as well as related methods designed for signed or directed netwroks, were implemented.

**Directed Unsigned Network Models and Layers**

* **[MagNet_node_classification](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.MagNet_node_classification.MagNet_node_classification)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[DiGCL](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DiGCL.DiGCL)** from Tong *et al.*: [Directed Graph Contrastive Learning.](https://proceedings.neurips.cc/paper/2021/file/a3048e47310d6efaa4b1eaf55227bc92-Paper.pdf) (NeurIPS 2021)

* **[DiGCN_Inception_Block_node_classification](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DiGCN_node_classification.DiGCN_node_classification)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DIGRAC_node_clustering](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DIGRAC_node_clustering.DIGRAC_node_clustering)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://proceedings.mlr.press/v198/he22b.html) (LoG 2022)


<details>
<summary><b>Expand to see all methods implemented for directed networks...</b></summary>

* **[DGCN_node_classification](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DGCN_node_classification.DGCN_node_classification)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)


* **[DiGCN_node_classification](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DiGCN_node_classification.DiGCN_node_classification)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[MagNet_link_prediction](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.MagNet_link_prediction.MagNet_link_prediction)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[DiGCN_link_prediction](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DiGCN_link_prediction.DiGCN_link_prediction)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DiGCN_Inception_Block_link_prediction](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DiGCN_Inception_Block_link_prediction.DiGCN_Inception_Block_link_prediction)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DGCN_link_prediction](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DGCN_link_prediction.DGCN_link_prediction)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)


* **[DiGCN_Inception_Block](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DiGCN_Inception_Block.DiGCN_InceptionBlock)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DGCNConv](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DGCNConv.DGCNConv)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)

* **[MagNetConv](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.MagNetConv.MagNetConv)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[DiGCNConv](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DiGCNConv.DiGCNConv)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[DIMPA](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.directed.DIMPA.DIMPA)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://proceedings.mlr.press/v198/he22b.html) (LoG 2022)
  

</details>

**Signed (Directed) Network Models and Layers**

* **[SSSNET_node_clustering](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.signed.SSSNET_node_clustering.SSSNET_node_clustering)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[SDGNN](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.signed.SDGNN.SDGNN)** from Huang *et al.*: [SDGNN: Learning Node Representation for Signed Directed Networks](https://arxiv.org/pdf/2101.02390.pdf) (AAAI 2021)

* **[SiGAT](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.signed.SiGAT.SiGAT)** from Huang *et al.*: [Signed Graph Attention Networks](https://arxiv.org/pdf/1906.10958.pdf) (ICANN 2019)


* **[MSGNN_link_prediction](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.general.MSGNN.MSGNN_link_prediction)** from He *et al.*: [MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian.](https://proceedings.mlr.press/v198/he22c.html) (LoG 2022)


<details>
<summary><b>Expand to see all methods implemented for signed networks...</b></summary>

* **[MSGNN_node_classification](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.general.MSGNN.MSGNN_node_classification)** from He *et al.*: [MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian.](https://proceedings.mlr.press/v198/he22c.html) (LoG 2022)

* **[MSConv](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.general.MSConv.MSConv)** from He *et al.*: [MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian.](https://proceedings.mlr.press/v198/he22c.html) (LoG 2022)

* **[SSSNET_link_prediction](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.signed.SSSNET_link_prediction.SSSNET_link_prediction)** adapted from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[SNEA](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.signed.SNEA.SNEA)** from Li *et al.*: [Learning Signed Network Embedding via Graph Attention](https://ojs.aaai.org/index.php/AAAI/article/view/5911) (AAAI 2020)

* **[SGCN](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.signed.SGCN.SGCN)** from Derr *et al.*: [Signed Graph Convolutional Networks](https://arxiv.org/pdf/1808.06354.pdf) (ICDM 2018)

* **[SNEAConv](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.signed.SNEAConv.SNEAConv)** from Li *et al.*: [Learning Signed Network Embedding via Graph Attention](https://ojs.aaai.org/index.php/AAAI/article/view/5911) (AAAI 2020)

* **[SGCNConv](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.signed.SGCNConv.SGCNConv)** from Derr *et al.*: [Signed Graph Convolutional Network](https://arxiv.org/pdf/1808.06354.pdf) (ICDM 2018)


* **[SIMPA](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.nn.signed.SIMPA.SIMPA)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)


</details>
  


**Network Generation Methods**

* **[Signed Stochastic Block Model(SSBM)](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.signed.SSBM.SSBM)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Polarized Signed Stochastic Block Model(POL-SSBM)](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.signed.polarized_SSBM.polarized_SSBM)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Directed Stochastic Block Model(DSBM)](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.directed.DSBM.DSBM)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://proceedings.mlr.press/v198/he22b.html) (LoG 2022)

* **[Signed Directed Stochastic Block Model(SDSBM)](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.general.SDSBM.SDSBM)** from He *et al.*: [MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian.](https://proceedings.mlr.press/v198/he22c.html) (LoG 2022)


**Data Loaders and Classes**

* **[load_signed_real_data](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.signed.load_signed_real_data.load_signed_real_data)** to load signed (directed) real-world data sets.

* **[load_directed_real_data](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.directed.load_directed_real_data.load_directed_real_data)** to load directed unsigned real-world data sets.

* **[SignedData](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.signed.SignedData.SignedData)** Signed Data Class.

* **[DirectedData](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.directed.DirectedData.DirectedData)** Directed Data Class.


<details>
<summary><b>Expand to see all data loaders and related methods...</b></summary>

* **[SSSNET_signed_real_data](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.signed.SSSNET_real_data.SSSNET_real_data)** to load signed real-world data sets from the SSSNET paper.

* **[SDGNN_signed_real_data](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.signed.SDGNN_real_data.SDGNN_real_data)** to load signed real-world data sets from the SDGNN paper.

* **[MSGNN_signed_directed_real_data](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.signed.MSGNN_real_data.MSGNN_real_data)** to load signed directed real-world data sets from the MSGNN paper.
  
* **[DIGRAC_directed_real_data](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.directed.DIGRAC_real_data.DIGRAC_real_data)** to load directed real-world data sets from the DIGRAC paper.

* **[Telegram](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.directed.Telegram.Telegram)** to load the Telegram data set.

* **[Cora_ml](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.directed.citation.Cora_ml)** to load the Cora_ML data set.

* **[Citeseer](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.directed.citation.Citeseer)** to load the CiteSeer data set.

* **[WikiCS](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.directed.WikiCS.WikiCS)** to load the WikiCS data set.

* **[WikipediaNetwork](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/data.html#torch_geometric_signed_directed.data.directed.WikipediaNetwork.WikipediaNetwork)** to load the WikipediaNetwork data set.
  
</details>

**Task-Specific Objectives and Evaluation Methods**

* **[Probabilistic Balanced Normalized Loss](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.signed.prob_balanced_normalized_loss.Prob_Balanced_Normalized_Loss)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)


* **[Probabilistic Imbalance Objective](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.directed.prob_imbalance_loss.Prob_Imbalance_Loss)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://proceedings.mlr.press/v198/he22b.html) (LoG 2022)


<details>
<summary><b>Expand to see all task-specific objectives and evaluation methods...</b></summary>

* **[Probabilistic Balanced Ratio Loss](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.signed.prob_balanced_ratio_loss.Prob_Balanced_Ratio_Loss)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)

* **[Unhappy Ratio](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.signed.unhappy_ratio.Unhappy_Ratio)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (SDM 2022)
  
* **[link_sign_prediction_logistic_function](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.signed.link_sign_prediction_logistic_function.link_sign_prediction_logistic_function)** for signed networks' link sign prediction task.

* **[link_sign_direction_prediction_logistic_function](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.general.link_sign_direction_prediction_logistic_function.link_sign_prediction_logistic_function)** for signed directed networks' link prediction task.

* **[triplet_loss_node_classification](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.general.triplet_loss.triplet_loss_node_classification)** for triplet loss in the node classification task.

* **[Sign_Triangle_Loss](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.utils.signed.link_sign_loss.Sign_Triangle_Loss)** from Huang *et al.*: [SDGNN: Learning Node Representation for Signed Directed Networks](https://arxiv.org/pdf/2101.02390.pdf) (AAAI 2021)

* **[Sign_Direction_Loss](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.utils.signed.link_sign_loss.Sign_Direction_Loss)** from Huang *et al.*: [SDGNN: Learning Node Representation for Signed Directed Networks](https://arxiv.org/pdf/2101.02390.pdf) (AAAI 2021)

* **[Sign_Product_Entropy_Loss](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.utils.signed.link_sign_loss.Sign_Product_Entropy_Loss)** from Huang *et al.*: [SDGNN: Learning Node Representation for Signed Directed Networks](https://arxiv.org/pdf/2101.02390.pdf) (AAAI 2021)

* **[Link_Sign_Product_Loss](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.utils.signed.link_sign_loss.Link_Sign_Product_Loss)** from Huang *et al.*: [Signed Graph Attention Networks](https://arxiv.org/pdf/1906.10958.pdf) (ICANN 2019)

* **[Link_Sign_Entropy_Loss](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.utils.signed.link_sign_loss.Link_Sign_Entropy_Loss)** from Derr *et al.*: [Signed Graph Convolutional Network](https://arxiv.org/pdf/1808.06354.pdf) (ICDM 2018)

* **[Sign_Structure_Loss](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/model.html#torch_geometric_signed_directed.utils.signed.link_sign_loss.Sign_Structure_Loss)** 
</details>

**Utilities and Preprocessing Methods**

* **[node_class_split](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.general.node_split.node_class_split)** to split nodes into training set etc..

* **[link_class_split](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.general.link_split.link_class_split)** to split edges into training set etc..

* **[get_magnetic_Laplacian](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.directed.get_magnetic_Laplacian.get_magnetic_Laplacian)** from from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[get_magnetic_signed_Laplacian](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.general.get_magnetic_signed_Laplacian.get_magnetic_signed_Laplacian)** from He *et al.*: [MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian.](https://proceedings.mlr.press/v198/he22c.html) (LoG 2022)

<details>
<summary><b>Expand to see all utilities and preprocessing methods...</b></summary>

* **[get_appr_directed_adj](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.directed.get_adjs_DiGCN.get_appr_directed_adj)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)
  
* **[meta_graph_generation](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.directed.meta_graph_generation.meta_graph_generation)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://proceedings.mlr.press/v198/he22b.html) (ArXiv 2021)

* **[extract_network](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.general.extract_network.extract_network)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://proceedings.mlr.press/v198/he22b.html) (LoG 2022)

* **[directed_features_in_out](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.directed.features_in_out.directed_features_in_out)** from Tong *et al.*: [Directed Graph Convolutional Network.](https://arxiv.org/pdf/2004.13970.pdf) (ArXiv 2020)

* **[get_second_directed_adj](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.directed.get_adjs_DiGCN.get_second_directed_adj)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)

* **[cal_fast_appr](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.directed.get_adjs_DiGCN.cal_fast_appr)** from Tong *et al.*: [Digraph Inception Convolutional Networks.](https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf) (NeurIPS 2020)


* **[scipy_sparse_to_torch_sparse](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.general.scipy_sparse_to_torch_sparse.scipy_sparse_to_torch_sparse)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://proceedings.mlr.press/v198/he22b.html) (LoG 2022)


* **[create spectral features](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/modules/utils.html#torch_geometric_signed_directed.utils.signed.create_spectral_features.create_spectral_features)**
  
</details>

--------------------------------------------------------------------------------

Head over to our [documentation](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/) to find out more!
If you notice anything unexpected, please open an [issue](https://github.com/SherylHYX/pytorch_geometric_signed_directed/issues). If you are missing a specific method, feel free to open a [feature request](https://github.com/SherylHYX/pytorch_geometric_signed_directed/issues).


--------------------------------------------------------------------------------

**Installation**

Binaries are provided for Python version >= 3.7 and NetworkX version < 2.7.

After installing [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), simply run

```sh
pip install torch-geometric-signed-directed
```
--------------------------------------------------------------------------------

**Running tests**

```
$ python setup.py test
```
--------------------------------------------------------------------------------

**License**

- [MIT License](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/master/LICENSE)

