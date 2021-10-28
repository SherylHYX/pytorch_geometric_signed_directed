[![CI](https://github.com/SherylHYX/pytorch_geometric_signed_directed/actions/workflows/main.yml/badge.svg)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/SherylHYX/pytorch_geometric_signed_directed/branch/main/graph/badge.svg?token=HZ4N607OBJ)](https://codecov.io/gh/SherylHYX/pytorch_geometric_signed_directed)
[![Documentation Status](https://readthedocs.org/projects/pytorch-geometric-signed-directed/badge/?version=latest)](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/?badge=latest)
[![BCH compliance](https://bettercodehub.com/edge/badge/SherylHYX/pytorch_geometric_signed_directed?branch=main&token=4780381faf1dfd39fa49d2d1a66857ae603f3f7a)](https://bettercodehub.com/)


*PyTorch Geometric Signed Directed* is a signed and directed extension library for [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).

<p align="justify">The library aims to consist of various signed and directed geometric deep learning, embedding, and clustering methods from a variety of published research papers and selected preprints. It is currently under development and we welcome your contribution!


--------------------------------------------------------------------------------

**Methods Included**

In detail, the following signed or directed graph neural networks, as well as related methods designed for signed or directed netwroks, were implemented.


**Signed Network Embedding and Clustering**

* **[SSSNET](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SSSNET.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (Complex Networks 2021)


<details>
<summary><b>Expand to see all methods implemented for signed networks...</b></summary>

more to come...

</details>
  
**Directed Network Embedding and Clustering**

* **[MagNet](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/MagNet.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[DIGRAC](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DIGRAC.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)


<details>
<summary><b>Expand to see all methods implemented for directed networks...</b></summary>

more to come...

</details>
  
**Auxiliary Network Embedding Methods**

* **[SIMPA](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/signed/SSSNET.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (Complex Networks 2021)

* **[MagNetConv](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/MagNet.py)** from Zhang *et al.*: [MagNet: A Neural Network for Directed Graphs.](https://arxiv.org/pdf/2102.11391.pdf) (NeurIPS 2021)

* **[DIMPA](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/nn/directed/DIGRAC.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)


<details>
<summary><b>Expand to see all auxiliary network embedding methods...</b></summary>
  
more to come...

  
</details>

**Network Generation Methods**

* **[Signed Stochastic Block Model(SSBM)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/signed_models.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (Complex Networks 2021)

* **[Polarized Signed Stochastic Block Model(POL-SSBM)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/signed_models.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (Complex Networks 2021)

* **[Directed Stochastic Block Model(DSBM)](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/data/directed_models.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)


<details>
<summary><b>Expand to see all auxiliary network generation methods...</b></summary>
  
more to come...

  
</details>

**Task-Specific Objectives and Evaluation Methods**

* **[Probablistic Balanced Normalized Cut](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/signed_metrics.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (Complex Networks 2021)

* **[Probablistic Balanced Ratio Cut](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/signed_metrics.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (Complex Networks 2021)

* **[Unhappy Ratio](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/signed_metrics.py)** from He *et al.*: [SSSNET: Semi-Supervised Signed Network Clustering](https://arxiv.org/pdf/2110.06623.pdf) (Complex Networks 2021)

* **[Probablistic Imbalance Objective](https://github.com/SherylHYX/pytorch_geometric_signed_directed/blob/main/torch_geometric_signed_directed/utils/directed_metrics.py)** from He *et al.*: [DIGRAC: Digraph Clustering Based on Flow Imbalance.](https://arxiv.org/pdf/2106.05194.pdf) (ArXiv 2021)


<details>
<summary><b>Expand to see all task-specific objectives and evaluation methods...</b></summary>
  
more to come...

  
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