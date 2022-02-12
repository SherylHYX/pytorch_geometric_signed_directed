Installation
============

The installation of PyTorch Geometric Signed Directed requires the presence of certain prerequisites. These are described in great detail in the installation description of PyTorch Geometric. Please follow the instructions laid out `here <https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/notes/installation.html>`_. You might also take a look at the `README file <https://github.com/SherylHYX/pytorch_geometric_signed_directed>`_ of the PyTorch Geometric Signed Directed repository.
Binaries are provided for Python version >= 3.6 and PyG version at least 2.

**PyTorch 1.10.0**

To install the binaries for PyTorch 1.10.0, simply run

    .. code-block:: none

        $ pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
        $ pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
        $ pip install torch-geometric
        $ pip install torch-geometric-signed-directed


where `${CUDA}` should be replaced by either `cpu`, `cu102`, or `cu113` depending on your PyTorch installation.

**Updating the Library**

The package itself can be installed via pip:

    .. code-block:: none

        $ pip install torch-geometric-signed-directed

Upgrade your outdated PyTorch Geometric Signed Directed version by using:

    .. code-block:: none

        $ pip install torch-geometric-signed-directed --upgrade


To check your current package version just simply run:

    .. code-block:: none

        $ pip freeze | grep torch-geometric-signed-directed

