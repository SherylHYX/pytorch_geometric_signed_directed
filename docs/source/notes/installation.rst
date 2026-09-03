Installation
============

PyGSD 1.2.0 requires Python >= 3.10, PyTorch >= 2.4, PyG >= 2.6, and
NetworkX >= 2.7. The current CPU reference environment uses Python 3.11 or
3.12, PyTorch 2.11.0, and PyG 2.8.0.

The PyTorch and PyG installation options depend on the operating system and
accelerator. Follow the `PyTorch installation guide
<https://pytorch.org/get-started/locally/>`_ and the `PyG installation guide
<https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_
for the appropriate commands.

After installing PyTorch and PyG, simply run

    .. code-block:: none

        $ pip install torch-geometric-signed-directed

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
