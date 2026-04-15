PyTorch Geometric Signed Directed Models
==================

.. contents:: Contents
    :local:

Directed (Unsigned) Network Models and Layers
--------------

This section documents directed graph models and core directed convolution
layers available in PyTorch Geometric Signed Directed. Each entry includes the
class/function signature and parameter documentation extracted from source
docstrings.

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.MagNet_node_classification.MagNet_node_classification
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DiGCN_node_classification.DiGCN_node_classification
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DiGCN_Inception_Block_node_classification.DiGCN_Inception_Block_node_classification
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DIGRAC_node_clustering.DIGRAC_node_clustering
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DGCN_node_classification.DGCN_node_classification
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DiGCL.DiGCL
    :members:
    :exclude-members: DiGCL_Encoder

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.MagNet_link_prediction.MagNet_link_prediction
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DiGCN_link_prediction.DiGCN_link_prediction
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DiGCN_Inception_Block_link_prediction.DiGCN_Inception_Block_link_prediction
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DGCN_link_prediction.DGCN_link_prediction
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.MagNetConv.MagNetConv
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DiGCNConv.DiGCNConv
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DiGCN_Inception_Block.DiGCN_InceptionBlock
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DIMPA.DIMPA
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.DGCNConv.DGCNConv
    :members:
    :exclude-members:

Signed (Directed) Network Models and Layers
--------------

This section covers methods tailored to signed graphs (including signed
directed settings), with links to model and layer level APIs.

.. autoapiclass:: torch_geometric_signed_directed.nn.signed.SSSNET_node_clustering.SSSNET_node_clustering
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.signed.SSSNET_link_prediction.SSSNET_link_prediction
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.signed.SIMPA.SIMPA
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.signed.SDGNN.SDGNN
    :members:
    :exclude-members: SDRLayer

.. autoapiclass:: torch_geometric_signed_directed.nn.signed.SiGAT.SiGAT
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.signed.SGCN.SGCN
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.signed.SNEA.SNEA
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.signed.SNEAConv.SNEAConv
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.signed.SGCNConv.SGCNConv
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.general.MSGNN.MSGNN_link_prediction
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.general.MSGNN.MSGNN_node_classification
    :members:
    :exclude-members:

.. autoapiclass:: torch_geometric_signed_directed.nn.general.MSConv.MSConv
    :members:
    :exclude-members:



Auxiliary Methods and Layers
--------------

This section contains supporting layers and helper operations used by the
directed and signed model families.

.. autoapiclass:: torch_geometric_signed_directed.nn.directed.complex_relu.complex_relu_layer
    :members:
    :exclude-members: