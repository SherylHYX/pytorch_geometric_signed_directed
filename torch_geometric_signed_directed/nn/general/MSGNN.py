from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..directed.complex_relu import complex_relu_layer
from .MSConv import MSConv

class MSGNN_link_prediction(nn.Module):
    r"""The MSGNN model for link prediction from the 
    `MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian <https://arxiv.org/pdf/2209.00546.pdf>`_ paper.
    
    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`True`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MSConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`0.5`)
        normalization (str, optional): The normalization scheme for the signed directed
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        conv_bias (bool, optional): Whether to use bias in the convolutional layers, default :obj:`True`.
        absolute_degree (bool, optional): Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
    """
    def __init__(self, num_features:int, hidden:int=2, q:float=0.25, K:int=2, label_dim:int=2, \
        activation:bool=True, trainable_q:bool=False, layer:int=2, dropout:float=0.5, normalization:str='sym', 
        cached: bool=False, conv_bias: bool=True, absolute_degree: bool=True):
        super(MSGNN_link_prediction, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MSConv(in_channels=num_features, out_channels=hidden, K=K, \
            q=q, trainable_q=trainable_q, normalization=normalization, bias=conv_bias))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MSConv(in_channels=hidden, out_channels=hidden, K=K,\
                q=q, trainable_q=trainable_q, normalization=normalization, \
                    bias=conv_bias, cached=cached, absolute_degree=absolute_degree))

        self.Chebs = chebs
        self.linear = nn.Linear(hidden*4, label_dim)      
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor, \
        query_edges: torch.LongTensor, edge_weight: Optional[torch.LongTensor]=None) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.
        
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * query_edges (PyTorch Long Tensor) - Edge indices for querying labels.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x = torch.cat((real[query_edges[:,0]], real[query_edges[:,1]], imag[query_edges[:,0]], imag[query_edges[:,1]]), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        self.z = x.clone()
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


class MSGNN_node_classification(nn.Module):
    r"""The MSGNN model for node classification from the 
    `MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian <https://arxiv.org/pdf/2209.00546.pdf>`_ paper.
    
    Args:
        num_features (int): Size of each input sample.
        hidden (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`False`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MSConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`False`)
        normalization (str, optional): The normalization scheme for the signed directed
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \bar{\mathbf{D}} - \mathbf{A} \odot \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \bar{\mathbf{D}}^{-1/2} \mathbf{A}
            \bar{\mathbf{D}}^{-1/2} \odot \exp(i \Theta^{(q)})`
            `\odot` denotes the element-wise multiplication.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        conv_bias (bool, optional): Whether to use bias in the convolutional layers, default :obj:`True`.
        absolute_degree (bool, optional): Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
    """
    def __init__(self, num_features:int, hidden:int=2, q:float=0.25, K:int=2, label_dim:int=2, \
        activation:bool=False, trainable_q:bool=False, layer:int=2, dropout:float=False, normalization:str='sym', 
        cached: bool=False, conv_bias: bool=True, absolute_degree: bool=True):
        super(MSGNN_node_classification, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MSConv(in_channels=num_features, out_channels=hidden, K=K, \
            q=q, trainable_q=trainable_q, bias=conv_bias, normalization=normalization))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MSConv(in_channels=hidden, out_channels=hidden, K=K,\
                q=q, trainable_q=trainable_q, normalization=normalization, \
                    cached=cached, bias=conv_bias, absolute_degree=absolute_degree))

        self.Chebs = chebs

        self.Conv = nn.Conv1d(2*hidden, label_dim, kernel_size=1)        
        self.dropout = dropout

    def reset_parameters(self):
        for cheb in self.Chebs:
            cheb.reset_parameters()
        self.Conv.reset_parameters()
        
    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor, \
        edge_weight: Optional[torch.LongTensor]=None) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model.
        
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        
        Return types:
            * **z** (PyTorch FloatTensor) - Embedding matrix, with shape (num_nodes, 2*hidden) for undirected graphs and (num_nodes, 4*hidden) for directed graphs.
            * **output** (PyTorch FloatTensor) - Log of prob, with shape (num_nodes, num_clusters).
            * **predictions_cluster** (PyTorch LongTensor) - Predicted labels.
            * **prob** (PyTorch FloatTensor) - Probability assignment matrix of different clusters, with shape (num_nodes, num_clusters).
        """
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x = torch.cat((real, imag), dim = -1)
        
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        z = torch.transpose(x[0], 0, 1).clone()
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)

        output = torch.transpose(x[0], 0, 1) # log_prob
        predictions_cluster = torch.argmax(output, dim=1)

        prob = F.softmax(output, dim=1)

        return F.normalize(z), output, predictions_cluster, prob