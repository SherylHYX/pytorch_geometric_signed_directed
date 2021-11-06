from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_relu import complex_relu_layer
from .MagNetConv import MagNetConv

class MagNet_node_classification(nn.Module):
    r"""The MagNet model for node classification from the
    `MagNet: A Neural Network for Directed Graphs." <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.
    Args:
        in_channels (int): Size of each input sample.
        num_filter (int, optional): Number of hidden channels.  Default: 2.
        K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        label_dim (int, optional): Number of output classes.  Default: 2.
        activation (bool, optional): whether to use activation function or not. (default: :obj:`False`)
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        layer (int, optional): Number of MagNetConv layers. Deafult: 2.
        dropout (float, optional): Dropout value. (default: :obj:`False`)
        normalization (str, optional): The normalization scheme for the magnetic
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} Hadamard \exp(i \Theta^{(q)})`
    """
    def __init__(self, in_channels:int, num_filter:int=2, q:float=0.25, K:int=2, label_dim:int=2, \
        activation:bool=False, trainable_q:bool=False, layer:int=2, dropout:float=False, normalization:str='sym'):
        super(MagNet_node_classification, self).__init__()

        chebs = nn.ModuleList()
        chebs.append(MagNetConv(in_channels=in_channels, out_channels=num_filter, K=K, \
            q=q, trainable_q=trainable_q, normalization=normalization))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MagNetConv(in_channels=num_filter, out_channels=num_filter, K=K,\
                q=q, trainable_q=trainable_q, normalization=normalization))

        self.Chebs = chebs

        self.Conv = nn.Conv1d(2*num_filter, label_dim, kernel_size=1)        
        self.dropout = dropout

    def forward(self, real: torch.FloatTensor, imag: torch.FloatTensor, edge_index: torch.LongTensor, \
        edge_weight: Optional[torch.LongTensor]=None) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet node classification model from the
    `MagNet: A Neural Network for Directed Graphs." <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.
        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * log_prob (PyTorch Float Tensor) - Logarithmic class probabilities for all nodes, with shape (num_nodes, num_classes).
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
        x = self.Conv(x)
        x = F.log_softmax(x, dim=1)
        return torch.transpose(x[0], 0, 1)