from typing import Optional

import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from ...utils.directed.get_magnetic_Laplacian import get_magnetic_Laplacian


class MagNetConv(MessagePassing):
    r"""The magnetic graph convolutional operator from the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper
    :math:`\mathbf{\hat{L}}` denotes the scaled and normalized magnetic Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        normalization (str, optional): The normalization scheme for the magnetic
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} \odot \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} \odot \exp(i \Theta^{(q)})`
            `\odot` denotes the element-wise multiplication.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int, q: float, trainable_q: bool,
                 normalization: str = 'sym', cached: bool = False, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MagNetConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym'], 'Invalid normalization'
        kwargs.setdefault('flow', 'target_to_source')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.cached = cached
        self.trainable_q = trainable_q
        if trainable_q:
            self.q = Parameter(torch.Tensor(1).fill_(q))
        else:
            self.q = q
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
        self.cached_q = None

    def __norm__(
        self,
        edge_index,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        q: float,
        normalization: Optional[str],
        lambda_max,
        dtype: Optional[int] = None
    ):
        """
        Get magnetic laplacian.

        Arg types:
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * num_nodes (int, Optional) - Node features.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.

        Return types:
            * edge_index_real, edge_index_imag, edge_weight_real, edge_weight_imag (PyTorch Float Tensor) - Magnetic laplacian tensor: real and imaginary edge indices and weights.
        """
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight_real, edge_weight_imag = get_magnetic_Laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes, q
        )

        edge_weight_real = (2.0 * edge_weight_real) / lambda_max
        edge_weight_real.masked_fill_(edge_weight_real == float("inf"), 0)
        edge_index_imag = edge_index.clone()

        edge_index_real, edge_weight_real = add_self_loops(
            edge_index, edge_weight_real, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_weight_real is not None

        edge_weight_imag = (2.0 * edge_weight_imag) / lambda_max
        edge_weight_imag.masked_fill_(edge_weight_imag == float("inf"), 0)

        assert edge_weight_imag is not None

        return edge_index_real, edge_index_imag, edge_weight_real, edge_weight_imag

    def forward(
        self,
        x_real: torch.FloatTensor,
        x_imag: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet Convolution layer.

        Arg types:
            * x_real, x_imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * out_real, out_imag (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (N_nodes, F_out).
        """
        if self.trainable_q:
            self.q = Parameter(torch.clamp(self.q, 0, 0.25))

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))
            if self.q != self.cached_q:
                raise RuntimeError(
                    'Cached q is {}, but found {} in input. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_q, self.q))
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.trainable_q:
                self.cached_q = self.q.detach().item()
            else:
                self.cached_q = self.q
            if self.normalization != 'sym' and lambda_max is None:
                if self.trainable_q:
                    raise RuntimeError(
                        'Cannot train q while not calculating maximum eigenvalue of Laplacian!')
                _, _, _, lambda_max = get_magnetic_Laplacian(
                    edge_index, edge_weight, None, q=self.q, return_lambda_max=True
                )

            if lambda_max is None:
                lambda_max = torch.tensor(
                    2.0, dtype=x_real.dtype, device=x_real.device)
            if not isinstance(lambda_max, torch.Tensor):
                lambda_max = torch.tensor(lambda_max, dtype=x_real.dtype,
                                          device=x_real.device)
            assert lambda_max is not None
            edge_index_real, edge_index_imag, norm_real, norm_imag = self.__norm__(edge_index, x_real.size(self.node_dim),
                                                             edge_weight, self.q, self.normalization,
                                                             lambda_max, dtype=x_real.dtype)
            self.cached_result = edge_index_real, edge_index_imag, norm_real, norm_imag

        edge_index_real, edge_index_imag, norm_real, norm_imag = self.cached_result

        Tx_0_real_real = x_real
        Tx_0_imag_imag = x_imag
        Tx_0_imag_real = x_real
        Tx_0_real_imag = x_imag
        out_real_real = torch.matmul(Tx_0_real_real, self.weight[0])
        out_imag_imag = torch.matmul(Tx_0_imag_imag, self.weight[0])
        out_imag_real = torch.matmul(Tx_0_imag_real, self.weight[0])
        out_real_imag = torch.matmul(Tx_0_real_imag, self.weight[0])

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.size(0) > 1:
            Tx_1_real_real = self.propagate(
                edge_index_real, x=x_real, norm=norm_real, size=None)
            out_real_real = out_real_real + \
                torch.matmul(Tx_1_real_real, self.weight[1])
            Tx_1_imag_imag = self.propagate(
                edge_index_imag, x=x_imag, norm=norm_imag, size=None)
            out_imag_imag = out_imag_imag + \
                torch.matmul(Tx_1_imag_imag, self.weight[1])
            Tx_1_imag_real = self.propagate(
                edge_index_real, x=x_real, norm=norm_real, size=None)
            out_imag_real = out_imag_real + \
                torch.matmul(Tx_1_imag_real, self.weight[1])
            Tx_1_real_imag = self.propagate(
                edge_index_imag, x=x_imag, norm=norm_imag, size=None)
            out_real_imag = out_real_imag + \
                torch.matmul(Tx_1_real_imag, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2_real_real = self.propagate(
                edge_index_real, x=Tx_1_real_real, norm=norm_real, size=None)
            Tx_2_real_real = 2. * Tx_2_real_real - Tx_0_real_real
            out_real_real = out_real_real + \
                torch.matmul(Tx_2_real_real, self.weight[k])
            Tx_0_real_real, Tx_1_real_real = Tx_1_real_real, Tx_2_real_real

            Tx_2_imag_imag = self.propagate(
                edge_index_imag, x=Tx_1_imag_imag, norm=norm_imag, size=None)
            Tx_2_imag_imag = 2. * Tx_2_imag_imag - Tx_0_imag_imag
            out_imag_imag = out_imag_imag + \
                torch.matmul(Tx_2_imag_imag, self.weight[k])
            Tx_0_imag_imag, Tx_1_imag_imag = Tx_1_imag_imag, Tx_2_imag_imag

            Tx_2_imag_real = self.propagate(
                edge_index_real, x=Tx_1_imag_real, norm=norm_real, size=None)
            Tx_2_imag_real = 2. * Tx_2_imag_real - Tx_0_imag_real
            out_imag_real = out_imag_real + \
                torch.matmul(Tx_2_imag_real, self.weight[k])
            Tx_0_imag_real, Tx_1_imag_real = Tx_1_imag_real, Tx_2_imag_real

            Tx_2_real_imag = self.propagate(
                edge_index_imag, x=Tx_1_real_imag, norm=norm_imag, size=None)
            Tx_2_real_imag = 2. * Tx_2_real_imag - Tx_0_real_imag
            out_real_imag = out_real_imag + \
                torch.matmul(Tx_2_real_imag, self.weight[k])
            Tx_0_real_imag, Tx_1_real_imag = Tx_1_real_imag, Tx_2_real_imag

        out_real = out_real_real - out_imag_imag
        out_imag = out_imag_real + out_real_imag

        if self.bias is not None:
            out_real += self.bias
            out_imag += self.bias

        return out_real, out_imag

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)
