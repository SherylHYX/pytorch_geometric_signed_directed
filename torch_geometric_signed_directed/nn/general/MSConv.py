from typing import Optional

import torch
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from ...utils.general.get_magnetic_signed_Laplacian import get_magnetic_signed_Laplacian

class MSConv(MessagePassing):
    r"""Magnetic Signed Laplacian Convolution Layer from the 
    `MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian <https://proceedings.mlr.press/v198/he22c.html>`_ paper.
    
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Order of the Chebyshev polynomial. The Chebyshev filter
            contains :math:`K + 1` terms.
        q (float, optional): Initial value of the phase parameter. Default: 0.25.
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        normalization (str, optional): The normalization scheme for the magnetic
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
            learning scenarios. Caching is bypassed when :obj:`q` is
            trainable. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        absolute_degree (bool, optional): Whether to calculate the degree matrix with respect to absolute entries of the adjacency matrix. (default: :obj:`True`)
        max_q (float, optional): Maximum value of a trainable phase parameter.
            A trainable :obj:`q` is clamped to :math:`[0, \mathrm{max\_q}]`
            before each forward pass. (default: :obj:`0.25`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels:int, out_channels:int, K:int, q:float, trainable_q:bool,
                 normalization:str='sym', bias:bool=True, cached: bool=False, absolute_degree: bool=True,
                 max_q: float=0.25, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super(MSConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym'], 'Invalid normalization'
        if max_q < 0:
            raise ValueError('max_q must be non-negative')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.cached = cached
        self.trainable_q = trainable_q
        self.absolute_degree = absolute_degree
        self.max_q = max_q

        if trainable_q:
            self.q = Parameter(torch.Tensor(1).fill_(q))
        else:
            self.q = q
        self.weight = Parameter(torch.Tensor(K+1, in_channels, out_channels))

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
        Get the magnetic signed Laplacian.
        
        Arg types:
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * num_nodes (int, Optional) - Node features.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * edge_index (PyTorch Long Tensor) - Magnetic signed Laplacian edge indices.
            * edge_weight (PyTorch Complex Tensor) - Complex magnetic signed Laplacian edge weights.
        """
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight_real, edge_weight_imag = get_magnetic_signed_Laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes, q, absolute_degree=self.absolute_degree
        )

        edge_weight = torch.complex(edge_weight_real, edge_weight_imag)
        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight = torch.where(
            torch.isinf(edge_weight), torch.zeros_like(edge_weight), edge_weight
        )
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_weight is not None

        return edge_index, edge_weight

    def forward(
        self,
        x_real: torch.FloatTensor, 
        x_imag: torch.FloatTensor, 
        edge_index: torch.LongTensor,
        edge_weight: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the Signed Directed Magnetic Laplacian Convolution layer.
        
        Arg types:
            * x_real, x_imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * out_real, out_imag (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (N_nodes, F_out).
        """
        if self.trainable_q:
            # Project the parameter without replacing it, so optimizers retain
            # their reference and gradients continue to update q.
            with torch.no_grad():
                self.q.clamp_(0, self.max_q)
        q = self.q
        use_cache = self.cached and not self.trainable_q

        if use_cache and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))
            q_value = q.detach().item() if isinstance(q, torch.Tensor) else q
            if q_value != self.cached_q:
                raise RuntimeError(
                    'Cached q is {}, but found {} in input. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_q, q_value))
        if not use_cache or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            self.cached_q = q.detach().item() if isinstance(q, torch.Tensor) else q
            if self.normalization != 'sym' and lambda_max is None:
                if self.trainable_q:
                    raise RuntimeError('Cannot train q while not calculating maximum eigenvalue of Laplacian!')
                _, _, _, lambda_max =  get_magnetic_signed_Laplacian(
                edge_index, edge_weight, None, q=q, return_lambda_max=True, absolute_degree=self.absolute_degree
            )

            if lambda_max is None:
                lambda_max = torch.tensor(2.0, dtype=x_real.dtype, device=x_real.device)
            if not isinstance(lambda_max, torch.Tensor):
                lambda_max = torch.tensor(lambda_max, dtype=x_real.dtype,
                                        device=x_real.device)
            assert lambda_max is not None
            edge_index_lap, norm = self.__norm__(
                edge_index, x_real.size(self.node_dim), edge_weight, q,
                self.normalization, lambda_max, dtype=x_real.dtype
            )
            self.cached_result = edge_index_lap, norm

        edge_index_lap, norm = self.cached_result

        x = torch.complex(x_real, x_imag)
        weight = self.weight.to(x.dtype)
        Tx_0 = x
        out = torch.matmul(Tx_0, weight[0])

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.size(0) > 1:
            Tx_1 = self.propagate(
                edge_index_lap, x=Tx_0, norm=norm, size=None
            )
            out = out + torch.matmul(Tx_1, weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2 = 2. * self.propagate(
                edge_index_lap, x=Tx_1, norm=norm, size=None
            ) - Tx_0
            out = out + torch.matmul(Tx_2, weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + torch.complex(self.bias, self.bias)

        return out.real, out.imag


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, filter size={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)
