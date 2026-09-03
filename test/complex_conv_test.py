import pytest
import torch

from torch_geometric_signed_directed.nn import MagNetConv, MSConv


CONV_TYPES = (MagNetConv, MSConv)


def _graph():
    edge_index = torch.tensor([
        [0, 1, 2, 0],
        [1, 2, 0, 2],
    ])
    edge_weight = torch.tensor([1.0, 0.7, 1.2, 0.4])
    return edge_index, edge_weight


@pytest.mark.parametrize('conv_type', CONV_TYPES)
@pytest.mark.parametrize('K', (1, 3))
def test_complex_chebyshev_recurrence(conv_type, K):
    edge_index, edge_weight = _graph()
    x_real = torch.tensor([
        [1.0, -0.5],
        [0.25, 0.75],
        [-1.0, 0.5],
    ])
    # Follow the MagNet input convention X + iX.
    x_imag = x_real.clone()
    conv = conv_type(
        2, 2, K=K, q=0.13, trainable_q=False, bias=False
    )
    with torch.no_grad():
        values = torch.arange((K + 1) * 4, dtype=x_real.dtype)
        conv.weight.copy_(values.view(K + 1, 2, 2) / 7 - 0.3)

    out_real, out_imag = conv(
        x_real, x_imag, edge_index, edge_weight
    )
    actual = torch.complex(out_real, out_imag)

    lap_edge_index, lap_weight = conv.__norm__(
        edge_index, x_real.size(0), edge_weight, 0.13, 'sym',
        torch.tensor(2.0), dtype=x_real.dtype
    )
    laplacian = torch.sparse_coo_tensor(
        lap_edge_index, lap_weight, (x_real.size(0), x_real.size(0))
    ).coalesce().to_dense()

    Tx_0 = torch.complex(x_real, x_imag)
    weight = conv.weight.to(Tx_0.dtype)
    expected = torch.matmul(Tx_0, weight[0])
    Tx_1 = torch.matmul(laplacian, Tx_0)
    expected = expected + torch.matmul(Tx_1, weight[1])
    for k in range(2, K + 1):
        Tx_2 = 2 * torch.matmul(laplacian, Tx_1) - Tx_0
        expected = expected + torch.matmul(Tx_2, weight[k])
        Tx_0, Tx_1 = Tx_1, Tx_2

    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize('conv_type', CONV_TYPES)
def test_trainable_q_keeps_optimizer_reference_and_is_clamped(conv_type):
    edge_index, edge_weight = _graph()
    x = torch.tensor([
        [1.0, -0.5],
        [0.25, 0.75],
        [-1.0, 0.5],
    ])
    conv = conv_type(
        2, 2, K=2, q=0.3, trainable_q=True, cached=True,
        bias=False, max_q=0.2
    )
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([
            [[0.2, -0.1], [0.3, 0.4]],
            [[-0.3, 0.5], [0.1, -0.2]],
            [[0.4, 0.2], [-0.5, 0.3]],
        ]))

    q_parameter = conv.q
    optimizer = torch.optim.SGD(conv.parameters(), lr=1e-4)
    assert any(q_parameter is parameter
               for group in optimizer.param_groups
               for parameter in group['params'])

    for _ in range(2):
        optimizer.zero_grad()
        out_real, out_imag = conv(x, x, edge_index, edge_weight)
        assert conv.q is q_parameter
        assert -1e-7 <= conv.q.item() <= conv.max_q + 1e-7
        loss = out_real.square().sum() + 0.37 * out_imag.square().sum()
        loss.backward()
        assert conv.q.grad is not None
        assert conv.q.grad.abs().item() > 0
        optimizer.step()

    # The next forward projects an optimizer update back into the valid range.
    conv(x, x, edge_index, edge_weight)
    assert conv.q is q_parameter
    assert -1e-7 <= conv.q.item() <= conv.max_q + 1e-7
