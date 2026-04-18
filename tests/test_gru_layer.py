"""Unit tests for GruLayer torch module."""

from __future__ import annotations

import torch
from aerocapture.training.rl.layers.gru import GruLayer


def test_gru_layer_shapes() -> None:
    layer = GruLayer(input_size=5, hidden_size=8)
    batch = 2
    x = torch.zeros(batch, 5)
    h = layer.new_state(batch, device="cpu")
    assert h.shape == (batch, 8)
    out, new_h = layer(x, h)
    assert out.shape == (batch, 8)
    assert new_h.shape == (batch, 8)
    # GRU output equals new hidden state (in-place semantics).
    torch.testing.assert_close(out, new_h, rtol=0, atol=0)


def test_gru_layer_zero_init_known_output() -> None:
    # All weights zeroed + h_prev non-zero => r=z=0.5, n=tanh(0)=0,
    # h_new = (1 - 0.5) * 0 + 0.5 * h_prev = 0.5 * h_prev.
    layer = GruLayer(input_size=2, hidden_size=3)
    with torch.no_grad():
        layer.weight_ih.zero_()
        layer.weight_hh.zero_()
        layer.bias_ih.zero_()
        layer.bias_hh.zero_()
    x = torch.tensor([[0.5, -0.5]])
    h_prev = torch.tensor([[1.0, 2.0, -1.0]])
    out, new_h = layer(x, h_prev)
    expected = h_prev * 0.5
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


def test_gru_layer_n_params_closed_form() -> None:
    # Input size I=4, hidden size H=8: 3H*I + 3H*H + 2*3H = 96 + 192 + 48 = 336.
    layer = GruLayer(input_size=4, hidden_size=8)
    total = sum(p.numel() for p in layer.parameters())
    assert total == 336
