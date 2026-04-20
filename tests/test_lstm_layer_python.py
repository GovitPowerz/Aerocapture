"""LstmLayer torch module matches nn.LSTMCell bit-for-bit on f64."""

from __future__ import annotations

import torch
from aerocapture.training.rl.layers.lstm import LstmLayer
from torch import nn


def test_lstm_layer_matches_nn_lstmcell() -> None:
    torch.manual_seed(0)
    I, H = 5, 4
    ours = LstmLayer(input_size=I, hidden_size=H).double()
    theirs = nn.LSTMCell(input_size=I, hidden_size=H).double()

    # Copy parameters: torch.nn.LSTMCell uses gate order (i, f, g, o) — matches ours.
    with torch.no_grad():
        ours.weight_ih.copy_(theirs.weight_ih)
        ours.weight_hh.copy_(theirs.weight_hh)
        ours.bias_ih.copy_(theirs.bias_ih)
        ours.bias_hh.copy_(theirs.bias_hh)

    B = 3
    x = torch.randn(B, I, dtype=torch.float64)
    h = torch.randn(B, H, dtype=torch.float64)
    c = torch.randn(B, H, dtype=torch.float64)

    h_ours, (h_ours_state, c_ours_state) = ours(x, (h, c))
    h_their, c_their = theirs(x, (h, c))

    assert torch.allclose(h_ours, h_their, atol=1e-12)
    assert torch.allclose(h_ours_state, h_their, atol=1e-12)
    assert torch.allclose(c_ours_state, c_their, atol=1e-12)


def test_lstm_layer_new_state_matches_dtype() -> None:
    I, H = 4, 6
    layer = LstmLayer(I, H).double()
    state = layer.new_state(batch_size=2, device="cpu")
    assert isinstance(state, tuple)
    assert len(state) == 2
    h, c = state
    assert h.dtype == torch.float64
    assert c.dtype == torch.float64
    assert h.shape == (2, H)
    assert c.shape == (2, H)
    assert torch.all(h == 0.0)
    assert torch.all(c == 0.0)
