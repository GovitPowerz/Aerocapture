"""_zero_state_where_done handles Dense (None), GRU/LSTM (2D / tuple-2D),
Window (3D), Mamba (3D), Transformer KV cache (tuple of 3D)."""

import torch

from aerocapture.training.rl.policy import _zero_state_where_done


def test_none_passthrough():
    out = _zero_state_where_done([None], torch.tensor([True, False]))
    assert out == [None]


def test_gru_2d_zeros_done_rows():
    h = torch.ones(3, 4)
    done = torch.tensor([True, False, True])
    out = _zero_state_where_done([h], done)[0]
    assert torch.allclose(out[0], torch.zeros(4))
    assert torch.allclose(out[1], torch.ones(4))
    assert torch.allclose(out[2], torch.zeros(4))


def test_lstm_tuple_of_2d():
    h = torch.ones(2, 4)
    c = torch.full((2, 4), 2.0)
    done = torch.tensor([False, True])
    out_h, out_c = _zero_state_where_done([(h, c)], done)[0]
    assert torch.allclose(out_h[1], torch.zeros(4))
    assert torch.allclose(out_c[1], torch.zeros(4))


def test_mamba_3d_zeros_done_rows():
    h = torch.ones(3, 4, 5)  # (B, input_size, d_state)
    done = torch.tensor([True, False, True])
    out = _zero_state_where_done([h], done)[0]
    assert torch.allclose(out[0], torch.zeros(4, 5))
    assert torch.allclose(out[1], torch.ones(4, 5))
    assert torch.allclose(out[2], torch.zeros(4, 5))


def test_window_3d_zeros_done_rows():
    h = torch.ones(2, 3, 4)  # (B, n_steps, input_size)
    done = torch.tensor([True, False])
    out = _zero_state_where_done([h], done)[0]
    assert torch.allclose(out[0], torch.zeros(3, 4))
    assert torch.allclose(out[1], torch.ones(3, 4))


def test_transformer_kv_cache_tuple_of_3d():
    k = torch.ones(2, 5, 8)
    v = torch.full((2, 5, 8), 2.0)
    done = torch.tensor([True, False])
    out_k, out_v = _zero_state_where_done([(k, v)], done)[0]
    assert torch.allclose(out_k[0], torch.zeros(5, 8))
    assert torch.allclose(out_v[1], torch.full((5, 8), 2.0))
