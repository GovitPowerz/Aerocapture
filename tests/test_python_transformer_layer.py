"""Unit tests for the Python TransformerLayer torch module.

Cross-language equivalence tests against Rust live in
test_rust_python_transformer_equivalence.py (Task 17).
"""

from __future__ import annotations

import math

import torch
from aerocapture.training.rl.layers.transformer import TransformerLayer


def test_transformer_layer_output_shape() -> None:
    layer = TransformerLayer(d_model=16, n_heads=2, d_ffn=32, n_seq=8).double()
    state = layer.new_state(batch_size=1)
    x = torch.randn(1, 16, dtype=torch.float64)
    out, new_state = layer(x, state)
    assert out.shape == (1, 16)
    assert new_state[0].shape == (1, 1, 16)
    assert new_state[1].shape == (1, 1, 16)


def test_transformer_cache_grows_to_n_seq_then_saturates() -> None:
    layer = TransformerLayer(d_model=8, n_heads=2, d_ffn=16, n_seq=3).double()
    state = layer.new_state(batch_size=1)
    for step in range(6):
        x = torch.randn(1, 8, dtype=torch.float64)
        _, state = layer(x, state)
        expected = min(step + 1, 3)
        assert state[0].shape[1] == expected, f"step {step}: got {state[0].shape[1]}"


def test_transformer_residual_dominates_when_weights_zero() -> None:
    # Zero all projection + FFN weights, LN gamma=1 / beta=0 -> output == input (residual).
    layer = TransformerLayer(d_model=4, n_heads=2, d_ffn=8, n_seq=3).double()
    with torch.no_grad():
        for lin in [layer.w_q, layer.w_k, layer.w_v, layer.w_o, layer.w_ffn1, layer.w_ffn2]:
            lin.weight.zero_()
            lin.bias.zero_()
        layer.ln1_gamma.fill_(1.0)
        layer.ln1_beta.zero_()
        layer.ln2_gamma.fill_(1.0)
        layer.ln2_beta.zero_()
    state = layer.new_state(batch_size=1)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    out, _ = layer(x, state)
    for i in range(4):
        assert abs(out[0, i].item() - x[0, i].item()) < 1e-12


def test_transformer_gelu_exact_vs_torch() -> None:
    # Our GELU matches torch.nn.functional.gelu (default = exact).
    z = torch.tensor([1.0, -1.0, 2.5], dtype=torch.float64)
    ours = 0.5 * z * (1.0 + torch.special.erf(z * (1.0 / math.sqrt(2.0))))
    theirs = torch.nn.functional.gelu(z)
    torch.testing.assert_close(ours, theirs, atol=1e-14, rtol=0)
