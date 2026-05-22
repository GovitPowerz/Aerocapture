"""Per-layer to_flat() matches Rust LayerWeights::to_flat round-trip.

For each layer: build a Python instance with randomized weights, extract
via to_flat(), serialize to v2 JSON via aerocapture_rs.flat_weights_to_json,
load back via aerocapture_rs.nn_forward, assert finite + matches Python
forward at <1e-10.

Shape tests assert that each layer's flat representation matches the byte
budget of the Rust LayerWeights::n_params contract (gold reference:
src/rust/src/data/neural.rs).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

aerocapture_rs = pytest.importorskip("aerocapture_rs")

from aerocapture.training.rl.layers import (  # noqa: E402
    DenseLayer,
    GruLayer,
    LstmLayer,
    MambaLayer,
    TransformerLayer,
    WindowLayer,
)


def _randomize(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.uniform_(-0.1, 0.1)


# -- Shape tests (one per layer type) -----------------------------------------


def test_dense_to_flat_shape() -> None:
    layer = DenseLayer(input_size=3, output_size=4, activation="tanh").double()
    _randomize(layer)
    flat = layer.to_flat()
    assert flat.shape == (3 * 4 + 4,)  # W + b
    assert flat.dtype == np.float64


def test_gru_to_flat_shape() -> None:
    layer = GruLayer(input_size=3, hidden_size=4).double()
    _randomize(layer)
    flat = layer.to_flat()
    # weight_ih (3H x I) + weight_hh (3H x H) + bias_ih (3H) + bias_hh (3H)
    assert flat.shape == (3 * 4 * 3 + 3 * 4 * 4 + 2 * 3 * 4,)
    assert flat.dtype == np.float64


def test_lstm_to_flat_shape() -> None:
    layer = LstmLayer(input_size=3, hidden_size=4).double()
    _randomize(layer)
    flat = layer.to_flat()
    # weight_ih (4H x I) + weight_hh (4H x H) + bias_ih (4H) + bias_hh (4H)
    assert flat.shape == (4 * 4 * 3 + 4 * 4 * 4 + 2 * 4 * 4,)
    assert flat.dtype == np.float64


def test_window_to_flat_empty() -> None:
    layer = WindowLayer(input_size=3, n_steps=4).double()
    flat = layer.to_flat()
    assert flat.shape == (0,)
    assert flat.dtype == np.float64


def test_transformer_to_flat_shape() -> None:
    d_model, n_heads, d_ffn, n_seq = 8, 2, 16, 4
    layer = TransformerLayer(d_model, n_heads, d_ffn, n_seq).double()
    _randomize(layer)
    flat = layer.to_flat()
    # 4 Q/K/V/O projections (each d_model*d_model + d_model)
    # + ffn1 (d_model -> d_ffn) and ffn2 (d_ffn -> d_model) with biases
    # + 4 LN vectors (gamma/beta x 2) of length d_model
    expected = 4 * (d_model * d_model + d_model) + (d_model * d_ffn + d_ffn) + (d_ffn * d_model + d_model) + 4 * d_model
    assert flat.shape == (expected,)
    assert flat.dtype == np.float64


def test_mamba_to_flat_shape() -> None:
    input_size, d_state, dt_rank = 8, 4, 2
    layer = MambaLayer(input_size, d_state, dt_rank).double()
    _randomize(layer)
    flat = layer.to_flat()
    # x_proj_w (dt_rank + 2*d_state, input_size)
    # + dt_proj_w (input_size, dt_rank) + dt_proj_b (input_size)
    # + a_log (input_size, d_state) + d_skip (input_size)
    expected = (dt_rank + 2 * d_state) * input_size + input_size * dt_rank + input_size + input_size * d_state + input_size
    assert flat.shape == (expected,)
    assert flat.dtype == np.float64


# -- Roundtrip tests: Python to_flat -> Rust JSON -> Rust nn_forward ---------


@pytest.mark.parametrize(
    "architecture, input_dim",
    [
        (
            [
                {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
                {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
            ],
            4,
        ),
        (
            [
                {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
                {"type": "gru", "input_size": 8, "hidden_size": 8},
                {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
            ],
            4,
        ),
    ],
)
def test_to_flat_roundtrip_via_rust(architecture: list[dict], input_dim: int, tmp_path) -> None:
    """Build small V2Policy, extract via per-layer to_flat, write through the
    Rust flat_weights_to_json helper, load via nn_forward, and assert the Rust
    forward matches the Python V2Policy forward at <1e-10 (machine epsilon)."""
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import LayerSpec
    from pydantic import TypeAdapter

    validated = TypeAdapter(list[LayerSpec]).validate_python(architecture)
    policy = V2Policy(architecture=validated, input_mask=None).double()
    _randomize(policy)

    flat_list: list[np.ndarray] = [layer.to_flat() for layer in policy.layers]
    flat = np.concatenate(flat_list)

    json_path = tmp_path / "model.json"
    aerocapture_rs.flat_weights_to_json(
        flat.tolist(),
        json.dumps(architecture),
        str(json_path),
        None,
    )

    x = np.linspace(-1.0, 1.0, input_dim, dtype=np.float64)
    rust_out = np.array(aerocapture_rs.nn_forward(str(json_path), x.tolist()), dtype=np.float64)

    x_t = torch.tensor(x, dtype=torch.float64).unsqueeze(0)  # (1, input_dim)
    state = policy.new_state(batch_size=1, device=None)
    mean, _ = policy(x_t, state)
    py_out = mean.detach().numpy().flatten()

    assert np.all(np.isfinite(rust_out)), f"Rust output not finite: {rust_out}"
    assert np.all(np.isfinite(py_out)), f"Python output not finite: {py_out}"
    max_diff = float(np.abs(rust_out - py_out).max())
    assert np.allclose(rust_out, py_out, atol=1e-10), f"Rust {rust_out} vs Python {py_out}, max abs diff {max_diff:.3e}"
