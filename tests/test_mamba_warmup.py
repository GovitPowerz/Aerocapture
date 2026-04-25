"""Mamba state warm-up test: state starts at zero and evolves deterministically.

Verifies:
  (a) Feeding the same input twice yields different outputs (state accumulates).
  (b) Repeated runs of the same sequence are bit-identical (determinism).
  (c) Cross-language Python/Rust agreement on the warm-up sequence (first 3 of 8 steps).

Architecture: Dense(4 -> 8, tanh) -> Mamba(8, d_state=4, dt_rank=2) -> Dense(8 -> 2, linear).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("aerocapture_rs")
import aerocapture_rs  # type: ignore[import-not-found]  # noqa: E402
from aerocapture.training.rl.layers.dense import DenseLayer  # noqa: E402
from aerocapture.training.rl.layers.mamba import MambaLayer  # noqa: E402


def _build_mamba_model(tmp_path: Path) -> tuple[Path, DenseLayer, MambaLayer, DenseLayer]:
    """Construct a small Dense -> Mamba -> Dense model, write JSON, return (path, layers)."""
    torch.manual_seed(7)
    dense_in = DenseLayer(input_size=4, output_size=8, activation="tanh").double()
    mamba = MambaLayer(input_size=8, d_state=4, dt_rank=2).double()
    dense_out = DenseLayer(input_size=8, output_size=2, activation="linear").double()

    with torch.no_grad():
        for lin in [dense_in.linear, dense_out.linear]:
            torch.nn.init.uniform_(lin.weight, -0.3, 0.3)
            torch.nn.init.uniform_(lin.bias, -0.1, 0.1)
        torch.nn.init.uniform_(mamba.x_proj_w, -0.3, 0.3)
        torch.nn.init.uniform_(mamba.dt_proj_w, -0.3, 0.3)
        torch.nn.init.uniform_(mamba.dt_proj_b, -0.5, 0.5)
        torch.nn.init.uniform_(mamba.a_log, 0.0, 2.0)
        mamba.d_skip.fill_(1.0)

    model_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2},
            {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {
                "w": dense_in.linear.weight.detach().tolist(),
                "b": dense_in.linear.bias.detach().tolist(),
            },
            "layer_1": {
                "x_proj_w": mamba.x_proj_w.detach().tolist(),
                "dt_proj_w": mamba.dt_proj_w.detach().tolist(),
                "dt_proj_b": mamba.dt_proj_b.detach().tolist(),
                "a_log": mamba.a_log.detach().tolist(),
                "d_skip": mamba.d_skip.detach().tolist(),
            },
            "layer_2": {
                "w": dense_out.linear.weight.detach().tolist(),
                "b": dense_out.linear.bias.detach().tolist(),
            },
        },
    }
    path = tmp_path / "mamba_warmup_model.json"
    path.write_text(json.dumps(model_json))
    return path, dense_in, mamba, dense_out


@pytest.mark.slow
def test_mamba_state_evolves_between_steps(tmp_path: Path) -> None:
    """Feed the SAME input twice; outputs must differ because state accumulates."""
    path, _, _, _ = _build_mamba_model(tmp_path)
    x = [0.5, -0.3, 0.1, 0.8]
    seq = [x, x]
    outs = np.asarray(aerocapture_rs.nn_forward_sequence(str(path), seq), dtype=np.float64)
    assert outs.shape == (2, 2)
    diff = np.max(np.abs(outs[0] - outs[1]))
    assert diff > 1e-10, f"state evolution: step-0 and step-1 outputs should differ; diff={diff:.3e}"


@pytest.mark.slow
def test_mamba_forward_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Same input sequence across two calls -> bit-identical output."""
    path, _, _, _ = _build_mamba_model(tmp_path)
    rng = np.random.default_rng(0)
    seq = rng.uniform(-1.0, 1.0, size=(20, 4)).astype(np.float64)
    seq_list = [row.tolist() for row in seq]
    out_a = np.asarray(aerocapture_rs.nn_forward_sequence(str(path), seq_list), dtype=np.float64)
    out_b = np.asarray(aerocapture_rs.nn_forward_sequence(str(path), seq_list), dtype=np.float64)
    assert np.array_equal(out_a, out_b), "Mamba Rust forward is non-deterministic!"


@pytest.mark.slow
def test_mamba_warmup_cross_language_agreement(tmp_path: Path) -> None:
    """Python and Rust outputs agree to < 1e-10 during the warm-up window (3 steps)."""
    path, dense_in, mamba, dense_out = _build_mamba_model(tmp_path)
    rng = np.random.default_rng(7)
    inputs = rng.standard_normal((3, 4)).astype(np.float64)

    rust_outs = np.asarray(aerocapture_rs.nn_forward_sequence(str(path), inputs.tolist()), dtype=np.float64)

    dense_in.eval()
    mamba.eval()
    dense_out.eval()
    h = mamba.new_state()
    py_outs = np.empty((3, 2), dtype=np.float64)

    with torch.no_grad():
        for t in range(3):
            x = torch.tensor(inputs[t], dtype=torch.float64)
            y0, _ = dense_in(x, None)
            y1, h = mamba(y0, h)
            y2, _ = dense_out(y1, None)
            py_outs[t] = y2.numpy()
            # State must be non-zero after first step.
            if t == 0:
                assert float(h.abs().max()) > 1e-10, "Mamba state is still zero after step 0"

    max_diff = float(np.abs(rust_outs - py_outs).max())
    assert max_diff < 1e-10, f"warm-up cross-language mismatch: {max_diff:.3e}"
