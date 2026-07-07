"""Cross-language bit-equivalence gate for the Mamba-3 ablation layer.

Architecture: Dense(4 -> 8, tanh) -> Mamba3(8, 4, 2, flags) -> Dense(8 -> 2, linear).
Exercises all 4 flag combos (euler|trapezoidal x real|complex). Exports a
Python-constructed v2 JSON with random f64 weights, loads it in Rust via
`aerocapture_rs.nn_forward_sequence`, feeds 100 random f64 inputs, and asserts
max abs diff < 1e-12 (complex adds a multiply vs the real Mamba path's 1e-14;
observed actual ~1e-14).

Diagnostics if it fails:
  - Constant drift across all steps -> flat-weight / field-name ordering mismatch
  - Growing drift over the sequence  -> state update bug (alpha, b_bar, cross term)
  - Only fails for complex/trapz     -> a_imag rotation or lambda cross-term bug
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
from aerocapture.training.rl.layers.mamba3 import Mamba3Layer  # noqa: E402


@pytest.mark.slow
@pytest.mark.parametrize(
    ("discretization", "state_mode"),
    [("euler", "real"), ("trapezoidal", "real"), ("euler", "complex"), ("trapezoidal", "complex")],
)
def test_mamba3_rust_python_equivalence_100_steps(discretization: str, state_mode: str, tmp_path: Path) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(seed=1234)
    trapezoidal = discretization == "trapezoidal"
    complex_mode = state_mode == "complex"

    dense_in = DenseLayer(input_size=4, output_size=8, activation="tanh").double()
    mamba = Mamba3Layer(input_size=8, d_state=4, dt_rank=2, trapezoidal=trapezoidal, complex=complex_mode).double()
    dense_out = DenseLayer(input_size=8, output_size=2, activation="linear").double()

    with torch.no_grad():
        for lin in (dense_in.linear, dense_out.linear):
            torch.nn.init.uniform_(lin.weight, -0.3, 0.3)
            torch.nn.init.uniform_(lin.bias, -0.3, 0.3)
        torch.nn.init.uniform_(mamba.x_proj_w, -0.3, 0.3)
        torch.nn.init.uniform_(mamba.dt_proj_w, -0.3, 0.3)
        torch.nn.init.uniform_(mamba.dt_proj_b, -0.5, 0.5)
        torch.nn.init.uniform_(mamba.a_log, 0.0, 2.0)
        mamba.d_skip.fill_(1.0)
        if mamba.a_imag is not None:
            torch.nn.init.uniform_(mamba.a_imag, -1.5, 1.5)  # rotation freq
        if mamba.lambda_logit is not None:
            torch.nn.init.uniform_(mamba.lambda_logit, -2.0, 6.0)  # sweep euler..trapz

    layer1: dict[str, object] = {
        "x_proj_w": mamba.x_proj_w.detach().tolist(),
        "dt_proj_w": mamba.dt_proj_w.detach().tolist(),
        "dt_proj_b": mamba.dt_proj_b.detach().tolist(),
        "a_log": mamba.a_log.detach().tolist(),
        "d_skip": mamba.d_skip.detach().tolist(),
    }
    if mamba.a_imag is not None:
        layer1["a_imag"] = mamba.a_imag.detach().tolist()
    if mamba.lambda_logit is not None:
        layer1["lambda_logit"] = mamba.lambda_logit.detach().tolist()

    model_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "mamba3", "input_size": 8, "d_state": 4, "dt_rank": 2, "discretization": discretization, "state_mode": state_mode},
            {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {"w": dense_in.linear.weight.detach().tolist(), "b": dense_in.linear.bias.detach().tolist()},
            "layer_1": layer1,
            "layer_2": {"w": dense_out.linear.weight.detach().tolist(), "b": dense_out.linear.bias.detach().tolist()},
        },
    }
    model_path = tmp_path / f"mamba3_{discretization}_{state_mode}.json"
    model_path.write_text(json.dumps(model_json))

    inputs = rng.standard_normal((100, 4)).astype(np.float64)
    rust_outs = np.asarray(
        aerocapture_rs.nn_forward_sequence(str(model_path), [row.tolist() for row in inputs]),
        dtype=np.float64,
    )
    assert rust_outs.shape == (100, 2)

    state = mamba.new_state()
    py_outs = np.empty((100, 2), dtype=np.float64)
    for layer in (dense_in, mamba, dense_out):
        layer.eval()
    with torch.no_grad():
        for t in range(100):
            x = torch.tensor(inputs[t], dtype=torch.float64)
            y0, _ = dense_in(x, None)
            y1, state = mamba.forward_unbatched(y0, state)
            y2, _ = dense_out(y1, None)
            py_outs[t] = y2.numpy()

    diff = float(np.abs(rust_outs - py_outs).max())
    print(f"Mamba3 {discretization}/{state_mode} cross-language max abs diff over 100 steps: {diff:.3e}")
    assert diff < 1e-12, (
        f"{discretization}/{state_mode} drift {diff:.3e} >= 1e-12; "
        f"step-0 {float(np.abs(rust_outs[0] - py_outs[0]).max()):.3e}, "
        f"step-99 {float(np.abs(rust_outs[99] - py_outs[99]).max()):.3e}"
    )
