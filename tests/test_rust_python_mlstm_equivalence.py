"""Cross-language bit-equivalence gate for the mLSTM probe layer.

Architecture: Dense(4 -> 8, tanh) -> Mlstm(8, 6) -> Dense(6 -> 2, linear).
Exports a Python-built v2 JSON with random f64 weights, runs 100 steps through
aerocapture_rs.nn_forward_sequence, and compares against the unbatched torch
mirror. Gate 1e-12 (observed expectation ~1e-15).
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
from aerocapture.training.rl.layers.mlstm import MlstmLayer  # noqa: E402


@pytest.mark.slow
def test_mlstm_rust_python_equivalence_100_steps(tmp_path: Path) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(seed=1234)

    dense_in = DenseLayer(input_size=4, output_size=8, activation="tanh").double()
    mlstm = MlstmLayer(input_size=8, hidden_size=6).double()
    dense_out = DenseLayer(input_size=6, output_size=2, activation="linear").double()

    with torch.no_grad():
        for lin in (dense_in.linear, dense_out.linear):
            torch.nn.init.uniform_(lin.weight, -0.3, 0.3)
            torch.nn.init.uniform_(lin.bias, -0.3, 0.3)
        for mat in (mlstm.w_q, mlstm.w_k, mlstm.w_v, mlstm.w_o, mlstm.w_i, mlstm.w_f):
            torch.nn.init.uniform_(mat, -0.5, 0.5)
        for vec in (mlstm.b_q, mlstm.b_k, mlstm.b_v, mlstm.b_o):
            torch.nn.init.uniform_(vec, -0.5, 0.5)
        mlstm.b_i.fill_(0.3)
        mlstm.b_f.fill_(2.0)

    model_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "mlstm", "input_size": 8, "hidden_size": 6},
            {"type": "dense", "input_size": 6, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {"w": dense_in.linear.weight.detach().tolist(), "b": dense_in.linear.bias.detach().tolist()},
            "layer_1": {
                "w_q": mlstm.w_q.detach().tolist(),
                "b_q": mlstm.b_q.detach().tolist(),
                "w_k": mlstm.w_k.detach().tolist(),
                "b_k": mlstm.b_k.detach().tolist(),
                "w_v": mlstm.w_v.detach().tolist(),
                "b_v": mlstm.b_v.detach().tolist(),
                "w_o": mlstm.w_o.detach().tolist(),
                "b_o": mlstm.b_o.detach().tolist(),
                "w_i": mlstm.w_i.detach().tolist(),
                "b_i": float(mlstm.b_i.detach()),
                "w_f": mlstm.w_f.detach().tolist(),
                "b_f": float(mlstm.b_f.detach()),
            },
            "layer_2": {"w": dense_out.linear.weight.detach().tolist(), "b": dense_out.linear.bias.detach().tolist()},
        },
    }
    model_path = tmp_path / "mlstm_eq.json"
    model_path.write_text(json.dumps(model_json))

    inputs = rng.standard_normal((100, 4)).astype(np.float64)
    rust_outs = np.asarray(
        aerocapture_rs.nn_forward_sequence(str(model_path), [row.tolist() for row in inputs]),
        dtype=np.float64,
    )
    assert rust_outs.shape == (100, 2)

    state = mlstm.new_state()
    py_outs = np.empty((100, 2), dtype=np.float64)
    for layer in (dense_in, mlstm, dense_out):
        layer.eval()
    with torch.no_grad():
        for t in range(100):
            x = torch.tensor(inputs[t], dtype=torch.float64)
            y0, _ = dense_in(x, None)
            y1, state = mlstm.forward_unbatched(y0, state)
            y2, _ = dense_out(y1, None)
            py_outs[t] = y2.numpy()

    diff = float(np.abs(rust_outs - py_outs).max())
    print(f"mLSTM cross-language max abs diff over 100 steps: {diff:.3e}")
    assert diff < 1e-12
