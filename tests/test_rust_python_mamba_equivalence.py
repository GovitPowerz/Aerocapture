"""Cross-language bit-equivalence gate for Mamba SSM layer.

Architecture: Dense(4 -> 8, tanh) -> Mamba(8, 4, 2) -> Dense(8 -> 2, linear).
Total params: 40 + (2+8)*8 + 8*2 + 8 + 8*4 + 8 + 18 = 40 + 80 + 16 + 8 + 32 + 8 + 18 = 202.

Exports a Python-constructed v2 JSON with random f64 weights, loads it in
Rust via `aerocapture_rs.nn_forward_sequence`, and feeds 100 random f64
inputs. Asserts max abs diff < 1e-14 (target actual: machine epsilon, ~1e-16).

If this test fails:
  - Constant drift across all steps  -> flat-weight ordering mismatch
  - Growing drift over sequence       -> state update bug (A_bar, B_bar, or h update)
  - Only fails past step N            -> warm-up / state-init bug
  - Any NaN / Inf                     -> numerical stability regression in softplus
                                         or expm1_over_x
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


@pytest.mark.slow
def test_mamba_rust_python_equivalence_100_steps(tmp_path: Path) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(seed=1234)

    # 1. Build Python layers in f64; bypass build_layer which rejects Mamba (PPO gate).
    dense_in = DenseLayer(input_size=4, output_size=8, activation="tanh").double()
    mamba = MambaLayer(input_size=8, d_state=4, dt_rank=2).double()
    dense_out = DenseLayer(input_size=8, output_size=2, activation="linear").double()

    # Randomize weights with bounded values to keep outputs well-scaled.
    with torch.no_grad():
        for lin in [dense_in.linear, dense_out.linear]:
            torch.nn.init.uniform_(lin.weight, -0.3, 0.3)
            torch.nn.init.uniform_(lin.bias, -0.3, 0.3)
        torch.nn.init.uniform_(mamba.x_proj_w, -0.3, 0.3)
        torch.nn.init.uniform_(mamba.dt_proj_w, -0.3, 0.3)
        torch.nn.init.uniform_(mamba.dt_proj_b, -0.5, 0.5)   # center matters for softplus
        torch.nn.init.uniform_(mamba.a_log, 0.0, 2.0)         # HiPPO-ish range; A < 0 strongly
        mamba.d_skip.fill_(1.0)

    # 2. Serialize to v2 JSON.
    #    Dense layers use `w` / `b` keys (matching NnLayerWeights in Rust from_v2_json).
    #    Mamba uses `x_proj_w`, `dt_proj_w`, `dt_proj_b`, `a_log`, `d_skip`.
    #    Matrices are row-major lists-of-lists (same convention as Transformer test).
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
                # x_proj_w: (dt_rank + 2*d_state, input_size) = (10, 8) row-major
                "x_proj_w": mamba.x_proj_w.detach().tolist(),
                # dt_proj_w: (input_size, dt_rank) = (8, 2) row-major
                "dt_proj_w": mamba.dt_proj_w.detach().tolist(),
                # dt_proj_b: (input_size,) = (8,)
                "dt_proj_b": mamba.dt_proj_b.detach().tolist(),
                # a_log: (input_size, d_state) = (8, 4) row-major
                "a_log": mamba.a_log.detach().tolist(),
                # d_skip: (input_size,) = (8,)
                "d_skip": mamba.d_skip.detach().tolist(),
            },
            "layer_2": {
                "w": dense_out.linear.weight.detach().tolist(),
                "b": dense_out.linear.bias.detach().tolist(),
            },
        },
    }

    model_path = tmp_path / "mamba_model.json"
    model_path.write_text(json.dumps(model_json))

    # 3. 100 random f64 inputs of shape (100, 4).
    inputs = rng.standard_normal((100, 4)).astype(np.float64)

    # 4. Rust: stateful across all 100 steps via nn_forward_sequence.
    rust_outs = np.asarray(
        aerocapture_rs.nn_forward_sequence(
            str(model_path),
            [row.tolist() for row in inputs],
        ),
        dtype=np.float64,
    )
    assert rust_outs.shape == (100, 2), f"unexpected rust_out shape {rust_outs.shape}"

    # 5. Python: thread Mamba state h across all 100 steps.
    h_mamba = mamba.new_state()
    py_outs = np.empty((100, 2), dtype=np.float64)

    dense_in.eval()
    mamba.eval()
    dense_out.eval()

    with torch.no_grad():
        for t in range(100):
            x = torch.tensor(inputs[t], dtype=torch.float64)  # (4,)
            y0, _ = dense_in(x, None)                          # (8,)
            y1, h_mamba = mamba(y0, h_mamba)                   # (8,), (8, 4)
            y2, _ = dense_out(y1, None)                        # (2,)
            py_outs[t] = y2.numpy()

    # 6. Assert cross-language bit-equivalence.
    diff = np.abs(rust_outs - py_outs)
    max_diff = float(diff.max())
    print(f"Mamba cross-language max abs diff over 100 steps: {max_diff:.3e}")
    print(
        f"  step  0 diff: {float(np.max(np.abs(rust_outs[0]  - py_outs[0]))):.3e}"
        f"  step  1 diff: {float(np.max(np.abs(rust_outs[1]  - py_outs[1]))):.3e}"
        f"  step 99 diff: {float(np.max(np.abs(rust_outs[99] - py_outs[99]))):.3e}"
    )
    assert max_diff < 1e-14, (
        f"cross-language drift: {max_diff:.3e} >= 1e-14. "
        f"Diagnostic: "
        f"step-0 diff = {float(np.max(np.abs(rust_outs[0] - py_outs[0]))):.3e}, "
        f"step-1 diff = {float(np.max(np.abs(rust_outs[1] - py_outs[1]))):.3e}, "
        f"step-99 diff = {float(np.max(np.abs(rust_outs[99] - py_outs[99]))):.3e}."
    )
