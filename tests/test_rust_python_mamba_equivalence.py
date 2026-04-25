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
        torch.nn.init.uniform_(mamba.dt_proj_b, -0.5, 0.5)  # center matters for softplus
        torch.nn.init.uniform_(mamba.a_log, 0.0, 2.0)  # HiPPO-ish range; A < 0 strongly
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
            y0, _ = dense_in(x, None)  # (8,)
            y1, h_mamba = mamba(y0, h_mamba)  # (8,), (8, 4)
            y2, _ = dense_out(y1, None)  # (2,)
            py_outs[t] = y2.numpy()

    # 6. Assert cross-language bit-equivalence.
    diff = np.abs(rust_outs - py_outs)
    max_diff = float(diff.max())
    print(f"Mamba cross-language max abs diff over 100 steps: {max_diff:.3e}")
    print(
        f"  step  0 diff: {float(np.max(np.abs(rust_outs[0] - py_outs[0]))):.3e}"
        f"  step  1 diff: {float(np.max(np.abs(rust_outs[1] - py_outs[1]))):.3e}"
        f"  step 99 diff: {float(np.max(np.abs(rust_outs[99] - py_outs[99]))):.3e}"
    )
    assert max_diff < 1e-14, (
        f"cross-language drift: {max_diff:.3e} >= 1e-14. "
        f"Diagnostic: "
        f"step-0 diff = {float(np.max(np.abs(rust_outs[0] - py_outs[0]))):.3e}, "
        f"step-1 diff = {float(np.max(np.abs(rust_outs[1] - py_outs[1]))):.3e}, "
        f"step-99 diff = {float(np.max(np.abs(rust_outs[99] - py_outs[99]))):.3e}."
    )


@pytest.mark.slow
def test_mamba_rust_python_equivalence_stacked_2_layers(tmp_path: Path) -> None:
    """Verify cross-language equivalence for the production 2x-Mamba architecture.

    Catches layer-index-dependent bugs that single-layer tests miss, especially
    any state-init / dt-bias-seed contamination across stacked Mamba layers.
    """
    torch.manual_seed(7)
    rng = np.random.default_rng(seed=5678)

    dense_in = DenseLayer(input_size=4, output_size=8, activation="tanh").double()
    mamba_a = MambaLayer(input_size=8, d_state=4, dt_rank=2).double()
    mamba_b = MambaLayer(input_size=8, d_state=4, dt_rank=2).double()
    dense_out = DenseLayer(input_size=8, output_size=2, activation="linear").double()

    with torch.no_grad():
        for lin in [dense_in.linear, dense_out.linear]:
            torch.nn.init.uniform_(lin.weight, -0.3, 0.3)
            torch.nn.init.uniform_(lin.bias, -0.3, 0.3)
        for m in (mamba_a, mamba_b):
            torch.nn.init.uniform_(m.x_proj_w, -0.3, 0.3)
            torch.nn.init.uniform_(m.dt_proj_w, -0.3, 0.3)
            torch.nn.init.uniform_(m.dt_proj_b, -0.5, 0.5)
            torch.nn.init.uniform_(m.a_log, 0.0, 2.0)
            m.d_skip.fill_(1.0)

    model_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2},
            {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2},
            {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {
                "w": dense_in.linear.weight.detach().tolist(),
                "b": dense_in.linear.bias.detach().tolist(),
            },
            "layer_1": {
                "x_proj_w": mamba_a.x_proj_w.detach().tolist(),
                "dt_proj_w": mamba_a.dt_proj_w.detach().tolist(),
                "dt_proj_b": mamba_a.dt_proj_b.detach().tolist(),
                "a_log": mamba_a.a_log.detach().tolist(),
                "d_skip": mamba_a.d_skip.detach().tolist(),
            },
            "layer_2": {
                "x_proj_w": mamba_b.x_proj_w.detach().tolist(),
                "dt_proj_w": mamba_b.dt_proj_w.detach().tolist(),
                "dt_proj_b": mamba_b.dt_proj_b.detach().tolist(),
                "a_log": mamba_b.a_log.detach().tolist(),
                "d_skip": mamba_b.d_skip.detach().tolist(),
            },
            "layer_3": {
                "w": dense_out.linear.weight.detach().tolist(),
                "b": dense_out.linear.bias.detach().tolist(),
            },
        },
    }

    model_path = tmp_path / "mamba_stacked_model.json"
    model_path.write_text(json.dumps(model_json))

    inputs = rng.standard_normal((100, 4)).astype(np.float64)
    rust_outs = np.asarray(
        aerocapture_rs.nn_forward_sequence(str(model_path), [row.tolist() for row in inputs]),
        dtype=np.float64,
    )
    assert rust_outs.shape == (100, 2)

    h_a = mamba_a.new_state()
    h_b = mamba_b.new_state()
    py_outs = np.empty((100, 2), dtype=np.float64)
    for layer in (dense_in, mamba_a, mamba_b, dense_out):
        layer.eval()
    with torch.no_grad():
        for t in range(100):
            x = torch.tensor(inputs[t], dtype=torch.float64)
            y0, _ = dense_in(x, None)
            y1, h_a = mamba_a(y0, h_a)
            y2, h_b = mamba_b(y1, h_b)
            y3, _ = dense_out(y2, None)
            py_outs[t] = y3.numpy()

    diff = np.abs(rust_outs - py_outs).max()
    print(f"Stacked 2x Mamba cross-language max abs diff over 100 steps: {float(diff):.3e}")
    assert diff < 1e-14, f"stacked-Mamba drift: {float(diff):.3e} >= 1e-14"


@pytest.mark.slow
def test_mamba_high_a_log_numerical_stability(tmp_path: Path) -> None:
    """Exercise the PSO-realistic high-exploration range for a_log.

    Default proptest uses a_log in [-1, 1]; PSO with bound_multiplier=2 around
    HiPPO centers `log(n+1)` can easily push a_log to ~5, giving A ~= -148 and
    discretization terms `exp(Δ*A) ~= exp(-300)` which underflow to 0. Verify
    the Rust forward stays finite (no NaN/Inf propagation) and matches the
    Python mirror bit-identically in this regime.
    """
    torch.manual_seed(13)
    rng = np.random.default_rng(seed=13)

    dense_in = DenseLayer(input_size=4, output_size=8, activation="tanh").double()
    mamba = MambaLayer(input_size=8, d_state=4, dt_rank=2).double()
    dense_out = DenseLayer(input_size=8, output_size=2, activation="linear").double()

    with torch.no_grad():
        for lin in [dense_in.linear, dense_out.linear]:
            torch.nn.init.uniform_(lin.weight, -0.3, 0.3)
            torch.nn.init.uniform_(lin.bias, -0.3, 0.3)
        torch.nn.init.uniform_(mamba.x_proj_w, -0.3, 0.3)
        torch.nn.init.uniform_(mamba.dt_proj_w, -0.3, 0.3)
        # Wide dt_proj_b -> softplus output can span ~1e-3 to ~10.
        torch.nn.init.uniform_(mamba.dt_proj_b, -3.0, 3.0)
        # Extreme a_log values: exp(5) = 148, so A = -148 elementwise.
        # ZOH: exp(Δ * A) with Δ ~ 10 -> exp(-1480) -> underflow to 0.
        torch.nn.init.uniform_(mamba.a_log, 3.0, 5.0)
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
    model_path = tmp_path / "mamba_high_a_log.json"
    model_path.write_text(json.dumps(model_json))

    inputs = rng.standard_normal((50, 4)).astype(np.float64)
    rust_outs = np.asarray(
        aerocapture_rs.nn_forward_sequence(str(model_path), [row.tolist() for row in inputs]),
        dtype=np.float64,
    )
    assert rust_outs.shape == (50, 2)
    assert np.all(np.isfinite(rust_outs)), f"Rust produced non-finite outputs under high a_log: {rust_outs}"

    # Python mirror must agree.
    h = mamba.new_state()
    py_outs = np.empty((50, 2), dtype=np.float64)
    for layer in (dense_in, mamba, dense_out):
        layer.eval()
    with torch.no_grad():
        for t in range(50):
            x = torch.tensor(inputs[t], dtype=torch.float64)
            y0, _ = dense_in(x, None)
            y1, h = mamba(y0, h)
            y2, _ = dense_out(y1, None)
            py_outs[t] = y2.numpy()
    assert np.all(np.isfinite(py_outs)), "Python mirror produced non-finite outputs"

    diff = np.abs(rust_outs - py_outs).max()
    print(f"High-a_log Mamba cross-language max abs diff: {float(diff):.3e}")
    # The normal-regime gate is 1e-14 (observed ~1e-16 in the baseline
    # equivalence test). Under extreme a_log + wide dt_proj_b, the per-step
    # magnitudes inside the SSM recurrence grow, and the Rust scalar inner
    # loop `acc += h[d,n] * c_vec[n]` vs Python `h @ c_vec` accumulate in
    # different orders. The resulting FP associativity drift is bounded by
    # `d_state * eps * max|h|*max|c|`, which scales with the forcing. We
    # accept 1e-12 here (still 4 orders of magnitude below any real signal);
    # the primary invariant is "no NaN / Inf propagation", asserted above.
    assert diff < 1e-12, f"high-a_log drift: {float(diff):.3e} >= 1e-12"
