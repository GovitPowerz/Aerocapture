"""Rust runtime <-> torch mirror equivalence, one row per layer type (`tests/nn_archs.py`).

Each row's mirror stack is written to a v2 model JSON (weights named by
`layer_schema`), driven 100 steps through `aerocapture_rs.nn_forward_sequence`
with ONE `NnState`, and compared against the torch mirror threading its own
state; the gate is the row's tolerance. The state-evolves / determinism checks
generalize the retired mamba warm-up test to every stateful row. The named
extras below are the genuinely per-type cases (stacked mamba, mamba under
PSO-realistic a_log, transformer cache growth, window zero-padding).

The `V2Policy` + `export_v2_policy_to_json` path (dense / gru / lstm) is a
different seam and stays in `test_v2_rust_python_equivalence.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.nn_archs import ARCHS, ArchCase, dense, init_dense, init_mamba_core, run_mirror, write_model_json

aerocapture_rs = pytest.importorskip("aerocapture_rs")

from aerocapture.training.torch_mirror.layers.dense import DenseLayer  # noqa: E402
from aerocapture.training.torch_mirror.layers.mamba import MambaLayer  # noqa: E402
from aerocapture.training.torch_mirror.layers.transformer import TransformerLayer  # noqa: E402
from aerocapture.training.torch_mirror.layers.window import WindowLayer  # noqa: E402

N_STEPS = 100


def _rows(stateful_only: bool = False) -> list[object]:
    return [pytest.param(c, id=c.name, marks=[pytest.mark.slow] if c.slow else []) for c in ARCHS.values() if c.stateful or not stateful_only]


def _rust_sequence(path: Path, inputs: np.ndarray) -> np.ndarray:
    return np.asarray(aerocapture_rs.nn_forward_sequence(str(path), [row.tolist() for row in inputs]), dtype=np.float64)


@pytest.mark.parametrize("case", _rows())
def test_stateful_100_steps(case: ArchCase, tmp_path: Path) -> None:
    modules = case.mirror()
    path = write_model_json(tmp_path / f"{case.name}.json", case.arch, modules)
    inputs = np.random.default_rng(1234).standard_normal((N_STEPS, case.n_inputs))

    rust_outs = _rust_sequence(path, inputs)
    assert rust_outs.shape == (N_STEPS, 2)
    py_outs = run_mirror(modules, inputs)

    per_step = np.abs(rust_outs - py_outs).max(axis=1)
    diff = float(per_step.max())
    print(f"{case.name} cross-language max abs diff over {N_STEPS} steps: {diff:.3e}")
    # Constant drift -> flat-weight / name ordering; growing drift -> state update; fails past step N -> warm-up.
    assert diff < case.tol, (
        f"{case.name}: drift {diff:.3e} >= {case.tol:g}; step-0 {per_step[0]:.3e}, step-1 {per_step[1]:.3e}, step-{N_STEPS - 1} {per_step[-1]:.3e}"
    )


@pytest.mark.parametrize("case", _rows(stateful_only=True))
def test_state_evolves(case: ArchCase, tmp_path: Path) -> None:
    """The same input twice must not give the same output twice: state accumulates."""
    path = write_model_json(tmp_path / f"{case.name}.json", case.arch, case.mirror())
    x = np.random.default_rng(3).uniform(-1.0, 1.0, size=case.n_inputs)
    outs = _rust_sequence(path, np.stack([x, x]))
    diff = float(np.abs(outs[0] - outs[1]).max())
    assert diff > 1e-10, f"{case.name}: step-0 and step-1 outputs equal on repeated input (diff={diff:.3e}); state is not evolving"


@pytest.mark.parametrize("case", _rows())
def test_deterministic(case: ArchCase, tmp_path: Path) -> None:
    path = write_model_json(tmp_path / f"{case.name}.json", case.arch, case.mirror())
    inputs = np.random.default_rng(0).uniform(-1.0, 1.0, size=(20, case.n_inputs))
    assert np.array_equal(_rust_sequence(path, inputs), _rust_sequence(path, inputs)), f"{case.name}: Rust forward is non-deterministic"


# ---- named extras: genuinely per-type ---------------------------------------


@pytest.mark.slow
def test_mamba_stacked_two_layers(tmp_path: Path) -> None:
    """The production 2x-Mamba stack: catches layer-index-dependent state-init /
    dt-bias-seed contamination that a single-layer row cannot."""
    torch.manual_seed(7)
    d0, m_a, m_b, d1 = dense(4, 8, "tanh"), MambaLayer(8, 4, 2).double(), MambaLayer(8, 4, 2).double(), dense(8, 2, "linear")
    modules: list[torch.nn.Module] = [d0, m_a, m_b, d1]
    with torch.no_grad():
        init_dense([d0, d1], 0.3, 0.3)
        init_mamba_core(m_a)
        init_mamba_core(m_b)
    mid = {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2}
    arch = [ARCHS["mamba"].arch[0], mid, mid, ARCHS["mamba"].arch[2]]
    path = write_model_json(tmp_path / "mamba_stacked.json", arch, modules)
    inputs = np.random.default_rng(5678).standard_normal((N_STEPS, 4))
    diff = float(np.abs(_rust_sequence(path, inputs) - run_mirror(modules, inputs)).max())
    print(f"stacked 2x mamba cross-language max abs diff: {diff:.3e}")
    assert diff < 1e-14, f"stacked-mamba drift {diff:.3e} >= 1e-14"


@pytest.mark.slow
def test_mamba_high_a_log_numerical_stability(tmp_path: Path) -> None:
    """PSO with bound_multiplier=2 around the HiPPO centers log(n+1) pushes a_log to ~5
    (A ~= -148), so exp(dt*A) underflows to 0. The Rust forward must stay finite and
    match the mirror in that regime."""
    torch.manual_seed(13)
    d0, m, d1 = dense(4, 8, "tanh"), MambaLayer(8, 4, 2).double(), dense(8, 2, "linear")
    modules: list[torch.nn.Module] = [d0, m, d1]
    with torch.no_grad():
        init_dense([d0, d1], 0.3, 0.3)
        init_mamba_core(m)
        torch.nn.init.uniform_(m.dt_proj_b, -3.0, 3.0)  # softplus output spans ~1e-3 .. ~10
        torch.nn.init.uniform_(m.a_log, 3.0, 5.0)  # A = -exp(a_log) down to -148
    path = write_model_json(tmp_path / "mamba_high_a_log.json", ARCHS["mamba"].arch, modules)
    inputs = np.random.default_rng(13).standard_normal((50, 4))
    rust_outs = _rust_sequence(path, inputs)
    assert np.all(np.isfinite(rust_outs)), f"Rust produced non-finite outputs under high a_log: {rust_outs}"
    py_outs = run_mirror(modules, inputs)
    assert np.all(np.isfinite(py_outs)), "Python mirror produced non-finite outputs"
    diff = float(np.abs(rust_outs - py_outs).max())
    print(f"high-a_log mamba cross-language max abs diff: {diff:.3e}")
    # The normal-regime gate is 1e-14 (observed ~1e-16). Under extreme a_log + wide
    # dt_proj_b the per-step magnitudes inside the SSM recurrence grow, and the Rust
    # scalar inner loop `acc += h[d,n] * c_vec[n]` vs Python `h @ c_vec` accumulate in
    # different orders. The FP associativity drift is bounded by
    # `d_state * eps * max|h|*max|c|`, which scales with the forcing. 1e-12 is still
    # 4 orders below any real signal; the primary invariant is "no NaN / Inf".
    assert diff < 1e-12, f"high-a_log drift {diff:.3e} >= 1e-12"


@pytest.mark.slow
def test_transformer_cache_warmup(tmp_path: Path) -> None:
    """The KV cache grows organically 0 -> n_seq with no zero-padding: driving fewer
    steps than n_seq must still agree with Rust, and the cache length must be t+1."""
    d_model, n_seq = 8, 4
    torch.manual_seed(1)
    tr = TransformerLayer(d_model=d_model, n_heads=2, d_ffn=16, n_seq=n_seq).double()
    d1 = dense(d_model, 2, "linear")
    with torch.no_grad():
        for lin in (tr.w_q, tr.w_k, tr.w_v, tr.w_o, tr.w_ffn1, tr.w_ffn2, d1.linear):
            torch.nn.init.uniform_(lin.weight, -0.1, 0.1)
            torch.nn.init.uniform_(lin.bias, -0.05, 0.05)
        # Default LN gamma=1, beta=0 stay.
    arch = [
        {"type": "transformer", "d_model": d_model, "n_heads": 2, "d_ffn": 16, "n_seq": n_seq},
        {"type": "dense", "input_size": d_model, "output_size": 2, "activation": "linear"},
    ]
    path = write_model_json(tmp_path / "transformer_warmup.json", arch, [tr, d1])
    inputs = np.random.default_rng(7).standard_normal((3, d_model))  # 3 < n_seq: warm-up phase only

    rust_outs = _rust_sequence(path, inputs)
    py_outs = np.empty_like(rust_outs)
    state = tr.new_state(batch_size=1)
    tr.eval()
    d1.eval()
    with torch.no_grad():
        for t in range(3):
            h, state = tr(torch.tensor(inputs[t : t + 1], dtype=torch.float64), state)
            y, _ = d1(h, None)
            py_outs[t] = y.squeeze(0).numpy()
            assert state[0].shape[1] == t + 1, f"step {t}: expected cache_len={t + 1}, got {state[0].shape[1]} (zero-padding leaked?)"
            assert state[1].shape[1] == t + 1
    diff = float(np.abs(rust_outs - py_outs).max())
    assert diff < 1e-10, f"warm-up cross-language mismatch: {diff:.2e}"


def test_window_buffer_warmup_zero_padded(tmp_path: Path) -> None:
    """Before n_steps ticks the window is zero-padded on both sides: a Dense that reads
    the OLDEST slot returns zeros for the first n_steps-1 ticks, then lags by n_steps-1."""
    input_size, n_steps = 2, 3
    reader = dense(input_size * n_steps, input_size, "linear")
    with torch.no_grad():
        reader.linear.weight.zero_()
        reader.linear.weight[0, 0] = 1.0  # buffer[0][0]
        reader.linear.weight[1, 1] = 1.0  # buffer[0][1]
        reader.linear.bias.zero_()
    arch = [
        {"type": "window", "input_size": input_size, "n_steps": n_steps},
        {"type": "dense", "input_size": input_size * n_steps, "output_size": input_size, "activation": "linear"},
    ]
    path = write_model_json(tmp_path / "window_warmup.json", arch, [WindowLayer(input_size=input_size, n_steps=n_steps).double(), reader])
    inputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    expected = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    rust_out = _rust_sequence(path, inputs)
    assert np.allclose(rust_out, expected, atol=1e-12), f"got {rust_out}"
