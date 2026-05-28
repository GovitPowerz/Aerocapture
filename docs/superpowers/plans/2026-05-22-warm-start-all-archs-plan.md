# Warm-Start for All NN Architectures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `aerocapture.training.warm_start` with a multi-supervisor (FTC + friends), all-architecture (dense / window / gru / lstm / transformer / mamba), per-trajectory chunked-BPTT pipeline targeting the signed final commanded bank, with per-algorithm initial-population seeding and conditional bound widening.

**Architecture:** Rust `collect_supervised` changes its capture point to the dispatch-level signed bank and returns per-trajectory `list[dict]` (seed + X + y_signed + dv + captured). Python orchestrates `for scheme in supervisor_schemes: collect; select best per seed; chunked BPTT via V2Policy.evaluate-style sequence forward; per-layer `to_flat()` mirrors Rust `LayerWeights::to_flat`; encode to normalized chromosome at `bound_multiplier=4.0` when warm-start is on; `train.py` dispatches per-algorithm seeding (replicate+jitter for GA/DE/PSO, seed mean + shrunken sigma0 for CMA-ES). Gen-0 validation MC baseline logged before the first algorithm step.

**Tech Stack:** Rust + PyO3 + nalgebra, Python 3.14, PyTorch (autograd for BPTT), pymoo (GA/CMA-ES/DE/PSO), pytest + hypothesis, `aerocapture_rs` PyO3 module.

**Spec:** `docs/superpowers/specs/2026-05-22-warm-start-all-archs-design.md`

---

## File Map

**Rust (modify):**
- `src/rust/src/simulation/tick.rs` — change supervised_trace push payload from `pre_lateral_magnitude` to `bank_angle` (signed final command).
- `src/rust/src/simulation/runner.rs` — extend `SimResult` (or equivalent) to carry per-sim `dv_total_m_s` and `captured` alongside `supervised_trace` (likely already present in `BatchResults` — verify).
- `src/rust/aerocapture-py/src/lib.rs` — change `collect_supervised` return shape from `(PyArray2, PyArray1)` to `Vec<PyDict>` (one dict per seed).

**Python (modify):**
- `src/python/aerocapture/training/rl/layers/__init__.py` — remove `NotImplementedError` for Window / Transformer / Mamba in `build_layer`; import `MambaLayer`.
- `src/python/aerocapture/training/rl/layers/dense.py` — add `to_flat()`.
- `src/python/aerocapture/training/rl/layers/gru.py` — add `to_flat()`.
- `src/python/aerocapture/training/rl/layers/lstm.py` — add `to_flat()`.
- `src/python/aerocapture/training/rl/layers/window.py` — add `to_flat()` (zero-param no-op); standardize `new_state(batch_size, device)` signature.
- `src/python/aerocapture/training/rl/layers/transformer.py` — add `to_flat()`; standardize `new_state(batch_size, device)`.
- `src/python/aerocapture/training/rl/layers/mamba.py` — add `to_flat()`; add batched `forward_batched(x, h)` + standardized `new_state(batch_size, device)`.
- `src/python/aerocapture/training/rl/policy.py` — extend `_zero_entry` to handle multi-dim Tensors (broadcast keep_bool to full rank); add `V2Policy.forward_seq_means(obs_seq, state_0, dones_seq) -> Tensor[T, B, out_dim]`.
- `src/python/aerocapture/training/config.py` — add `WarmStartConfig` dataclass and `TrainingConfig.warm_start` field.
- `src/python/aerocapture/training/warm_start.py` — rewrite: `_policy_to_flat_weights_v2`, `_select_best_teacher_per_seed`, `_chunked_bptt_train`, `build_warm_start_chromosome` with multi-supervisor + cache-key extensions + per-epoch loss logging.
- `src/python/aerocapture/training/train.py` — replace warm-start block (~lines 580-606): per-algorithm seeding, conditional bound widening, gen-0 validation baseline.
- Training TOML configs (one `[warm_start]` block per NN scheme): `configs/training/msr_aller_nn_train_consolidated.toml`, `msr_aller_nn_joint_train.toml`, `msr_aller_gru_pso_train.toml`, `msr_aller_gru_pso_magonly_train.toml`, `msr_aller_lstm_pso_train.toml`, `msr_aller_window_pso_train.toml`, `msr_aller_transformer_pso_train.toml`, `msr_aller_mamba_pso_train.toml`.

**Python (create):**
- `tests/test_warm_start_v2_to_flat_roundtrip.py` — Python `to_flat` → Rust `flat_weights_to_json` → Rust `nn_forward` parity per layer type.
- `tests/test_warm_start_selection.py` — `_select_best_teacher_per_seed` synthetic.
- `tests/test_warm_start_optimizer_seeding.py` — per-algorithm seeding shape/mean/std.
- `tests/test_warm_start_failures.py` — all failure modes from spec.
- `tests/test_warm_start_cache.py` — cache hit/miss/invalidation matrix.
- `tests/test_warm_start_per_arch.py` — slow per-architecture E2E.
- `tests/test_warm_start_validation_baseline.py` — slow E2E gen-0 baseline log.
- `tests/test_warm_start_equivalence_gate.py` — slow magnitude_only "at least as good" regression.

---

## Task 1: Rust — change `collect_supervised` capture point and return shape

**Files:**
- Modify: `src/rust/src/simulation/tick.rs` (around line 167, the `supervised_trace.push` call)
- Modify: `src/rust/src/simulation/runner.rs` (verify SimResult has `dv_total_m_s` + `captured`)
- Modify: `src/rust/aerocapture-py/src/lib.rs` (`collect_supervised` function, ~line 400-485)
- Test: extend existing `tests/test_v2_rust_python_equivalence.py` or add inline assertion

- [ ] **Step 1: Read the current supervised_trace push site in tick.rs**

Run: `rg -n "supervised_trace" src/rust/src/simulation/tick.rs`

Confirm the push is `supervised_trace.push((nn_input, guidance_out.pre_lateral_magnitude))`. The replacement uses `guidance_out.bank_angle` (the final signed dispatch output, after thermal limiter / lateral / command shaper).

- [ ] **Step 2: Change the capture point in tick.rs**

In `src/rust/src/simulation/tick.rs`, locate the supervised_trace push and change the second tuple field:

```rust
state
    .supervised_trace
    .push((nn_input, guidance_out.bank_angle));
```

The signed value is what the pilot integrates. `magnitude_only` derives the unsigned target Python-side via `.abs()`.

- [ ] **Step 3: Verify `BatchResult` / per-sim output already carries `dv_total_m_s` and `captured`**

Run: `rg -n "dv_total_m_s|captured:" src/rust/src/simulation/runner.rs | head -30`

Both fields exist on `BatchResult` (used by `run_batch`). If `collect_supervised`'s per-sim outputs don't already expose them, find where `outputs` is iterated in `aerocapture-py/src/lib.rs::collect_supervised` and read them off `output.final_record` (column index for `dv_total_m_s`) and `output.captured`.

- [ ] **Step 4: Rewrite the `collect_supervised` return shape in `aerocapture-py/src/lib.rs`**

Replace the existing return-builder block (lines ~470-485) with per-seed dict construction:

```rust
use pyo3::types::PyDict;

let result_list = pyo3::types::PyList::empty(py);
for (seed_idx, output) in outputs.iter().enumerate() {
    let n_steps = output.supervised_trace.len();
    let mut x_rows: Vec<Vec<f64>> = Vec::with_capacity(n_steps);
    let mut y_signed: Vec<f64> = Vec::with_capacity(n_steps);
    for (nn_input, bank) in &output.supervised_trace {
        x_rows.push(nn_input.clone());
        y_signed.push(*bank);
    }
    let x_array = numpy::PyArray2::from_vec2(py, &x_rows)?;
    let y_array = numpy::PyArray1::from_vec(py, y_signed);

    let dict = PyDict::new(py);
    dict.set_item("seed", seeds[seed_idx])?;
    dict.set_item("X", x_array)?;
    dict.set_item("y_signed", y_array)?;
    dict.set_item("dv", output.dv_total_m_s)?;
    dict.set_item("captured", output.captured)?;
    result_list.append(dict)?;
}
Ok(result_list.unbind().into())
```

Update the function signature return type from `PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)>` to `PyResult<Py<PyList>>`.

The outer loop that pushed `output.supervised_trace` items into one big `all_x_rows` / `all_y` is replaced. The `outputs` vec is now iterated to construct one dict per result; we use the parallel `seeds` vec for the seed key (index alignment matches the `for seed in seeds` driving loop above).

- [ ] **Step 5: Rebuild the PyO3 module**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`
Expected: clean build, `aerocapture_rs` reinstalled.

- [ ] **Step 6: Quick smoke from Python REPL**

Run:
```bash
uv run python -c "
import aerocapture_rs as r
out = r.collect_supervised(
    'configs/training/msr_aller_ftc_train.toml',
    seeds=[42, 43],
    scheme='ftc',
)
assert isinstance(out, list) and len(out) == 2, out
for d in out:
    assert set(d.keys()) == {'seed', 'X', 'y_signed', 'dv', 'captured'}, d.keys()
    assert d['X'].shape[1] == 21
    assert d['X'].shape[0] == d['y_signed'].shape[0]
print('OK:', [(d[\"seed\"], d[\"X\"].shape, d[\"captured\"], d[\"dv\"]) for d in out])
"
```
Expected: prints two seeds, finite DV values, X.shape[1] == 21.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/simulation/tick.rs src/rust/aerocapture-py/src/lib.rs
git commit -m "feat(rust): collect_supervised returns per-trajectory signed bank dicts

Change supervised_trace capture from pre_lateral_magnitude to the signed
final bank_angle after thermal limiter + lateral + command shaper.
PyO3 binding returns list[dict] per seed with X, y_signed, dv, captured —
trajectory boundaries preserved for downstream BPTT training.

Only caller is warm_start.py (updated in Task 11)."
```

---

## Task 2: Python — enable Window / Transformer / Mamba in `build_layer` + standardize `new_state` signature

**Files:**
- Modify: `src/python/aerocapture/training/rl/layers/__init__.py` (`build_layer` dispatch + Mamba import)
- Modify: `src/python/aerocapture/training/rl/layers/window.py` (signature already `(batch_size, device=None)`, OK)
- Modify: `src/python/aerocapture/training/rl/layers/transformer.py` (`new_state(batch_size)` → `new_state(batch_size, device=None)`)
- Modify: `src/python/aerocapture/training/rl/layers/mamba.py` (`new_state()` → `new_state(batch_size, device=None)`, batched zero-init)
- Test: `tests/test_warm_start_build_layer.py`

- [ ] **Step 1: Write a failing test that build_layer accepts all six layer types**

Create `tests/test_warm_start_build_layer.py`:

```python
"""build_layer accepts all six layer specs for warm-start training."""

import pytest

from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import (
    DenseSpec, GruSpec, LstmSpec, MambaSpec, TransformerSpec, WindowSpec,
)


@pytest.mark.parametrize(
    "spec",
    [
        DenseSpec(input_size=4, output_size=2, activation="tanh"),
        GruSpec(input_size=4, hidden_size=8),
        LstmSpec(input_size=4, hidden_size=8),
        WindowSpec(input_size=4, n_steps=3),
        TransformerSpec(d_model=8, n_heads=2, d_ffn=16, n_seq=4),
        MambaSpec(input_size=8, d_state=4, dt_rank=2),
    ],
)
def test_build_layer_constructs_all_types(spec):
    layer = build_layer(spec)
    assert layer is not None


@pytest.mark.parametrize(
    "spec",
    [
        DenseSpec(input_size=4, output_size=2, activation="tanh"),
        GruSpec(input_size=4, hidden_size=8),
        LstmSpec(input_size=4, hidden_size=8),
        WindowSpec(input_size=4, n_steps=3),
        TransformerSpec(d_model=8, n_heads=2, d_ffn=16, n_seq=4),
        MambaSpec(input_size=8, d_state=4, dt_rank=2),
    ],
)
def test_new_state_accepts_batch_size_and_device(spec):
    layer = build_layer(spec)
    state = layer.new_state(batch_size=2, device=None)
    assert state is None or state is not None  # contract: callable without error
```

- [ ] **Step 2: Run the test — expect NotImplementedError on Window / Transformer / Mamba**

Run: `uv run pytest tests/test_warm_start_build_layer.py -v`
Expected: `test_build_layer_constructs_all_types[WindowSpec...]`, `[TransformerSpec...]`, `[MambaSpec...]` fail with `NotImplementedError`. `new_state` cases also fail for Transformer / Mamba due to signature mismatch.

- [ ] **Step 3: Remove the NotImplementedError gates and import MambaLayer**

Edit `src/python/aerocapture/training/rl/layers/__init__.py`:

```python
"""Torch mirrors of Rust layer types. One file per layer variant."""

from __future__ import annotations

from torch import nn

from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.layers.gru import GruLayer
from aerocapture.training.rl.layers.lstm import LstmLayer
from aerocapture.training.rl.layers.mamba import MambaLayer
from aerocapture.training.rl.layers.transformer import TransformerLayer
from aerocapture.training.rl.layers.window import WindowLayer
from aerocapture.training.rl.schemas import (
    DenseSpec,
    GruSpec,
    LayerSpec,
    LstmSpec,
    MambaSpec,
    TransformerSpec,
    WindowSpec,
)

__all__ = [
    "DenseLayer", "GruLayer", "LstmLayer", "MambaLayer",
    "TransformerLayer", "WindowLayer", "build_layer",
]


def build_layer(spec: LayerSpec) -> nn.Module:
    """Dispatch a LayerSpec to its torch module constructor."""
    if isinstance(spec, DenseSpec):
        return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    if isinstance(spec, GruSpec):
        return GruLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, LstmSpec):
        return LstmLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, WindowSpec):
        return WindowLayer(spec.input_size, spec.n_steps)
    if isinstance(spec, TransformerSpec):
        return TransformerLayer(spec.d_model, spec.n_heads, spec.d_ffn, spec.n_seq)
    if isinstance(spec, MambaSpec):
        return MambaLayer(spec.input_size, spec.d_state, spec.dt_rank)
    raise ValueError(f"Unknown layer spec: {spec!r}")
```

- [ ] **Step 4: Standardize `TransformerLayer.new_state` signature**

In `src/python/aerocapture/training/rl/layers/transformer.py`, change:

```python
def new_state(self, batch_size: int, device: object | None = None) -> tuple[Tensor, Tensor]:
    target_device = device if device is not None else self.w_q.weight.device
    dtype = self.w_q.weight.dtype
    empty = torch.zeros(batch_size, 0, self.d_model, device=target_device, dtype=dtype)
    return (empty.clone(), empty.clone())
```

- [ ] **Step 5: Standardize `MambaLayer.new_state` signature to batched**

In `src/python/aerocapture/training/rl/layers/mamba.py`, replace `new_state` with:

```python
def new_state(self, batch_size: int, device: object | None = None) -> Tensor:
    """Return zero-initialized batched state (batch_size, input_size, d_state)."""
    target_device = device if device is not None else self.x_proj_w.device
    return torch.zeros(
        batch_size,
        self.input_size,
        self.d_state,
        dtype=self.x_proj_w.dtype,
        device=target_device,
    )
```

Note the new 3D shape `(B, input_size, d_state)`. The unbatched contract `forward(x: (input_size,), h: (input_size, d_state))` is preserved for the existing cross-language equivalence test (which calls it directly without going through V2Policy). Task 3 adds the batched forward.

- [ ] **Step 6: Re-run the test**

Run: `uv run pytest tests/test_warm_start_build_layer.py -v`
Expected: all 12 cases pass.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/rl/layers/__init__.py \
        src/python/aerocapture/training/rl/layers/transformer.py \
        src/python/aerocapture/training/rl/layers/mamba.py \
        tests/test_warm_start_build_layer.py
git commit -m "feat(layers): enable Window/Transformer/Mamba in build_layer; standardize new_state(batch_size, device)

Required by warm-start training, which constructs V2Policy via build_layer
for all six architecture types. Mamba new_state returns batched (B, *) zeros
to align with the GRU/LSTM contract; the unbatched forward signature stays
intact for the existing cross-language equivalence test."
```

---

## Task 3: Python — batched Mamba forward

**Files:**
- Modify: `src/python/aerocapture/training/rl/layers/mamba.py` (add `forward` with batched shape; rename old unbatched to `forward_unbatched` and call from `forward` when batch dim is absent — see Step 3)
- Test: `tests/test_mamba_batched_forward.py`

- [ ] **Step 1: Write failing test for batched forward**

Create `tests/test_mamba_batched_forward.py`:

```python
"""MambaLayer.forward accepts batched (B, input_size) and matches unbatched."""

import torch
import pytest

from aerocapture.training.rl.layers import MambaLayer


@pytest.fixture
def mamba():
    torch.manual_seed(0)
    layer = MambaLayer(input_size=8, d_state=4, dt_rank=2).double()
    # Randomize params so the test exercises non-trivial weights
    with torch.no_grad():
        for p in layer.parameters():
            p.uniform_(-0.1, 0.1)
    return layer


def test_batched_forward_matches_unbatched(mamba):
    B = 3
    xs = torch.randn(B, 8, dtype=torch.float64)
    h_batched = mamba.new_state(batch_size=B)
    assert h_batched.shape == (B, 8, 4)

    # Unbatched: loop one at a time
    expected_y = []
    expected_h_new = []
    for b in range(B):
        # Use a fresh unbatched state matching the batched zero-init
        y_b, h_b_new = mamba.forward_unbatched(xs[b], h_batched[b])
        expected_y.append(y_b)
        expected_h_new.append(h_b_new)
    expected_y_stack = torch.stack(expected_y, dim=0)
    expected_h_stack = torch.stack(expected_h_new, dim=0)

    # Batched call
    y_batched, h_new_batched = mamba.forward(xs, h_batched)
    assert y_batched.shape == (B, 8)
    assert h_new_batched.shape == (B, 8, 4)
    assert torch.allclose(y_batched, expected_y_stack, atol=1e-14)
    assert torch.allclose(h_new_batched, expected_h_stack, atol=1e-14)
```

- [ ] **Step 2: Run — expect AttributeError on forward_unbatched + shape error**

Run: `uv run pytest tests/test_mamba_batched_forward.py -v`
Expected: fail.

- [ ] **Step 3: Rename existing forward to `forward_unbatched`; add batched `forward`**

In `src/python/aerocapture/training/rl/layers/mamba.py`:

Rename the existing single-step `forward(self, x: Tensor, h: Tensor)` to `forward_unbatched(self, x, h)` (preserves the existing cross-language equivalence test's call path).

Add new batched `forward`:

```python
def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
    """Batched single-step forward.

    Args:
        x: (batch, input_size) input vectors.
        h: (batch, input_size, d_state) per-env state.

    Returns:
        y:     (batch, input_size) output vectors.
        h_new: (batch, input_size, d_state) updated state.
    """
    if x.ndim == 1:
        # Unbatched fallback for the cross-language equivalence test.
        return self.forward_unbatched(x, h)
    B = x.shape[0]
    assert x.shape == (B, self.input_size)
    assert h.shape == (B, self.input_size, self.d_state)

    # 1. x_proj: (dt_rank + 2*d_state, input_size) @ x.T -> (dt_rank + 2*d_state, B)
    proj = x @ self.x_proj_w.t()  # (B, dt_rank + 2*d_state)
    dt_pre = proj[:, : self.dt_rank]                                       # (B, dt_rank)
    b_vec = proj[:, self.dt_rank : self.dt_rank + self.d_state]            # (B, d_state)
    c_vec = proj[:, self.dt_rank + self.d_state : self.dt_rank + 2 * self.d_state]  # (B, d_state)

    # 2. dt_proj + softplus
    dt_lifted = dt_pre @ self.dt_proj_w.t() + self.dt_proj_b               # (B, input_size)
    delta = _softplus(dt_lifted)

    # 3. ZOH discretization
    a = -torch.exp(self.a_log)                                             # (input_size, d_state)
    # za: (B, input_size, d_state) = delta(B, input_size, 1) * a(1, input_size, d_state)
    za = delta.unsqueeze(-1) * a.unsqueeze(0)
    a_bar = torch.exp(za)
    # b_bar: (B, input_size, d_state) = delta(B, in, 1) * b_vec(B, 1, d_state) * expm1_over_x(za)
    b_bar = delta.unsqueeze(-1) * b_vec.unsqueeze(1) * _expm1_over_x(za)
    # h_new: (B, input_size, d_state)
    h_new = a_bar * h + b_bar * x.unsqueeze(-1)
    # y: (B, input_size) = sum over d_state of (h_new * c_vec)
    y = (h_new * c_vec.unsqueeze(1)).sum(dim=-1) + self.d_skip * x
    return y, h_new
```

- [ ] **Step 4: Re-run test**

Run: `uv run pytest tests/test_mamba_batched_forward.py -v`
Expected: pass at machine epsilon (< 1e-14).

- [ ] **Step 5: Verify cross-language equivalence test still passes**

Run: `uv run pytest tests/test_v2_rust_python_equivalence.py -v -k mamba`
Expected: still passes (unbatched forward path unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/rl/layers/mamba.py tests/test_mamba_batched_forward.py
git commit -m "feat(layers): MambaLayer batched forward for warm-start BPTT

forward(x: (B, input_size), h: (B, input_size, d_state)) matches the
unbatched forward (now forward_unbatched, preserved for the cross-language
equivalence test) row-by-row at machine epsilon."
```

---

## Task 4: Python — extend `_zero_state_where_done` for multi-dim Tensor and Transformer KV cache

**Files:**
- Modify: `src/python/aerocapture/training/rl/policy.py` (`_zero_entry`, lines 163-174)
- Test: `tests/test_zero_state_where_done.py`

- [ ] **Step 1: Write failing test for Mamba (3D) and Transformer (tuple of 3D)**

Create `tests/test_zero_state_where_done.py`:

```python
"""_zero_state_where_done handles Dense (None), GRU/LSTM (2D / tuple-2D),
Window (3D), Mamba (3D), Transformer KV cache (tuple of 3D)."""

import torch
import pytest

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
```

- [ ] **Step 2: Run — expect failures on Mamba/Window/Transformer cases**

Run: `uv run pytest tests/test_zero_state_where_done.py -v`
Expected: Mamba/Window/Transformer cases fail because `keep_bool` has shape `(B, 1)` and won't broadcast correctly with `(B, *, *)`.

- [ ] **Step 3: Fix `_zero_entry` to broadcast to arbitrary rank**

In `src/python/aerocapture/training/rl/policy.py`, replace the `_zero_entry` function:

```python
def _zero_entry(s: Any, keep_bool: Tensor) -> Any:
    if s is None:
        return None
    if isinstance(s, Tensor):
        # keep_bool starts as (B, 1); reshape to (B, 1, 1, ..., 1) to broadcast
        # against any trailing dims (Mamba 3D, Window 3D, Transformer KV-cache 3D, ...).
        extra = s.ndim - keep_bool.ndim
        if extra > 0:
            shape = keep_bool.shape + (1,) * extra
            broadcast = keep_bool.view(shape)
        else:
            broadcast = keep_bool
        return s * broadcast.to(dtype=s.dtype, device=s.device)
    if isinstance(s, tuple):
        return tuple(_zero_entry(sub, keep_bool) for sub in s)
    raise TypeError(
        f"_zero_state_where_done: unsupported state entry type {type(s).__name__!r}; "
        "only None, Tensor, or tuple supported."
    )
```

- [ ] **Step 4: Re-run test**

Run: `uv run pytest tests/test_zero_state_where_done.py -v`
Expected: all 6 cases pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/policy.py tests/test_zero_state_where_done.py
git commit -m "fix(policy): _zero_state_where_done broadcasts keep_bool to arbitrary rank

Mamba state is (B, input_size, d_state), Window is (B, n_steps, input_size),
Transformer KV-cache halves are (B, cache_len, d_model). Old code's
keep_bool (B, 1) only broadcast correctly against 2D states. Now reshapes
to (B, 1, 1, ..., 1) per state rank."
```

---

## Task 5: Python — add `to_flat()` method to all six layer modules + cross-language round-trip test

**Files:**
- Modify: `src/python/aerocapture/training/rl/layers/dense.py`
- Modify: `src/python/aerocapture/training/rl/layers/gru.py`
- Modify: `src/python/aerocapture/training/rl/layers/lstm.py`
- Modify: `src/python/aerocapture/training/rl/layers/window.py`
- Modify: `src/python/aerocapture/training/rl/layers/transformer.py`
- Modify: `src/python/aerocapture/training/rl/layers/mamba.py`
- Test: `tests/test_warm_start_v2_to_flat_roundtrip.py`

- [ ] **Step 1: Write failing test for `to_flat` per layer type**

Create `tests/test_warm_start_v2_to_flat_roundtrip.py`:

```python
"""Per-layer to_flat() matches Rust LayerWeights::to_flat round-trip.

For each layer: build a Python instance with randomized weights, extract
via to_flat(), serialize to v2 JSON via aerocapture_rs.flat_weights_to_json,
load back via aerocapture_rs.nn_forward, assert finite + matches Python
forward at <1e-10.
"""

import json
import tempfile
from pathlib import Path

import aerocapture_rs as r
import numpy as np
import pytest
import torch

from aerocapture.training.rl.layers import (
    DenseLayer, GruLayer, LstmLayer, MambaLayer, TransformerLayer, WindowLayer,
)


def _randomize(layer):
    with torch.no_grad():
        for p in layer.parameters():
            p.uniform_(-0.1, 0.1)


def test_dense_to_flat_shape():
    layer = DenseLayer(input_size=3, output_size=4, activation="tanh").double()
    _randomize(layer)
    flat = layer.to_flat()
    assert flat.shape == (3 * 4 + 4,)  # W + b
    assert flat.dtype == np.float64


def test_gru_to_flat_shape():
    layer = GruLayer(input_size=3, hidden_size=4).double()
    _randomize(layer)
    flat = layer.to_flat()
    assert flat.shape == (3 * 4 * 3 + 3 * 4 * 4 + 2 * 3 * 4,)  # ih + hh + 2 biases


def test_lstm_to_flat_shape():
    layer = LstmLayer(input_size=3, hidden_size=4).double()
    _randomize(layer)
    flat = layer.to_flat()
    assert flat.shape == (4 * 4 * 3 + 4 * 4 * 4 + 2 * 4 * 4,)


def test_window_to_flat_empty():
    layer = WindowLayer(input_size=3, n_steps=4).double()
    flat = layer.to_flat()
    assert flat.shape == (0,)
    assert flat.dtype == np.float64


def test_transformer_to_flat_shape():
    d_model, n_heads, d_ffn, n_seq = 8, 2, 16, 4
    layer = TransformerLayer(d_model, n_heads, d_ffn, n_seq).double()
    _randomize(layer)
    flat = layer.to_flat()
    # 4 projections (W+b each, d_model x d_model + d_model) = 4*(d_model^2 + d_model)
    # 2 FFN linears: (d_model x d_ffn + d_ffn) + (d_ffn x d_model + d_model)
    # 2 layer norms: 2*(d_model + d_model) = 4*d_model
    expected = (
        4 * (d_model * d_model + d_model)
        + (d_model * d_ffn + d_ffn) + (d_ffn * d_model + d_model)
        + 4 * d_model
    )
    assert flat.shape == (expected,)


def test_mamba_to_flat_shape():
    input_size, d_state, dt_rank = 8, 4, 2
    layer = MambaLayer(input_size, d_state, dt_rank).double()
    _randomize(layer)
    flat = layer.to_flat()
    expected = (
        (dt_rank + 2 * d_state) * input_size       # x_proj_w
        + input_size * dt_rank                     # dt_proj_w
        + input_size                               # dt_proj_b
        + input_size * d_state                     # a_log
        + input_size                               # d_skip
    )
    assert flat.shape == (expected,)


@pytest.mark.parametrize(
    "architecture, input_dim",
    [
        ([{"type": "dense", "input_size": 4, "output_size": 2, "activation": "tanh"}], 4),
        ([
            {"type": "gru", "input_size": 4, "hidden_size": 8},
            {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
        ], 4),
    ],
)
def test_to_flat_roundtrip_via_rust(architecture, input_dim, tmp_path):
    """Build small policy, extract via to_flat, write via Rust, load via nn_forward,
    compare to Python forward at <1e-10."""
    from pydantic import TypeAdapter

    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import LayerSpec

    validated = TypeAdapter(list[LayerSpec]).validate_python(architecture)
    policy = V2Policy(architecture=validated, input_mask=None).double()
    _randomize(policy)

    # Extract flat weights
    flat_list = []
    for entry, layer in zip(architecture, policy.layers, strict=True):
        flat_list.append(layer.to_flat())
    flat = np.concatenate(flat_list)

    # Write via Rust
    json_path = tmp_path / "model.json"
    r.flat_weights_to_json(architecture, flat.tolist(), str(json_path))

    # Forward via Rust on a sample input
    x = np.linspace(-1.0, 1.0, input_dim, dtype=np.float64)
    rust_out = np.array(r.nn_forward(str(json_path), x.tolist()))

    # Python forward (single step, batch=1)
    x_t = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
    state = policy.new_state(batch_size=1, device=None)
    mean, _ = policy(x_t, state)
    py_out = mean.detach().numpy().flatten()

    assert np.allclose(rust_out, py_out, atol=1e-10), \
        f"Rust {rust_out} vs Python {py_out}, max diff {np.abs(rust_out - py_out).max()}"
```

- [ ] **Step 2: Run — expect AttributeError on `to_flat`**

Run: `uv run pytest tests/test_warm_start_v2_to_flat_roundtrip.py -v`
Expected: fail with "DenseLayer has no attribute 'to_flat'".

- [ ] **Step 3: Add `to_flat()` to DenseLayer**

In `src/python/aerocapture/training/rl/layers/dense.py`:

```python
def to_flat(self) -> np.ndarray:
    """Canonical flat weight order: W (row-major, [out, in]) then b. Matches Rust DenseLayer::to_flat."""
    w = self.linear.weight.detach().cpu().numpy().astype(np.float64)
    b = self.linear.bias.detach().cpu().numpy().astype(np.float64)
    return np.concatenate([w.ravel(), b])
```

Add `import numpy as np` at top of file.

- [ ] **Step 4: Add `to_flat()` to GruLayer**

In `src/python/aerocapture/training/rl/layers/gru.py`:

```python
def to_flat(self) -> np.ndarray:
    """Canonical flat order: weight_ih row-major, weight_hh row-major, bias_ih, bias_hh."""
    return np.concatenate([
        self.weight_ih.detach().cpu().numpy().astype(np.float64).ravel(),
        self.weight_hh.detach().cpu().numpy().astype(np.float64).ravel(),
        self.bias_ih.detach().cpu().numpy().astype(np.float64),
        self.bias_hh.detach().cpu().numpy().astype(np.float64),
    ])
```

- [ ] **Step 5: Add `to_flat()` to LstmLayer**

In `src/python/aerocapture/training/rl/layers/lstm.py`:

```python
def to_flat(self) -> np.ndarray:
    """Canonical flat order: weight_ih row-major, weight_hh row-major, bias_ih, bias_hh. Matches Rust LstmLayer::to_flat."""
    return np.concatenate([
        self.weight_ih.detach().cpu().numpy().astype(np.float64).ravel(),
        self.weight_hh.detach().cpu().numpy().astype(np.float64).ravel(),
        self.bias_ih.detach().cpu().numpy().astype(np.float64),
        self.bias_hh.detach().cpu().numpy().astype(np.float64),
    ])
```

- [ ] **Step 6: Add `to_flat()` to WindowLayer (zero-param no-op)**

In `src/python/aerocapture/training/rl/layers/window.py`:

```python
def to_flat(self) -> np.ndarray:
    """Zero trainable parameters; flat representation is empty."""
    return np.array([], dtype=np.float64)
```

Add `import numpy as np` at top.

- [ ] **Step 7: Add `to_flat()` to TransformerLayer**

In `src/python/aerocapture/training/rl/layers/transformer.py`:

```python
def to_flat(self) -> np.ndarray:
    """Canonical flat order matching Rust LayerWeights<TransformerLayer>::to_flat:
    w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
    w_ffn1, b_ffn1, w_ffn2, b_ffn2,
    ln1_gamma, ln1_beta, ln2_gamma, ln2_beta.
    All 2D weights row-major."""
    parts: list[np.ndarray] = []
    for linear in (self.w_q, self.w_k, self.w_v, self.w_o, self.w_ffn1, self.w_ffn2):
        parts.append(linear.weight.detach().cpu().numpy().astype(np.float64).ravel())
        parts.append(linear.bias.detach().cpu().numpy().astype(np.float64))
    for ln in (self.ln1_gamma, self.ln1_beta, self.ln2_gamma, self.ln2_beta):
        parts.append(ln.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(parts)
```

Add `import numpy as np` at top.

- [ ] **Step 8: Add `to_flat()` to MambaLayer**

In `src/python/aerocapture/training/rl/layers/mamba.py`:

```python
def to_flat(self) -> np.ndarray:
    """Canonical flat order matching Rust LayerWeights<MambaLayer>::to_flat:
    x_proj_w row-major, dt_proj_w row-major, dt_proj_b, a_log row-major, d_skip."""
    return np.concatenate([
        self.x_proj_w.detach().cpu().numpy().astype(np.float64).ravel(),
        self.dt_proj_w.detach().cpu().numpy().astype(np.float64).ravel(),
        self.dt_proj_b.detach().cpu().numpy().astype(np.float64),
        self.a_log.detach().cpu().numpy().astype(np.float64).ravel(),
        self.d_skip.detach().cpu().numpy().astype(np.float64),
    ])
```

Add `import numpy as np` at top.

- [ ] **Step 9: Re-run roundtrip test**

Run: `uv run pytest tests/test_warm_start_v2_to_flat_roundtrip.py -v`
Expected: shape tests pass, roundtrip-via-Rust tests pass at < 1e-10.

If the roundtrip tests fail with order mismatches, double-check the per-layer canonical order against `src/rust/src/data/neural.rs` `LayerWeights for {Dense,Gru,Lstm,Transformer,Mamba}Layer::to_flat` — that's the ground truth.

- [ ] **Step 10: Commit**

```bash
git add src/python/aerocapture/training/rl/layers/ tests/test_warm_start_v2_to_flat_roundtrip.py
git commit -m "feat(layers): to_flat() per layer mirrors Rust LayerWeights::to_flat

Per-layer canonical flat order required for warm-start chromosome encoding.
Round-trip test (Python to_flat -> Rust flat_weights_to_json ->
aerocapture_rs.nn_forward) matches Python V2Policy forward at <1e-10 for
each layer type. Window is a zero-param no-op."
```

---

## Task 6: Python — add `V2Policy.forward_seq_means` for supervised BPTT

**Files:**
- Modify: `src/python/aerocapture/training/rl/policy.py`
- Test: `tests/test_v2_policy_forward_seq_means.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_v2_policy_forward_seq_means.py`:

```python
"""V2Policy.forward_seq_means computes (T, B, out_dim) mean predictions
for supervised warm-start, with done-mask state zeroing matching evaluate."""

import torch

from pydantic import TypeAdapter

from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import LayerSpec


def _build_policy(arch):
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    return V2Policy(architecture=validated, input_mask=None).double()


def test_forward_seq_means_dense_shape_and_finite():
    arch = [{"type": "dense", "input_size": 4, "output_size": 2, "activation": "tanh"}]
    policy = _build_policy(arch)
    T, B = 5, 3
    obs = torch.randn(T, B, 4, dtype=torch.float64)
    state_0 = policy.new_state(batch_size=B, device=None)
    dones = torch.zeros(T, B, dtype=torch.bool)
    means = policy.forward_seq_means(obs, state_0, dones)
    assert means.shape == (T, B, 2)
    assert torch.isfinite(means).all()


def test_forward_seq_means_gru_state_propagates():
    arch = [
        {"type": "gru", "input_size": 4, "hidden_size": 8},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
    ]
    policy = _build_policy(arch)
    T, B = 4, 2
    obs = torch.randn(T, B, 4, dtype=torch.float64)
    state_0 = policy.new_state(batch_size=B, device=None)
    dones = torch.zeros(T, B, dtype=torch.bool)
    means = policy.forward_seq_means(obs, state_0, dones)
    assert means.shape == (T, B, 2)
    assert torch.isfinite(means).all()


def test_forward_seq_means_done_zeros_state():
    """When done[t]=True, the GRU hidden state at t+1 is zeroed."""
    arch = [
        {"type": "gru", "input_size": 2, "hidden_size": 4},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
    ]
    policy = _build_policy(arch)
    T, B = 3, 1
    obs = torch.ones(T, B, 2, dtype=torch.float64)

    # done at t=1: state at t=2 should be the same as starting fresh
    state_0 = policy.new_state(batch_size=B, device=None)
    dones_with = torch.tensor([[False], [True], [False]])
    means_with = policy.forward_seq_means(obs, state_0, dones_with)

    # If we replay only t=2 with fresh state, we should get the same output
    state_fresh = policy.new_state(batch_size=B, device=None)
    means_fresh = policy.forward_seq_means(obs[2:3], state_fresh, torch.zeros(1, B, dtype=torch.bool))
    assert torch.allclose(means_with[2], means_fresh[0], atol=1e-14)
```

- [ ] **Step 2: Run — expect AttributeError**

Run: `uv run pytest tests/test_v2_policy_forward_seq_means.py -v`
Expected: fail.

- [ ] **Step 3: Add `forward_seq_means` to `V2Policy`**

In `src/python/aerocapture/training/rl/policy.py`, add to the `V2Policy` class (after `evaluate`):

```python
def forward_seq_means(
    self,
    obs_seq: Tensor,
    state_0: list[Any],
    dones_seq: Tensor,
) -> Tensor:
    """Supervised-warm-start forward over a time chunk.

    Returns the layer-stack final output (the policy mean) per step. Used by
    the chunked-BPTT supervised pretraining loop in warm_start.py; this is the
    autograd-friendly mirror of `evaluate` minus the Gaussian log-prob math.

    Args:
        obs_seq:   (T, B, obs_dim)
        state_0:   list of per-layer state tensors. Caller is responsible for
                   `.detach()` before passing across chunk boundaries.
        dones_seq: (T, B) bool. When True at time t, the per-env state
                   entering step t+1 is zeroed.

    Returns:
        means: (T, B, out_dim)
    """
    T = obs_seq.shape[0]
    means_list: list[Tensor] = []
    state = state_0
    for t in range(T):
        mean, state = self.forward(obs_seq[t], state)
        means_list.append(mean)
        if t + 1 < T:
            done_mask = dones_seq[t]
            if done_mask.any():
                state = _zero_state_where_done(state, done_mask)
    return torch.stack(means_list, dim=0)
```

- [ ] **Step 4: Re-run test**

Run: `uv run pytest tests/test_v2_policy_forward_seq_means.py -v`
Expected: pass (including the done-mask zeroing test, which exercises Task 4's broadcast fix indirectly via the GRU path).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/policy.py tests/test_v2_policy_forward_seq_means.py
git commit -m "feat(policy): V2Policy.forward_seq_means for supervised BPTT chunks

Mirrors evaluate() but returns (T, B, out_dim) means instead of Gaussian
log-probs. Used by warm_start.py's chunked-BPTT pretraining loop. Done-mask
state zeroing matches evaluate (per-env state at t+1 zeroed when done[t])."
```

---

## Task 7: Python — `_policy_to_flat_weights_v2` dispatcher

**Files:**
- Modify: `src/python/aerocapture/training/warm_start.py` (replace `_policy_to_flat_weights` with v2 version)
- Test: `tests/test_warm_start_policy_to_flat_v2.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_warm_start_policy_to_flat_v2.py`:

```python
"""_policy_to_flat_weights_v2 dispatches per layer type and matches per-layer to_flat."""

import numpy as np
import torch
from pydantic import TypeAdapter

from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import LayerSpec
from aerocapture.training.warm_start import _policy_to_flat_weights_v2


def _build(arch):
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    return V2Policy(architecture=validated, input_mask=None).double()


def test_dense_only_matches_concat_of_to_flat():
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
    ]
    policy = _build(arch)
    with torch.no_grad():
        for p in policy.parameters():
            p.uniform_(-0.1, 0.1)
    expected = np.concatenate([layer.to_flat() for layer in policy.layers])
    actual = _policy_to_flat_weights_v2(policy, arch)
    assert np.allclose(actual, expected, atol=0.0)  # bitwise equal


def test_mixed_arch_dense_gru_dense():
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "gru", "input_size": 8, "hidden_size": 8},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
    ]
    policy = _build(arch)
    with torch.no_grad():
        for p in policy.parameters():
            p.uniform_(-0.1, 0.1)
    expected = np.concatenate([layer.to_flat() for layer in policy.layers])
    actual = _policy_to_flat_weights_v2(policy, arch)
    assert np.allclose(actual, expected, atol=0.0)
```

- [ ] **Step 2: Run — expect ImportError or NotImplementedError**

Run: `uv run pytest tests/test_warm_start_policy_to_flat_v2.py -v`
Expected: fail (function doesn't exist yet, or existing `_policy_to_flat_weights` raises).

- [ ] **Step 3: Add `_policy_to_flat_weights_v2` to `warm_start.py`**

In `src/python/aerocapture/training/warm_start.py`, replace the existing `_policy_to_flat_weights` function (lines 136-156) with the v2 dispatcher:

```python
def _policy_to_flat_weights_v2(policy: V2Policy, architecture: list[dict]) -> npt.NDArray[np.float64]:
    """Extract physical weights from a V2Policy in canonical flat order.

    Dispatches per-layer via each layer module's `to_flat()` method (which
    mirrors Rust `LayerWeights::to_flat` for that variant). Concatenates the
    per-layer flat slabs in architecture order.

    Window contributes an empty slab (zero trainable params); the v2 chromosome
    width is the sum across non-empty layers.
    """
    parts: list[npt.NDArray[np.float64]] = []
    for i, (entry, layer_module) in enumerate(zip(architecture, policy.layers, strict=True)):
        if not hasattr(layer_module, "to_flat"):
            raise RuntimeError(
                f"layer {i} ({entry.get('type', '?')}) has no to_flat() method; "
                "ensure the layer module mirrors Rust LayerWeights::to_flat"
            )
        parts.append(np.asarray(layer_module.to_flat(), dtype=np.float64))
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)
```

Remove the old dense-only `_policy_to_flat_weights` function (it's the only caller of `DenseLayer` import here; clean that up).

- [ ] **Step 4: Re-run test**

Run: `uv run pytest tests/test_warm_start_policy_to_flat_v2.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/warm_start.py tests/test_warm_start_policy_to_flat_v2.py
git commit -m "feat(warm_start): _policy_to_flat_weights_v2 dispatches per layer type

Walks policy.layers and calls each layer's to_flat() (added in Task 5).
Replaces the dense-only _policy_to_flat_weights which raised
NotImplementedError on non-dense layers. Window contributes an empty slab."
```

---

## Task 8: Python — `WarmStartConfig` dataclass + TOML parsing

**Files:**
- Modify: `src/python/aerocapture/training/config.py` (add `WarmStartConfig`, add `TrainingConfig.warm_start` field)
- Modify: `src/python/aerocapture/training/train.py` (TOML loader to populate `cfg.warm_start`)
- Test: `tests/test_warm_start_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_warm_start_config.py`:

```python
"""WarmStartConfig defaults + TOML parsing."""

import textwrap

import pytest

from aerocapture.training.config import WarmStartConfig, TrainingConfig


def test_defaults():
    cfg = WarmStartConfig()
    assert cfg.supervisor_schemes == [
        "ftc", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag",
    ]
    assert cfg.bptt_length == 32
    assert cfg.n_warm_seeds == 200
    assert cfg.n_epochs == 10
    assert cfg.bound_multiplier == 4.0
    assert cfg.jitter == 0.02
    assert cfg.cmaes_sigma0 == 0.1
    assert cfg.params_paths == {}


def test_training_config_has_warm_start_field():
    cfg = TrainingConfig()
    assert isinstance(cfg.warm_start, WarmStartConfig)


def test_from_dict():
    d = {
        "supervisor_schemes": ["ftc", "fnpag"],
        "bptt_length": 16,
        "n_warm_seeds": 100,
        "params_paths": {"ftc": "/some/path/best_params.json"},
    }
    cfg = WarmStartConfig.from_dict(d)
    assert cfg.supervisor_schemes == ["ftc", "fnpag"]
    assert cfg.bptt_length == 16
    assert cfg.n_warm_seeds == 100
    assert cfg.params_paths == {"ftc": "/some/path/best_params.json"}
    # Unspecified keys use defaults
    assert cfg.bound_multiplier == 4.0


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        WarmStartConfig.from_dict({"typo_key": 5})
```

- [ ] **Step 2: Run — expect ImportError on `WarmStartConfig`**

Run: `uv run pytest tests/test_warm_start_config.py -v`
Expected: fail.

- [ ] **Step 3: Add `WarmStartConfig` to `config.py`**

In `src/python/aerocapture/training/config.py`, after the imports and before `NetworkConfig`:

```python
@dataclass
class WarmStartConfig:
    """Multi-supervisor warm-start configuration.

    All fields are optional with documented defaults. Activation of warm-start
    itself is gated by `[guidance.neural_network] warm_start_from` being set;
    this block only configures supervisor list, BPTT mechanics, and optimizer
    seeding tunables.
    """

    supervisor_schemes: list[str] = field(
        default_factory=lambda: ["ftc", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag"]
    )
    bptt_length: int = 32
    n_warm_seeds: int = 200
    n_epochs: int = 10
    bound_multiplier: float = 4.0
    jitter: float = 0.02
    cmaes_sigma0: float = 0.1
    params_paths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WarmStartConfig":
        known = {
            "supervisor_schemes", "bptt_length", "n_warm_seeds", "n_epochs",
            "bound_multiplier", "jitter", "cmaes_sigma0", "params_paths",
        }
        unknown = set(d.keys()) - known
        if unknown:
            raise ValueError(f"unknown [warm_start] keys: {sorted(unknown)}")
        return cls(**d)
```

Add `warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)` to `TrainingConfig`:

```python
@dataclass
class TrainingConfig:
    """Complete training configuration."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    save_dir: str = "training_output"
    guidance_type: str = "neural_network"
    # ...
```

- [ ] **Step 4: Wire `[warm_start]` TOML loading in `train.py`**

In `src/python/aerocapture/training/train.py`, near the existing TOML→config loader (search for `cfg.network.warm_start_from`), add:

```python
if "warm_start" in toml_doc:
    cfg.warm_start = WarmStartConfig.from_dict(toml_doc["warm_start"])
```

Add `from aerocapture.training.config import WarmStartConfig` to imports.

- [ ] **Step 5: Re-run config test**

Run: `uv run pytest tests/test_warm_start_config.py -v`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/config.py \
        src/python/aerocapture/training/train.py \
        tests/test_warm_start_config.py
git commit -m "feat(config): WarmStartConfig dataclass + [warm_start] TOML loader

New TrainingConfig.warm_start field with defaults: 5 supervisor schemes,
bptt_length=32, n_warm_seeds=200, n_epochs=10, bound_multiplier=4.0,
jitter=0.02, cmaes_sigma0=0.1. Unknown keys raise."
```

---

## Task 9: Python — `_select_best_teacher_per_seed`

**Files:**
- Modify: `src/python/aerocapture/training/warm_start.py`
- Test: `tests/test_warm_start_selection.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_warm_start_selection.py`:

```python
"""_select_best_teacher_per_seed picks lowest-DV captured trajectory per seed,
drops seeds with no captures."""

import numpy as np

from aerocapture.training.warm_start import _select_best_teacher_per_seed


def _traj(seed, dv, captured, n_steps=10):
    return {
        "seed": seed,
        "X": np.zeros((n_steps, 21)),
        "y_signed": np.zeros(n_steps),
        "dv": dv,
        "captured": captured,
    }


def test_picks_lowest_dv_captured_per_seed():
    by_scheme = {
        "ftc": [_traj(1, 100.0, True), _traj(2, 200.0, True)],
        "fnpag": [_traj(1, 50.0, True), _traj(2, 250.0, True)],
    }
    selected = _select_best_teacher_per_seed(by_scheme)
    by_seed = {s["seed"]: s for s in selected}
    assert by_seed[1]["scheme"] == "fnpag"
    assert by_seed[1]["dv"] == 50.0
    assert by_seed[2]["scheme"] == "ftc"
    assert by_seed[2]["dv"] == 200.0


def test_drops_seeds_with_no_captures():
    by_scheme = {
        "ftc": [_traj(1, 100.0, True), _traj(2, 999.0, False)],
        "fnpag": [_traj(1, 50.0, True), _traj(2, 888.0, False)],
    }
    selected = _select_best_teacher_per_seed(by_scheme)
    by_seed = {s["seed"]: s for s in selected}
    assert 1 in by_seed
    assert 2 not in by_seed


def test_mixed_capture_falls_back_to_captured_only():
    """A seed where one scheme captures and another fails: capture wins
    even if its DV is higher than the failure's nominal DV."""
    by_scheme = {
        "ftc": [_traj(1, 999.0, False)],   # failed, high DV
        "fnpag": [_traj(1, 500.0, True)],  # captured
    }
    selected = _select_best_teacher_per_seed(by_scheme)
    assert len(selected) == 1
    assert selected[0]["scheme"] == "fnpag"


def test_ignores_seeds_outside_intersection_gracefully():
    """If schemes have different seed coverage, union is taken."""
    by_scheme = {
        "ftc": [_traj(1, 100.0, True)],
        "fnpag": [_traj(2, 50.0, True)],
    }
    selected = _select_best_teacher_per_seed(by_scheme)
    assert len(selected) == 2
```

- [ ] **Step 2: Run — expect ImportError**

Run: `uv run pytest tests/test_warm_start_selection.py -v`
Expected: fail.

- [ ] **Step 3: Add `_select_best_teacher_per_seed` to `warm_start.py`**

In `src/python/aerocapture/training/warm_start.py`:

```python
def _select_best_teacher_per_seed(
    results_by_scheme: dict[str, list[dict]],
) -> list[dict]:
    """Across schemes, pick the captured trajectory with the lowest DV per seed.

    Returns a list of dicts with the original (seed, X, y_signed, dv, captured)
    fields plus a "scheme" field naming the winner. Seeds where no scheme
    captures are dropped (warm-start should teach winning behavior).
    """
    all_seeds: set[int] = set()
    for results in results_by_scheme.values():
        for r in results:
            all_seeds.add(r["seed"])

    selected: list[dict] = []
    for seed in sorted(all_seeds):
        candidates: list[tuple[str, dict]] = []
        for scheme, results in results_by_scheme.items():
            for r in results:
                if r["seed"] == seed and r["captured"]:
                    candidates.append((scheme, r))
        if not candidates:
            continue
        scheme, r = min(candidates, key=lambda sr: float(sr[1]["dv"]))
        selected.append({"scheme": scheme, **r})
    return selected
```

- [ ] **Step 4: Re-run test**

Run: `uv run pytest tests/test_warm_start_selection.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/warm_start.py tests/test_warm_start_selection.py
git commit -m "feat(warm_start): _select_best_teacher_per_seed picks lowest-DV capture per seed

Drops seeds where no supervisor captures (cleaner warm-start signal than
teaching the least-bad failure). Returns per-seed dict with winning scheme
name attached for downstream logging."
```

---

## Task 10: Python — chunked-BPTT supervised training loop with per-epoch loss logging

**Files:**
- Modify: `src/python/aerocapture/training/warm_start.py` (replace `_supervised_pretrain`)
- Test: `tests/test_warm_start_chunked_bptt.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_warm_start_chunked_bptt.py`:

```python
"""_chunked_bptt_train runs end-to-end on a synthetic 2-trajectory corpus,
returns a trained V2Policy + per-epoch losses."""

import numpy as np
import pytest
import torch
from pydantic import TypeAdapter

from aerocapture.training.config import NetworkConfig
from aerocapture.training.rl.schemas import LayerSpec
from aerocapture.training.warm_start import _chunked_bptt_train


def _make_trajectories(n_trajectories=2, T=64, input_dim=4):
    """Synthetic: y_signed = sin(X[:, 0]) so a small MLP can fit it quickly."""
    trajs = []
    rng = np.random.default_rng(0)
    for i in range(n_trajectories):
        X = rng.standard_normal((T, input_dim))
        y = np.sin(X[:, 0])
        trajs.append({"seed": i, "X": X, "y_signed": y, "dv": 100.0, "captured": True, "scheme": "ftc"})
    return trajs


def test_dense_runs_and_loss_decreases():
    trajs = _make_trajectories()
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )
    policy, losses = _chunked_bptt_train(
        trajectories=trajs,
        network=network,
        bptt_length=16,
        n_epochs=5,
        lr=1e-2,
    )
    assert len(losses) == 5
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]  # loss decreased


def test_gru_runs_and_loss_finite():
    trajs = _make_trajectories(T=32)
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "gru", "input_size": 8, "hidden_size": 8},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(4)),
        output_parameterization="acos_tanh",
    )
    policy, losses = _chunked_bptt_train(
        trajectories=trajs,
        network=network,
        bptt_length=8,
        n_epochs=3,
        lr=1e-2,
    )
    assert len(losses) == 3
    assert all(np.isfinite(losses))


def test_bptt_length_greater_than_n_seq_raises_for_transformer():
    trajs = _make_trajectories(T=32, input_dim=8)
    network = NetworkConfig(
        architecture=[
            {"type": "dense", "input_size": 8, "output_size": 8, "activation": "tanh"},
            {"type": "transformer", "d_model": 8, "n_heads": 2, "d_ffn": 16, "n_seq": 4},
            {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
        ],
        input_mask=list(range(8)),
        output_parameterization="acos_tanh",
    )
    with pytest.raises(ValueError, match="bptt_length.*n_seq"):
        _chunked_bptt_train(
            trajectories=trajs,
            network=network,
            bptt_length=16,  # > n_seq=4
            n_epochs=1,
            lr=1e-2,
        )
```

- [ ] **Step 2: Run — expect ImportError on `_chunked_bptt_train`**

Run: `uv run pytest tests/test_warm_start_chunked_bptt.py -v`
Expected: fail.

- [ ] **Step 3: Implement `_chunked_bptt_train`**

In `src/python/aerocapture/training/warm_start.py`, replace the existing `_supervised_pretrain` function with:

```python
def _chunked_bptt_train(
    trajectories: list[dict],
    network: NetworkConfig,
    bptt_length: int,
    n_epochs: int,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[V2Policy, list[float]]:
    """Chunked truncated-BPTT supervised pretraining.

    Each trajectory is split into `bptt_length`-sized chunks; per-chunk forward
    is via `V2Policy.forward_seq_means`. Hidden state from chunk c is detached
    and carried as the start state for chunk c+1. Loss is MSE between the
    predicted output parameterization (cos(y) for acos_tanh, (sin,cos) for
    atan2_signed) and the target.

    For magnitude_only mode, callers pre-process `y_signed -> abs(y_signed)`.
    """
    import torch
    from pydantic import TypeAdapter
    from torch import nn

    from aerocapture.training.rl.layers.transformer import TransformerLayer
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.schemas import LayerSpec

    if network.architecture is None:
        raise ValueError("_chunked_bptt_train requires a v2 architecture (network.architecture is None)")

    validated_arch = TypeAdapter(list[LayerSpec]).validate_python(network.architecture)
    policy = V2Policy(architecture=validated_arch, input_mask=network.input_mask).double()

    # Validate bptt_length <= n_seq for any Transformer layer
    for i, layer in enumerate(policy.layers):
        if isinstance(layer, TransformerLayer) and bptt_length > layer.n_seq:
            raise ValueError(
                f"bptt_length={bptt_length} > layer {i} Transformer n_seq={layer.n_seq}; "
                "reduce bptt_length or increase n_seq"
            )

    output_param = network.output_parameterization or "atan2_signed"
    input_mask = network.input_mask if network.input_mask is not None else list(range(21))

    # Build chunks: list of (X_chunk[T_c, input_dim], y_chunk[T_c]) per trajectory.
    chunks: list[tuple[np.ndarray, np.ndarray, int]] = []  # (X, y, traj_id)
    for tid, traj in enumerate(trajectories):
        X = np.asarray(traj["X"])[:, input_mask]
        y = np.asarray(traj["y_signed"])
        # Drop non-finite rows
        finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[finite]
        y = y[finite]
        T = X.shape[0]
        # Slice into bptt_length chunks; trailing partial chunk dropped (clean BPTT)
        n_chunks = T // bptt_length
        for c in range(n_chunks):
            s = c * bptt_length
            e = s + bptt_length
            chunks.append((X[s:e], y[s:e], tid))

    if not chunks:
        raise RuntimeError("no usable BPTT chunks; check bptt_length vs trajectory lengths")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses: list[float] = []

    for epoch in range(n_epochs):
        # Shuffle chunks; minibatch as the chunk-batch dim
        order = rng.permutation(len(chunks))
        # Group into minibatches of up to 32 chunks; each minibatch is forwarded together.
        # Different trajectories' chunks can be batched freely because we re-init state per chunk.
        batch_size = min(32, len(chunks))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            X_batch = np.stack([chunks[i][0] for i in batch_idx], axis=0)  # (B, T, in)
            y_batch = np.stack([chunks[i][1] for i in batch_idx], axis=0)  # (B, T)
            # Time-major
            obs_seq = torch.tensor(X_batch.transpose(1, 0, 2), dtype=torch.float64)  # (T, B, in)
            y_t = torch.tensor(y_batch.transpose(1, 0), dtype=torch.float64)         # (T, B)

            B = obs_seq.shape[1]
            state_0 = policy.new_state(batch_size=B, device=None)
            dones = torch.zeros(obs_seq.shape[0], B, dtype=torch.bool)  # no dones within a chunk

            optimizer.zero_grad()
            means = policy.forward_seq_means(obs_seq, state_0, dones)  # (T, B, out_dim)

            if output_param == "acos_tanh":
                pred = torch.tanh(means[..., 0])              # (T, B)
                target = torch.cos(y_t)                        # (T, B)
                loss = nn.functional.mse_loss(pred, target)
            elif output_param == "atan2_signed":
                # means: (T, B, 2). Target = (sin(y), cos(y)).
                target = torch.stack([torch.sin(y_t), torch.cos(y_t)], dim=-1)
                loss = nn.functional.mse_loss(means, target)
            else:
                raise ValueError(f"unknown output_parameterization {output_param!r}")

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return policy, losses
```

- [ ] **Step 4: Re-run test**

Run: `uv run pytest tests/test_warm_start_chunked_bptt.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/warm_start.py tests/test_warm_start_chunked_bptt.py
git commit -m "feat(warm_start): _chunked_bptt_train replaces shuffled-step _supervised_pretrain

Per-trajectory chunks of bptt_length; minibatching is across the chunk-batch
axis; forward via V2Policy.forward_seq_means. Loss is MSE on cos(y) for
acos_tanh or (sin, cos) for atan2_signed. Validates bptt_length <= Transformer
n_seq at training-time. Returns trained policy + per-epoch mean loss list."
```

---

## Task 11: Python — rewrite `build_warm_start_chromosome` end-to-end (multi-supervisor + cache + log)

**Files:**
- Modify: `src/python/aerocapture/training/warm_start.py` (replace `build_warm_start_chromosome`, update `_cache_key`)
- Test: extended in Task 17 per-arch smoke; quick smoke here

- [ ] **Step 1: Write a synthetic end-to-end test (no Rust sim, mocked collect_supervised)**

Create the end-to-end test in `tests/test_warm_start_end_to_end.py`:

```python
"""End-to-end build_warm_start_chromosome with mocked collect_supervised."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from aerocapture.training.config import (
    NetworkConfig, OptimizerConfig, SimConfig, TrainingConfig, WarmStartConfig,
)
from aerocapture.training.warm_start import build_warm_start_chromosome


@pytest.fixture
def synthetic_supervisor_data():
    rng = np.random.default_rng(0)
    def _collect(toml_path, seeds, overrides, scheme, sim_timeout_secs=None):
        results = []
        for seed in seeds:
            T = 50
            results.append({
                "seed": int(seed),
                "X": rng.standard_normal((T, 21)),
                "y_signed": np.sin(rng.standard_normal(T)),
                "dv": float(rng.uniform(50, 500)),
                "captured": True,
            })
        return results
    return _collect


@pytest.fixture
def temp_ftc_params(tmp_path):
    p = tmp_path / "ftc_best_params.json"
    p.write_text(json.dumps({
        "k_alt": 1.0,
        "lateral.tau": 5.0,
        "exit.dpdyn_target": 100.0,
        "nav.density_filter_gain": 0.5,
        "thermal.heat_flux_activation": 0.8,
        "shaping.max_bank_acceleration": 30.0,
    }))
    return p


@pytest.fixture
def cfg(tmp_path, temp_ftc_params):
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
        {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
    ]
    return TrainingConfig(
        network=NetworkConfig(
            architecture=arch,
            input_mask=[0, 1, 2, 3],
            output_parameterization="acos_tanh",
            warm_start_from=str(temp_ftc_params),
        ),
        warm_start=WarmStartConfig(
            supervisor_schemes=["ftc"],
            params_paths={"ftc": str(temp_ftc_params)},
            n_warm_seeds=4,
            n_epochs=2,
            bptt_length=16,
        ),
        sim=SimConfig(toml_config="dummy.toml"),
        save_dir=str(tmp_path / "warm_out"),
    )


def test_end_to_end_with_mocked_collect(cfg, synthetic_supervisor_data, tmp_path):
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=synthetic_supervisor_data):
        chromo = build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
    assert chromo.dtype == np.float64
    assert chromo.ndim == 1
    assert (chromo >= 0.0).all() and (chromo <= 1.0).all()
    # Loss log written
    assert (Path(cfg.save_dir) / "warm_start_loss.json").exists()
    # Chromosome cached
    assert (Path(cfg.save_dir) / "warm_start_chromosome.npy").exists()
    assert (Path(cfg.save_dir) / "warm_start_cache_key.json").exists()


def test_cache_hit_skips_recomputation(cfg, synthetic_supervisor_data):
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=synthetic_supervisor_data) as mock:
        chromo1 = build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        n_calls_first = mock.call_count
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=synthetic_supervisor_data) as mock2:
        chromo2 = build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock2.call_count == 0  # cache hit, no calls
    assert np.array_equal(chromo1, chromo2)
```

- [ ] **Step 2: Run — expect failures (build_warm_start_chromosome signature mismatch or no warm_start cfg field plumbed)**

Run: `uv run pytest tests/test_warm_start_end_to_end.py -v`
Expected: fail.

- [ ] **Step 3: Rewrite `build_warm_start_chromosome` end-to-end**

In `src/python/aerocapture/training/warm_start.py`, replace `build_warm_start_chromosome` with:

```python
def build_warm_start_chromosome(
    cfg: TrainingConfig,
    base_mc_seed: int,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Multi-supervisor warm-start: collect per-seed best teacher, chunked-BPTT, encode.

    Configuration is fully read from cfg.warm_start (supervisor_schemes,
    bptt_length, n_warm_seeds, n_epochs, bound_multiplier, params_paths)
    and cfg.network (architecture, input_mask, output_parameterization,
    optimize_scaffolding, warm_start_from for the scaffolding source).

    `base_mc_seed` MUST be the resolved value train.py uses for
    validation/final-eval pools so warm-start seeds are disjoint
    (`WARM_START_SEED_OFFSET = 4M`).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ws = cfg.warm_start
    network = cfg.network

    # 1. Resolve supervisor paths
    resolved_paths: dict[str, Path] = {}
    for scheme in ws.supervisor_schemes:
        override = ws.params_paths.get(scheme)
        path = Path(override) if override else Path(f"training_output/{scheme}/best_params.json")
        if not path.exists():
            raise FileNotFoundError(
                f"warm-start supervisor '{scheme}' params not found at '{path}'. "
                f"Train {scheme} first or set [warm_start.params_paths].{scheme}."
            )
        resolved_paths[scheme] = path

    # Scaffolding source (for the 17-slot tail when optimize_scaffolding)
    scaffolding_source_path = Path(network.warm_start_from) if network.warm_start_from else resolved_paths[ws.supervisor_schemes[0]]
    if not scaffolding_source_path.exists():
        raise FileNotFoundError(f"scaffolding source params not found at '{scaffolding_source_path}'")

    # 2. Cache check
    cache_key = _cache_key(cfg, resolved_paths, scaffolding_source_path)
    cached = _cache_hit(save_dir, cache_key)
    if cached is not None:
        return cached

    # 3. Collect per scheme
    seeds = make_reserved_seeds(base_mc_seed, WARM_START_SEED_OFFSET, ws.n_warm_seeds)
    results_by_scheme: dict[str, list[dict]] = {}
    for scheme, path in resolved_paths.items():
        with open(path) as f:
            source_params = json.load(f)
        overrides = _build_overrides_for_source(source_params)
        results_by_scheme[scheme] = _aero_rs.collect_supervised(
            toml_path=cfg.sim.toml_config,
            seeds=seeds,
            overrides=overrides,
            scheme=scheme,
        )

    # 4. Pick best per seed
    selected = _select_best_teacher_per_seed(results_by_scheme)
    min_corpus = max(20, ws.n_warm_seeds // 4)
    if len(selected) < min_corpus:
        raise RuntimeError(
            f"warm-start corpus too small: {len(selected)} captures across {ws.n_warm_seeds} seeds "
            f"(threshold {min_corpus}). Widen MC dispersions, check the TOML, or revise supervisor_schemes."
        )

    # 5. Magnitude_only mode: derive |y| Python-side
    mode = _resolve_nn_mode(cfg)
    for traj in selected:
        if mode == "magnitude_only":
            traj["y_signed"] = np.abs(traj["y_signed"])

    # 6. Chunked-BPTT supervised pretraining
    policy, losses = _chunked_bptt_train(
        trajectories=selected,
        network=network,
        bptt_length=ws.bptt_length,
        n_epochs=ws.n_epochs,
    )
    (save_dir / "warm_start_loss.json").write_text(json.dumps(
        [{"epoch": i, "mean_mse": float(loss)} for i, loss in enumerate(losses)],
        indent=2,
    ))
    print(f"  [warm_start] supervised MSE: {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} epochs")

    # 7. Extract flat weights and encode to normalized chromosome at warm-start bound_multiplier
    flat_weights = _policy_to_flat_weights_v2(policy, network.architecture)
    from pydantic import TypeAdapter
    from aerocapture.training.rl.schemas import LayerSpec
    validated_arch = TypeAdapter(list[LayerSpec]).validate_python(network.architecture)
    weight_specs = nn_param_specs_from_v2(validated_arch, bound_multiplier=ws.bound_multiplier)

    weight_chromo = np.empty(len(weight_specs), dtype=np.float64)
    n_clipped = 0
    for i, s in enumerate(weight_specs):
        v = float(flat_weights[i])
        normalized = (v - s.p_min) / (s.p_max - s.p_min)
        if normalized < 0.0 or normalized > 1.0:
            n_clipped += 1
        weight_chromo[i] = np.clip(normalized, 0.0, 1.0)

    clip_rate = n_clipped / max(len(weight_specs), 1)
    if clip_rate > 0.05:
        raise RuntimeError(
            f"warm-start clip rate {100 * clip_rate:.1f}% ({n_clipped}/{len(weight_specs)}) exceeds 5% threshold. "
            "Widen [warm_start] bound_multiplier, reduce n_epochs, or lower lr."
        )
    elif n_clipped > 0:
        print(f"  [warm_start] {n_clipped}/{len(weight_specs)} weights clipped ({100 * clip_rate:.2f}%).")

    chromo = weight_chromo
    if network.optimize_scaffolding:
        with open(scaffolding_source_path) as f:
            scaff_params = json.load(f)
        scaff_chromo = encode_to_normalized(scaff_params, list(_NN_SCAFFOLDING_PARAMS))
        chromo = np.concatenate([weight_chromo, scaff_chromo])

    np.save(save_dir / "warm_start_chromosome.npy", chromo)
    (save_dir / "warm_start_cache_key.json").write_text(json.dumps(cache_key, indent=2))
    return chromo


def _resolve_nn_mode(cfg: TrainingConfig) -> str:
    """Read [guidance.neural_network] mode from the TOML; default 'full_neural'."""
    # The TOML loader populates a private cfg field; if absent, fall back to TOML re-read.
    mode = getattr(cfg.network, "neural_network_mode", None)
    if mode is not None:
        return str(mode)
    if cfg.sim.toml_config is None:
        return "full_neural"
    try:
        import tomllib
        with open(cfg.sim.toml_config, "rb") as f:
            doc = tomllib.load(f)
        return str(doc.get("guidance", {}).get("neural_network", {}).get("mode", "full_neural"))
    except Exception:
        return "full_neural"
```

Update `_cache_key` to take `resolved_paths` and `scaffolding_source_path`:

```python
def _cache_key(
    cfg: TrainingConfig,
    resolved_paths: dict[str, Path],
    scaffolding_source_path: Path,
) -> dict:
    return {
        "architecture": cfg.network.architecture,
        "input_mask": cfg.network.input_mask,
        "output_parameterization": cfg.network.output_parameterization or "atan2_signed",
        "optimize_scaffolding": bool(cfg.network.optimize_scaffolding),
        "toml_config": str(cfg.sim.toml_config) if cfg.sim.toml_config else None,
        "supervisor_schemes": sorted(cfg.warm_start.supervisor_schemes),
        "supervisor_params": {
            scheme: {"path": str(p), "mtime": p.stat().st_mtime}
            for scheme, p in sorted(resolved_paths.items())
        },
        "scaffolding_source_path": str(scaffolding_source_path),
        "scaffolding_source_mtime": scaffolding_source_path.stat().st_mtime,
        "n_warm_seeds": cfg.warm_start.n_warm_seeds,
        "n_epochs": cfg.warm_start.n_epochs,
        "bptt_length": cfg.warm_start.bptt_length,
        "bound_multiplier": cfg.warm_start.bound_multiplier,
        "mode": _resolve_nn_mode(cfg),
    }
```

- [ ] **Step 4: Re-run end-to-end test**

Run: `uv run pytest tests/test_warm_start_end_to_end.py -v`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/warm_start.py tests/test_warm_start_end_to_end.py
git commit -m "feat(warm_start): rewrite build_warm_start_chromosome for multi-supervisor + BPTT

Multi-supervisor data collection (Python orchestrates one collect_supervised
call per scheme), best-teacher-per-seed selection, magnitude_only mode
derivation via .abs(), chunked-BPTT pretraining via _chunked_bptt_train,
per-arch _policy_to_flat_weights_v2 extraction, conditional bound widening
via cfg.warm_start.bound_multiplier (default 4.0).

Cache key extended: supervisor list, per-scheme (path, mtime), bptt_length,
bound_multiplier, mode. Per-epoch loss persisted to warm_start_loss.json.
Clip rate >5% is now a hard error (was a warning)."
```

---

## Task 12: Python — `train.py` per-algorithm initial-population seeding

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (~lines 580-606, the warm-start block)
- Test: `tests/test_warm_start_optimizer_seeding.py`

- [ ] **Step 1: Write failing test for the four-algorithm seeding contract**

Create `tests/test_warm_start_optimizer_seeding.py`:

```python
"""Per-algorithm warm-start seeding contract: GA/DE/PSO replicate+jitter,
CMA-ES seeds mean + shrunken sigma0."""

import numpy as np

from aerocapture.training.train import _seed_initial_population


def test_ga_replicate_and_jitter():
    chromo = np.full(50, 0.5)
    n_pop = 30
    rng = np.random.default_rng(0)
    pop = _seed_initial_population(
        algorithm_name="ga",
        chromosome=chromo,
        n_pop=n_pop,
        jitter=0.02,
        rng=rng,
    )
    assert pop.shape == (n_pop, 50)
    assert pop.mean(axis=0) == pytest_approx(0.5, abs=0.01)
    assert pop.std(axis=0).mean() == pytest_approx(0.02, abs=0.005)
    assert (pop >= 0.0).all() and (pop <= 1.0).all()


def test_cma_es_singleton_seeded():
    """CMA-ES seeding returns a single chromosome row (the mean); sigma0 is
    applied via OptimizerConfig override (separate path)."""
    chromo = np.full(50, 0.5)
    rng = np.random.default_rng(0)
    pop = _seed_initial_population(
        algorithm_name="cma_es",
        chromosome=chromo,
        n_pop=20,
        jitter=0.02,
        rng=rng,
    )
    # CMA-ES uses the chromosome as initial mean; we still tile to n_pop for setup,
    # but contract says all rows equal the chromosome (mean) — sigma is applied by pymoo.
    assert pop.shape[0] >= 1
    assert np.allclose(pop[0], chromo)


def test_de_and_pso_match_ga_contract():
    chromo = np.full(20, 0.7)
    n_pop = 10
    for algo in ("de", "pso"):
        rng = np.random.default_rng(0)
        pop = _seed_initial_population(algo, chromo, n_pop, jitter=0.02, rng=rng)
        assert pop.shape == (n_pop, 20)
        assert pop.mean(axis=0) == pytest_approx(0.7, abs=0.02)


# Helper since `pytest.approx` is more readable but the module-level import is heavier
def pytest_approx(value, abs=0.01):
    import pytest
    return pytest.approx(value, abs=abs)
```

- [ ] **Step 2: Run — expect ImportError on `_seed_initial_population`**

Run: `uv run pytest tests/test_warm_start_optimizer_seeding.py -v`
Expected: fail.

- [ ] **Step 3: Extract the seeding logic into `_seed_initial_population` in `train.py`**

In `src/python/aerocapture/training/train.py`, add a module-level function:

```python
def _seed_initial_population(
    algorithm_name: str,
    chromosome: np.ndarray,
    n_pop: int,
    jitter: float,
    rng: np.random.Generator,
    n_weights: int | None = None,
) -> np.ndarray:
    """Build the initial population from a warm-started chromosome.

    GA / DE / PSO: tile chromosome to `n_pop` rows; add per-row N(0, jitter)
    noise to the first `n_weights` columns (or all columns if `n_weights` is
    None); clip to [0, 1]. The scaffolding tail (if optimize_scaffolding is on,
    `chromosome[n_weights:]`) is NOT jittered here — caller is responsible for
    overriding that slab with `scaffolding_slab` when applicable.

    CMA-ES: tile chromosome to `n_pop` rows without jitter; pymoo's CMA-ES
    uses the population mean as its initial mean. sigma0 is configured via
    `OptimizerConfig.cma_es.sigma0` (separate path in create_algorithm).
    """
    pop = np.tile(chromosome, (n_pop, 1))
    if algorithm_name == "cma_es":
        return pop
    if algorithm_name not in ("ga", "de", "pso"):
        raise ValueError(f"unknown algorithm {algorithm_name!r} for warm-start seeding")
    nw = n_weights if n_weights is not None else chromosome.size
    pop[:, :nw] += rng.normal(0.0, jitter, size=(n_pop, nw))
    pop[:, :nw] = np.clip(pop[:, :nw], 0.0, 1.0)
    return pop
```

- [ ] **Step 4: Wire it into the existing warm-start block (around lines 580-606)**

Replace the existing block:

```python
            if config.network.warm_start_from:
                from aerocapture.training.warm_start import build_warm_start_chromosome

                warm_chromo = build_warm_start_chromosome(
                    cfg=config,
                    base_mc_seed=base_mc_seed,
                    rng=rng,
                )
                n_scaff = 17 if config.network.optimize_scaffolding else 0
                n_weights = len(warm_chromo) - n_scaff
                # CMA-ES: also shrink the initial step size for warm-started run
                if config.optimizer.algorithm == "cma_es":
                    config.optimizer.cma_es.sigma0 = config.warm_start.cmaes_sigma0
                pop_array = _seed_initial_population(
                    algorithm_name=config.optimizer.algorithm,
                    chromosome=warm_chromo,
                    n_pop=config.optimizer.n_pop,
                    jitter=config.warm_start.jitter,
                    rng=rng,
                    n_weights=n_weights,
                )
                if scaffolding_slab is not None:
                    pop_array[:, n_weights:] = scaffolding_slab
            else:
                pop_array = build_initial_population_for_v2(
                    config.network.architecture,
                    config.optimizer.n_pop,
                    bound_multiplier=2.0,
                    rng=rng,
                    param_specs=param_specs,
                    scaffolding_slab=scaffolding_slab,
                )
```

- [ ] **Step 5: Re-run test**

Run: `uv run pytest tests/test_warm_start_optimizer_seeding.py -v`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/train.py tests/test_warm_start_optimizer_seeding.py
git commit -m "feat(train): per-algorithm warm-start initial-population seeding

GA/DE/PSO: tile chromosome to n_pop and jitter the weight slab by
N(0, warm_start.jitter) in normalized space (default 0.02). CMA-ES: tile
without jitter (pymoo uses the population mean as initial mean) and shrink
the step size via OptimizerConfig.cma_es.sigma0 = warm_start.cmaes_sigma0
(default 0.1, vs the default 0.3)."
```

---

## Task 13: Python — conditional `bound_multiplier` widening + gen-0 validation baseline

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (use `cfg.warm_start.bound_multiplier` when warm-start is on; add gen-0 validation baseline writer)
- Create: `src/python/aerocapture/training/_warm_start_baseline.py` (small helper to keep the train.py edit minimal)
- Test: `tests/test_warm_start_baseline_writer.py`

- [ ] **Step 1: Write the failing test for the baseline writer helper**

Create `tests/test_warm_start_baseline_writer.py`:

```python
"""Gen-0 validation MC writer for warm-started chromosomes."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from aerocapture.training._warm_start_baseline import write_gen0_baseline


def test_writes_baseline_file_with_mean_rms_n_sims(tmp_path):
    save_dir = tmp_path
    # Mock run_batch to return a deterministic (n_sims, n_final_columns) array
    n_sims = 8
    fake_dv = np.array([100.0, 200.0, 150.0, 50.0, 75.0, 125.0, 175.0, 225.0])

    def _fake_run_batch(toml_path, overrides_list, n_threads=None, include_trajectories=False, sim_timeout_secs=None):
        class _Result:
            final_records = np.tile(np.zeros(52), (n_sims, 1))
        # final_records[:, dv_col] gets dv values
        _Result.final_records[:, 39] = fake_dv  # dv_total_m_s index (adjust if column differs)
        return _Result

    with patch("aerocapture.training._warm_start_baseline._aero_rs.run_batch", side_effect=_fake_run_batch):
        path = write_gen0_baseline(
            save_dir=save_dir,
            toml_path="dummy.toml",
            overrides=[{}],
            n_sims=n_sims,
            dv_column_index=39,
        )

    assert path == save_dir / "warm_start_baseline.json"
    data = json.loads(path.read_text())
    assert data["n_sims"] == n_sims
    assert data["mean"] == pytest.approx(float(np.mean(fake_dv)))
    assert data["rms"] == pytest.approx(float(np.sqrt(np.mean(fake_dv ** 2))))
```

- [ ] **Step 2: Run the test — expect ImportError**

Run: `uv run pytest tests/test_warm_start_baseline_writer.py -v`
Expected: fail.

- [ ] **Step 3: Implement the helper**

Create `src/python/aerocapture/training/_warm_start_baseline.py`:

```python
"""Gen-0 validation baseline helper for warm-started chromosomes.

Kept as a small standalone module so train.py's warm-start integration is
one import + one function call, and the validation-MC plumbing can be tested
in isolation against a mocked aerocapture_rs.run_batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import aerocapture_rs as _aero_rs
except ImportError as e:
    raise ImportError("warm_start baseline requires aerocapture_rs PyO3 module") from e


def write_gen0_baseline(
    save_dir: Path,
    toml_path: str,
    overrides: list[dict[str, Any]],
    n_sims: int,
    dv_column_index: int = 39,
) -> Path:
    """Run validation MC on a single warm-started overrides set and persist
    mean/RMS DV to `<save_dir>/warm_start_baseline.json`.

    Args:
        save_dir: training output dir.
        toml_path: TOML config path.
        overrides: a single-element list of override dicts; the chromosome's
            decoded params should already be in the dict.
        n_sims: number of validation sims.
        dv_column_index: column index of dv_total_m_s in final_records.
            Default 39 matches the current Rust FINAL_CSV_COLUMNS layout.

    Returns:
        Path to the written JSON file.
    """
    # n_sims is enforced via the overrides dict.
    overrides_with_n = [{**overrides[0], "simulation.n_sims": int(n_sims)}]
    result = _aero_rs.run_batch(
        toml_path,
        overrides_with_n,
        include_trajectories=False,
    )
    dv_values = np.asarray(result.final_records[:, dv_column_index], dtype=np.float64)
    payload = {
        "n_sims": int(n_sims),
        "mean": float(np.mean(dv_values)),
        "rms": float(np.sqrt(np.mean(dv_values ** 2))),
    }
    out_path = save_dir / "warm_start_baseline.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
```

- [ ] **Step 4: Re-run helper test**

Run: `uv run pytest tests/test_warm_start_baseline_writer.py -v`
Expected: pass.

- [ ] **Step 5: Thread `warm_start.bound_multiplier` through `train.py`**

In `src/python/aerocapture/training/train.py`, find the param-spec construction site (e.g. `param_specs = nn_param_specs_from_v2(...)` near the v2-architecture branch). Replace any hardcoded `bound_multiplier=2.0` with:

```python
bound_mult = config.warm_start.bound_multiplier if config.network.warm_start_from else 2.0
param_specs = nn_param_specs_from_v2(validated_architecture, bound_multiplier=bound_mult)
```

Also update the `build_initial_population_for_v2(... bound_multiplier=2.0 ...)` call (lines around 599-606 from Task 12's edit) to use the same `bound_mult` variable.

- [ ] **Step 6: Wire the baseline writer into the warm-start block**

In `src/python/aerocapture/training/train.py`, after the warm-start block from Task 12 builds `pop_array` and before `algorithm.setup(problem, pop=initial_pop)`, add:

```python
            # Gen-0 validation baseline: write the bare warm-started chromosome's MC mean/RMS.
            # Activated only when warm-start is on.
            if config.network.warm_start_from:
                from aerocapture.training._warm_start_baseline import write_gen0_baseline
                # Decode chromosome's parameters into overrides (re-use problem._build_overrides
                # if exposed, otherwise call decode_normalized then problem helper directly).
                warm_overrides = problem._build_overrides(
                    params=problem._decode_individual(warm_chromo),
                    mc_seed=base_mc_seed,
                ) if hasattr(problem, "_decode_individual") else [{}]
                baseline_path = write_gen0_baseline(
                    save_dir=Path(config.save_dir),
                    toml_path=config.sim.toml_config,
                    overrides=[warm_overrides] if isinstance(warm_overrides, dict) else warm_overrides,
                    n_sims=config.optimizer.validation_n_sims,
                )
                if verbose:
                    import json
                    baseline = json.loads(baseline_path.read_text())
                    print(f"  [warm_start] gen-0 validation baseline: mean={baseline['mean']:.3f}, rms={baseline['rms']:.3f}, n_sims={baseline['n_sims']}")
```

The `problem._build_overrides` / `problem._decode_individual` pair is already how the training loop produces validation overrides per individual; reuse it. If the AerocaptureProblem API differs from what's sketched above (likely — check `src/python/aerocapture/training/problem.py`), use the actual public method; the test in Step 1 only exercises the helper, so this wiring just needs to call `write_gen0_baseline` with a valid overrides set.

- [ ] **Step 7: Run a 1-gen training smoke to verify no regression**

Run:
```bash
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_nn_train_consolidated.toml \
    --n-gen 1 --n-pop 4 --no-tui --skip-report
```
Expected: exits cleanly. If `warm_start_from` is set in the TOML, a `warm_start_baseline.json` file appears in the save dir and a baseline line is printed.

- [ ] **Step 8: Commit**

```bash
git add src/python/aerocapture/training/_warm_start_baseline.py \
        src/python/aerocapture/training/train.py \
        tests/test_warm_start_baseline_writer.py
git commit -m "feat(train): conditional bound widening + gen-0 validation baseline writer

When [guidance.neural_network] warm_start_from is set, v2 param specs use
warm_start.bound_multiplier (default 4.0, vs 2.0 without warm-start). The
wider search space matches the warm-started population's drift past Xavier
bounds.

A new _warm_start_baseline.py helper runs validation MC on the bare warm-
started chromosome and writes mean/RMS to <save_dir>/warm_start_baseline.json
before generation 0. Lets users see 'did warm-start help?' quantitatively."
```

---

## Task 14: Python — failure mode tests

**Files:**
- Test: `tests/test_warm_start_failures.py`

- [ ] **Step 1: Write the failure-mode test battery**

Create `tests/test_warm_start_failures.py`:

```python
"""Failure modes from the spec: missing supervisor params, zero captures,
clip rate > 5%, bptt_length > Transformer n_seq."""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from aerocapture.training.config import (
    NetworkConfig, SimConfig, TrainingConfig, WarmStartConfig,
)
from aerocapture.training.warm_start import (
    build_warm_start_chromosome, _select_best_teacher_per_seed,
)


def _basic_cfg(tmp_path, supervisor_schemes=None, params_paths=None):
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
        {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
    ]
    return TrainingConfig(
        network=NetworkConfig(
            architecture=arch,
            input_mask=[0, 1, 2, 3],
            output_parameterization="acos_tanh",
            warm_start_from=str(tmp_path / "ftc_params.json") if params_paths is None else None,
        ),
        warm_start=WarmStartConfig(
            supervisor_schemes=supervisor_schemes or ["ftc"],
            params_paths=params_paths or {},
            n_warm_seeds=4,
            n_epochs=1,
            bptt_length=8,
        ),
        sim=SimConfig(toml_config="dummy.toml"),
        save_dir=str(tmp_path / "warm_out"),
    )


def test_missing_supervisor_params_raises_filenotfound(tmp_path):
    cfg = _basic_cfg(tmp_path, supervisor_schemes=["ftc"], params_paths={"ftc": str(tmp_path / "missing.json")})
    cfg.network.warm_start_from = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="ftc"):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)


def test_zero_captures_raises(tmp_path):
    p = tmp_path / "ftc.json"
    p.write_text("{}")
    cfg = _basic_cfg(tmp_path, params_paths={"ftc": str(p)})
    cfg.network.warm_start_from = str(p)

    def _all_fail(toml_path, seeds, overrides, scheme, sim_timeout_secs=None):
        return [
            {"seed": int(s), "X": np.zeros((5, 21)), "y_signed": np.zeros(5),
             "dv": 999.0, "captured": False}
            for s in seeds
        ]

    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_all_fail):
        with pytest.raises(RuntimeError, match="too small"):
            build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)


def test_clip_rate_above_threshold_raises(tmp_path):
    """Force clip rate > 5% by training with extreme target values and
    a tiny bound_multiplier so weights blow out of bounds."""
    p = tmp_path / "ftc.json"
    p.write_text("{}")
    cfg = _basic_cfg(tmp_path, params_paths={"ftc": str(p)})
    cfg.network.warm_start_from = str(p)
    cfg.warm_start.bound_multiplier = 0.01  # absurdly tight; will clip everything
    cfg.warm_start.n_epochs = 50            # ensure weights drift

    rng = np.random.default_rng(0)
    def _strong_targets(toml_path, seeds, overrides, scheme, sim_timeout_secs=None):
        return [
            {"seed": int(s),
             "X": rng.standard_normal((20, 21)),
             "y_signed": rng.uniform(-3.0, 3.0, size=20),  # large bank values
             "dv": 50.0, "captured": True}
            for s in seeds
        ]

    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_strong_targets):
        with pytest.raises(RuntimeError, match="clip rate"):
            build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)


def test_bptt_length_greater_than_n_seq_raises(tmp_path):
    p = tmp_path / "ftc.json"
    p.write_text("{}")
    cfg = _basic_cfg(tmp_path, params_paths={"ftc": str(p)})
    cfg.network.warm_start_from = str(p)
    cfg.network.architecture = [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "transformer", "d_model": 8, "n_heads": 2, "d_ffn": 16, "n_seq": 4},
        {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
    ]
    cfg.warm_start.bptt_length = 16  # > n_seq=4

    rng = np.random.default_rng(0)
    def _ok(toml_path, seeds, overrides, scheme, sim_timeout_secs=None):
        return [
            {"seed": int(s), "X": rng.standard_normal((40, 21)),
             "y_signed": np.zeros(40), "dv": 50.0, "captured": True}
            for s in seeds
        ]

    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_ok):
        with pytest.raises(ValueError, match="bptt_length.*n_seq"):
            build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_warm_start_failures.py -v`
Expected: all 4 cases pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_warm_start_failures.py
git commit -m "test(warm_start): failure-mode regression battery

Covers: missing supervisor params, zero captures across all seeds (corpus
too small), clip rate > 5% threshold, bptt_length > Transformer n_seq."
```

---

## Task 15: Python — cache invalidation tests

**Files:**
- Test: `tests/test_warm_start_cache.py`

- [ ] **Step 1: Write cache invalidation tests**

Create `tests/test_warm_start_cache.py`:

```python
"""Cache hit/miss matrix: changes to supervisor mtime, bound_multiplier,
architecture, input_mask, output_param, mode each invalidate the cache."""

import json
import time

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from aerocapture.training.config import (
    NetworkConfig, SimConfig, TrainingConfig, WarmStartConfig,
)
from aerocapture.training.warm_start import build_warm_start_chromosome


def _basic_cfg(tmp_path):
    p = tmp_path / "ftc_params.json"
    p.write_text(json.dumps({"k_alt": 1.0}))
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
        {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
    ]
    return TrainingConfig(
        network=NetworkConfig(
            architecture=arch, input_mask=[0, 1, 2, 3],
            output_parameterization="acos_tanh", warm_start_from=str(p),
        ),
        warm_start=WarmStartConfig(
            supervisor_schemes=["ftc"], params_paths={"ftc": str(p)},
            n_warm_seeds=4, n_epochs=1, bptt_length=8,
        ),
        sim=SimConfig(toml_config="dummy.toml"),
        save_dir=str(tmp_path / "out"),
    )


def _mock_collect(toml_path, seeds, overrides, scheme, sim_timeout_secs=None):
    rng = np.random.default_rng(int(seeds[0]) if len(seeds) else 0)
    return [
        {"seed": int(s), "X": rng.standard_normal((10, 21)),
         "y_signed": np.sin(rng.standard_normal(10)),
         "dv": 50.0, "captured": True}
        for s in seeds
    ]


def test_unchanged_cfg_hits_cache(tmp_path):
    cfg = _basic_cfg(tmp_path)
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect) as mock:
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count == 1
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect) as mock:
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count == 0  # cache hit


def test_supervisor_mtime_change_invalidates(tmp_path):
    cfg = _basic_cfg(tmp_path)
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
    # Touch the supervisor file
    time.sleep(0.01)
    Path(cfg.warm_start.params_paths["ftc"]).touch()
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect) as mock:
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count == 1  # cache miss


def test_bound_multiplier_change_invalidates(tmp_path):
    cfg = _basic_cfg(tmp_path)
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
    cfg.warm_start.bound_multiplier = 3.0
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect) as mock:
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count == 1


def test_architecture_change_invalidates(tmp_path):
    cfg = _basic_cfg(tmp_path)
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
    cfg.network.architecture[0]["output_size"] = 8
    cfg.network.architecture[1]["input_size"] = 8
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect) as mock:
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count == 1
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_warm_start_cache.py -v`
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_warm_start_cache.py
git commit -m "test(warm_start): cache hit/miss matrix"
```

---

## Task 16: Python — per-architecture smoke tests (slow)

**Files:**
- Test: `tests/test_warm_start_per_arch.py`

- [ ] **Step 1: Write the per-arch smoke**

Create `tests/test_warm_start_per_arch.py`:

```python
"""Per-architecture warm-start smoke: each of 6 layer types completes
end-to-end on a tiny config and produces a valid chromosome."""

import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch
from pydantic import TypeAdapter

import aerocapture_rs as r

from aerocapture.training.config import (
    NetworkConfig, SimConfig, TrainingConfig, WarmStartConfig,
)
from aerocapture.training.encoding import nn_param_specs_from_v2
from aerocapture.training.rl.schemas import LayerSpec
from aerocapture.training.warm_start import build_warm_start_chromosome


def _ftc_params(tmp_path):
    p = tmp_path / "ftc_params.json"
    p.write_text(json.dumps({"k_alt": 1.0}))
    return p


def _mock_collect(traj_T=40, input_dim=21):
    rng = np.random.default_rng(0)
    def _inner(toml_path, seeds, overrides, scheme, sim_timeout_secs=None):
        return [
            {"seed": int(s), "X": rng.standard_normal((traj_T, input_dim)),
             "y_signed": np.sin(rng.standard_normal(traj_T)),
             "dv": 50.0, "captured": True}
            for s in seeds
        ]
    return _inner


@pytest.mark.slow
@pytest.mark.parametrize("arch_name, arch", [
    ("dense", [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
    ]),
    ("window", [
        {"type": "window", "input_size": 4, "n_steps": 3},
        {"type": "dense", "input_size": 12, "output_size": 8, "activation": "tanh"},
        {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
    ]),
    ("gru", [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "gru", "input_size": 8, "hidden_size": 8},
        {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
    ]),
    ("lstm", [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "lstm", "input_size": 8, "hidden_size": 8},
        {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
    ]),
    ("transformer", [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "transformer", "d_model": 8, "n_heads": 2, "d_ffn": 16, "n_seq": 16},
        {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
    ]),
    ("mamba", [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2},
        {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
    ]),
])
def test_warm_start_per_arch_smoke(arch_name, arch, tmp_path):
    p = _ftc_params(tmp_path)
    cfg = TrainingConfig(
        network=NetworkConfig(
            architecture=arch, input_mask=[0, 1, 2, 3],
            output_parameterization="acos_tanh", warm_start_from=str(p),
        ),
        warm_start=WarmStartConfig(
            supervisor_schemes=["ftc"], params_paths={"ftc": str(p)},
            n_warm_seeds=4, n_epochs=1, bptt_length=8,
            bound_multiplier=10.0,  # generous; smoke focuses on plumbing, not clipping
        ),
        sim=SimConfig(toml_config="dummy.toml"),
        save_dir=str(tmp_path / f"warm_out_{arch_name}"),
    )

    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised",
               side_effect=_mock_collect()):
        chromo = build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)

    # Width matches param specs
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    expected_width = len(nn_param_specs_from_v2(validated, bound_multiplier=10.0))
    assert chromo.shape == (expected_width,)
    assert (chromo >= 0.0).all() and (chromo <= 1.0).all()

    # Quick decode + Rust forward: build a JSON via flat_weights_to_json from the un-normalized weights
    # and assert nn_forward returns a finite output.
    weight_specs = nn_param_specs_from_v2(validated, bound_multiplier=10.0)
    flat = np.array([s.p_min + chromo[i] * (s.p_max - s.p_min) for i, s in enumerate(weight_specs)])
    json_path = tmp_path / f"model_{arch_name}.json"
    r.flat_weights_to_json(arch, flat.tolist(), str(json_path))
    out = r.nn_forward(str(json_path), [0.1, 0.2, 0.3, 0.4])
    assert all(np.isfinite(out)), f"arch={arch_name} nn_forward output {out} not finite"
```

- [ ] **Step 2: Run (slow)**

Run: `uv run pytest tests/test_warm_start_per_arch.py -v -m slow`
Expected: all 6 cases pass.

If a layer type's roundtrip fails because the chromosome decodes outside Xavier-ish bounds, widen `bound_multiplier` in the test to ~20 — the smoke is about plumbing correctness, not bound tightness.

- [ ] **Step 3: Commit**

```bash
git add tests/test_warm_start_per_arch.py
git commit -m "test(warm_start): per-architecture smoke for all 6 layer types"
```

---

## Task 17: Python — equivalence gate ("at least as good") regression (slow)

**Files:**
- Test: `tests/test_warm_start_equivalence_gate.py`

- [ ] **Step 1: Write the regression**

Create `tests/test_warm_start_equivalence_gate.py`:

```python
"""magnitude_only warm-start with supervisor_schemes=['ftc'] should be at
least as good as the pre-refactor pipeline (within 5% slack on validation
RMS after 20 GA generations under a fixed seed).

Slow / E2E: requires Rust + a trained FTC scheme available at
training_output/ftc/best_params.json. Skipped if the FTC artifact is missing.
"""

import os
import subprocess
import json

import pytest
from pathlib import Path


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("training_output/ftc/best_params.json").exists(),
    reason="requires trained FTC scheme",
)
def test_magnitude_only_at_least_as_good(tmp_path):
    # Run a fresh 20-gen training under fixed seed; compare validation RMS
    # to the snapshot baseline file (`tests/fixtures/warm_start_magonly_baseline.json`).
    # The baseline is the pre-refactor validation RMS for the same config.
    baseline_path = Path("tests/fixtures/warm_start_magonly_baseline.json")
    if not baseline_path.exists():
        pytest.skip("baseline snapshot missing; record one with the pre-refactor pipeline first")
    baseline = json.loads(baseline_path.read_text())["val_rms_after_20_gens"]

    out_dir = tmp_path / "training_run"
    cmd = [
        "uv", "run", "python", "-m", "aerocapture.training.train",
        "configs/training/msr_aller_nn_train_consolidated.toml",
        "--n-gen", "20", "--n-pop", "32",
        "--no-tui", "--skip-report",
        "--output-dir", str(out_dir),
    ]
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
    assert result.returncode == 0, f"train.py failed: {result.stderr}"

    # Parse best validation RMS from JSONL log
    jsonl_paths = sorted(out_dir.glob("*.jsonl"))
    assert jsonl_paths, f"no JSONL log in {out_dir}"
    best_rms = float("inf")
    for line in jsonl_paths[0].read_text().splitlines():
        rec = json.loads(line)
        rms = rec.get("validation", {}).get("rms_cost")
        if rms is not None and rms < best_rms:
            best_rms = rms
    assert best_rms != float("inf"), "no validation RMS in log"
    assert best_rms <= baseline * 1.05, (
        f"warm-start regression: post-refactor RMS {best_rms:.3f} > baseline {baseline:.3f} + 5% slack"
    )
```

- [ ] **Step 2: Record the baseline once (manual step, before merging)**

Run the pre-refactor pipeline (the existing code on `main` before this branch is merged) with the same 20-gen, 32-pop, seed-fixed config; extract best validation RMS; write to `tests/fixtures/warm_start_magonly_baseline.json`:

```json
{
  "val_rms_after_20_gens": 14.69,
  "recorded_at": "2026-05-22",
  "config": "msr_aller_nn_train_consolidated.toml",
  "n_gen": 20,
  "n_pop": 32
}
```

(Replace the value with the actual measured RMS from a pre-refactor run. The 14.69 from CLAUDE.md is the long-run best, not 20-gen; the 20-gen value will be higher.)

- [ ] **Step 3: Run the regression**

Run: `uv run pytest tests/test_warm_start_equivalence_gate.py -v -m slow`
Expected: pass when post-refactor RMS is ≤ baseline + 5%.

- [ ] **Step 4: Commit**

```bash
git add tests/test_warm_start_equivalence_gate.py tests/fixtures/warm_start_magonly_baseline.json
git commit -m "test(warm_start): equivalence gate against pre-refactor magnitude_only baseline

20-gen GA run under fixed seed must achieve validation RMS within 5% of
the pre-refactor pipeline. Baseline snapshot recorded once from the
pre-refactor codebase; failures here indicate the new target signal
(abs(final_signed_bank) vs pre_lateral_magnitude) materially hurt the
magnitude_only basin and warrant investigation."
```

---

## Task 18: Configs + final smart-commit

**Files:**
- Modify: each NN training TOML to add a `[warm_start]` block

- [ ] **Step 1: Inventory the NN training configs**

Run: `ls configs/training/msr_aller_*nn*.toml configs/training/msr_aller_*gru*.toml configs/training/msr_aller_*lstm*.toml configs/training/msr_aller_*transformer*.toml configs/training/msr_aller_*mamba*.toml configs/training/msr_aller_*window*.toml configs/training/msr_aller_*ftc*.toml 2>/dev/null`

Confirmed list:
- `configs/training/msr_aller_nn_train_consolidated.toml`
- `configs/training/msr_aller_nn_joint_train.toml`
- `configs/training/msr_aller_gru_pso_train.toml`
- `configs/training/msr_aller_gru_pso_magonly_train.toml`
- `configs/training/msr_aller_gru_ppo_train.toml` (PPO — no warm-start applies here, skip)
- `configs/training/msr_aller_lstm_pso_train.toml`
- `configs/training/msr_aller_window_pso_train.toml`
- `configs/training/msr_aller_transformer_pso_train.toml`
- `configs/training/msr_aller_mamba_pso_train.toml`
- (`msr_aller_lstm_ppo_train.toml` — PPO, skip)

- [ ] **Step 2: For each non-PPO NN config, append a `[warm_start]` block**

For each TOML in the list above (excluding `*_ppo_*`), add at the end of the file:

```toml
[warm_start]
supervisor_schemes = ["ftc", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag"]
bptt_length = 32
n_warm_seeds = 200
n_epochs = 10
bound_multiplier = 4.0
jitter = 0.02
cmaes_sigma0 = 0.1
```

For Transformer configs specifically, ensure `bptt_length <= n_seq` (transformer config has `n_seq = 64`, so `bptt_length = 32` is fine).

For Window configs, the chromosome width is determined by post-Window Dense layers only (Window contributes 0 trainable params), so warm-start works without special handling.

- [ ] **Step 3: Run a 1-gen smoke per arch to verify the configs load**

For each updated config:
```bash
uv run python -m aerocapture.training.train <config.toml> --n-gen 1 --n-pop 4 --no-tui --skip-report
```
Expected: each run completes without config-load errors. Warm-start activates when `warm_start_from` is set (which is the existing behavior). For configs that don't set `warm_start_from`, the `[warm_start]` block is dormant.

- [ ] **Step 4: Commit configs**

```bash
git add configs/training/msr_aller_*.toml
git commit -m "feat(configs): add [warm_start] block to all PSO/GA NN training configs

5 supervisor schemes by default, bptt_length=32 (compatible with the
transformer n_seq=64), bound_multiplier=4.0. Warm-start still activates
only when [guidance.neural_network] warm_start_from is set — this block
just configures the tunables when it does."
```

- [ ] **Step 5: Final invocation of `smart-commit` for the whole branch (per user's CLAUDE.md)**

Per the user's planning convention, end the plan with the `smart-commit` skill applied to the branch. Run the skill via the Skill tool:

```
Skill(smart-commit)
```

Instruct smart-commit to take the whole `feature/warm_start_full_neural_with_ftc_and_friends` branch into account so any straggler docs (CLAUDE.md memory note about the new pipeline, TODO.md status update) get updated and committed.

---

## Validation Checklist (post-merge)

These are not bite-sized tasks — they are the user's responsibility to verify after the implementation merges:

- [ ] Run `train_all.sh nn` end-to-end and confirm convergence speed improvement vs the pre-refactor pipeline (the main reason for this work — full_neural NN should converge faster from warm-start than from random Xavier init).
- [ ] Spot-check `<save_dir>/warm_start_loss.json` for a trained run and verify the loss curve is sensible (decreasing, not still falling at the last epoch).
- [ ] Spot-check the gen-0 validation baseline log for one run and confirm the value is consistent with the supervised target (`abs(final_signed_bank)` for magnitude_only NNs should give a baseline close to FTC's own validation RMS).
- [ ] If results regress, the first knob to try is `warm_start.bound_multiplier` (widen further) or `n_epochs` (reduce to avoid teacher overfit).
