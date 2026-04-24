# Phase 4a -- Mamba Selective SSM MVP (PSO-only)

**Date:** 2026-04-24
**Status:** Design approved, ready for implementation planning
**Parent effort:** LSTM / Transformer / Mamba architectures for neural guidance (see `TODO.md`).
**Predecessors:** Phase 0 (stateful NN runtime infrastructure), Phase 1 (PSO-GRU), Phase 1.5 (PPO-GRU + truncated BPTT), Phase 2a (LSTM MVP + activation-aware init), Phase 2b (Window-MLP, PSO-only), Phase 3a (Transformer MVP, PSO-only).

## 1. Context

The paper experimental grid has one remaining architecture cell unshipped: **Mamba**, the selective state-space model from Gu & Dao 2023 ("Mamba: Linear-Time Sequence Modeling with Selective State Spaces"). Phases 0/1/1.5/2a/2b/3a have shipped:

- Runtime infrastructure for stateful NN layers with JSON format v2.
- The extensibility contract for scalar-state, multi-tensor-state, zero-trainable-parameter, and PSO-only layer types.
- Activation-aware init (Phase 2a carry-over).
- Derived-at-load-time per-layer fields (Phase 3a's PE-offset precompute pattern).
- End-to-end training on five of six paper architectures (MLP baseline, Window-MLP, GRU, LSTM, Transformer).

Phase 4a adds **Mamba**: a single-layer selective SSM core with diagonal `A`, input-dependent `Δ / B / C` projections, ZOH discretization, and HiPPO-style A init. Per-tick inference computes input-dependent discretization parameters from `x_t`, updates a per-channel SSM state `h ∈ R^{d_inner × d_state}`, and emits `y = h @ C + D * x`.

The TODO grid schedules Mamba under both PSO and BPTT-PPO. Phase 4a scopes **PSO only**, matching the Phase 2b / 3a staging. Rationale:

1. **Selectivity is novel numerical ground.** ZOH discretization introduces per-step `exp()` and the `(exp(z) - 1) / z` Taylor-crossover numerical trap; input-dependent `Δ / B / C` requires matching the softplus + dt_proj bias init bit-for-bit across Rust and Python; HiPPO `A = -exp(A_log)` reparameterization must thread cleanly through both the chromosome encoding and the JSON round-trip. Pinning PSO first validates the runtime before stacking BPTT on top.
2. **State shape is new.** Mamba's state is a single 2D tensor per layer (`(d_inner, d_state)`), not the flat vectors GRU uses or the tuple LSTM uses. PPO rollout buffers would need a `(T, B, d_inner, d_state)` slab (ndim==4) -- a new dispatch branch in `_zero_state_where_done`, `hidden_shapes`, `_np_state_to_torch` / `_torch_state_to_np`, and `ppo_update_bptt`. That dispatch deserves its own phase.
3. **Paper result already covered.** The MLP vs Mamba PSO comparison fully populates the "PSO row" of the paper's experimental grid once Phase 4a lands.

The PPO path gets a **clean error at build time**: `build_layer(MambaSpec)` raises `NotImplementedError` with a pointer to this spec. Same for `load_policy_from_json`. PSO bypasses `V2Policy` entirely (Rust forward via `aerocapture_rs.nn_forward`), so PSO training is unaffected.

**Variant choice (recorded for paper narrative):** Phase 4a ships the **selective SSM core only**, not the full Mamba block. The full block (conv1d pre-filter + SiLU gating + in/out expansion linears + block-level residual) is deferred to Phase 4c-or-never. Rationale: selectivity (input-dependent `Δ / B / C`) is what distinguishes Mamba from S4; conv1d + gating are block-level machinery that roughly triples parameter count and complicates the flat-weight layout without adding much to the aerocapture trajectory story (a smooth, low-bandwidth control signal). Users who want dim adaptation stack Dense layers before/after the Mamba layer, same pattern as GRU / LSTM / Transformer.

## 2. Scope

**In scope:**

- Rust `MambaLayer` struct with fields for `x_proj_w` (fused Δ/B/C projection), `dt_proj_w` + `dt_proj_b` (Δ bottleneck + softplus bias), `a_log` (HiPPO diagonal A reparameterization), `d_skip` (residual scalar per channel). Stored dims: `input_size` (= d_inner = d_out), `d_state`, `dt_rank`.
- `Layer::Mamba(Box<MambaLayer>)` boxed per Phase 3a's `large_enum_variant` clippy precedent.
- `LayerSpec::Mamba { input_size, d_state, dt_rank }` (all `usize`; no activation enum -- the layer's nonlinearities are softplus on Δ and ZOH exp on A, both hard-coded).
- `LayerState::Mamba { h: DMatrix<f64> }` -- single-tensor 2D state of shape `(input_size, d_state)`, zero-initialized via `DMatrix::zeros(...)` in `LayerState::for_layer`.
- `TomlLayerSpec::Mamba { input_size: usize, d_state: usize, dt_rank: Option<usize> }` with `to_layer_spec` resolving `dt_rank.unwrap_or(max(1, input_size / 16))`. Validator: `input_size > 0`, `d_state > 0`, resolved `dt_rank > 0`, `dt_rank <= input_size`.
- `LayerWeights for MambaLayer`: canonical flat ordering documented in section 3.3; `from_flat` reconstructs the layer directly (no derived-at-load-time fields in Phase 4a -- `A = -exp(A_log)` is evaluated per forward, not cached).
- `NeuralNetModel::save_json` / `from_v2_json` / `from_flat_weights_v2` Mamba arms. JSON v2 entry includes spec fields plus a `weights` dict with 5 flat arrays: `x_proj_w`, `dt_proj_w`, `dt_proj_b`, `a_log`, `d_skip`.
- `NeuralNetModel::forward` routes through `Layer::Mamba` with the single-token forward defined in section 3.4: fused `x_proj` -> split into (Δ_pre, B, C) -> `dt_proj` + softplus -> ZOH discretization -> state update -> `y = h @ C + D * x`. Module-level free helpers `softplus(x: f64)` and `expm1_over_x(z: f64)` (both `pub(crate)` for unit testing and reuse).
- PyTorch `MambaLayer` module in `rl/layers/mamba.py` with forward `(x: Tensor, h: Tensor) -> (y: Tensor, h_new: Tensor)`. Manual softplus (matching the numerically stable form), manual `expm1_over_x` with `torch.where` Taylor fallback, manual ZOH (no `mamba_ssm` package dep, no `selective_scan_cuda`). Consumed only by the cross-language equivalence test; PPO rejects the layer.
- `MambaSpec` pydantic class appended to the `LayerSpec` discriminated union on the `type` field. Fields: `input_size: PositiveInt`, `d_state: PositiveInt`, `dt_rank: int | None = None`. A `model_validator(mode='after')` resolves `dt_rank` to `max(1, input_size // 16)` when None, then enforces `1 <= dt_rank <= input_size`. All downstream code reads `spec.dt_rank` as the resolved value.
- `build_layer` in `rl/layers/__init__.py` dispatches `MambaSpec` to **raise `NotImplementedError`** with message "Mamba is PSO-only in Phase 4a; PPO use deferred -- see docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md".
- `_layer_param_specs` Mamba dispatch returns the PSO ParamSpec list documented in section 3.6 (Xavier on projections, paper-style `dt_rank^{-0.5}` scaling on `dt_proj_w`, per-channel `inv_softplus(uniform(dt_min, dt_max))` init centers for `dt_proj_b`, HiPPO `log(n+1)` init centers for `a_log`, init value 1.0 for `d_skip`).
- `nn_param_specs_from_v2` naturally picks up the Mamba arm through `_layer_param_specs`.
- `config.py::_layer_n_params` Mamba arm returns `input_size * (3 * d_state + 2 * dt_rank + 2)`; `_layer_output_size` returns `input_size`.
- `describe_architecture` Mamba arm alongside dense / gru / lstm / window / transformer.
- `export_v2_policy_to_json`: Mamba branch writes the 5-key `weights` dict flat at layer level (matching Rust `NnLayerWeights` schema). Obs-norm bake-in guard extended to reject `MambaSpec` as layer 0 (affine absorption would need to touch both `x_proj_w` and `dt_proj_b` through the softplus nonlinearity, nontrivial and not required for PSO).
- `load_policy_from_json`: raises `NotImplementedError` when any layer is `MambaSpec`, consistent with `build_layer`.
- `init_v2_population`: Mamba dispatch writes per-element init values matching the `init_center` field of the ParamSpec list from `_mamba_specs`. See section 3.6 for the exact per-slice init rules.
- Training config `configs/training/msr_aller_mamba_pso_train.toml`: Dense(23 -> 32, swish) -> Mamba(d_inner=32, d_state=16) -> Mamba(d_inner=32, d_state=16) -> Dense(32 -> 2, asinh). `dt_rank` omitted in both Mamba layers (resolves to 2 via `max(1, 32/16)`). PSO `n_pop=64 n_gen=2000 seed_strategy="adaptive"`, `training_n_sims=20`, `validation_n_sims=1000`. **Total: 4290 trainable params.**
- `compare_guidance.SCHEMES` / `SCHEME_TRAINING_CONFIGS` / `_NN_DEPLOY_SCHEMES` registration as `neural_network_mamba_pso` (goes through the Rust `neural_network` runtime, same as GRU-PSO / LSTM-PSO / Window-PSO / Transformer-PSO).
- `train_all.sh` aliases: `mamba_pso`, `nn_mamba_pso`, `mamba`.
- Cross-language equivalence test `test_rust_python_mamba_equivalence.py`: architecture = Dense(4 -> 8, tanh) -> Mamba(d_inner=8, d_state=4, dt_rank=2) -> Dense(8 -> 2, linear). Uses `aerocapture_rs.nn_forward_sequence` to thread a single `NnState` across 100 random f64 inputs; Python forward threads `h` through `V2Policy.forward_mean_logstd` explicitly. Asserts max abs diff < 1e-14, target machine epsilon.
- Warm-up test `test_mamba_warmup.py`: verifies state starts at zero and evolves deterministically, catching state-init bugs that could silently break cross-language gates.
- PSO smoke test `test_mamba_pso_smoke.py` (@slow, python-pyo3 CI job): 2-gen PSO on reduced arch Dense(23 -> 8, tanh) -> Mamba(8, 4, 1) -> Dense(8 -> 2, linear) (~440 params), `n_pop=4`, `training_n_sims=4`, asserts `best_model.json` is v2 with `["dense", "mamba", "dense"]` and `nn_forward` returns a finite 2-tuple.
- PPO-rejection test `test_mamba_ppo_rejection.py` (@fast, main python job): constructs a minimal Mamba v2 JSON and asserts `load_policy_from_json` + `build_layer` both raise `NotImplementedError` with the expected message fragment.
- `ci.yml` workflow extension: four new tests added (`test_rust_python_mamba_equivalence.py`, `test_mamba_warmup.py`, `test_mamba_pso_smoke.py` in python-pyo3 job; `test_mamba_ppo_rejection.py` in the main python job).
- `CLAUDE.md` + `TODO.md` sync after landing: Phase 4a checkbox done, extensibility contract updated to document Mamba's role as the first 2D-single-tensor state layer, Phase 4b (Mamba PPO) confirmed open.

**Out of scope:**

- **PPO-BPTT for Mamba.** Deferred to Phase 4b. PPO path errors at `build_layer` time.
- **`_zero_state_where_done` extension for 2D single-tensor state.** Helper already handles flat `Tensor` (Gru) and `tuple` (Lstm). Mamba's 2D state would slot in as another tensor branch (ndim==3 in the rollout buffer: `(B, d_inner, d_state)`). Phase 4a doesn't need this because Mamba never reaches V2Policy.
- **Rollout buffer `hidden_shapes` / ndim==4 dispatch for Mamba.** Same reasoning; Phase 4b.
- **Obs-normalizer bake-in into Mamba as layer 0.** `export_v2_policy_to_json` raises `NotImplementedError`. Deferred until Phase 4b or later; requires deriving the shift for `dt_proj_b` through softplus, which isn't closed-form.
- **Full Mamba block** (conv1d pre-filter + SiLU gating branch + in/out projections + block residual). Phase 4c-or-never. Would ship as `LayerSpec::MambaBlock` distinct from `LayerSpec::Mamba`.
- **Parallel scan / selective_scan_cuda.** Per-step recurrence only. The paper's parallel scan optimization is a training-time throughput win irrelevant to PSO (we run the full rollout env-by-env per step anyway) and to the online guidance use case (we emit one bank angle per GNC tick).
- **Alternative A parameterizations** (full matrix, complex diagonal, S5 block-diagonal). Phase 4a ships diagonal real with HiPPO init -- the paper's own "Mamba S6" configuration.
- **Activation-aware init beyond Xavier + paper-Mamba init.** Mamba has no user-facing activation dial; softplus and exp are hard-coded. No He / LeCun variant needed.
- **SAC-Mamba.** SAC umbrella still deferred since Phase 1.6.
- **Stacked Mamba beyond 2 layers in the default config.** Stacking is a TOML-level concern; users can add more `[[network.architecture]] type = "mamba"` blocks with no runtime change. The default config ships 2 layers as the paper baseline.

## 3. Detailed design

### 3.1 Rust LayerSpec + Layer + MambaLayer

```rust
// src/rust/src/data/neural.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LayerSpec {
    Dense       { input_size: usize, output_size: usize, activation: Activation },
    Gru         { input_size: usize, hidden_size: usize },
    Lstm        { input_size: usize, hidden_size: usize },
    Window      { input_size: usize, n_steps: usize },
    Transformer { d_model: usize, n_heads: usize, d_ffn: usize, n_seq: usize },
    Mamba       { input_size: usize, d_state: usize, dt_rank: usize },
}

#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
    Lstm(LstmLayer),
    Window(WindowLayer),
    Transformer(Box<TransformerLayer>),
    Mamba(Box<MambaLayer>),
}

#[derive(Debug, Clone)]
pub struct MambaLayer {
    pub input_size: usize,     // d_inner -- layer fan-in and fan-out
    pub d_state:    usize,     // N in paper
    pub dt_rank:    usize,

    pub x_proj_w:   DMatrix<f64>,  // (dt_rank + 2 * d_state, input_size), no bias
    pub dt_proj_w:  DMatrix<f64>,  // (input_size, dt_rank)
    pub dt_proj_b:  DVector<f64>,  // (input_size,)
    pub a_log:      DMatrix<f64>,  // (input_size, d_state)  -- A = -exp(A_log)
    pub d_skip:     DVector<f64>,  // (input_size,)          -- scalar residual per channel
}
```

### 3.2 LayerState + TomlLayerSpec

```rust
// src/rust/src/data/nn_state.rs

#[derive(Debug, Clone)]
pub enum LayerState {
    None,
    Gru(Vec<f64>),
    Lstm { h: Vec<f64>, c: Vec<f64> },
    Window { buffer: VecDeque<Vec<f64>> },
    Transformer { k_cache: VecDeque<Vec<f64>>, v_cache: VecDeque<Vec<f64>> },
    Mamba { h: DMatrix<f64> },  // shape (input_size, d_state)
}

impl LayerState {
    pub fn for_layer(layer: &Layer) -> Self {
        match layer {
            Layer::Dense(_)       => LayerState::None,
            Layer::Gru(g)         => LayerState::Gru(vec![0.0; g.hidden_size]),
            Layer::Lstm(l)        => LayerState::Lstm { h: vec![0.0; l.hidden_size], c: vec![0.0; l.hidden_size] },
            Layer::Window(w)      => LayerState::Window { buffer: VecDeque::from(vec![vec![0.0; w.input_size]; w.n_steps]) },
            Layer::Transformer(_) => LayerState::Transformer { k_cache: VecDeque::new(), v_cache: VecDeque::new() },
            Layer::Mamba(m)       => LayerState::Mamba { h: DMatrix::zeros(m.input_size, m.d_state) },
        }
    }

    pub fn reset(&mut self) {
        match self {
            // ... existing arms ...
            LayerState::Mamba { h } => h.fill(0.0),
        }
    }
}
```

```rust
// src/rust/src/config.rs

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TomlLayerSpec {
    // ... existing variants ...
    Mamba {
        input_size: usize,
        d_state:    usize,
        #[serde(default)]
        dt_rank:    Option<usize>,
    },
}

impl TomlLayerSpec {
    pub fn to_layer_spec(&self) -> Result<LayerSpec, String> {
        match self {
            TomlLayerSpec::Mamba { input_size, d_state, dt_rank } => {
                if *input_size == 0 { return Err("Mamba: input_size must be > 0".into()); }
                if *d_state    == 0 { return Err("Mamba: d_state must be > 0".into()); }
                let resolved = dt_rank.unwrap_or_else(|| (*input_size / 16).max(1));
                if resolved == 0              { return Err("Mamba: dt_rank must be > 0".into()); }
                if resolved > *input_size     { return Err(format!("Mamba: dt_rank ({resolved}) must be <= input_size ({input_size})")); }
                Ok(LayerSpec::Mamba { input_size: *input_size, d_state: *d_state, dt_rank: resolved })
            }
            // ... existing arms ...
        }
    }
}
```

### 3.3 Canonical flat-weight ordering (PSO chromosome layout)

The entire PSO path relies on Rust `LayerWeights::to_flat` / `from_flat` and Python `_mamba_specs` / `export_v2_policy_to_json` emitting the exact same f64 sequence. Canonical order:

| # | Field        | Shape                                     | Params                              | Flatten order |
|---|--------------|-------------------------------------------|-------------------------------------|---------------|
| 1 | `x_proj_w`   | `(dt_rank + 2 * d_state, input_size)`     | `input_size * (dt_rank + 2*d_state)`| row-major     |
| 2 | `dt_proj_w`  | `(input_size, dt_rank)`                   | `input_size * dt_rank`              | row-major     |
| 3 | `dt_proj_b`  | `(input_size,)`                           | `input_size`                        | contiguous    |
| 4 | `a_log`      | `(input_size, d_state)`                   | `input_size * d_state`              | row-major     |
| 5 | `d_skip`     | `(input_size,)`                           | `input_size`                        | contiguous    |

**Total:** `input_size * (3 * d_state + 2 * dt_rank + 2)`.

Example: `(input_size=32, d_state=16, dt_rank=2)` -> `32 * (48 + 4 + 2) = 1728 params`.

```rust
impl LayerWeights for MambaLayer {
    fn n_params(&self) -> usize {
        self.input_size * (3 * self.d_state + 2 * self.dt_rank + 2)
    }

    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_params());
        // 1. x_proj_w row-major
        for i in 0..self.x_proj_w.nrows() { for j in 0..self.x_proj_w.ncols() { out.push(self.x_proj_w[(i, j)]); } }
        // 2. dt_proj_w row-major
        for i in 0..self.dt_proj_w.nrows() { for j in 0..self.dt_proj_w.ncols() { out.push(self.dt_proj_w[(i, j)]); } }
        // 3. dt_proj_b
        for i in 0..self.dt_proj_b.len() { out.push(self.dt_proj_b[i]); }
        // 4. a_log row-major
        for i in 0..self.a_log.nrows() { for j in 0..self.a_log.ncols() { out.push(self.a_log[(i, j)]); } }
        // 5. d_skip
        for i in 0..self.d_skip.len() { out.push(self.d_skip[i]); }
        out
    }

    fn from_flat(spec: &LayerSpec, flat: &[f64]) -> Result<(Self, usize), String> {
        let LayerSpec::Mamba { input_size, d_state, dt_rank } = spec else {
            return Err("from_flat called with non-Mamba spec".into());
        };
        let (input_size, d_state, dt_rank) = (*input_size, *d_state, *dt_rank);
        let expected = input_size * (3 * d_state + 2 * dt_rank + 2);
        if flat.len() < expected {
            return Err(format!("Mamba: flat slice too short (need {expected}, got {})", flat.len()));
        }

        let mut cursor = 0;
        // 1. x_proj_w
        let rows = dt_rank + 2 * d_state;
        let cols = input_size;
        let x_proj_w = DMatrix::from_row_slice(rows, cols, &flat[cursor .. cursor + rows * cols]);
        cursor += rows * cols;
        // 2. dt_proj_w
        let dt_proj_w = DMatrix::from_row_slice(input_size, dt_rank, &flat[cursor .. cursor + input_size * dt_rank]);
        cursor += input_size * dt_rank;
        // 3. dt_proj_b
        let dt_proj_b = DVector::from_row_slice(&flat[cursor .. cursor + input_size]);
        cursor += input_size;
        // 4. a_log
        let a_log = DMatrix::from_row_slice(input_size, d_state, &flat[cursor .. cursor + input_size * d_state]);
        cursor += input_size * d_state;
        // 5. d_skip
        let d_skip = DVector::from_row_slice(&flat[cursor .. cursor + input_size]);
        cursor += input_size;

        Ok((MambaLayer { input_size, d_state, dt_rank, x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip }, cursor))
    }
}
```

### 3.4 Forward pass (single-token inference)

```rust
// src/rust/src/data/neural.rs

pub(crate) fn softplus(x: f64) -> f64 {
    // Numerically stable: max(x, 0) + log1p(exp(-|x|))
    // Python mirror (`rl/layers/mamba.py::_softplus`) uses the identical manual form, NOT
    // torch.nn.functional.softplus (which has a threshold=20 linear-branch fallback we do not want).
    // Bit-identical across Rust libm and PyTorch f64 on the same platform.
    let a = x.abs();
    x.max(0.0) + (-a).exp().ln_1p()
}

pub(crate) fn expm1_over_x(z: f64) -> f64 {
    // (exp(z) - 1) / z, stable at z -> 0 via Taylor
    // For |z| < 1e-8: 1 + z/2 + z^2/6  (error ~ z^3/24, machine epsilon for |z| < ~1e-5)
    // For |z| >= 1e-8: use expm1(z) / z  (libm expm1 is accurate down to this range)
    if z.abs() < 1e-8 {
        1.0 + z * 0.5 + z * z / 6.0
    } else {
        z.exp_m1() / z
    }
}

impl MambaLayer {
    pub fn forward(&self, x: &[f64], h: &mut DMatrix<f64>) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.input_size);
        debug_assert_eq!(h.nrows(), self.input_size);
        debug_assert_eq!(h.ncols(), self.d_state);

        let x_vec = DVector::from_row_slice(x);

        // 1. Fused x_proj: (dt_rank + 2*d_state,)
        let proj = &self.x_proj_w * &x_vec;
        let dt_pre: Vec<f64> = (0..self.dt_rank).map(|i| proj[i]).collect();
        let b_vec:  Vec<f64> = (0..self.d_state).map(|i| proj[self.dt_rank + i]).collect();
        let c_vec:  Vec<f64> = (0..self.d_state).map(|i| proj[self.dt_rank + self.d_state + i]).collect();

        // 2. dt_proj + softplus
        let dt_pre_v = DVector::from_row_slice(&dt_pre);
        let dt_lifted = &self.dt_proj_w * &dt_pre_v + &self.dt_proj_b;  // (input_size,)
        let delta: Vec<f64> = (0..self.input_size).map(|i| softplus(dt_lifted[i])).collect();

        // 3. Per-channel, per-state ZOH discretization + state update
        //    Ā[d, n] = exp(Δ[d] * A[d, n])  where A = -exp(a_log)
        //    B̄[d, n] = Δ[d] * B[n] * expm1_over_x(Δ[d] * A[d, n])
        //    h[d, n] = Ā[d, n] * h[d, n] + B̄[d, n] * x[d]
        //    y[d]    = Σ_n (h[d, n] * C[n])  +  D[d] * x[d]
        let mut y = vec![0.0; self.input_size];
        for d in 0..self.input_size {
            let delta_d = delta[d];
            let x_d = x[d];
            let mut acc = 0.0;
            for n in 0..self.d_state {
                let a_dn = -self.a_log[(d, n)].exp();              // A = -exp(a_log)
                let za = delta_d * a_dn;                            // Δ·A
                let a_bar = za.exp();                               // Ā
                let b_bar = delta_d * b_vec[n] * expm1_over_x(za);  // B̄
                h[(d, n)] = a_bar * h[(d, n)] + b_bar * x_d;
                acc += h[(d, n)] * c_vec[n];
            }
            y[d] = acc + self.d_skip[d] * x_d;
        }
        y
    }
}
```

**Numerical contract:** `softplus`, `expm1_over_x`, and the ZOH recurrence must produce bit-identical f64 output against the PyTorch mirror. See section 3.5 for the Python mirror and section 4.1 for the equivalence gate.

### 3.5 Python mirror (`rl/layers/mamba.py`)

```python
# src/python/aerocapture/training/rl/layers/mamba.py

import math
import torch
from torch import Tensor, nn


def _softplus(x: Tensor) -> Tensor:
    # Matches Rust softplus(): max(x, 0) + log1p(exp(-|x|))
    return x.clamp_min(0.0) + torch.log1p(torch.exp(-x.abs()))


def _expm1_over_x(z: Tensor) -> Tensor:
    # (exp(z) - 1) / z with Taylor fallback for |z| < 1e-8
    taylor = 1.0 + 0.5 * z + (z * z) / 6.0
    exact = torch.expm1(z) / z.where(z != 0.0, torch.ones_like(z))  # avoid div-by-zero on zero entries
    return torch.where(z.abs() < 1e-8, taylor, exact)


class MambaLayer(nn.Module):
    def __init__(self, input_size: int, d_state: int, dt_rank: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.x_proj_w  = nn.Parameter(torch.zeros(dt_rank + 2 * d_state, input_size))
        self.dt_proj_w = nn.Parameter(torch.zeros(input_size, dt_rank))
        self.dt_proj_b = nn.Parameter(torch.zeros(input_size))
        self.a_log     = nn.Parameter(torch.zeros(input_size, d_state))
        self.d_skip    = nn.Parameter(torch.zeros(input_size))

    def new_state(self) -> Tensor:
        # shape (input_size, d_state), dtype tracks parameter dtype so policy.double() propagates
        return torch.zeros(self.input_size, self.d_state, dtype=self.x_proj_w.dtype, device=self.x_proj_w.device)

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        # x: (input_size,), h: (input_size, d_state)
        assert x.shape == (self.input_size,)
        assert h.shape == (self.input_size, self.d_state)

        # 1. Fused x_proj
        proj = self.x_proj_w @ x                                # (dt_rank + 2*d_state,)
        dt_pre = proj[: self.dt_rank]
        b_vec  = proj[self.dt_rank : self.dt_rank + self.d_state]                     # (d_state,)
        c_vec  = proj[self.dt_rank + self.d_state : self.dt_rank + 2 * self.d_state]  # (d_state,)

        # 2. dt_proj + softplus
        dt_lifted = self.dt_proj_w @ dt_pre + self.dt_proj_b    # (input_size,)
        delta = _softplus(dt_lifted)                             # (input_size,)

        # 3. ZOH discretization + state update (vectorized over d, n)
        a = -torch.exp(self.a_log)                               # (input_size, d_state)
        za = delta.unsqueeze(1) * a                              # (input_size, d_state)
        a_bar = torch.exp(za)
        b_bar = delta.unsqueeze(1) * b_vec.unsqueeze(0) * _expm1_over_x(za)
        h_new = a_bar * h + b_bar * x.unsqueeze(1)               # (input_size, d_state)
        y = h_new @ c_vec + self.d_skip * x                      # (input_size,)
        return y, h_new
```

**Determinism note:** the vectorized `h @ c_vec` in Python is one `matmul`; Rust's inner loop accumulates `acc += h[d, n] * c_vec[n]` in a Kahan-free scalar sum. For `d_state = 4..64`, accumulation order and associativity produce differences well below 1e-14 in f64, so the equivalence gate at `< 1e-14` passes with room to spare (expected actual: ~1e-16, matching Phase 3a Transformer's 4.16e-17).

### 3.6 PSO ParamSpec generator (`_mamba_specs`)

```python
# src/python/aerocapture/training/encoding.py

def _mamba_specs(spec: MambaSpec, bound_multiplier: float) -> list[ParamSpec]:
    d_inner = spec.input_size
    d_state = spec.d_state
    dt_rank = spec.dt_rank
    rng_seed_local = 0  # init values are deterministic where possible; uniform(dt_min, dt_max) is the one stochastic slice

    specs: list[ParamSpec] = []

    # 1. x_proj_w: Xavier uniform with fan_in=d_inner, fan_out=dt_rank + 2*d_state
    fan_in_xp = d_inner
    fan_out_xp = dt_rank + 2 * d_state
    bound_xp = math.sqrt(6.0 / (fan_in_xp + fan_out_xp)) * bound_multiplier
    for _ in range(fan_out_xp * d_inner):
        specs.append(ParamSpec(name="x_proj_w", low=-bound_xp, high=+bound_xp, init_center=0.0, scale="linear"))

    # 2. dt_proj_w: Xavier with paper's dt_rank^{-0.5} scaling
    fan_in_dt  = dt_rank
    fan_out_dt = d_inner
    bound_dt   = math.sqrt(6.0 / (fan_in_dt + fan_out_dt)) * (1.0 / math.sqrt(max(dt_rank, 1))) * bound_multiplier
    for _ in range(d_inner * dt_rank):
        specs.append(ParamSpec(name="dt_proj_w", low=-bound_dt, high=+bound_dt, init_center=0.0, scale="linear"))

    # 3. dt_proj_b: per-channel init center = inv_softplus(uniform(dt_min, dt_max))
    #    Paper defaults: dt_min = 1e-3, dt_max = 1e-1. Draw deterministically with a fixed sub-seed.
    dt_min, dt_max = 1e-3, 1e-1
    local_rng = np.random.default_rng(_MAMBA_DT_BIAS_SEED)  # const defined in encoding.py
    for _ in range(d_inner):
        dt_draw = float(local_rng.uniform(dt_min, dt_max))
        # inv_softplus(y) = log(exp(y) - 1); stable: log(expm1(y))
        init_center = math.log(math.expm1(dt_draw))
        specs.append(ParamSpec(name="dt_proj_b", low=init_center - bound_multiplier, high=init_center + bound_multiplier, init_center=init_center, scale="linear"))

    # 4. a_log: HiPPO diagonal init center = log(n + 1) for n in [0, d_state), broadcast across d_inner
    for _ in range(d_inner):
        for n in range(d_state):
            init_center = math.log(n + 1)
            specs.append(ParamSpec(name="a_log", low=init_center - bound_multiplier, high=init_center + bound_multiplier, init_center=init_center, scale="linear"))

    # 5. d_skip: init center 1.0 (identity skip per paper)
    for _ in range(d_inner):
        specs.append(ParamSpec(name="d_skip", low=1.0 - bound_multiplier, high=1.0 + bound_multiplier, init_center=1.0, scale="linear"))

    return specs
```

**Load-bearing invariant:** `_mamba_specs` and `init_v2_population`'s Mamba arm must produce identical init values when both consume the same `_MAMBA_DT_BIAS_SEED`. Since `init_v2_population` writes the starting PSO population and `_mamba_specs` defines the [low, high] bounds, the init value **must** fall inside the bound window. With `bound_multiplier = 1.0` and the centers above, all centers land at the midpoint of their windows (0.5 in unit-hypercube encoding), except `x_proj_w` and `dt_proj_w` whose centers are 0 and bounds are symmetric (also 0.5 in encoding). Consistent.

**Param count formula match:**
- `x_proj_w`: `d_inner * (dt_rank + 2 * d_state)`
- `dt_proj_w`: `d_inner * dt_rank`
- `dt_proj_b`: `d_inner`
- `a_log`: `d_inner * d_state`
- `d_skip`: `d_inner`
- Total: `d_inner * (3 * d_state + 2 * dt_rank + 2)` ✓ matches section 3.3.

### 3.7 init_v2_population Mamba arm

```python
# src/python/aerocapture/training/initialization_v2.py

_INIT_JITTER_STD = 0.01  # matches Phase 2a LSTM forget-bias jitter (1.0 + N(0, 0.01 * mul))

def _init_mamba_layer(spec: MambaSpec, n_pop: int, bound_multiplier: float, rng: np.random.Generator) -> np.ndarray:
    d_inner = spec.input_size
    d_state = spec.d_state
    dt_rank = spec.dt_rank
    n_params = d_inner * (3 * d_state + 2 * dt_rank + 2)
    pop = np.empty((n_pop, n_params), dtype=np.float64)

    # Per-channel dt_proj_b centers: draw ONCE (population-wide target), then jitter per individual.
    # Fixed sub-RNG matches _mamba_specs so ParamSpec bounds agree with these centers.
    local = np.random.default_rng(_MAMBA_DT_BIAS_SEED)
    dt_bias_centers = np.log(np.expm1(local.uniform(1e-3, 1e-1, size=d_inner)))  # (d_inner,)
    # HiPPO A_log centers: deterministic log(n+1), broadcast across d_inner, row-major flatten.
    a_log_centers = np.broadcast_to(np.log(np.arange(d_state) + 1.0), (d_inner, d_state)).copy().ravel()

    jitter_std = _INIT_JITTER_STD * bound_multiplier

    for i in range(n_pop):
        buf = []
        # 1. x_proj_w: Xavier uniform around 0 (per-individual, full Xavier spread)
        fan_in_xp = d_inner; fan_out_xp = dt_rank + 2 * d_state
        bound_xp = math.sqrt(6.0 / (fan_in_xp + fan_out_xp))
        buf.extend(rng.uniform(-bound_xp, +bound_xp, size=fan_out_xp * d_inner))
        # 2. dt_proj_w: Xavier * dt_rank^{-0.5} (per-individual)
        fan_in_dt = dt_rank; fan_out_dt = d_inner
        bound_dt = math.sqrt(6.0 / (fan_in_dt + fan_out_dt)) / math.sqrt(max(dt_rank, 1))
        buf.extend(rng.uniform(-bound_dt, +bound_dt, size=d_inner * dt_rank))
        # 3. dt_proj_b: shared center + per-individual jitter
        buf.extend(dt_bias_centers + rng.normal(0.0, jitter_std, size=d_inner))
        # 4. a_log: HiPPO centers + per-individual jitter
        buf.extend(a_log_centers + rng.normal(0.0, jitter_std, size=d_inner * d_state))
        # 5. d_skip: 1.0 center + per-individual jitter
        buf.extend(1.0 + rng.normal(0.0, jitter_std, size=d_inner))

        pop[i] = np.asarray(buf, dtype=np.float64)

    return pop
```

**Critical:** `dt_proj_b` draws its per-channel centers once via `_MAMBA_DT_BIAS_SEED`, so `_mamba_specs` (ParamSpec bounds) and `_init_mamba_layer` agree on center values. Around those centers, per-individual `N(0, 0.01 * bound_multiplier)` jitter provides PSO initial population diversity -- mirroring Phase 2a LSTM's forget-bias init pattern. Without jitter, `dt_proj_b` / `a_log` / `d_skip` would be identical across the population, collapsing PSO exploration to the Xavier-random projection slices only.

**Invariant (load-bearing):** `jitter_std * 3 << bound_multiplier`. With `jitter_std = 0.01 * bound_multiplier`, 3-sigma jitter = `0.03 * bound_multiplier`, well inside the `[center - bound_multiplier, center + bound_multiplier]` window from `_mamba_specs`. PSO starts with valid normalized values in [0, 1].

### 3.8 JSON v2 schema for Mamba layer

```json
{
  "format_version": 2,
  "input_mask": [0, 1, ...],
  "layers": [
    {
      "type": "dense",
      "input_size": 23,
      "output_size": 32,
      "activation": "swish",
      "weights": { "w": [...], "b": [...] }
    },
    {
      "type": "mamba",
      "input_size": 32,
      "d_state": 16,
      "dt_rank": 2,
      "weights": {
        "x_proj_w":  [[...], ...],
        "dt_proj_w": [[...], ...],
        "dt_proj_b": [...],
        "a_log":     [[...], ...],
        "d_skip":    [...]
      }
    }
  ]
}
```

`weights` is flat at layer level (5 top-level keys), matching Rust `NnLayerWeights` schema precedent from Phase 3a Transformer (which uses `ln1_gamma`, `ln1_beta`, `ln2_gamma`, `ln2_beta` as top-level keys rather than nested).

### 3.9 Training config defaults

`configs/training/msr_aller_mamba_pso_train.toml` (full):

```toml
base = "common.toml"

[simulation]
guidance = "neural_network"

# --- Architecture ---
[[network.architecture]]
type = "dense"
input_size = 23
output_size = 32
activation = "swish"

[[network.architecture]]
type = "mamba"
input_size = 32
d_state = 16
# dt_rank omitted -> max(1, 32/16) = 2

[[network.architecture]]
type = "mamba"
input_size = 32
d_state = 16

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 2
activation = "asinh"

# --- Optimizer ---
[optimizer]
algorithm = "pso"
n_pop = 64
n_gen = 2000
seed_strategy = "adaptive"
training_n_sims = 20
validation_n_sims = 1000
seed_pool_interval = 50
curation_sample_size = 1000
curation_top_k = 5

[optimizer.pso]
# defaults from common.toml; no overrides needed
```

Param count: `768 + 1728 + 1728 + 66 = 4290`.

## 4. Tests

### 4.1 `tests/test_rust_python_mamba_equivalence.py` (@slow, CI)

Architecture: Dense(4 -> 8, tanh) -> Mamba(8, 4, 2) -> Dense(8 -> 2, linear). Dense(4 -> 8) = 40, Mamba(8, 4, 2) = 8*(12 + 4 + 2) = 144, Dense(8 -> 2) = 18. **Total: 202 params.**

1. Instantiate `V2Policy` (via pydantic architecture list), call `policy.double()` for f64.
2. Draw a fixed-seed random input sequence shape `(100, 4)`, dtype f64.
3. Export policy to JSON v2, load in Rust via `aerocapture_rs.nn_forward_sequence(json_path, input_sequence)`.
4. Python forward: iterate 100 steps, thread `h` through `policy.layers[1].forward(x, h)`.
5. Assert `np.max(np.abs(rust_out - py_out)) < 1e-14`.

Expected actual: ~1e-16 (sub machine epsilon, matching Phase 3a Transformer's 4.16e-17).

### 4.2 `tests/test_mamba_warmup.py` (@slow)

Same arch. Build fresh `NnState` for the Rust runtime, feed the same input 10 times, assert:
- Step 0 output (zero state) differs from step 1 output by more than 1e-10.
- Repeated runs with identical input sequences produce identical output to machine epsilon.

### 4.3 `tests/test_mamba_pso_smoke.py` (@slow, python-pyo3 CI job)

Reduced arch: Dense(23 -> 8, tanh) -> Mamba(8, 4, 1) -> Dense(8 -> 2, linear). Dense(23 -> 8) = 192, Mamba(8, 4, 1) = 8*(12 + 2 + 2) = 128, Dense(8 -> 2) = 18. **Total: 338 params.**

Config: `n_pop=4`, `training_n_sims=4`, `n_gen=2`.

Asserts:
- `best_model.json` written with `format_version: 2` and architecture `["dense", "mamba", "dense"]`.
- `aerocapture_rs.nn_forward(best_model.json, x0)` returns a finite 2-tuple for a zero input.
- Gen-2 best cost is finite.

### 4.4 `tests/test_mamba_ppo_rejection.py` (@fast, main python CI job)

Constructs a minimal Mamba v2 JSON in-memory, asserts:
- `build_layer(MambaSpec(...))` raises `NotImplementedError` with message fragment `"Mamba is PSO-only in Phase 4a"`.
- `load_policy_from_json(tmp_json_path)` raises `NotImplementedError` with a similar fragment.

### 4.5 Rust unit tests (inline in `src/rust/src/data/neural.rs`)

- `MambaLayer::forward` hand-verified 2-step trajectory for `(d_inner=2, d_state=2, dt_rank=1)` with known weights.
- `to_flat` -> `from_flat` round-trip for `(d_inner=8, d_state=4, dt_rank=2)` with `proptest` random f64 chromosomes.
- `softplus` at `x = -100.0, -1.0, 0.0, 1.0, 100.0`: asserts no overflow, matches `(1.0 + x.exp()).ln()` in the `|x| < 20` regime.
- `expm1_over_x` at `z = 0.0, 1e-12, 1e-10, 1e-8, 1e-4, 0.5, 5.0`: verifies Taylor-vs-exact crossover is smooth (adjacent values differ by < 1e-15).
- `proptest`: for `(d_inner, d_state, dt_rank)` drawn from small bounds, `forward` produces finite output on finite input + finite weights.

### 4.6 Existing gates that must stay green

- All 10 Rust guidance golden regressions bit-identical (Mamba is additive; no touch to existing schemes).
- PSO-GRU / PPO-GRU / LSTM-PSO / LSTM-PPO / Window-PSO / Transformer-PSO smoke tests all still pass.
- `test_v2_rust_python_equivalence.py` (Phase 1/2a baselines) still passes.
- `ci.yml` fmt / clippy / mypy / ruff gates all green.

## 5. Out-of-scope / Phase 4b+ handoff

**Phase 4b (Mamba PPO + BPTT), defined here so it doesn't leak into 4a:**

- `MambaLayer` forward gains batch dim support: `h: Tensor[(B, d_inner, d_state)]`, 3D with batch prepended.
- `_zero_state_where_done` extends to handle 2D single-tensor state (ndim==3 in rollout buffer). Current helper handles `None`, flat `Tensor` (GRU, ndim==2), and `tuple` (LSTM, ndim==2 stacked to (B, 2, H)). Mamba is the first 2D single-tensor state.
- `hidden_shapes` packer in `train.py::_derive_hidden_shapes` gets a Mamba arm returning `(d_inner, d_state)`.
- `_np_state_to_torch` / `_torch_state_to_np` in `train.py` handle ndim==4 (`(T, B, d_inner, d_state)`) slabs.
- `ppo_update_bptt` chunk-boundary reconstruction adds the 2D-state branch (detach a 3D tensor at chunk boundaries).
- Obs-normalizer bake-in: either skip (safe) or derive the shift for `dt_proj_b` through softplus (nontrivial; requires linearizing softplus around the current bias value).
- New test: `tests/test_mamba_ppo_smoke.py` (5 PPO updates, `bptt_length=32`).
- New config: `configs/training/msr_aller_mamba_ppo_train.toml` (Dense(23 -> 32, tanh) -> Mamba(32, 16, 2) -> Dense(32 -> 2, linear), `bptt_length=32 rollout_steps=2048`).

**Phase 4c (Full Mamba block, deferred -- not guaranteed):** conv1d pre-filter + SiLU gating + in/out expansion linears + block-level residual. Would ship as `LayerSpec::MambaBlock` distinct from `LayerSpec::Mamba`, so Phase 4a's selective-SSM core remains usable standalone. Not currently on the paper roadmap.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `softplus` f64 differences between Rust `x.max(0).0 + (-x.abs()).exp().ln_1p()` and Python `_softplus` | Unit-test both at matching `x` inputs in the equivalence gate; the stable form is identical IEEE 754 across `libm` and PyTorch on the same platform |
| `expm1_over_x` Taylor crossover discontinuity near `|z| = 1e-8` | Unit-test adjacent values at `1e-9, 1e-8, 1e-7` asserting smoothness; the relative error of Taylor at `|z| = 1e-8` is `z^3 / 24 ≈ 4e-26`, far below machine eps |
| HiPPO `A_log = log(n+1)` chromosome init mismatch between `_mamba_specs.init_center` and `init_v2_population` | Both helpers compute the init from the same deterministic formula (no RNG for HiPPO); `dt_proj_b` uses a module-level fixed sub-seed |
| `dt_proj_b` centered far from 0 breaks PSO's implicit assumption that chromosome 0 is a valid point | PSO operates on normalized [0,1] unit hypercube via `decode_normalized`; init centers are encoded to 0.5 unit-value regardless of physical offset, so PSO sees uniform-in-[0,1] bounds |
| `exp(z)` overflow for very large `z = Δ·A` during PSO exploration | Δ is bounded from below by 0 (softplus) and from above by `softplus(dt_proj_b + dt_proj_w @ dt_pre)` which is bounded by the chromosome bounds; `A = -exp(a_log)` is bounded by `-exp(bound)` which is finite. Worst-case `z` at `bound_multiplier=1.0` is around `-exp(log(16)+1) * softplus(inv_softplus(1e-1)+1) ≈ -43 * 1.36 ≈ -58`, so `exp(z)` is tiny but finite. No overflow risk. |
| Boxed `Layer::Mamba(Box<...>)` allocation cost in hot GNC loop | Mamba layer is ~10kB for realistic dims; box is once per Arc-shared `NeuralNetModel`, not per tick. No runtime cost. |
| Cross-language drift from accumulation order (Python vectorized `h @ c_vec`, Rust scalar `acc +=`) | For `d_state <= 64` in f64, accumulation-order differences bound by `d_state * eps * max|h|*max|c| ≈ 64 * 2e-16 * 1 ≈ 1e-14`. Equivalence gate at `< 1e-14` is intentionally loose; expected actual `< 1e-16`. |
| Obs-norm bake-in guard forgotten, producing silently-wrong PPO training in Phase 4b | Explicit `NotImplementedError` in `export_v2_policy_to_json` guards the Mamba-as-layer-0 case now; Phase 4b MUST resolve before lifting the guard. |

## 7. Rollout and acceptance

Acceptance criteria for Phase 4a landing:

1. All tests in section 4 pass locally and in CI (Rust + main Python + python-pyo3 jobs).
2. All 10 existing Rust guidance golden regressions bit-identical.
3. All 5 existing NN equivalence gates (Dense, GRU, LSTM, Window, Transformer) still pass.
4. `compare_guidance --schemes neural_network_mamba_pso neural_network_transformer_pso --n-sims 500` produces the expected three-way trajectory classification with no crashes.
5. `train_all.sh mamba` runs end-to-end on a reduced-generation config (e.g. `n_gen=5`), produces `best_model.json`, `report.pdf`, and `final_eval.parquet` without errors.
6. `CLAUDE.md` updated with a Phase 4a paragraph mirroring Phases 1/2a/3a style (dims, params, flat-weight ordering, gates).
7. `TODO.md` Phase 4a checkbox flipped; Phase 4b (Mamba PPO) left open.

## 8. Post-phase extensibility contract update (for CLAUDE.md)

Mamba establishes the first **2D single-tensor state** layer. For Phase 4b (Mamba PPO) or any future layer with a single 2D state tensor, the rollout-buffer extension recipe is:

1. Add `LayerSpec::X` / `Layer::X` / `XLayer` / `LayerWeights for XLayer` / `save_json` / `from_v2_json` / `from_flat_weights_v2` arms in `neural.rs`.
2. Add `LayerState::X { h: DMatrix<f64> }` variant + `for_layer` / `reset` arms in `nn_state.rs`.
3. Add `TomlLayerSpec::X` variant + `to_layer_spec` arm in `config.rs`.
4. Add `rl/layers/x.py` + `__init__.py` dispatch line (raise `NotImplementedError` for PSO-only MVP, implement forward for PPO phase).
5. Add `XSpec` to `rl/schemas.py` discriminated union.
6. Add `encoding._layer_param_specs`, `config._layer_n_params`, `config._layer_output_size` arms.
7. Add `rl/export.py` + `model_io.py` isinstance branches (raise `NotImplementedError` for obs-norm bake-in if the layer is nonlinear in layer-0 position).
8. Add `initialization_v2._init_X_layer` with deterministic init values matching the ParamSpec centers.
9. For PPO extension (Phase 4b-style): extend `_zero_state_where_done` tensor branch; add `hidden_shapes` arm returning `(d_inner, d_state)`; extend `_np_state_to_torch` / `_torch_state_to_np` for ndim==4; extend `ppo_update_bptt` chunk-boundary reconstruction for 3D state.

**Multi-tensor hidden states** (Phase 2a LSTM precedent) are orthogonal to **2D single-tensor states** (Phase 4a Mamba): they use different shape-pack conventions (stacked `(B, 2, H)` vs natural `(B, D, N)`). Both patterns are now locked in the runtime.
