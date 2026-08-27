# Phase 2b -- Window-MLP (PSO-only)

**Date:** 2026-04-20
**Status:** Design approved, ready for implementation planning
**Parent effort:** LSTM / Transformer / Mamba architectures for neural guidance (see `TODO.md`).
**Predecessors:** Phase 0 (stateful NN runtime infrastructure, PR #37), Phase 1 (PSO-GRU MVP, `feature/gru-mvp`), Phase 1.5 (PPO-GRU + truncated BPTT, PR #38), Phase 2a (LSTM MVP + activation-aware init, PR #39).

## 1. Context

The paper experimental grid has six architectures (MLP baseline, Window-MLP, GRU, LSTM, Transformer, Mamba) across two training axes (PSO, BPTT-PPO). Phases 0/1/1.5/2a have shipped the runtime infrastructure, the extensibility contract for scalar-state and multi-tensor-state layers, activation-aware init, and both gated recurrent architectures on both training axes.

Phase 2b adds **Window-MLP**: a trivial stateful layer that maintains a FIFO ring buffer of the last `n_steps` inputs and concatenates them into a wider vector for the next Dense layer. The layer itself has **zero trainable parameters**. This makes the paper hypothesis crisp: "does short-term history matter *on its own*, without gating?" The Dense trunk stays identical to the baseline MLP, so any improvement over MLP attributes to input-history availability rather than model expressivity.

The TODO grid schedules Window-MLP under PSO only (no BPTT-PPO). This is a deliberate scope choice:

1. **Paper experiment already covered.** The 2x(MLP vs Window) comparison is apples-to-apples under PSO.
2. **No new training-axis signal.** With zero trainable params in the Window layer itself, PSO is the natural fit -- the chromosome is 100% downstream Dense weights.
3. **Tuple-state PPO rollout buffer.** The Window buffer state is a flat `(n_steps, input_size)` tensor, which would work with the existing `(B, n_steps, input_size)` ndim==3 pack convention, but plumbing it through `ppo_update_bptt` / `hidden_shapes` / `_np_state_to_torch` / `_torch_state_to_np` and writing the PPO smoke + BPTT chunk-invariant tests is meaningful additional surface area. Defer until a paper reviewer asks for it.

The PPO path gets a **clean error at build time** rather than a runtime surprise: `build_layer(WindowSpec)` raises `NotImplementedError` with a pointer to this spec. Same for `load_policy_from_json` on a Window v2 JSON file. PSO bypasses V2Policy entirely (Rust forward via `aerocapture_rs.nn_forward`), so the PSO path stays clean.

## 2. Scope

**In scope:**

- Rust `WindowLayer` struct (fields: `input_size: usize`, `n_steps: usize`, no weights).
- `Layer::Window`, `LayerSpec::Window { input_size: usize, n_steps: usize }`, `LayerState::Window { buffer: VecDeque<Vec<f64>> }` (pre-filled with `n_steps` zero vectors of length `input_size`), `TomlLayerSpec::Window { input_size: usize, n_steps: usize }`.
- `LayerWeights for WindowLayer` with `n_params() == 0`, `to_flat() == Vec::new()`, `from_flat(&[])` no-op. Consistent with the trait contract so `from_flat_weights_v2` keeps working unchanged.
- `NeuralNetModel::save_json` / `from_v2_json` / `from_flat_weights_v2` Window arms. JSON v2 entry is spec-only: `{"type": "window", "input_size": N, "n_steps": K}` -- no weight dict.
- `NeuralNetModel::forward` routes through `Layer::Window` which performs: push `input` to `buffer` back, `pop_front()`, output = flattened buffer contents (length `n_steps * input_size`). `LayerState::Window::for_layer` pre-fills the `VecDeque` with `n_steps` zero vectors of length `input_size` so the buffer is at steady-state capacity from tick 0.
- PyTorch `WindowLayer` module in `rl/layers/window.py` with forward `(x: Tensor, state: Tensor) -> (out: Tensor, new_state: Tensor)` where `state` shape is `(batch, n_steps, input_size)`. `new_state(batch, device)` returns a zero tensor with the module's parameter dtype (tracks dtype via a registered non-persistent buffer since the layer has no `nn.Parameter`). Cross-language equivalence test consumes this module; no PPO use.
- `WindowSpec` pydantic class appended to the `LayerSpec` discriminated union on the `type` field.
- `build_layer` in `rl/layers/__init__.py` dispatches `WindowSpec` to **raise `NotImplementedError`** with a clear message ("Window-MLP is PSO-only in Phase 2b; PPO use deferred -- see docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md"). This is the PPO-rejection guard.
- `nn_param_specs_from_v2` Window arm appends an empty list (zero trainable params).
- `encoding._layer_param_specs` Window dispatch that returns `[]`.
- `config.py::_layer_n_params` window arm returns `0`; `_layer_output_size` window arm returns `spec.n_steps * spec.input_size`.
- `describe_architecture` window arm alongside dense/gru/lstm.
- `export_v2_policy_to_json`: Window branch writes only `{"type": "window", "input_size", "n_steps"}` to the layer entry (no weights dict). Obs-norm bake-in guard extended to reject `WindowSpec` as layer 0 (same invariant as GRU/LSTM; a window buffer can't absorb an affine shift bake-in without per-buffer-slot rescaling which violates the "no new matmul" discipline).
- `load_policy_from_json`: raises `NotImplementedError` when any layer is Window, consistent with `build_layer`.
- `init_v2_population`: Window dispatch is a one-line `elif isinstance(spec, WindowSpec): continue`. Since `_layer_n_params(WindowSpec) == 0` and `_layer_param_specs(WindowSpec)` returns `[]`, the existing concatenation logic produces the correct total chromosome length with no Window-specific code in the init math itself. Regression-tested against mixed Window + Dense architectures.
- Training config `configs/training/msr_aller_window_pso_train.toml`: Dense(16x8=128 -> 32, swish) -> Dense(32 -> 8, swish) -> Dense(8 -> 2, linear), `output_interpretation = "atan2"`, `input_mask = [0..16]` (matches baseline MLP input convention).
- `compare_guidance.SCHEMES` / `SCHEME_TRAINING_CONFIGS` / `_NN_DEPLOY_SCHEMES` registration as `neural_network_window_pso` (goes through the Rust `neural_network` runtime, same as GRU-PSO / LSTM-PSO).
- `train_all.sh` aliases: `window_pso`, `nn_window_pso`, `window`.
- Cross-language equivalence test `test_rust_python_window_equivalence.py`: architecture = Window(4, 4) -> Dense(16 -> 4, tanh) -> Dense(4 -> 2, linear). Uses `aerocapture_rs.nn_forward_sequence` to thread a single `NnState` across 100 random f64 inputs; Python forward threads the `(batch=1, n_steps=4, input_size=4)` tensor state through `WindowLayer.forward` explicitly. Asserts max abs diff < 1e-10, target machine epsilon.
- PSO smoke test `test_window_pso_smoke.py` (@slow, python-pyo3 CI job): 2-gen PSO on reduced Window(4, 4) -> Dense(16 -> 4, swish) -> Dense(4 -> 2, linear) (~40 params), asserts `best_model.json` is v2 with `["window", "dense", "dense"]` arch and `nn_forward` returns a finite 2-tuple.
- PPO-rejection test `test_window_ppo_rejection.py` (@fast): builds a minimal Window TOML architecture, calls `load_policy_from_json` on a v2 JSON file containing a Window layer, asserts `NotImplementedError` is raised with the expected message fragment.
- `ci.yml` workflow extension: three new tests added to the python-pyo3 job (`test_rust_python_window_equivalence.py`, `test_window_pso_smoke.py`, `test_window_ppo_rejection.py`). The PPO-rejection test also runs in the main python job since it doesn't need PyO3.
- `CLAUDE.md` + `TODO.md` sync after landing: Phase 2b checkbox done, scalar-state extensibility contract updated to show Window landed, remaining phases (Transformer, Mamba) confirmed still open.

**Out of scope:**

- **PPO-BPTT for Window-MLP.** Deferred to a future "Phase 2b.5" iff paper reviewers ask for it. PPO path errors at `build_layer` time (no silent fallback).
- **`_zero_state_where_done` extension.** Window never reaches V2Policy because `build_layer` errors out, so the helper sees no Window state. No change needed.
- **Rollout buffer `hidden_shapes` / ndim-dispatch extension for Window.** Same reasoning as above.
- **Activation-aware init changes.** Window contributes zero parameters to the init population, so `init_v2_population` dense/gru/lstm arms are unchanged. Window only adds a "continue" branch.
- **Recurrent-critic or SAC-Window work.** Tracked as Phase 1.5/1.6 carry-overs, orthogonal.
- **Supporting Window as a non-first layer in PSO training configs.** The Rust `WindowLayer::forward` is position-agnostic (buffers whatever input it receives), but the initial training config places Window first per the paper design. A future Phase may explore "windowed latent" (Dense -> Window -> Dense) once the "pure window-MLP" baseline is validated.
- **Per-step-history observability tooling.** Trajectory logging records the bank command, not the per-layer state. Adding buffer-content logging is a debugging convenience, not a correctness requirement.

## 3. Detailed design

### 3.1 Rust LayerSpec + Layer + WindowLayer

```rust
// src/rust/src/data/neural.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LayerSpec {
    Dense  { input_size: usize, output_size: usize, activation: Activation },
    Gru    { input_size: usize, hidden_size: usize },
    Lstm   { input_size: usize, hidden_size: usize },
    Window { input_size: usize, n_steps: usize },
}

#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
    Lstm(LstmLayer),
    Window(WindowLayer),
}

#[derive(Debug, Clone)]
pub struct WindowLayer {
    pub input_size: usize,
    pub n_steps: usize,
}

impl WindowLayer {
    pub fn forward(&self, input: &[f64], state: &mut LayerState) -> Vec<f64> {
        let buffer = match state {
            LayerState::Window { buffer } => buffer,
            _ => panic!("WindowLayer::forward called with non-Window state"),
        };
        assert_eq!(input.len(), self.input_size);
        buffer.pop_front();
        buffer.push_back(input.to_vec());
        let mut out = Vec::with_capacity(self.n_steps * self.input_size);
        for slot in buffer.iter() {
            out.extend_from_slice(slot);
        }
        out
    }
}

impl LayerWeights for WindowLayer {
    fn n_params(&self) -> usize { 0 }
    fn to_flat(&self) -> Vec<f64> { Vec::new() }
    fn from_flat(&mut self, flat: &[f64]) {
        assert!(flat.is_empty(), "WindowLayer takes no weights");
    }
}
```

Rationale for `VecDeque`: O(1) push_back / pop_front; fixed capacity = `n_steps`; serde-serializable (via `Vec` conversion at save/load). The VecDeque is the right data structure here; a flat `Vec<Vec<f64>>` with manual index rotation would save nothing measurable in simulator hot path (guidance tick is milliseconds).

### 3.2 LayerState::Window

```rust
// src/rust/src/data/nn_state.rs

#[derive(Debug, Clone)]
pub enum LayerState {
    None,
    Gru    { h: Vec<f64> },
    Lstm   { h: Vec<f64>, c: Vec<f64> },
    Window { buffer: VecDeque<Vec<f64>> },
}

impl LayerState {
    pub fn for_layer(layer: &Layer) -> Self {
        match layer {
            Layer::Dense(_) => LayerState::None,
            Layer::Gru(g)   => LayerState::Gru { h: vec![0.0; g.hidden_size] },
            Layer::Lstm(l)  => LayerState::Lstm {
                h: vec![0.0; l.hidden_size],
                c: vec![0.0; l.hidden_size],
            },
            Layer::Window(w) => {
                let mut buffer = VecDeque::with_capacity(w.n_steps);
                for _ in 0..w.n_steps {
                    buffer.push_back(vec![0.0; w.input_size]);
                }
                LayerState::Window { buffer }
            }
        }
    }
}
```

Pre-filling the buffer with zero vectors (zero-padded convention, approved in brainstorming) keeps `forward` branchless: every tick is `push_back + pop_front + flatten`, no warm-up special case.

### 3.3 TomlLayerSpec + config parser

```rust
// src/rust/src/config.rs

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum TomlLayerSpec {
    Dense  { input_size: usize, output_size: usize, activation: Activation },
    Gru    { input_size: usize, hidden_size: usize },
    Lstm   { input_size: usize, hidden_size: usize },
    Window { input_size: usize, n_steps: usize },
}

fn to_layer_spec(t: &TomlLayerSpec) -> LayerSpec {
    match t {
        // ...existing arms unchanged...
        TomlLayerSpec::Window { input_size, n_steps } => {
            assert!(*n_steps > 0, "Window layer n_steps must be positive");
            assert!(*input_size > 0, "Window layer input_size must be positive");
            LayerSpec::Window { input_size: *input_size, n_steps: *n_steps }
        }
    }
}
```

TOML form:

```toml
[[network.architecture]]
type = "window"
input_size = 16
n_steps = 8
```

Both fields are required (consistent with the GRU / LSTM convention -- `input_size` is explicit, not inferred).

### 3.4 JSON v2 read/write

```rust
// save_json: Window arm writes spec-only entry
match spec {
    LayerSpec::Window { input_size, n_steps } => json!({
        "type": "window",
        "input_size": input_size,
        "n_steps": n_steps,
    }),
    // ...other arms...
}

// from_v2_json: Window arm validates both fields and constructs the stateless layer
"window" => {
    let input_size = parse_usize("input_size")?;
    let n_steps    = parse_usize("n_steps")?;
    if input_size == 0 || n_steps == 0 {
        return Err(DataError::InvalidArchitecture(
            "(window) input_size and n_steps must both be positive".into()
        ));
    }
    (LayerSpec::Window { input_size, n_steps },
     Layer::Window(WindowLayer { input_size, n_steps }))
}

// from_flat_weights_v2: Window arm consumes zero flat weights
LayerSpec::Window { input_size, n_steps } => {
    Layer::Window(WindowLayer { input_size: *input_size, n_steps: *n_steps })
}
```

### 3.5 Python WindowSpec + WindowLayer module

```python
# src/python/aerocapture/training/rl/schemas.py

class WindowSpec(BaseModel):
    type: Literal["window"]
    input_size: int
    n_steps: int

    @field_validator("input_size", "n_steps")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("input_size and n_steps must be positive")
        return v

LayerSpec = Annotated[
    DenseSpec | GruSpec | LstmSpec | WindowSpec,
    Discriminator("type"),
]
```

```python
# src/python/aerocapture/training/rl/layers/window.py

class WindowLayer(nn.Module):
    def __init__(self, input_size: int, n_steps: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.n_steps = n_steps
        # Register a no-op parameter just to track dtype/device for new_state.
        # Zero-param layers need some way to report the policy's dtype; a
        # non-trainable buffer is the idiomatic torch approach.
        self.register_buffer("_dtype_anchor", torch.zeros(1), persistent=False)

    def forward(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        # x:     (batch, input_size)
        # state: (batch, n_steps, input_size)
        new_state = torch.cat(
            [state[:, 1:], x.unsqueeze(1)], dim=1
        )  # (batch, n_steps, input_size)
        out = new_state.reshape(x.shape[0], -1)  # (batch, n_steps * input_size)
        return out, new_state

    def new_state(self, batch_size: int) -> Tensor:
        return torch.zeros(
            batch_size, self.n_steps, self.input_size,
            dtype=self._dtype_anchor.dtype,
            device=self._dtype_anchor.device,
        )
```

```python
# src/python/aerocapture/training/rl/layers/__init__.py

def build_layer(spec: LayerSpec) -> nn.Module:
    if isinstance(spec, DenseSpec):
        return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    if isinstance(spec, GruSpec):
        return GruLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, LstmSpec):
        return LstmLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, WindowSpec):
        raise NotImplementedError(
            "Window-MLP is PSO-only in Phase 2b; PPO use deferred. "
            "See docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md"
        )
    raise TypeError(f"Unknown layer spec: {spec!r}")
```

The `NotImplementedError` is the PPO-rejection guard. V2Policy is only built by PPO / SAC paths (PSO bypasses V2Policy entirely, invoking `aerocapture_rs.nn_forward` directly on the Rust runtime). So raising here means: PPO configurations with Window layers fail loudly at policy-construction time, with a clear pointer to this spec. PSO training is unaffected.

### 3.6 Config helpers, encoding, export/load

```python
# src/python/aerocapture/training/config.py

def _layer_n_params(spec: LayerSpec) -> int:
    if isinstance(spec, DenseSpec):  return spec.input_size * spec.output_size + spec.output_size
    if isinstance(spec, GruSpec):    return 3 * spec.hidden_size * spec.input_size \
                                          + 3 * spec.hidden_size * spec.hidden_size \
                                          + 6 * spec.hidden_size
    if isinstance(spec, LstmSpec):   return 4 * spec.hidden_size * spec.input_size \
                                          + 4 * spec.hidden_size * spec.hidden_size \
                                          + 8 * spec.hidden_size
    if isinstance(spec, WindowSpec): return 0
    raise TypeError(...)

def _layer_output_size(spec: LayerSpec) -> int:
    if isinstance(spec, DenseSpec):  return spec.output_size
    if isinstance(spec, GruSpec):    return spec.hidden_size
    if isinstance(spec, LstmSpec):   return spec.hidden_size
    if isinstance(spec, WindowSpec): return spec.n_steps * spec.input_size
    raise TypeError(...)
```

```python
# src/python/aerocapture/training/encoding.py

def _layer_param_specs(spec: LayerSpec, bound_multiplier: float) -> list[ParamSpec]:
    if isinstance(spec, DenseSpec):  return _dense_specs(spec, bound_multiplier)
    if isinstance(spec, GruSpec):    return _gru_specs(spec, bound_multiplier)
    if isinstance(spec, LstmSpec):   return _lstm_specs(spec, bound_multiplier)
    if isinstance(spec, WindowSpec): return []  # zero trainable params
    raise TypeError(...)
```

`nn_param_specs_from_v2` does `sum(_layer_param_specs(spec, mul) for spec in architecture, [])` -- the Window empty-list contribution is absorbed naturally.

```python
# src/python/aerocapture/training/rl/export.py::export_v2_policy_to_json

for i, (spec, layer) in enumerate(zip(policy.architecture, policy.layers)):
    if isinstance(spec, WindowSpec):
        # Spec-only entry; no weights.
        arch_entry = {"type": "window", "input_size": spec.input_size, "n_steps": spec.n_steps}
        architecture_json.append(arch_entry)
        # No weights entry under weights["layer_i"].
        continue
    # ...existing Dense/GRU/LSTM arms...

# Obs-norm bake-in guard:
if obs_normalizer is not None and isinstance(policy.architecture[0], (GruSpec, LstmSpec, WindowSpec)):
    raise NotImplementedError(
        f"Obs normalizer bake-in into layer 0 is only supported for DenseSpec, "
        f"got {type(policy.architecture[0]).__name__}. Export without the bake-in."
    )
```

```python
# src/python/aerocapture/training/model_io.py::load_policy_from_json

if any(isinstance(spec, WindowSpec) for spec in architecture):
    raise NotImplementedError(
        "Window-MLP is PSO-only in Phase 2b; load_policy_from_json is a PPO/SAC entry point "
        "that cannot construct V2Policy with Window layers. "
        "See docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md"
    )
```

### 3.7 Training config and registration

```toml
# configs/training/msr_aller_window_pso_train.toml

base = [
    "./common.toml",
    "../missions/mars.toml",
]

[simulation]
mission = "msr_aller"
guidance = "neural_network"

[data]
neural_network = "training_output/neural_network_window_pso/best_model.json"

[network]
# 16-input baseline (matches MLP consolidated config) -- no bounce-gated exit inputs,
# no reference trajectory interpolations; Window provides history via raw observation stack.
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
output_interpretation = "atan2"

[[network.architecture]]
type = "window"
input_size = 16
n_steps = 8

[[network.architecture]]
type = "dense"
input_size = 128   # 16 * 8
output_size = 32
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 8
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 8
output_size = 2
activation = "linear"

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
```

Parameter count: `128*32 + 32 + 32*8 + 8 + 8*2 + 2 = 4410`. Smaller than GRU-PSO (6946) and LSTM-PSO (14353); matches the paper hypothesis that Window-MLP is the cheapest stateful architecture.

```python
# src/python/aerocapture/training/compare_guidance.py

SCHEMES = [..., "neural_network_window_pso"]
SCHEME_TRAINING_CONFIGS = {
    ...,
    "neural_network_window_pso": "configs/training/msr_aller_window_pso_train.toml",
}
_NN_DEPLOY_SCHEMES = {..., "neural_network_window_pso"}
```

```bash
# train_all.sh
    window_pso|nn_window_pso|window)
        scheme="neural_network_window_pso"
        ;;
```

### 3.8 Tests

```python
# tests/test_rust_python_window_equivalence.py
# architecture: Window(4, 4) -> Dense(16, 4, tanh) -> Dense(4, 2, linear)
# 100 random f64 inputs, stateful forward via nn_forward_sequence vs V2Policy.forward_mean_logstd
# max abs diff < 1e-10, target machine epsilon

# tests/test_window_pso_smoke.py  (@slow, python-pyo3 CI)
# 2-gen PSO on reduced Window(4, 4) -> Dense(16, 4, swish) -> Dense(4, 2, linear)
# ~40 trainable params (all in Dense)
# asserts best_model.json is v2 with ["window", "dense", "dense"] arch
# asserts nn_forward returns finite 2-tuple

# tests/test_window_ppo_rejection.py  (@fast)
# Constructs a v2 JSON with a Window layer, calls load_policy_from_json,
# asserts NotImplementedError is raised with "Window-MLP is PSO-only" in the message.
# Also tests build_layer(WindowSpec(...)) directly.
```

**CI wiring:** three new test files added to `.github/workflows/ci.yml` python-pyo3 job. `test_window_ppo_rejection.py` also runs in the main python job (no PyO3 needed).

## 4. Backward compatibility

- JSON v1 files continue to load unchanged (pre-existing).
- JSON v2 files without Window layers load unchanged.
- TOML training configs without `[[network.architecture]] type = "window"` entries unchanged.
- All 10 existing guidance schemes unchanged; 10 golden-regression tests bit-identical.
- PyO3 API (`run`, `run_mc`, `run_batch`, `run_with_draws`, `nn_forward`, `nn_forward_sequence`, `flat_weights_to_json`) signatures unchanged.
- `compare_guidance` output format unchanged; adds `neural_network_window_pso` as a new scheme key.

## 5. Extensibility contract

Post-Phase-2b scalar-state layer contract (the minimal touch set for the **next** scalar-state layer -- Attention with reset-per-episode KV, LayerNorm, residual):

- `neural.rs`: `LayerSpec` variant + `Layer` variant + `XxxLayer` struct + `LayerWeights` impl + `save_json` arm + `from_v2_json` arm + `from_flat_weights_v2` arm.
- `nn_state.rs`: `LayerState` variant (scalar or zero-capacity) + `for_layer` arm + `reset` arm.
- `config.rs`: `TomlLayerSpec` variant + `to_layer_spec` arm.
- `rl/layers/<type>.py`: new file with the torch module.
- `rl/layers/__init__.py`: `build_layer` dispatch line.
- `rl/schemas.py`: `Spec` pydantic class + union entry.
- `encoding.py`: `_layer_param_specs` dispatch arm.
- `rl/export.py` + `model_io.py`: `isinstance` branches.
- `config.py`: `_layer_n_params` + `_layer_output_size` arms.

**No changes to** `problem.py`, `dispatch.rs`, `runner.rs`, `train.py`, `ppo.py`, `policy.py` (beyond the cases already handled), `hidden_shapes`, `_np_state_to_torch`, `_torch_state_to_np`, `ppo_update_bptt`. These paths are already stable by Phase 2a.

Phase 2b's specific contribution to the contract: **a scalar-state layer can have zero trainable parameters**. The `_layer_param_specs` empty-list arm, the `LayerWeights::from_flat` no-op, and the `init_v2_population` `continue` branch lock this in as a supported case.

## 6. Success criteria

Phase 2b is complete when:

1. `./check_all.sh` passes on the branch; ruff + mypy clean; full Python test suite passes.
2. `cargo test --workspace` passes; release build succeeds.
3. Cross-language Window equivalence test passes at machine epsilon (target max abs diff < 1e-10; Phase 1 GRU hit 4.4e-16 and Phase 2a LSTM hit ~1e-16).
4. PSO-Window smoke test runs in under 3s wall-clock in the python-pyo3 CI job.
5. PPO-rejection test confirms `build_layer` and `load_policy_from_json` raise `NotImplementedError` with the expected message.
6. 10/10 guidance-scheme golden regressions bit-identical.
7. Training `neural_network_window_pso` to convergence reaches a DV distribution within the same order of magnitude as the MLP baseline (strict improvement not required for MVP; the paper comparison happens post-merge).
8. `CLAUDE.md` and `TODO.md` updated: Phase 2b checkbox marked done, next-phase (Transformer / Mamba) carry-overs still listed.

## 7. Open questions

None. Brainstorming settled scope (PSO-only), initial buffer (zero-pad), architecture position (Window-first, N=8, Dense trunk matching MLP baseline).

## 8. Follow-ups (explicit non-goals for Phase 2b)

- **Phase 2b.5 (conditional):** PPO-BPTT for Window. Requires `hidden_shapes` ndim==3 dispatch (already present for LSTM's `(2, H)` pack), `_np_state_to_torch` / `_torch_state_to_np` branches, and a `ppo_update_bptt` unpack for `(|mb|, n_steps, input_size)` tensors. Triggered if paper reviewers request PPO-Window parity.
- **Phase 3 (Transformer):** Multi-head attention with sinusoidal position encoding, causal window attention (fixed N=64 token buffer), manual attention implementation in PyTorch for bit-identical Rust equivalence. Separate spec.
- **Phase 4 (Mamba / S6):** Input-dependent SSM with sequential scan. Separate spec.
- **Phase 5 (paper):** Unified 10-cell grid comparison, figures, arXiv draft.
- **SAC-GRU / SAC-LSTM / SAC-Window:** Phase 1.6, orthogonal to the paper grid. Window-specific SAC is even further deferred since SAC still runs on `GaussianPolicy` (not `V2Policy`).
