# Phase 0 -- Stateful Neural Network Runtime Infrastructure

**Date:** 2026-04-17
**Status:** Design approved, ready for implementation planning
**Parent effort:** LSTM / Transformer / Mamba architectures for neural guidance (see `TODO.md`).

## 1. Context

The current NN guidance runtime (`src/rust/src/gnc/guidance/neural.rs`, `src/rust/src/data/neural.rs`) assumes every layer is a dense matmul + activation. The JSON model format v1 reflects that: a single `architecture.layers` size array + parallel `architecture.activations` array + a flat `weights` map. The forward pass is stateless -- each guidance tick runs `nn.forward(input)` and discards any intermediate quantities.

This phase adds the runtime infrastructure required to run stateful architectures (GRU, LSTM, attention-based, state-space models) while preserving bit-level behavior for existing v1 models. It is deliberately scoped narrow: **no new architectures ship in this phase**. The deliverable is the plumbing -- JSON v2 schema, stateful `forward`, per-sim `NnState`, PSO chromosome round-trip for heterogeneous layers, PyTorch mirror base class, exporter with obs-norm bake-in. Phase 1 (GRU MVP) validates the infrastructure on one real architecture.

## 2. Scope

**In scope:**
- JSON format v2 schema (tagged-layer list) + v1 backward-compatible loader.
- Rust `NeuralNetModel` stores heterogeneous layer types; stateful `forward(&self, state: &mut NnState, input: &[f64]) -> Vec<f64>` signature.
- `NnState` type with per-layer-variant state. `Clone`, reset via construction (`GuidanceState::new(..., Option<&NeuralNetModel>)`).
- Threading `&mut NnState` through the single-sim runner, `BatchedSimulation` per-env state, and PSO MC eval.
- Extended `to_flat_weights` / `from_flat_weights` with canonical per-layer flat ordering.
- Extended `nn_param_specs_from_v2(architecture, bound_multiplier)` in `encoding.py` with per-layer-type PSO bound generation.
- PyTorch `V2Policy` mirror base class with step-wise `forward(x, state)` API.
- Exporter `export_policy_to_json(policy, path, obs_normalizer=None)` writing v2 format.
- Unit + integration tests: Rust forward matches PyTorch forward to 1e-10; episode reset invariants; v1 models load and run unchanged.

**Out of scope for Phase 0:**
- Any new layer types (GRU, LSTM, attention, SSM, window). Those land in Phases 1-4.
- RL rollout buffer changes to carry hidden state. Phase 1 concern.
- New training configs. Phase 1+ concern.
- Paper artifact work. Phase 5.

**What Phase 0 must enable:**
- Adding a new layer type in a subsequent phase requires only: (a) adding a variant to the Rust `Layer` enum, (b) adding its serde tag, (c) implementing its `forward` + `to_flat`/`from_flat`, (d) mirroring in PyTorch, (e) adding its `ParamSpec` generator. No changes to the dispatch layer, the sim runner, `BatchedSimulation`, or PSO training code.

## 3. Design decisions

The five decisions locked during brainstorming:

### 3.1 JSON format v2: tagged-layer list

v2 replaces the parallel `layers: [sizes]` + `activations: [...]` arrays with a single ordered list of tagged layer specs. Each spec carries its type and the params specific to that type. v1 is detected via `format_version: 1` (or absence, treated as 1) and continues to load unchanged.

```json
{
  "format_version": 2,
  "architecture": [
    { "type": "dense", "input_size": 23, "output_size": 32, "activation": "tanh" },
    { "type": "dense", "input_size": 32, "output_size": 2, "activation": "linear" }
  ],
  "weights": {
    "layer_0": { "w": [...], "b": [...] },
    "layer_1": { "w": [...], "b": [...] }
  },
  "output_interpretation": "atan2",
  "input_mask": [0, 1, 2, ...],
  "ablated_input": null
}
```

The `weights` map is keyed by position (`layer_0`, `layer_1`, ...) so positional ordering is explicit. Contents are layer-type-dependent (`{w, b}` for dense; gate-specific keys for GRU/LSTM; Q/K/V/out keys for attention; etc. -- defined per layer type in subsequent phases).

Rust deserialization uses `#[serde(tag = "type", rename_all = "snake_case")]` on a `LayerSpec` enum. Python uses a Pydantic `discriminated_union` on the `type` field.

### 3.2 `NnState` outside the model, clonable, reset at construction

```rust
pub struct NnState {
    layer_states: Vec<LayerState>,
}

pub enum LayerState {
    None,                                   // dense, layer_norm, residual -- stateless
    Gru    { h: Vec<f64> },
    Lstm   { h: Vec<f64>, c: Vec<f64> },
    Window { buffer: VecDeque<Vec<f64>> },  // window-MLP and attention KV-cache
    Ssm    { h: Vec<f64> },                 // Mamba
}

impl NnState {
    pub fn for_model(model: &NeuralNetModel) -> Self { ... }  // eager init
    pub fn reset(&mut self) { ... }                            // zero all layer states
}

// NnState: Clone required for RL rollout buffer snapshots.
```

Phase 0 ships only `LayerState::None` (since no stateful layer types exist yet). Subsequent phases add the other variants. The point of defining the enum shape now is that the match arms in forward-pass dispatch, `for_model`, `reset`, and `Clone` are stable going forward -- later phases add variants without changing existing match arms in signed code.

Reset fires **only at episode start** (via `build_sim_state` rebuilding `GuidanceState`). No reset on bounce, roll reversal, or phase transition -- the policy sees those as input-level signals and learns to use them together with its hidden state.

### 3.3 Eager init + two-mode state ownership

`GuidanceState::new` gains an `Option<&NeuralNetModel>` parameter:

```rust
impl GuidanceState {
    pub fn new(
        initial_bank: f64,
        initial_aoa: f64,
        nn_model: Option<&NeuralNetModel>,
    ) -> Self {
        let nn_state = nn_model.map(NnState::for_model);
        Self {
            bank_angle_commanded: initial_bank,
            bank_angle_realized: initial_bank,
            aoa_commanded: initial_aoa,
            command_shaper: CommandShaper::new(),
            lateral_state: LateralState::default(),
            nn_state,
            ...
        }
    }
}
```

`NnState` shape matches the model at construction time. Forward-pass hot path has no init branch and no Option unwrap (the caller already knows the NN scheme is active when it invokes `nn_bank_angle`).

**Two-mode state ownership:**

- **PSO + deployed MC eval path:** Rust-side `NnState` is live. Guidance runs in Rust. Bank angle comes from `nn_bank_angle(nav, model, &mut nn_state, ...)`.
- **RL training path:** Python-side `V2Policy` carries hidden state in torch tensors. Bank angle is computed in Python and pushed into `BatchedSimulation.step(actions)`. Rust-side `NnState` is allocated (because the scheme is `neural_network`) but untouched -- the guidance code is not reached since the RL env bypasses it.

This split keeps the two training methods cleanly separated without forking the runtime. The deployment path (MC eval of a trained model, regardless of training method) always exercises the Rust-side `NnState` because `best_model.json` is the handoff.

### 3.4 PSO chromosome: canonical flat ordering + unified bounds

**Canonical flat order per layer type** (subsequent phases add rows; Phase 0 ships only `dense`):

| Layer type | Flat order |
|---|---|
| `dense` | `W` (row-major, `[out, in]`), then `b` |
| `gru` | `W_ir, W_iz, W_in, W_hr, W_hz, W_hn, b_ir, b_iz, b_in, b_hr, b_hz, b_hn` |
| `lstm` | `W_ii, W_if, W_ig, W_io, W_hi, W_hf, W_hg, W_ho, b_ii, b_if, b_ig, b_io, b_hi, b_hf, b_hg, b_ho` |
| `attention` | `W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o` |
| `layer_norm` | `gamma, beta` |
| `ssm` | `W_in_proj, W_x_proj, W_dt_proj, A_log, D, b_in_proj, b_dt_proj` |
| `window` | (no weights) |

Ordering matches PyTorch naming so the chromosome layout is identical on both sides. **Phase 0 implements only the `dense` row**; subsequent phases add rows as their layer types land.

Rust `LayerWeights` trait per enum variant:

```rust
trait LayerWeights {
    fn to_flat(&self) -> Vec<f64>;
    fn from_flat(&mut self, flat: &[f64], offset: usize) -> usize;  // returns new offset
}
```

`NeuralNetModel::to_flat_weights` / `from_flat_weights` iterate layers and delegate. Phase 0 implements only the dense variant (matching v1 behavior exactly).

**Bounds (Xavier-uniform universal, two special cases):**

- All weight matrices: `bound = bound_multiplier * sqrt(6 / (fan_in + fan_out))` on a uniform `[-bound, +bound]` `ParamSpec`.
- All biases: uniform `[-0.1, +0.1]` scaled by `bound_multiplier`.
- Special: Mamba `A_log` uniform `[-5, 0]` (Phase 4).
- Special: LayerNorm `gamma` uniform `[0.5, 1.5]`, `beta` uniform `[-0.1, 0.1]` (Phase 3).

Python `nn_param_specs_from_v2(architecture: list[LayerSpec], bound_multiplier: float) -> list[ParamSpec]` dispatches per `type`. Old `nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier)` signature kept as a thin wrapper that constructs a v2 all-dense architecture and calls the new function.

### 3.5 PyTorch mirror + export contract

**Architecture invariant:** every v2 net begins with either `dense` (layer 0) or `window → dense` (layers 0-1). Obs-norm bake-in applies to the first `dense` layer; for window-first architectures, the exporter tiles the `(mean, std)` pattern across the N window slots into the embedding layer.

**Mirror API:**

```python
class V2Policy(nn.Module):
    def __init__(self, architecture, output_interpretation, input_mask):
        super().__init__()
        self.layers = nn.ModuleList([build_layer(spec) for spec in architecture])
        self.output_interpretation = output_interpretation
        self.input_mask = input_mask
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # exploration noise, not exported

    def forward(self, x: Tensor, state: list[LayerState]) -> tuple[Tensor, list[LayerState]]:
        for i, layer in enumerate(self.layers):
            x, state[i] = layer(x, state[i])
        return x, state

    def new_state(self, batch_size: int, device) -> list[LayerState]:
        return [layer.new_state(batch_size, device) for layer in self.layers]
```

Step-wise forward (single `x_t`) matches the Rust API. BPTT-over-sequence is an explicit Python loop in the training code; `torch.autograd` tracks the unroll.

`log_std` is a `nn.Parameter` for PPO/SAC exploration noise; **never written to JSON**. `best_model.json` is deterministic-policy-only.

**Exporter:**

```python
def export_policy_to_json(
    policy: V2Policy,
    path: str,
    obs_normalizer: ObsNormalizer | None = None,
) -> None:
    """
    Serialize policy to JSON v2. If obs_normalizer provided, bake (mean, std)
    into the first dense layer (or into the embedding following the window layer).
    """
```

**Loader:**

```python
def load_policy_from_json(path: str, device) -> V2Policy:
    """Parse JSON v2 and construct V2Policy. Round-trips with export."""
```

Loader is used by `report_rl.py` for post-training analysis and by test code for Rust/Python equivalence checks.

## 4. Rust runtime changes (concrete)

### 4.1 Module layout

```
src/rust/src/data/neural.rs        -- expanded with v2 types
src/rust/src/gnc/guidance/neural.rs -- signature change on nn_bank_angle
src/rust/src/gnc/guidance/dispatch.rs -- GuidanceState gains nn_state field
src/rust/src/simulation/runner.rs  -- build_sim_state threads model into GuidanceState::new
```

### 4.2 Key types

```rust
// src/rust/src/data/neural.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LayerSpec {
    Dense { input_size: usize, output_size: usize, activation: Activation },
    // Variants below reserved for later phases; not implemented in Phase 0:
    // Gru { input_size, hidden_size },
    // Lstm { input_size, hidden_size },
    // Attention { d_model, n_heads, window },
    // LayerNorm { size },
    // Ssm { d_model, d_state, dt_rank },
    // Window { input_size, n_steps },
}

#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    // ...future variants...
}

pub struct DenseLayer {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub activation: Activation,
}

pub struct NeuralNetModel {
    pub architecture: Vec<LayerSpec>,   // parsed spec (unchanged post-load)
    pub layers: Vec<Layer>,              // instantiated layers
    pub output_interpretation: String,
    pub input_mask: Option<Vec<usize>>,
    pub ablated_input: Option<usize>,
}

impl NeuralNetModel {
    pub fn forward(&self, state: &mut NnState, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();
        for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            x = layer.forward(x, layer_state);
        }
        x
    }
}

// src/rust/src/gnc/guidance/nn_state.rs (new file)

pub enum LayerState {
    None,
    // Reserved for later phases:
    // Gru { h: Vec<f64> },
    // Lstm { h: Vec<f64>, c: Vec<f64> },
    // Window { buffer: VecDeque<Vec<f64>> },
    // Ssm { h: Vec<f64> },
}

pub struct NnState {
    pub layer_states: Vec<LayerState>,
}

impl NnState {
    pub fn for_model(model: &NeuralNetModel) -> Self {
        let layer_states = model.layers.iter().map(LayerState::for_layer).collect();
        Self { layer_states }
    }

    pub fn reset(&mut self) {
        for s in self.layer_states.iter_mut() {
            s.reset();
        }
    }
}

impl Clone for NnState { /* per-variant clone */ }
```

### 4.3 v1 compatibility

`NeuralNetModel::load(path)` branches on `format_version`:
- `1`: parse as v1 struct, convert internally to v2 (all layers become `LayerSpec::Dense`), populate `Layer::Dense`.
- `2`: parse as v2 directly.
- Any other value (including missing): error out with a clear message.

Validation (input_mask, ablated_input, output_interpretation) applies identically. The existing 7 golden files (eqglide, energy_ctrl, pred_guid, fnpag, ftc, neural + regression) use v1 format and must continue to load and produce identical numerical output.

### 4.4 Guidance dispatch signature change

```rust
// Before:
pub fn nn_bank_angle(
    nav: &NavigationOutput,
    nn: &NeuralNetModel,
    data: &SimData,
    planet: &PlanetConfig,
    target_inclination: f64,
    ref_velocity_latched: f64,
) -> f64 { ... }

// After:
pub fn nn_bank_angle(
    nav: &NavigationOutput,
    nn: &NeuralNetModel,
    nn_state: &mut NnState,
    data: &SimData,
    planet: &PlanetConfig,
    target_inclination: f64,
    ref_velocity_latched: f64,
) -> f64 { ... }
```

Callers in `dispatch.rs` pass `&mut guidance_state.nn_state.as_mut().expect(...)`. The expect is infallible because this code path is only reached when the scheme is `neural_network`, and `build_sim_state` is required to set up `nn_state: Some(...)` when loading a neural model. This invariant is a debug-assert at `build_sim_state`.

## 5. Python / PyO3 changes

### 5.1 Module layout

```
src/python/aerocapture/training/rl/policy.py      -- V2Policy + build_layer dispatch
src/python/aerocapture/training/rl/export.py      -- export_policy_to_json v2
src/python/aerocapture/training/rl/layers/        -- new subpackage
  __init__.py
  dense.py                                         -- DenseLayer torch module (Phase 0)
  # gru.py, lstm.py, attention.py, ssm.py, window.py come in later phases
src/python/aerocapture/training/encoding.py       -- nn_param_specs_from_v2
src/python/aerocapture/training/model_io.py       -- new: load_policy_from_json (shared between RL and analysis)
```

### 5.2 Pydantic schemas

```python
# src/python/aerocapture/training/rl/schemas.py

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Discriminator

class DenseSpec(BaseModel):
    type: Literal["dense"]
    input_size: int
    output_size: int
    activation: Literal["tanh", "relu", "sigmoid", "asinh", "linear", "swish", "mish"]

# Future variants (Phase 1+):
# class GruSpec(BaseModel): ...
# class LstmSpec(BaseModel): ...
# class AttentionSpec(BaseModel): ...
# class LayerNormSpec(BaseModel): ...
# class SsmSpec(BaseModel): ...
# class WindowSpec(BaseModel): ...

LayerSpec = Annotated[Union[DenseSpec], Discriminator("type")]

class ArchitectureV2(BaseModel):
    format_version: Literal[2]
    architecture: list[LayerSpec]
    weights: dict[str, dict]  # keyed by "layer_<i>", contents schema per layer type
    output_interpretation: Literal["atan2", "direct"]
    input_mask: list[int] | None = None
    ablated_input: int | None = None
```

### 5.3 V2Policy

Step-wise forward, `new_state(batch_size)`, and `log_std` as non-exported parameter. Implementation is ~60 lines for Phase 0 since only `DenseLayer` exists. The subpackage structure (`layers/`) means adding a new layer type is one new file + one line in `build_layer`.

### 5.4 Export/load round-trip

Invariant: `load(export(policy))` produces a torch model with identical weights to the input, to bitwise precision. Round-trip test covers this in Phase 0 for dense-only architectures.

## 6. PSO integration

`encoding.py` changes:

```python
def nn_param_specs_from_v2(
    architecture: list[LayerSpec],
    bound_multiplier: float = 1.0,
) -> list[ParamSpec]:
    specs = []
    for layer in architecture:
        specs.extend(_layer_param_specs(layer, bound_multiplier))
    return specs

def _layer_param_specs(layer: LayerSpec, mult: float) -> list[ParamSpec]:
    match layer.type:
        case "dense":
            return _dense_specs(layer, mult)
        # ...future variants
```

`_dense_specs` generates Xavier-uniform bounds per weight + uniform `[-0.1, +0.1]` per bias, matching the existing `nn_param_specs_from_architecture` semantics to numerical precision. The old wrapper stays in place to avoid breaking the 4 existing NN training configs.

`problem.py` (AerocaptureProblem) is unchanged -- it still receives a flat `np.ndarray` decision vector and writes it to `best_model.json` via `NeuralNetModel::from_flat_weights` in Rust. v2 models round-trip the same way as v1 (flat weights are layer-ordered and layer-type-aware).

## 7. RL integration (minimal)

Phase 0 ships the `V2Policy` class and exporter. It does **not** rewire `train.py`, the rollout buffer, or the PPO/SAC loops to actually use stateful policies -- that's Phase 1 work (since stateful policies require a layer type that Phase 0 does not ship).

What Phase 0 does require in the RL module: the exporter must accept the optional `obs_normalizer` argument (to bake mean/std into layer 0) and produce a v2 JSON that `aerocapture_rs` (Rust loader) reads identically to the v1-exported JSON. The obs-normalization bake-in logic is the same math as today -- it just writes v2 schema. This gives Phase 1 a clean target to extend.

## 8. Testing strategy

### 8.1 Rust

- **Unit:** `DenseLayer::forward` identical to v1 `forward` on golden inputs.
- **Unit:** v1 JSON -> internal v2 representation preserves weights exactly.
- **Unit:** `NnState::for_model` produces one `LayerState::None` per dense layer.
- **Unit:** `NnState::Clone` produces an independent state (mutation of clone does not affect original).
- **Integration:** Existing 6 golden regression files (eqglide, energy_ctrl, pred_guid, fnpag, ftc, neural) run end-to-end and match their CSVs bit-for-bit.
- **Integration:** `BatchedSimulation` per-env state is per-env (no aliasing) -- verified by running two envs with different inputs and checking independence.
- **Proptest:** `NnState::reset` followed by identical inputs produces identical outputs.

### 8.2 Python

- **Unit:** Pydantic round-trip (JSON -> model -> JSON) is bitwise-identical.
- **Unit:** `nn_param_specs_from_v2` on an all-dense architecture produces specs numerically identical to `nn_param_specs_from_architecture`.
- **Unit:** Export -> load round-trip preserves weights bit-for-bit: `load_policy_from_json(export_policy_to_json(p, norm=None))` has the same `state_dict()` as `p`.
- **Unit:** `export_policy_to_json(policy, obs_normalizer=norm)` applies the bake-in correctly -- verified by feeding a known input through both `norm(x) -> policy(x)` and the baked policy, comparing outputs to 1e-12.

### 8.3 Cross-language equivalence

- **Integration (new):** `tests/test_v2_rust_python_equivalence.py` -- build the same architecture and weights in both PyTorch (`V2Policy`) and Rust (`NeuralNetModel` via JSON), feed 100 random inputs, assert max abs diff < 1e-10. Phase 0 covers `dense`-only architectures; each subsequent phase extends this test to cover the new layer type.

### 8.4 Regression

- **Integration:** Re-run `./check_all.sh` -- all existing Rust tests pass, all Python tests pass, clippy clean, fmt clean.
- **Integration:** Re-train `neural_network` for 5 generations with v1 config + existing best_model.json as seed -- verify identical convergence behavior to main-branch.

## 9. Backward compatibility

- v1 JSON files load unchanged. All 7 existing `best_model.json` / `best_params.json` files continue to work.
- v1 training configs (4 existing NN configs) produce identical training behavior.
- Rust CLI binary behavior unchanged for all guidance schemes.
- PyO3 API (`run`, `run_mc`, `run_batch`, `run_with_draws`) signatures unchanged.
- `compare_guidance.py` behavior unchanged (no new schemes registered in Phase 0).

## 10. Non-goals and deferred items

- Adding GRU, LSTM, attention, SSM, or window layer types. Phase 1 ships GRU; Phase 2 LSTM + Window; Phase 3 Transformer; Phase 4 Mamba.
- Rewiring RL training loop for stateful policies. Phase 1 concern.
- Training-time performance optimization of stateful forward (e.g., fused GRU kernel, cuDNN calls in PyTorch). Use naive manual implementations until a profiler says otherwise.
- Rust-side `torch::jit` or ONNX interop. Not needed; the JSON format is the source of truth.

## 11. Success criteria

Phase 0 is complete when:
1. `./check_all.sh` passes on a branch containing all Phase 0 changes.
2. All existing `neural_network` training and eval paths produce bit-identical results to main-branch.
3. A synthetic 2-layer dense model can be defined in Python via `V2Policy`, exported to JSON v2, loaded by Rust `NeuralNetModel`, and run with outputs matching PyTorch forward to 1e-10 on 100 random inputs.
4. Adding a new layer type requires zero changes to `dispatch.rs`, `runner.rs`, `BatchedSimulation`, `problem.py`, or `train.py` (only new files + one-line dispatch additions).

Criterion 4 is the real gate -- it proves the infrastructure holds.
