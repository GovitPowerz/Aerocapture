# Phase 1 -- PSO-GRU MVP

**Date:** 2026-04-17
**Status:** Design approved, ready for implementation planning
**Parent effort:** LSTM / Transformer / Mamba architectures for neural guidance (see `TODO.md`).
**Predecessor:** Phase 0 (stateful NN runtime infrastructure, merged as PR #37).

## 1. Context

Phase 0 shipped the infrastructure: JSON format v2, stateful `NeuralNetModel::forward(&mut NnState, &[f64])`, `LayerState` enum (Phase 0 variant: `None`), `LayerSpec` enum (Phase 0 variant: `Dense`), `LayerWeights` trait for flat-weight round-trip, `V2Policy` PyTorch mirror, cross-language equivalence gate at machine epsilon. Criterion #4 of the Phase 0 spec held: adding a new layer type touches only `data/neural.rs`, `data/nn_state.rs`, a new file under `rl/layers/`, one dispatch line in `rl/layers/__init__.py`, one `*Spec` class in `rl/schemas.py`, and one branch in `encoding.py::_layer_param_specs`.

Phase 1 exercises that extension point by adding the first stateful architecture: a small GRU sandwiched between dense layers. The scientific purpose is to test the paper's central hypothesis, that temporal memory improves PSO-trained aerocapture guidance over the 2008 feedforward baseline. If PSO-GRU beats PSO-MLP on the reserved-seed DV distribution, the paper has its primary signal; if not, we learn that the aerocapture pass is too short / too Markovian for recurrent memory to help, which is also publishable.

Phase 1 is scoped to PSO-GRU only. The BPTT-via-PPO axis for GRU is deferred to Phase 1.5 once the rollout-buffer hidden-state work has its own spec. The PPO/SAC infrastructure changes needed for stateful policies are non-trivial (per-step hidden-state snapshots, truncation-aware bootstrap with `V(terminal_obs, h_terminal)`), and shipping the PSO primary signal first keeps the PR reviewable.

## 2. Scope

**In scope:**

- Rust `LayerSpec::Gru { input_size, hidden_size }` variant.
- Rust `Layer` enum split: `DenseLayer` struct (rename of existing dense fields) + new `GruLayer` struct + `Layer` enum wrapping both. `NeuralNetModel::forward` iterates `&self.layers` and dispatches per variant.
- Rust `LayerState::Gru { h: Vec<f64> }` variant. `LayerState::for_layer` and `reset` gain Gru arms.
- Rust `LayerWeights for GruLayer` impl with canonical flat ordering.
- Rust v2 JSON read and write for Gru layers (`weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`).
- PyO3 `aerocapture_rs.flat_weights_to_json` helper routing all PSO output through Rust (closes Phase 0 carry-over item 2: trait adoption).
- Rust config parser accepts `[[network.architecture]]` TOML array-of-tables for heterogeneous architectures; falls back to v1 `layer_sizes` + `activations` when absent.
- Python `GruLayer` torch module (manual gate math, no `nn.GRUCell`) under `src/python/aerocapture/training/rl/layers/gru.py`.
- Python `GruSpec` Pydantic class; `LayerSpec` becomes a real discriminated union `Annotated[Union[DenseSpec, GruSpec], Discriminator("type")]` (reverting the Phase 0 single-variant alias workaround).
- Python `build_layer` dispatch gains a `gru` branch.
- Python `_layer_param_specs` gains a GRU branch producing Xavier-uniform bounds for all gate matrices and biases.
- Python `load_policy_from_json` and `export_v2_policy_to_json` gain GRU branches.
- PSO training config `configs/training/msr_aller_gru_pso_train.toml`.
- Scheme registration `neural_network_gru_pso` in `compare_guidance.py`.
- Test coverage: Rust unit tests for forward math and flat-weight roundtrip; Python unit tests for mirror forward + export round-trip; cross-language equivalence test extended with a GRU architecture and a non-None `input_mask` case (closes carry-over items 3 + 4); training smoke test that runs 5 PSO gens on a minimal GRU config and verifies checkpoint output.

**Out of scope for Phase 1:**

- PPO-GRU (rollout-buffer hidden-state snapshots, truncation-aware bootstrap). Deferred to Phase 1.5.
- LSTM, attention (Transformer), SSM (Mamba), window-MLP layer types. Phases 2-4.
- Workspace-wide clippy cleanup for pre-existing `aerocapture-py` warnings (Phase 0 carry-over item 5). Separate one-line fix, lands as its own commit before or parallel to Phase 1.
- Widening `load_policy_from_json` to accept v1 (Phase 0 carry-over item 1). No Phase 1 caller needs it. Defer until the v1-loading need materializes.
- `deny_unknown_fields` on TOML config structs to catch silent key drops (pre-existing issue noted in project memory). Orthogonal concern.

## 3. Architecture

Every Phase 1 GRU config conforms to this shape:

```
Input x_t (23 masked scalars, or fewer per input_mask)
  -> Dense(input_dim -> H, tanh)         layer 0: embedding + obs-norm bake-in
  -> GRU(H -> H)                         layer 1: recurrent
  -> Dense(H -> 2, linear)               layer 2: atan2 output head
```

Default `H = 32`. Configurable via TOML. Default layer-1 activation is implicit (GRU has no post-activation). Dense 0 uses tanh (standard embedding nonlinearity). Dense 2 uses linear + `output_interpretation = "atan2"` so the runtime computes `atan2(out[0], out[1])` for signed bank angle.

Total params at default H=32:

- Dense 0: (23 * 32) + 32 = 768
- GRU: 3 * ((32 * 32) + (32 * 32)) weights + 3 * (32 + 32) biases = 6144 + 192 = 6336
- Dense 2: (32 * 2) + 2 = 66
- Total: ~7170 parameters. Well within PSO-friendly range (<10k).

The dense-embedding-first constraint carries over from Phase 0 section 3.5 and simplifies obs-normalizer bake-in. Bake-in logic is unchanged from Phase 0: mean-std transform folded into Dense 0's W and b at export time.

## 4. GRU mathematical specification

The PyTorch `nn.GRUCell` convention, two biases per gate:

```
r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
h_t = (1 - z_t) * n_t + z_t * h_{t-1}
```

Note the `r_t * (W_hn @ h + b_hn)` grouping. This matches `nn.GRUCell` and differs from the Cho-2014 paper which has `W_hn @ (r_t * h)`. The PyTorch convention is the modern standard; matching it makes PyTorch-to-Rust weight transfer trivial.

Reset gate `r_t` controls how much prior hidden state leaks into the candidate `n_t`. Update gate `z_t` controls the blend between new candidate and old state. Output is the new hidden state directly (GRU, unlike LSTM, has no separate output gate).

The two biases per gate (`b_ir, b_hr`, etc.) are mathematically redundant since they always appear summed. PyTorch keeps both for initialization-symmetry reasons and because some downstream tooling expects the split. We match PyTorch to simplify cross-language equivalence checks and to avoid surprising anyone reading the JSON who expects PyTorch-shaped weights.

## 5. JSON v2 schema for GRU layers

Per-layer JSON entry:

```json
{
  "type": "gru",
  "input_size": 32,
  "hidden_size": 32
}
```

Weights dict entry for this layer (under the top-level `weights` map keyed by `layer_<i>`):

```json
{
  "weight_ih": [[...], ...],
  "weight_hh": [[...], ...],
  "bias_ih":   [...],
  "bias_hh":   [...]
}
```

Shape contract:

- `weight_ih`: `[3H, input_size]`, row-major. Rows `0..H` are `W_ir`, `H..2H` are `W_iz`, `2H..3H` are `W_in`. This matches `torch.nn.GRUCell.weight_ih` layout.
- `weight_hh`: `[3H, H]`, row-major. Rows `0..H` are `W_hr`, `H..2H` are `W_hz`, `2H..3H` are `W_hn`.
- `bias_ih`: `[3H]`. Order `b_ir, b_iz, b_in`.
- `bias_hh`: `[3H]`. Order `b_hr, b_hz, b_hn`.

Loading in PyTorch is a direct `weight_ih.copy_(json_tensor)` because `nn.GRUCell` uses this exact layout. Loading in Rust stores them as `Vec<Vec<f64>>` row-major for consistency with `DenseLayer.w`.

## 6. Flat weight ordering for PSO

`LayerWeights for GruLayer` defines the canonical flat layout:

```
weight_ih row 0, weight_ih row 1, ..., weight_ih row 3H-1,
weight_hh row 0, weight_hh row 1, ..., weight_hh row 3H-1,
bias_ih[0], bias_ih[1], ..., bias_ih[3H-1],
bias_hh[0], bias_hh[1], ..., bias_hh[3H-1]
```

In element terms, equivalent to Phase 0 spec section 3.4's per-gate listing:

```
W_ir, W_iz, W_in,  (each row-major, 3*H*input_size elements total)
W_hr, W_hz, W_hn,  (each row-major, 3*H*H elements total)
b_ir, b_iz, b_in,  (3H elements)
b_hr, b_hz, b_hn   (3H elements)
```

Total per-layer params: `2 * 3 * H * (input_size + H) / 2 * 2 + 6H` ... simpler: `3*H*input_size + 3*H*H + 6*H = 3*H*(input_size + H) + 6*H`.

For H=32 and input_size=32: `3*32*(32+32) + 6*32 = 6144 + 192 = 6336` params per GRU layer.

Python `_dense_specs` equivalent `_gru_specs` generates per-weight `ParamSpec`s: Xavier-uniform with `bound = compute_layer_bound(3*H, input_size, activation="tanh") * bound_multiplier` for `W_i*` (treating the 3H-output as the gate-concatenated fan-out), same for `W_h*`, plus fixed `+/- 0.1 * bound_multiplier` for biases. This is a deliberate simplification over layer-by-layer tuning: PSO does not need fine-grained init calibration, it just needs a reasonable sampling box.

## 7. Rust implementation

### 7.1 Module changes

```
src/rust/src/data/neural.rs       -- Layer enum split, GruLayer + LayerWeights impl,
                                     extended from_v2_json for gru, extended save_json
                                     for gru, extended from_flat_weights_v2 (new, accepts
                                     architecture spec rather than layer_sizes+activations).
src/rust/src/data/nn_state.rs     -- LayerState::Gru variant, for_layer and reset arms.
src/rust/src/gnc/guidance/neural.rs  -- no changes; nn_bank_angle already threads &mut NnState.
src/rust/src/config.rs            -- TOML reader for [[network.architecture]] array-of-tables.
src/rust/aerocapture-py/src/lib.rs -- new flat_weights_to_json PyO3 helper.
```

### 7.2 Key types

```rust
// data/neural.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LayerSpec {
    Dense { input_size: usize, output_size: usize, activation: Activation },
    Gru { input_size: usize, hidden_size: usize },
}

#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
}

// Existing Layer struct becomes DenseLayer:
pub struct DenseLayer {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub activation: Activation,
}

pub struct GruLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Vec<Vec<f64>>,  // [3H, I]
    pub weight_hh: Vec<Vec<f64>>,  // [3H, H]
    pub bias_ih:   Vec<f64>,        // [3H]
    pub bias_hh:   Vec<f64>,        // [3H]
}
```

### 7.3 GRU forward (Rust)

Per-step, gate-by-gate to match PyTorch exactly:

```rust
impl GruLayer {
    pub fn forward(&self, h_prev: &[f64], x: &[f64]) -> Vec<f64> {
        let h_size = self.hidden_size;
        let mut h_new = vec![0.0; h_size];

        // r, z gates (rows 0..2H apply to input x; rows 0..2H of weight_hh apply to h_prev)
        // n gate rows 2H..3H
        for i in 0..h_size {
            // r_i
            let mut s_ih = self.bias_ih[i];
            for k in 0..self.input_size { s_ih += self.weight_ih[i][k] * x[k]; }
            let mut s_hh = self.bias_hh[i];
            for k in 0..h_size { s_hh += self.weight_hh[i][k] * h_prev[k]; }
            let r = sigmoid(s_ih + s_hh);

            // z_i (row i + h_size)
            let mut sz_ih = self.bias_ih[i + h_size];
            for k in 0..self.input_size { sz_ih += self.weight_ih[i + h_size][k] * x[k]; }
            let mut sz_hh = self.bias_hh[i + h_size];
            for k in 0..h_size { sz_hh += self.weight_hh[i + h_size][k] * h_prev[k]; }
            let z = sigmoid(sz_ih + sz_hh);

            // n_i (row i + 2*h_size)
            let mut sn_ih = self.bias_ih[i + 2 * h_size];
            for k in 0..self.input_size { sn_ih += self.weight_ih[i + 2 * h_size][k] * x[k]; }
            let mut sn_hh = self.bias_hh[i + 2 * h_size];
            for k in 0..h_size { sn_hh += self.weight_hh[i + 2 * h_size][k] * h_prev[k]; }
            let n = (sn_ih + r * sn_hh).tanh();

            h_new[i] = (1.0 - z) * n + z * h_prev[i];
        }
        h_new
    }
}
```

This is the dumb, readable reference implementation. Optimization (fused gate matmuls, SIMD) deferred until profiling says it matters. For 600-step episodes at H=32 the per-tick cost is ~2500 multiplies, negligible compared to the sim step.

### 7.4 NeuralNetModel::forward dispatch

```rust
pub fn forward(&self, state: &mut NnState, input: &[f64]) -> Vec<f64> {
    // Input length matches layer[0]'s expected input_size
    // (DenseLayer.w[0].len() or GruLayer.input_size depending on variant).
    assert_eq!(input.len(), self.layers[0].input_size());
    assert_eq!(state.layer_states.len(), self.layers.len());
    let mut current = input.to_vec();
    for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
        match (layer, layer_state) {
            (Layer::Dense(d), LayerState::None) => {
                current = d.forward(&current);
            }
            (Layer::Gru(g), LayerState::Gru { h }) => {
                let h_new = g.forward(h, &current);
                *h = h_new.clone();
                current = h_new;  // GRU output = hidden state
            }
            _ => unreachable!("layer/state variant mismatch (construction invariant)"),
        }
    }
    current
}
```

The `(Dense, None)` and `(Gru, Gru {h})` pairings are guaranteed by `NnState::for_model` which calls `LayerState::for_layer(layer)` per-layer. A future layer type adds both a `LayerSpec` arm and a matching `LayerState` arm; the `unreachable!` stays defensive but never fires.

### 7.5 LayerWeights for GruLayer

```rust
impl LayerWeights for GruLayer {
    fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        for row in &self.weight_ih { v.extend_from_slice(row); }
        for row in &self.weight_hh { v.extend_from_slice(row); }
        v.extend_from_slice(&self.bias_ih);
        v.extend_from_slice(&self.bias_hh);
        v
    }
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let mut idx = 0;
        for row in self.weight_ih.iter_mut() {
            row.copy_from_slice(&flat[idx..idx + self.input_size]);
            idx += self.input_size;
        }
        for row in self.weight_hh.iter_mut() {
            row.copy_from_slice(&flat[idx..idx + self.hidden_size]);
            idx += self.hidden_size;
        }
        let three_h = 3 * self.hidden_size;
        self.bias_ih.copy_from_slice(&flat[idx..idx + three_h]);
        idx += three_h;
        self.bias_hh.copy_from_slice(&flat[idx..idx + three_h]);
        idx += three_h;
        idx
    }
    fn n_params(&self) -> usize {
        3 * self.hidden_size * self.input_size
            + 3 * self.hidden_size * self.hidden_size
            + 2 * 3 * self.hidden_size
    }
}
```

### 7.6 from_flat_weights_v2

New Rust constructor:

```rust
pub fn from_flat_weights_v2(
    flat: &[f64],
    architecture: &[LayerSpec],
    output_interpretation: &str,
    input_mask: Option<Vec<usize>>,
) -> Result<Self, DataError> {
    // For each spec, construct a zero-initialized Layer variant,
    // call layer.from_flat(&flat[offset..]), accumulate offset.
}
```

The existing v1 `from_flat_weights(flat, layer_sizes, activations)` stays as a thin wrapper that builds an all-Dense architecture and calls this.

### 7.7 save_json extension

`NeuralNetModel::save_json` already emits v2 via `NnJsonFileV2`. Extend `NnLayerWeights` (or a per-variant JSON type) to write `weight_ih`/`weight_hh`/`bias_ih`/`bias_hh` instead of `w`/`b` for Gru layers. Simplest: use `serde_json::Value` flexibility, writing whichever keys match the variant.

### 7.8 TOML schema extension

`config.rs::TomlNetwork` gains an optional `architecture` field:

```rust
pub struct TomlNetwork {
    pub layer_sizes: Option<Vec<usize>>,          // v1 path
    pub activations: Option<Vec<String>>,          // v1 path
    pub architecture: Option<Vec<TomlLayerSpec>>,  // v2 path (NEW)
    pub input_mask: Option<Vec<usize>>,
    pub ablated_input: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TomlLayerSpec {
    Dense { input_size: usize, output_size: usize, activation: String },
    Gru { input_size: usize, hidden_size: usize },
}
```

When `architecture` is present, it wins and `layer_sizes`/`activations` are ignored. When absent, existing v1 path unchanged. Validation: `input_mask.len()` must equal `architecture[0].input_size` (when architecture provided) or `layer_sizes[0]` (v1 path).

TOML config excerpt:

```toml
[network]
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
output_interpretation = "atan2"

[[network.architecture]]
type = "dense"
input_size = 16
output_size = 32
activation = "tanh"

[[network.architecture]]
type = "gru"
input_size = 32
hidden_size = 32

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 2
activation = "linear"
```

## 8. Python implementation

### 8.1 Module changes

```
src/python/aerocapture/training/rl/layers/gru.py         -- new: GruLayer torch module.
src/python/aerocapture/training/rl/layers/__init__.py    -- add build_layer gru branch.
src/python/aerocapture/training/rl/schemas.py            -- add GruSpec; restore LayerSpec discriminated union.
src/python/aerocapture/training/rl/export.py             -- add GRU JSON-write branch.
src/python/aerocapture/training/model_io.py              -- add GRU JSON-read branch.
src/python/aerocapture/training/rl/policy.py             -- no changes; V2Policy already dispatches per-layer.
src/python/aerocapture/training/encoding.py              -- add _gru_specs helper.
src/python/aerocapture/training/evaluate.py              -- route PSO chromosome write through PyO3 flat_weights_to_json.
```

### 8.2 GruLayer torch module

```python
# src/python/aerocapture/training/rl/layers/gru.py
import torch
from torch import Tensor, nn


class GruLayer(nn.Module):
    """GRU cell matching nn.GRUCell + Rust GruLayer bit-for-bit.

    Canonical flat weight order (LayerWeights trait + PSO chromosome):
    weight_ih row-major, weight_hh row-major, bias_ih, bias_hh.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih   = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh   = nn.Parameter(torch.empty(3 * hidden_size))
        # Init matches PyTorch's nn.GRUCell default: uniform(-1/sqrt(H), +1/sqrt(H)).
        stdv = hidden_size ** -0.5
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        # x: [batch, input_size]; h: [batch, hidden_size]
        H = self.hidden_size
        gates_x = x @ self.weight_ih.t() + self.bias_ih        # [batch, 3H]
        gates_h = h @ self.weight_hh.t() + self.bias_hh        # [batch, 3H]
        r = torch.sigmoid(gates_x[:, :H] + gates_h[:, :H])
        z = torch.sigmoid(gates_x[:, H:2*H] + gates_h[:, H:2*H])
        n = torch.tanh(gates_x[:, 2*H:3*H] + r * gates_h[:, 2*H:3*H])
        h_new = (1 - z) * n + z * h
        return h_new, h_new                                    # (output, new_state); both = h_new

    def new_state(self, batch_size: int, device) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)
```

Forward uses batched matmul (`x @ W.t()`) for PyTorch efficiency but computes gates with explicit index slicing to mirror the Rust loop exactly. The cross-language equivalence test will verify the bit-for-bit match at 1e-10.

### 8.3 Pydantic schemas

```python
class GruSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["gru"]
    input_size: int = Field(ge=1)
    hidden_size: int = Field(ge=1)


LayerSpec = Annotated[Union[DenseSpec, GruSpec], Discriminator("type")]
```

The Phase 0 single-variant alias workaround is removed.

### 8.4 encoding.py _gru_specs

```python
def _gru_specs(layer: GruSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    H = layer.hidden_size
    # Xavier-uniform on gate-concatenated matrices: fan_out = 3H.
    w_ih_bound = compute_layer_bound(layer.input_size, 3 * H, "tanh") * bound_multiplier
    w_hh_bound = compute_layer_bound(H, 3 * H, "tanh") * bound_multiplier
    b_bound = 0.1 * bound_multiplier
    specs: list[ParamSpec] = []
    # Mirror _dense_specs ParamSpec construction: name strings match the
    # flat-weight position so downstream logging / weight_stats stay coherent.
    for j in range(3 * H * layer.input_size):
        specs.append(ParamSpec(name=f"w_ih{layer_idx}_{j}", p_min=-w_ih_bound, p_max=+w_ih_bound, default=0.0, log_scale=False, is_integer=False))
    for j in range(3 * H * H):
        specs.append(ParamSpec(name=f"w_hh{layer_idx}_{j}", p_min=-w_hh_bound, p_max=+w_hh_bound, default=0.0, log_scale=False, is_integer=False))
    for j in range(3 * H):
        specs.append(ParamSpec(name=f"b_ih{layer_idx}_{j}", p_min=-b_bound, p_max=+b_bound, default=0.0, log_scale=False, is_integer=False))
    for j in range(3 * H):
        specs.append(ParamSpec(name=f"b_hh{layer_idx}_{j}", p_min=-b_bound, p_max=+b_bound, default=0.0, log_scale=False, is_integer=False))
    return specs
```

`_layer_param_specs` is updated to pass `layer_idx` through to the helper. `_dense_specs` already accepts it per Phase 0's encoding.py changes.

The "treat 3H as fan_out" choice is deliberate. Standard GRU init (including PyTorch's default `stdv = 1/sqrt(H)`) ignores the gate-concatenation and uses the hidden_size alone. We use Xavier with the concatenated fan-out because PSO samples uniformly and a slightly tighter box helps. This is a tuning knob, not a physics constraint -- empirical observation can revise.

### 8.5 evaluate.py routing through PyO3

```python
# src/python/aerocapture/training/evaluate.py
import aerocapture_rs  # already imported elsewhere for PyO3 sim calls

def write_nn_json(flat: np.ndarray, architecture: list[LayerSpec], path: Path, ...):
    arch_dicts = [spec.model_dump() for spec in architecture]
    aerocapture_rs.flat_weights_to_json(
        flat=flat,
        architecture=arch_dicts,
        path=str(path),
        output_interpretation="atan2",
        input_mask=input_mask,
    )
```

The existing Python-side v1 JSON writer is removed entirely. All PSO NN output now goes through Rust. This is the `LayerWeights` trait adoption the Phase 0 review asked for -- and it closes the "no production caller" concern.

### 8.6 PyO3 helper

```rust
#[pyfunction]
fn flat_weights_to_json(
    flat: Vec<f64>,
    architecture_json: String,   // JSON-serialized list[dict] from Python
    path: String,
    output_interpretation: String,
    input_mask: Option<Vec<usize>>,
) -> PyResult<()> {
    let specs: Vec<LayerSpec> = serde_json::from_str(&architecture_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!(
            "flat_weights_to_json: architecture_json parse error: {}", e
        )))?;
    let model = NeuralNetModel::from_flat_weights_v2(
        &flat, &specs, &output_interpretation, input_mask,
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    model.save_json(&path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(())
}
```

Python call site in `evaluate.py`:

```python
import json
arch_json = json.dumps([spec.model_dump() for spec in architecture])
aerocapture_rs.flat_weights_to_json(
    flat=flat.tolist(),
    architecture_json=arch_json,
    path=str(path),
    output_interpretation="atan2",
    input_mask=input_mask,
)
```

The JSON-string passthrough avoids adding `pythonize` to the `aerocapture-py` Cargo dependencies. `serde_json` is already in the tree (used by `data/neural.rs` for v2 parsing). One serialize/deserialize hop, but PSO NN-write is not hot-path (once per generation's best individual, not once per sim), so the cost is trivial.

## 9. Training config

`configs/training/msr_aller_gru_pso_train.toml`:

```toml
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "neural_network"

[data]
neural_network = "training_output/neural_network_gru_pso/best_model.json"
results_suffix = ".train_gru_pso"

[network]
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
output_interpretation = "atan2"

[[network.architecture]]
type = "dense"
input_size = 16
output_size = 32
activation = "tanh"

[[network.architecture]]
type = "gru"
input_size = 32
hidden_size = 32

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 2
activation = "linear"

[optimizer]
algorithm = "pso"
n_pop = 64
n_gen = 1000
seed_strategy = "adaptive"
training_n_sims = 20
validation_n_sims = 1000
```

The input_mask defaults to the 16 pre-bounce candidate inputs (matches the current neural_network baseline) so the comparison with PSO-MLP is apples-to-apples. Extending to the full 23-input mask or the 4 reference-trajectory inputs is a follow-up tuning experiment, not a Phase 1 concern.

## 10. Scheme registration

`src/python/aerocapture/training/compare_guidance.py` gains a `neural_network_gru_pso` entry. The entry points to `training_output/neural_network_gru_pso/best_model.json` (trained by the new TOML config) and uses the same `neural_network` Rust dispatch under the hood (the scheme name is a Python label; Rust sees the model file and dispatches per its architecture).

## 11. Tests

### 11.1 Rust unit

- `data::neural::tests::gru_forward_known_output`: hand-computed `GruLayer::forward` on a fixed `(W_ih, W_hh, b_ih, b_hh, x, h)`. One forward step produces a specific vector; assert to f64 machine epsilon.
- `data::neural::tests::gru_flat_weights_roundtrip`: construct GruLayer with random weights, call `to_flat`, `from_flat` into a zero-initialized twin, assert weights equal bit-for-bit.
- `data::neural::tests::v2_gru_json_roundtrip`: JSON v2 serialize GruLayer via `save_json`, reload via `from_json_str`, assert forward outputs match.
- `data::neural::tests::from_flat_weights_v2_mixed_arch`: flat weights -> NeuralNetModel with architecture [Dense, Gru, Dense] -> forward pass produces finite output.
- `data::nn_state::tests::clone_is_behaviorally_independent`: construct NnState with a Gru variant, mutate the clone's h vector, assert original unchanged. Closes Phase 0 carry-over item 4 (NnState::Clone structural -> behavioral).

### 11.2 Python unit

- `tests/test_gru_layer.py`: GruLayer forward shape, determinism, state shape from `new_state`, parameter count matches closed-form.
- `tests/test_v2_export.py`: extend with a mixed Dense+Gru+Dense architecture export/load roundtrip.
- `tests/test_nn_schemas_v2.py`: extend with a GruSpec validation test and a discriminated-union dispatch test (valid gru parses as GruSpec, unknown variant rejected).
- `tests/test_nn_param_specs_v2.py`: extend with a GruSpec param-specs test asserting total param count matches closed-form.

### 11.3 Cross-language equivalence

Extend `tests/test_v2_rust_python_equivalence.py` with two new cases:

- `test_rust_python_gru_equivalence`: Dense(5->8,tanh) + Gru(8->8) + Dense(8->2,linear). Feed 100 random inputs. Assert max abs diff < 1e-10.
- `test_rust_python_dense_equivalence_with_input_mask`: existing dense test with input_mask = [0, 2, 4] over a 5-input raw vector (runs only 3 inputs through the network). Closes Phase 0 carry-over item 3.

### 11.4 Training smoke

`tests/test_gru_pso_smoke.py`: invoke `train.py` via the Python API on a minimal GRU config (n_pop=8, n_gen=5, training_n_sims=2) and assert:

- Training completes without error.
- A `best_model.json` file is produced under the expected output dir.
- The produced JSON has `format_version: 2` and contains a `gru` layer.
- Loading the file back via `aerocapture_rs.nn_forward` on a zeros input returns finite outputs.

Smoke only. Not a convergence test. The scientific validation (PSO-GRU vs PSO-MLP) is a separate benchmark, not a CI gate.

### 11.5 Regression

- 6 guidance-regression golden files (eqglide, energy_ctrl, pred_guid, fnpag, ftc, neural) must still pass bit-identically. The existing `neural` golden uses v1 JSON; the v1 read path is untouched by Phase 1 so this is zero-risk.
- `./check_all.sh` and `./lint_code.sh` clean.

## 12. Success criteria

1. `./check_all.sh` + `./lint_code.sh` pass on `feature/gru-mvp` branch.
2. 6 guidance-regression golden files bit-identical to pre-branch.
3. Cross-language equivalence test covers GRU at max abs diff < 1e-10 on 100 random f64 inputs.
4. `tests/test_gru_pso_smoke.py` passes: 5 PSO generations complete on a minimal GRU config, producing a valid v2 JSON that loads and runs in Rust.
5. Scientific gate (not CI-gated, informal): a 500+ generation PSO run on `msr_aller_gru_pso_train.toml` produces a `best_model.json` whose reserved-seed validation cost is competitive with the PSO-MLP baseline. "Competitive" = within 10% of MLP DV or better. If dramatically worse, investigate before merging.

Criterion 5 is the scientific signal; criteria 1-4 are the engineering gate. The PR can merge on engineering criteria alone, with the scientific signal captured in a separate validation report.

## 13. Phase 0 carry-overs addressed

From TODO.md:

- Item 2 (LayerWeights trait adopt-or-delete): **Adopted.** `evaluate.py` writes NN JSON through the new `aerocapture_rs.flat_weights_to_json` PyO3 helper backed by `NeuralNetModel::from_flat_weights_v2 + save_json`. The trait gains a real production caller.
- Item 3 (cross-language test input_mask case): **Added** as `test_rust_python_dense_equivalence_with_input_mask`.
- Item 4 (NnState::Clone behavioral coverage): **Added** as `data::nn_state::tests::clone_is_behaviorally_independent` once `LayerState::Gru { h }` exists.
- Item 1 (v1 loader widening): **Deferred**, no Phase 1 caller.
- Item 5 (workspace clippy cleanup): **Deferred**, orthogonal fix.

## 14. Non-goals

- PPO-GRU (rollout-buffer hidden-state snapshots, truncation-aware bootstrap). Phase 1.5.
- LSTM, attention, SSM, window-MLP. Phases 2-4.
- Optimized GRU forward (fused gate matmuls, SIMD in Rust). Profile first; dumb loop is fine until throughput matters.
- BPTT-specific PyTorch policy code (per-timestep hidden state in the policy class for gradient tracking). Not needed for PSO training; Phase 1.5 adds it.
- Changing the input mask defaults or the reference-trajectory inputs. Architecture-only change; keep the input surface identical to the PSO-MLP baseline for fair comparison.
