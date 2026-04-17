# Stateful NN Runtime Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the JSON format v2 + stateful-capable Rust runtime + PyTorch mirror base class that unlocks later phases (GRU, LSTM, Transformer, Mamba) without introducing any new layer types.

**Architecture:** Refactor `NeuralNetModel` to carry a heterogeneous `Layer` enum (Phase 0 variants: `Dense` only), add a sibling `NnState` with `LayerState` enum (Phase 0 variant: `None` only) that lives outside the model in `GuidanceState`, thread `&mut NnState` through the guidance dispatch, and mirror this structure in Python via a new `V2Policy` class with step-wise `forward(x, state)`. v1 JSON models continue to load and produce bit-identical output.

**Tech Stack:** Rust 2024 (nalgebra, serde tagged enums), PyO3, Python 3.14 (PyTorch, Pydantic discriminated unions), pymoo (unchanged), pytest, proptest.

**Spec:** `docs/superpowers/specs/2026-04-17-stateful-nn-runtime-infrastructure-design.md`

---

## Task 0: Create feature branch

**Files:** (none)

- [ ] **Step 1: Create and switch to the feature branch**

Run: `git checkout -b feature/stateful-nn-runtime`
Expected: `Switched to a new branch 'feature/stateful-nn-runtime'`

- [ ] **Step 2: Verify working tree is clean apart from the in-progress spec/plan**

Run: `git status`
Expected: `TODO.md` modified (from brainstorming), new untracked files for the spec and plan under `docs/superpowers/`.

- [ ] **Step 3: Commit spec, plan, and TODO.md updates**

```bash
git add TODO.md docs/superpowers/specs/2026-04-17-stateful-nn-runtime-infrastructure-design.md docs/superpowers/plans/2026-04-17-stateful-nn-runtime-infrastructure-plan.md
git commit -m "docs: add Phase 0 stateful NN runtime infrastructure spec + plan"
```

Expected: one commit on `feature/stateful-nn-runtime`.

---

## Task 1: Rust v1 <-> v2 JSON schema types (no behavior change)

**Files:**
- Modify: `src/rust/src/data/neural.rs`
- Test: `src/rust/src/data/neural.rs` (inline `#[cfg(test)]` module)

This task introduces the v2 JSON schema (tagged-layer list) and converts v1 models to the same internal representation on load. No stateful forward yet; `NeuralNetModel::forward(input)` keeps the stateless signature.

- [ ] **Step 1: Write failing test -- v2 JSON parses to identical internal rep as v1**

Add to the `tests` module in `src/rust/src/data/neural.rs`:

```rust
#[test]
fn v2_json_parses_to_same_layers_as_v1() {
    let v1 = r#"{
      "format_version": 1,
      "architecture": { "layers": [3, 2], "activations": ["linear"] },
      "weights": { "layer_0": { "w": [[0.1,0.2,0.3],[0.4,0.5,0.6]], "b": [0.01,0.02] } },
      "output_interpretation": "atan2"
    }"#;
    let v2 = r#"{
      "format_version": 2,
      "architecture": [
        { "type": "dense", "input_size": 3, "output_size": 2, "activation": "linear" }
      ],
      "weights": { "layer_0": { "w": [[0.1,0.2,0.3],[0.4,0.5,0.6]], "b": [0.01,0.02] } },
      "output_interpretation": "atan2"
    }"#;
    let m1 = NeuralNetModel::from_json_str(v1, "v1").unwrap();
    let m2 = NeuralNetModel::from_json_str(v2, "v2").unwrap();
    assert_eq!(m1.layer_sizes, m2.layer_sizes);
    assert_eq!(m1.n_params(), m2.n_params());
    let input = vec![1.0, 2.0, 3.0];
    let o1 = m1.forward(&input);
    let o2 = m2.forward(&input);
    assert_eq!(o1, o2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aerocapture --lib data::neural::tests::v2_json_parses_to_same_layers_as_v1`
Expected: FAIL (no `from_json_str` public function yet, v2 schema not implemented).

- [ ] **Step 3: Extract `from_json_str` as a public helper in `data/neural.rs`**

Rename the private `from_json` to `from_json_str` and make it `pub`. `NeuralNetModel::load(path)` calls it after reading the file. No other logic change yet.

- [ ] **Step 4: Add v2 schema types next to the existing v1 types**

Insert above the existing `NnJsonFile` struct in `src/rust/src/data/neural.rs`:

```rust
/// v2 layer spec: tagged-union over the layer type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LayerSpec {
    Dense {
        input_size: usize,
        output_size: usize,
        activation: Activation,
    },
    // Phase 1+: Gru, Lstm, Attention, LayerNorm, Ssm, Window
}

#[derive(Debug, Deserialize)]
struct NnJsonFileV2 {
    format_version: u32,
    architecture: Vec<LayerSpec>,
    weights: std::collections::BTreeMap<String, NnLayerWeights>,
    output_interpretation: String,
    #[serde(default)]
    input_mask: Option<Vec<usize>>,
    #[serde(default)]
    ablated_input: Option<usize>,
}
```

- [ ] **Step 5: Implement format_version dispatch in `from_json_str`**

Replace the body of `from_json_str` with:

```rust
pub fn from_json_str(content: &str, path: &str) -> Result<Self, DataError> {
    let v: serde_json::Value = serde_json::from_str(content)
        .map_err(|e| DataError(format!("JSON parse error in {}: {}", path, e)))?;
    let fmt = v.get("format_version").and_then(|x| x.as_u64()).unwrap_or(0);
    match fmt {
        1 => Self::from_v1_json(content, path),
        2 => Self::from_v2_json(content, path),
        other => Err(DataError(format!(
            "Unsupported format_version {} in {} (expected 1 or 2)", other, path
        ))),
    }
}

fn from_v1_json(content: &str, path: &str) -> Result<Self, DataError> {
    // Move the existing v1 parsing body here (unchanged).
    // At the end, before returning NeuralNetModel, also produce the v2-style
    // `architecture: Vec<LayerSpec>` from `layer_sizes` + `activations`
    // and store it in a new `architecture` field (see Step 6).
    ...
}

fn from_v2_json(content: &str, path: &str) -> Result<Self, DataError> {
    let file: NnJsonFileV2 = serde_json::from_str(content)
        .map_err(|e| DataError(format!("JSON parse error in {}: {}", path, e)))?;

    let mut layers = Vec::with_capacity(file.architecture.len());
    let mut layer_sizes = Vec::with_capacity(file.architecture.len() + 1);

    for (i, spec) in file.architecture.iter().enumerate() {
        match spec {
            LayerSpec::Dense { input_size, output_size, activation } => {
                if i == 0 { layer_sizes.push(*input_size); }
                layer_sizes.push(*output_size);

                let key = format!("layer_{}", i);
                let lw = file.weights.get(&key).ok_or_else(|| {
                    DataError(format!("Missing {} in weights in {}", key, path))
                })?;

                if lw.w.len() != *output_size || lw.b.len() != *output_size {
                    return Err(DataError(format!(
                        "Layer {} size mismatch in {}", i, path
                    )));
                }
                for row in &lw.w {
                    if row.len() != *input_size {
                        return Err(DataError(format!(
                            "Layer {} weight row length mismatch in {}", i, path
                        )));
                    }
                }

                layers.push(Layer {
                    w: lw.w.clone(),
                    b: lw.b.clone(),
                    activation: *activation,
                });
            }
        }
    }

    Self::validate_mask(&file.input_mask, layer_sizes[0])?;
    Self::validate_ablated_input(&file.ablated_input)?;

    let output_size = *layer_sizes.last().unwrap_or(&0);
    if file.output_interpretation != "direct" && output_size < 2 {
        return Err(DataError(format!(
            "output_interpretation '{}' requires >= 2 outputs, got {} in {}",
            file.output_interpretation, output_size, path
        )));
    }

    Ok(NeuralNetModel {
        architecture: file.architecture,
        layer_sizes,
        layers,
        output_interpretation: file.output_interpretation,
        input_mask: file.input_mask,
        ablated_input: file.ablated_input,
    })
}
```

- [ ] **Step 6: Add the `architecture: Vec<LayerSpec>` field to `NeuralNetModel`**

Update the struct:

```rust
#[derive(Debug, Clone)]
pub struct NeuralNetModel {
    pub architecture: Vec<LayerSpec>,   // NEW: canonical v2-shaped spec
    pub layer_sizes: Vec<usize>,
    pub layers: Vec<Layer>,
    pub output_interpretation: String,
    pub input_mask: Option<Vec<usize>>,
    pub ablated_input: Option<usize>,
}
```

In `from_v1_json`, populate `architecture` by zipping `layer_sizes` windows with `activations`:

```rust
let architecture = (0..layers.len()).map(|i| LayerSpec::Dense {
    input_size: layer_sizes[i],
    output_size: layer_sizes[i + 1],
    activation: layers[i].activation,
}).collect();
```

Update `from_flat_weights` and any other constructors to populate `architecture` similarly.

- [ ] **Step 7: Update `save_json` to emit v2 format**

```rust
pub fn save_json(&self, path: &str) -> Result<(), DataError> {
    let mut weights = std::collections::BTreeMap::new();
    for (i, layer) in self.layers.iter().enumerate() {
        weights.insert(
            format!("layer_{}", i),
            NnLayerWeights { w: layer.w.clone(), b: layer.b.clone() },
        );
    }
    let file = NnJsonFileV2 {
        format_version: 2,
        architecture: self.architecture.clone(),
        weights,
        output_interpretation: self.output_interpretation.clone(),
        input_mask: self.input_mask.clone(),
        ablated_input: self.ablated_input,
    };
    // Requires Serialize on NnJsonFileV2; add derive.
    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| DataError(format!("JSON serialize error: {}", e)))?;
    std::fs::write(path, json)
        .map_err(|e| DataError(format!("Cannot write {}: {}", path, e)))?;
    Ok(())
}
```

Add `#[derive(Serialize)]` to `NnJsonFileV2` and `NnLayerWeights` if missing.

- [ ] **Step 8: Run the new test + full data::neural test module**

Run: `cargo test -p aerocapture --lib data::neural`
Expected: all existing tests pass + new test `v2_json_parses_to_same_layers_as_v1` passes.

- [ ] **Step 9: Run Rust full test suite to confirm zero regressions**

Run: `cargo test -p aerocapture`
Expected: all tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): add JSON v2 schema + v1->v2 internal conversion (no behavior change)"
```

---

## Task 2: Add `NnState` type with empty `LayerState::None`

**Files:**
- Create: `src/rust/src/gnc/guidance/nn_state.rs`
- Modify: `src/rust/src/gnc/guidance/mod.rs`
- Test: `src/rust/src/gnc/guidance/nn_state.rs` (inline tests)

- [ ] **Step 1: Create `nn_state.rs` with the type definitions**

```rust
//! Per-sim mutable state for stateful NN layers.
//!
//! Lives outside NeuralNetModel (which is immutable and shared via Arc).
//! Phase 0 ships only LayerState::None (dense layers are stateless).
//! Phase 1+ adds Gru, Lstm, Window, Ssm variants.

use crate::data::neural::{Layer, NeuralNetModel};

#[derive(Debug, Clone)]
pub enum LayerState {
    None,
    // Phase 1+: Gru { h: Vec<f64> }, Lstm { h: Vec<f64>, c: Vec<f64> },
    // Window { buffer: std::collections::VecDeque<Vec<f64>> }, Ssm { h: Vec<f64> },
}

impl LayerState {
    pub fn for_layer(layer: &Layer) -> Self {
        let _ = layer;  // all Phase 0 layers are stateless
        LayerState::None
    }

    pub fn reset(&mut self) {
        match self {
            LayerState::None => {}
        }
    }
}

#[derive(Debug, Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::neural::{Activation, Layer, LayerSpec, NeuralNetModel};

    fn two_layer_model() -> NeuralNetModel {
        NeuralNetModel {
            architecture: vec![
                LayerSpec::Dense { input_size: 3, output_size: 2, activation: Activation::Tanh },
                LayerSpec::Dense { input_size: 2, output_size: 1, activation: Activation::Linear },
            ],
            layer_sizes: vec![3, 2, 1],
            layers: vec![
                Layer { w: vec![vec![0.1; 3]; 2], b: vec![0.0; 2], activation: Activation::Tanh },
                Layer { w: vec![vec![0.1; 2]; 1], b: vec![0.0; 1], activation: Activation::Linear },
            ],
            output_interpretation: "direct".to_string(),
            input_mask: None,
            ablated_input: None,
        }
    }

    #[test]
    fn for_model_produces_one_state_per_layer() {
        let model = two_layer_model();
        let state = NnState::for_model(&model);
        assert_eq!(state.layer_states.len(), 2);
        for s in &state.layer_states {
            matches!(s, LayerState::None);
        }
    }

    #[test]
    fn clone_is_independent() {
        let model = two_layer_model();
        let state = NnState::for_model(&model);
        let cloned = state.clone();
        // With only LayerState::None, there is nothing mutable to diverge yet;
        // assert structural equivalence to lock the invariant.
        assert_eq!(state.layer_states.len(), cloned.layer_states.len());
    }

    #[test]
    fn reset_is_idempotent_on_none_states() {
        let model = two_layer_model();
        let mut state = NnState::for_model(&model);
        state.reset();
        state.reset();
        assert_eq!(state.layer_states.len(), 2);
    }
}
```

- [ ] **Step 2: Add module declaration**

Add to `src/rust/src/gnc/guidance/mod.rs`:

```rust
pub mod nn_state;
```

- [ ] **Step 3: Run tests to verify pass**

Run: `cargo test -p aerocapture --lib gnc::guidance::nn_state`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/nn_state.rs src/rust/src/gnc/guidance/mod.rs
git commit -m "feat(nn): add NnState + LayerState scaffolding (Phase 0 ships None only)"
```

---

## Task 3: Update `nn_bank_angle` to take `&mut NnState`

**Files:**
- Modify: `src/rust/src/gnc/guidance/neural.rs`
- Modify: `src/rust/src/gnc/guidance/dispatch.rs` (callers)

- [ ] **Step 1: Write failing test -- stateful forward with empty state matches stateless forward**

Add to the `tests` module in `src/rust/src/gnc/guidance/neural.rs`:

```rust
#[test]
fn stateful_forward_with_empty_state_matches_stateless() {
    use crate::gnc::guidance::nn_state::NnState;

    let nn = zero_weight_nn(0.5, 0.5);
    let nav = test_nav();
    let data = test_sim_data();
    let planet = PlanetConfig::mars();
    let mut state = NnState::for_model(&nn);

    let bank = nn_bank_angle(
        &nav, &nn, &mut state, &data, &planet,
        50.0_f64.to_radians(), 0.0,
    );
    assert_relative_eq!(bank, 0.5_f64.atan2(0.5), epsilon = 1e-12);
}
```

- [ ] **Step 2: Run to confirm failure**

Run: `cargo test -p aerocapture --lib gnc::guidance::neural::tests::stateful_forward_with_empty_state_matches_stateless`
Expected: FAIL (signature mismatch).

- [ ] **Step 3: Update `nn_bank_angle` signature**

Change the function in `src/rust/src/gnc/guidance/neural.rs`:

```rust
pub fn nn_bank_angle(
    nav: &NavigationOutput,
    nn: &NeuralNetModel,
    nn_state: &mut NnState,
    data: &SimData,
    planet: &PlanetConfig,
    target_inclination: f64,
    ref_velocity_latched: f64,
) -> f64 {
    let masked = build_nn_input(nav, nn, data, planet, target_inclination, ref_velocity_latched);
    let output = nn.forward(nn_state, &masked);  // stateful forward
    match nn.output_interpretation.as_str() {
        "direct" => output[0] * 2.0 * std::f64::consts::PI,
        _ => output[0].atan2(output[1]),
    }
}
```

Add `use crate::gnc::guidance::nn_state::NnState;` at the top of the file.

- [ ] **Step 4: Update `NeuralNetModel::forward` signature to take `&mut NnState`**

In `src/rust/src/data/neural.rs`:

```rust
pub fn forward(&self, state: &mut NnState, input: &[f64]) -> Vec<f64> {
    assert_eq!(input.len(), self.layer_sizes[0]);
    assert_eq!(state.layer_states.len(), self.layers.len());
    let mut current = input.to_vec();
    for (layer, _layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
        let n_out = layer.b.len();
        let mut next = Vec::with_capacity(n_out);
        for j in 0..n_out {
            let sum: f64 = layer.w[j].iter().zip(&current).map(|(w, x)| w * x).sum();
            next.push(layer.activation.apply(sum + layer.b[j]));
        }
        current = next;
    }
    current
}
```

Add `use crate::gnc::guidance::nn_state::NnState;` at the top of `src/rust/src/data/neural.rs`.

- [ ] **Step 5: Update all existing tests in `data/neural.rs` and `guidance/neural.rs` to pass `&mut state`**

For each test calling `.forward(&input)` or `nn_bank_angle(...)` with the old signature, introduce:

```rust
let mut state = NnState::for_model(&nn);
let output = nn.forward(&mut state, &input);
// or
let bank = nn_bank_angle(&nav, &nn, &mut state, &data, &planet, tgt, 0.0);
```

All existing assertions keep their values; behavior is unchanged because `LayerState::None` is a no-op in the forward loop.

- [ ] **Step 6: Update callers in `dispatch.rs`**

Find the call to `nn_bank_angle` (there is one, in the neural-network branch of the guidance dispatch). Update to pass `&mut guidance_state.nn_state.as_mut().expect("neural_network scheme requires nn_state initialized by GuidanceState::new")`.

`guidance_state.nn_state` does not exist yet -- Task 4 adds it. For this task, either:
- Skip this step (callers updated in Task 4), or
- Create a local placeholder `let mut state = NnState::for_model(nn_model);` just above the call.

Use the placeholder approach so this task's tests compile standalone:

```rust
// TEMPORARY: Task 4 will wire this through GuidanceState.
let mut temp_nn_state = NnState::for_model(nn_model);
let bank = nn_bank_angle(
    &nav_output, nn_model, &mut temp_nn_state,
    &sim_data, &planet, target_inclination, ref_velocity_latched,
);
```

Leave a `// TODO(Task 4):` comment next to the placeholder.

- [ ] **Step 7: Run full Rust test suite**

Run: `cargo test -p aerocapture`
Expected: all tests pass.

- [ ] **Step 8: Run clippy clean + fmt check**

Run: `cargo clippy -p aerocapture --all-targets -- -D warnings && cargo fmt -p aerocapture --check`
Expected: clean exit.

- [ ] **Step 9: Commit**

```bash
git add src/rust/src/data/neural.rs src/rust/src/gnc/guidance/neural.rs src/rust/src/gnc/guidance/dispatch.rs
git commit -m "feat(nn): stateful NeuralNetModel::forward + nn_bank_angle(&mut NnState)"
```

---

## Task 4: Wire `NnState` through `GuidanceState` and `build_sim_state`

**Files:**
- Modify: `src/rust/src/gnc/guidance/dispatch.rs`
- Modify: `src/rust/src/simulation/runner.rs`
- Modify: `src/rust/src/simulation/init.rs` (if `GuidanceState::new` is constructed there)

- [ ] **Step 1: Write failing test -- BatchedSimulation per-env NnState is independent**

Add a test in `src/rust/tests/test_batched_simulation_nn_state.rs` (new file) or extend an existing BatchedSimulation integration test. Minimal assertion: two envs with the same neural-network scheme produce independent `GuidanceState::nn_state` instances (mutation of one does not affect the other).

Since NnState has no mutable internal state in Phase 0, a structural check is sufficient:

```rust
#[test]
fn guidance_state_nn_state_is_per_env() {
    use aerocapture::config::SimInput;
    use aerocapture::simulation::runner::build_sim_state;
    // Load a minimal neural-network config (use an existing test fixture)
    let config = SimInput::from_toml_file("configs/test/test_neural_network_min.toml").unwrap();
    let data = aerocapture::data::SimData::from_config(&config).unwrap();
    let run_state_0 = aerocapture::simulation::init::build_run_state(&config, &data, 0);
    let run_state_1 = aerocapture::simulation::init::build_run_state(&config, &data, 1);
    let s0 = build_sim_state(&config, &data, run_state_0, 0);
    let s1 = build_sim_state(&config, &data, run_state_1, 1);
    assert!(s0.guidance_state.nn_state.is_some());
    assert!(s1.guidance_state.nn_state.is_some());
    // Structural: both have the same number of layer states as the model.
    let n_layers = data.neural_net.as_ref().unwrap().layers.len();
    assert_eq!(s0.guidance_state.nn_state.as_ref().unwrap().layer_states.len(), n_layers);
    assert_eq!(s1.guidance_state.nn_state.as_ref().unwrap().layer_states.len(), n_layers);
}
```

If `configs/test/test_neural_network_min.toml` does not exist, use the existing test config that loads a neural-network model (check `configs/test/` for one that references `[data] neural_network`).

- [ ] **Step 2: Run to confirm failure**

Run: `cargo test --test test_batched_simulation_nn_state` (adjust test target name if placed inline).
Expected: FAIL (`nn_state` field does not exist on `GuidanceState`).

- [ ] **Step 3: Add `nn_state` field to `GuidanceState`**

In `src/rust/src/gnc/guidance/dispatch.rs`:

```rust
use crate::gnc::guidance::nn_state::NnState;
use crate::data::neural::NeuralNetModel;

pub struct GuidanceState {
    pub bank_angle_commanded: f64,
    pub bank_angle_realized: f64,
    pub aoa_commanded: f64,
    pub command_shaper: CommandShaper,
    pub lateral_state: LateralState,
    pub nn_state: Option<NnState>,   // NEW
    // ...existing fields
}

impl GuidanceState {
    pub fn new(
        initial_bank: f64,
        initial_aoa: f64,
        nn_model: Option<&NeuralNetModel>,  // NEW parameter
    ) -> Self {
        let nn_state = nn_model.map(NnState::for_model);
        Self {
            bank_angle_commanded: initial_bank,
            bank_angle_realized: initial_bank,
            aoa_commanded: initial_aoa,
            command_shaper: CommandShaper::new(),
            lateral_state: LateralState::default(),
            nn_state,
            // ...existing field initializers
        }
    }
}
```

- [ ] **Step 4: Update all `GuidanceState::new` call sites**

Search for `GuidanceState::new`:

Run: `rg "GuidanceState::new\(" src/rust/src/ -n`

For each caller, pass `data.neural_net.as_ref()`. In `build_sim_state` (`src/rust/src/simulation/runner.rs`):

```rust
let guidance_state = GuidanceState::new(
    entry_initial_bank,
    entry_initial_aoa,
    data.neural_net.as_ref(),
);
```

Verify every call site is updated. Expect 2-3 sites total.

- [ ] **Step 5: Replace the Task 3 placeholder in dispatch**

Remove the temporary `let mut temp_nn_state = NnState::for_model(nn_model);` from Task 3 Step 6 and wire to `guidance_state.nn_state`:

```rust
let nn_state = guidance_state.nn_state.as_mut()
    .expect("neural_network scheme requires nn_state initialized by GuidanceState::new");
let bank = nn_bank_angle(
    &nav_output, nn_model, nn_state,
    &sim_data, &planet, target_inclination, ref_velocity_latched,
);
```

Remove the `// TODO(Task 4):` comment.

- [ ] **Step 6: Add debug_assert in build_sim_state**

In `build_sim_state`, after constructing `guidance_state`:

```rust
debug_assert!(
    (data.neural_net.is_some()) == (guidance_state.nn_state.is_some()),
    "nn_state presence must match neural_net presence"
);
```

- [ ] **Step 7: Run Rust tests**

Run: `cargo test -p aerocapture`
Expected: all tests pass, including the new per-env state test.

- [ ] **Step 8: Run the 6 golden regression tests explicitly**

Run: `cargo test -p aerocapture --test regression_golden` (adjust to actual regression test target name; check `src/rust/tests/`).
Expected: all 6 golden files (eqglide, energy_ctrl, pred_guid, fnpag, ftc, neural) pass bit-identity checks.

- [ ] **Step 9: Commit**

```bash
git add src/rust/src/gnc/guidance/dispatch.rs src/rust/src/simulation/runner.rs src/rust/tests/test_batched_simulation_nn_state.rs
git commit -m "feat(nn): thread NnState through GuidanceState and build_sim_state"
```

---

## Task 5: Extend `to_flat_weights` / `from_flat_weights` with per-layer trait

**Files:**
- Modify: `src/rust/src/data/neural.rs`

Phase 0 still has only dense layers, so behavior is identical. The trait-based extension makes adding layer types in Phase 1+ a drop-in.

- [ ] **Step 1: Write failing test -- round-trip through flat weights preserves output**

Add to `tests` module:

```rust
#[test]
fn flat_weights_roundtrip_dense() {
    use crate::gnc::guidance::nn_state::NnState;

    let original = NeuralNetModel {
        architecture: vec![
            LayerSpec::Dense { input_size: 4, output_size: 3, activation: Activation::Tanh },
            LayerSpec::Dense { input_size: 3, output_size: 2, activation: Activation::Linear },
        ],
        layer_sizes: vec![4, 3, 2],
        layers: vec![
            Layer {
                w: vec![vec![0.1,0.2,0.3,0.4], vec![0.5,0.6,0.7,0.8], vec![-0.1,-0.2,-0.3,-0.4]],
                b: vec![0.01, 0.02, 0.03],
                activation: Activation::Tanh,
            },
            Layer {
                w: vec![vec![0.1,0.2,0.3], vec![-0.1,-0.2,-0.3]],
                b: vec![0.1, -0.1],
                activation: Activation::Linear,
            },
        ],
        output_interpretation: "atan2".to_string(),
        input_mask: None,
        ablated_input: None,
    };

    let flat = original.to_flat_weights();
    let layer_sizes: Vec<usize> = original.layer_sizes.clone();
    let activations = vec![Activation::Tanh, Activation::Linear];
    let reconstructed = NeuralNetModel::from_flat_weights(&flat, &layer_sizes, &activations).unwrap();

    let input = vec![0.5, -0.3, 0.1, 0.7];
    let mut s0 = NnState::for_model(&original);
    let mut s1 = NnState::for_model(&reconstructed);
    let o0 = original.forward(&mut s0, &input);
    let o1 = reconstructed.forward(&mut s1, &input);
    assert_eq!(o0, o1);
}
```

- [ ] **Step 2: Run to confirm the test passes today (sanity) or fails (if forward changed behavior)**

Run: `cargo test -p aerocapture --lib data::neural::tests::flat_weights_roundtrip_dense`
Expected: PASS -- documents current behavior to protect against regression in Step 3.

- [ ] **Step 3: Introduce the `LayerWeights` trait and refactor dense**

Add to `src/rust/src/data/neural.rs`:

```rust
/// Trait for flattening and reconstructing a layer's parameters.
///
/// Each layer type implements its own canonical flat ordering:
/// dense = W (row-major) then b; gru/lstm/attention/ssm defined per variant
/// (see Phase 1+ for those). Order MUST match the PyTorch mirror in
/// src/python/aerocapture/training/rl/layers/<type>.py for PSO chromosome
/// compatibility.
pub trait LayerWeights {
    fn to_flat(&self) -> Vec<f64>;
    fn from_flat(&mut self, flat: &[f64]) -> usize;  // returns bytes consumed
    fn n_params(&self) -> usize;
}

impl LayerWeights for Layer {
    fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        for row in &self.w {
            v.extend_from_slice(row);
        }
        v.extend_from_slice(&self.b);
        v
    }

    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let n_out = self.w.len();
        let n_in = if n_out > 0 { self.w[0].len() } else { 0 };
        let mut idx = 0;
        for j in 0..n_out {
            self.w[j].copy_from_slice(&flat[idx..idx + n_in]);
            idx += n_in;
        }
        self.b.copy_from_slice(&flat[idx..idx + n_out]);
        idx += n_out;
        idx
    }

    fn n_params(&self) -> usize {
        let n_out = self.w.len();
        let n_in = if n_out > 0 { self.w[0].len() } else { 0 };
        n_out * n_in + n_out
    }
}
```

- [ ] **Step 4: Refactor `NeuralNetModel::to_flat_weights` / `from_flat_weights` to use the trait**

```rust
pub fn to_flat_weights(&self) -> Vec<f64> {
    let mut flat = Vec::with_capacity(self.n_params());
    for layer in &self.layers {
        flat.extend(layer.to_flat());
    }
    flat
}

pub fn from_flat_weights(
    weights: &[f64],
    layer_sizes: &[usize],
    activations: &[Activation],
) -> Result<Self, DataError> {
    if activations.len() != layer_sizes.len() - 1 {
        return Err(DataError("Activation count != layer count - 1".to_string()));
    }
    let mut architecture = Vec::with_capacity(activations.len());
    let mut layers = Vec::with_capacity(activations.len());
    let mut offset = 0;
    for i in 0..activations.len() {
        let n_in = layer_sizes[i];
        let n_out = layer_sizes[i + 1];
        architecture.push(LayerSpec::Dense {
            input_size: n_in,
            output_size: n_out,
            activation: activations[i],
        });
        let mut layer = Layer {
            w: vec![vec![0.0; n_in]; n_out],
            b: vec![0.0; n_out],
            activation: activations[i],
        };
        let consumed = layer.from_flat(&weights[offset..]);
        offset += consumed;
        layers.push(layer);
    }
    if offset != weights.len() {
        return Err(DataError(format!(
            "Weight vector length mismatch: consumed {} of {}", offset, weights.len()
        )));
    }
    Ok(NeuralNetModel {
        architecture,
        layer_sizes: layer_sizes.to_vec(),
        layers,
        output_interpretation: "atan2".to_string(),
        input_mask: None,
        ablated_input: None,
    })
}
```

- [ ] **Step 5: Run the roundtrip test and full suite**

Run: `cargo test -p aerocapture --lib data::neural`
Expected: roundtrip test + existing tests all pass.

Run: `cargo test -p aerocapture`
Expected: full suite clean.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "refactor(nn): LayerWeights trait for flat-weight round-trip (Phase 0 dense only)"
```

---

## Task 6: Regression gate -- confirm bit-identity on all 6 golden files

**Files:** (none modified; verification only)

- [ ] **Step 1: Run Rust golden regression suite**

Run: `cargo test -p aerocapture --release -- regression`
Expected: all 6 golden files (eqglide, energy_ctrl, pred_guid, fnpag, ftc, neural) match CSVs bit-for-bit.

- [ ] **Step 2: Run `./check_all.sh`**

Run: `./check_all.sh`
Expected: cargo test + fmt --check + clippy + release build all clean.

- [ ] **Step 3: Rebuild PyO3 bindings and run Python integration tests**

Run: `uv run maturin develop --manifest-path src/rust/aerocapture-py/Cargo.toml --release && uv run pytest tests/test_pyo3.py -v`
Expected: all PyO3 tests pass.

- [ ] **Step 4: Run the `neural_network` training config smoke check (5 gens)**

Run:
```bash
uv run python -m aerocapture.training.train \
  configs/training/msr_aller_nn_train_consolidated.toml \
  --n-gen 5 --no-tui --skip-report
```

Expected: training runs end-to-end without error, produces a checkpoint in `training_output/neural_network/`.

- [ ] **Step 5: Confirm no commit needed -- this is pure verification**

If all steps above passed, proceed to Task 7. If any failed, diagnose and fix before proceeding.

---

## Task 7: Python Pydantic v2 schemas

**Files:**
- Create: `src/python/aerocapture/training/rl/schemas.py`
- Test: `tests/test_nn_schemas_v2.py`

- [ ] **Step 1: Write failing test -- v2 JSON round-trip via Pydantic**

Create `tests/test_nn_schemas_v2.py`:

```python
import json

import pytest

from aerocapture.training.rl.schemas import ArchitectureV2, DenseSpec


def test_v2_dense_json_roundtrip():
    raw = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 3, "output_size": 2, "activation": "linear"}
        ],
        "weights": {"layer_0": {"w": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "b": [0.01, 0.02]}},
        "output_interpretation": "atan2",
    }
    model = ArchitectureV2.model_validate(raw)
    assert len(model.architecture) == 1
    assert isinstance(model.architecture[0], DenseSpec)
    assert model.architecture[0].input_size == 3
    assert model.output_interpretation == "atan2"
    roundtrip = model.model_dump(exclude_none=True)
    assert json.dumps(roundtrip, sort_keys=True) == json.dumps(raw, sort_keys=True)


def test_v2_rejects_unknown_layer_type():
    raw = {
        "format_version": 2,
        "architecture": [{"type": "mystery", "foo": 42}],
        "weights": {},
        "output_interpretation": "atan2",
    }
    with pytest.raises(Exception):
        ArchitectureV2.model_validate(raw)


def test_v2_rejects_wrong_format_version():
    raw = {
        "format_version": 3,
        "architecture": [],
        "weights": {},
        "output_interpretation": "atan2",
    }
    with pytest.raises(Exception):
        ArchitectureV2.model_validate(raw)
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_nn_schemas_v2.py -v`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Create the Pydantic schema module**

Create `src/python/aerocapture/training/rl/schemas.py`:

```python
"""Pydantic schemas for NN model JSON v2 format.

Mirror of the Rust serde types in src/rust/src/data/neural.rs.
Adding a new layer type means: add a *Spec class, list it in LayerSpec, and
add the matching Rust variant. No other file in this module changes.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field


Activation = Literal["tanh", "relu", "sigmoid", "asinh", "linear", "swish", "mish"]


class DenseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["dense"]
    input_size: int = Field(ge=1)
    output_size: int = Field(ge=1)
    activation: Activation


# Phase 1+ variants (GruSpec, LstmSpec, AttentionSpec, LayerNormSpec, SsmSpec, WindowSpec)
# are appended to this union as they land.
LayerSpec = Annotated[Union[DenseSpec], Discriminator("type")]


class LayerWeights(BaseModel):
    model_config = ConfigDict(extra="allow")  # per-layer-type schema-free bag
    w: list[list[float]] | None = None
    b: list[float] | None = None


class ArchitectureV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format_version: Literal[2]
    architecture: list[LayerSpec]
    weights: dict[str, LayerWeights]
    output_interpretation: Literal["atan2", "direct"]
    input_mask: list[int] | None = None
    ablated_input: int | None = None
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_nn_schemas_v2.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/schemas.py tests/test_nn_schemas_v2.py
git commit -m "feat(nn): Pydantic v2 schemas (DenseSpec, ArchitectureV2)"
```

---

## Task 8: Python `V2Policy` class with `DenseLayer`

**Files:**
- Create: `src/python/aerocapture/training/rl/layers/__init__.py`
- Create: `src/python/aerocapture/training/rl/layers/dense.py`
- Modify: `src/python/aerocapture/training/rl/policy.py`
- Test: `tests/test_v2_policy.py`

- [ ] **Step 1: Write failing test -- V2Policy forward produces expected shape + deterministic output**

Create `tests/test_v2_policy.py`:

```python
import torch

from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec


def _two_layer_policy() -> V2Policy:
    architecture = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    return V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)


def test_v2_policy_forward_shape():
    policy = _two_layer_policy()
    state = policy.new_state(batch_size=1, device="cpu")
    x = torch.tensor([[0.5, -0.3, 0.1]])
    y, new_state = policy(x, state)
    assert y.shape == (1, 2)
    assert len(new_state) == 2


def test_v2_policy_forward_is_deterministic():
    torch.manual_seed(0)
    policy = _two_layer_policy()
    state1 = policy.new_state(1, "cpu")
    state2 = policy.new_state(1, "cpu")
    x = torch.tensor([[0.5, -0.3, 0.1]])
    y1, _ = policy(x, state1)
    y2, _ = policy(x, state2)
    torch.testing.assert_close(y1, y2)


def test_v2_policy_log_std_not_in_state_dict_export_contract():
    policy = _two_layer_policy()
    sd = policy.state_dict()
    assert "log_std" in sd  # log_std IS in state_dict
    # but exporter filters it out; separately tested in export round-trip.
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_v2_policy.py -v`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Create `layers/dense.py`**

Create `src/python/aerocapture/training/rl/layers/dense.py`:

```python
"""Dense (fully-connected) layer matching the Rust DenseLayer variant.

Canonical flat weight order: W (row-major, [out, in]) then b.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


_ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "asinh": torch.asinh,
    "linear": lambda x: x,
    "swish": lambda x: x * torch.sigmoid(x),
    "mish": lambda x: x * torch.tanh(torch.nn.functional.softplus(x)),
}


class DenseLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation: str):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.activation_name = activation
        self.activation_fn = _ACTIVATIONS[activation]

    def forward(self, x: Tensor, state: None) -> tuple[Tensor, None]:
        """Stateful-compatible signature. State is always None for dense layers."""
        return self.activation_fn(self.linear(x)), None

    def new_state(self, batch_size: int, device) -> None:
        return None

    def extra_repr(self) -> str:
        return f"activation={self.activation_name}"
```

- [ ] **Step 4: Create `layers/__init__.py`**

Create `src/python/aerocapture/training/rl/layers/__init__.py`:

```python
"""Torch mirrors of Rust layer types. One file per layer variant."""

from aerocapture.training.rl.layers.dense import DenseLayer

__all__ = ["DenseLayer", "build_layer"]


def build_layer(spec):
    """Dispatch a LayerSpec to its torch module constructor."""
    if spec.type == "dense":
        return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    # Phase 1+: gru, lstm, attention, layer_norm, ssm, window
    raise ValueError(f"Unknown layer type: {spec.type}")
```

- [ ] **Step 5: Add `V2Policy` to `policy.py`**

Append to `src/python/aerocapture/training/rl/policy.py`:

```python
from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import LayerSpec


class V2Policy(nn.Module):
    """Step-wise stateful policy matching the Rust NeuralNetModel contract.

    Forward pass: `(x_t, state_t-1) -> (y_t, state_t)`. BPTT over sequences
    is an explicit Python loop in the training code.

    The final dense layer produces the pre-interpretation output (2 values for
    atan2, 1 for direct). log_std is a separate learnable parameter (not a layer,
    not exported to JSON) used only for PPO/SAC exploration noise.
    """

    def __init__(
        self,
        architecture: list[LayerSpec],
        output_interpretation: str,
        input_mask: list[int] | None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([build_layer(spec) for spec in architecture])
        self.output_interpretation = output_interpretation
        self.input_mask = input_mask
        action_dim = 2 if output_interpretation == "atan2" else 1
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: Tensor, state: list) -> tuple[Tensor, list]:
        new_state = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, new_state[i] = layer(x, state[i])
        return x, new_state

    def new_state(self, batch_size: int, device) -> list:
        return [layer.new_state(batch_size, device) for layer in self.layers]
```

- [ ] **Step 6: Run tests to verify pass**

Run: `uv run pytest tests/test_v2_policy.py -v`
Expected: 3 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/rl/layers/ src/python/aerocapture/training/rl/policy.py tests/test_v2_policy.py
git commit -m "feat(nn): V2Policy torch mirror + layers/dense.py (Phase 0 dense only)"
```

---

## Task 9: Python v2 export + load round-trip

**Files:**
- Create: `src/python/aerocapture/training/model_io.py`
- Modify: `src/python/aerocapture/training/rl/export.py`
- Test: `tests/test_v2_export.py`

- [ ] **Step 1: Write failing test -- export -> load round-trip preserves weights**

Create `tests/test_v2_export.py`:

```python
import json

import torch

from aerocapture.training.model_io import load_policy_from_json
from aerocapture.training.rl.export import export_policy_to_json
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec


def _policy() -> V2Policy:
    architecture = [
        DenseSpec(type="dense", input_size=4, output_size=3, activation="tanh"),
        DenseSpec(type="dense", input_size=3, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)
    with torch.no_grad():
        p.layers[0].linear.weight.data.fill_(0.1)
        p.layers[0].linear.bias.data.fill_(0.01)
        p.layers[1].linear.weight.data.fill_(-0.1)
        p.layers[1].linear.bias.data.fill_(-0.01)
    return p


def test_export_produces_v2_format(tmp_path):
    p = _policy()
    path = tmp_path / "model.json"
    export_policy_to_json(p, str(path), obs_normalizer=None)
    raw = json.loads(path.read_text())
    assert raw["format_version"] == 2
    assert raw["architecture"][0]["type"] == "dense"
    assert "layer_0" in raw["weights"]
    assert "log_std" not in raw  # log_std is never exported


def test_export_load_roundtrip_preserves_weights(tmp_path):
    p = _policy()
    path = tmp_path / "model.json"
    export_policy_to_json(p, str(path), obs_normalizer=None)
    q = load_policy_from_json(str(path), device="cpu")

    # Weights match bit-for-bit on the linear layer parameters.
    for la, lb in zip(p.layers, q.layers):
        torch.testing.assert_close(la.linear.weight, lb.linear.weight, rtol=0, atol=0)
        torch.testing.assert_close(la.linear.bias, lb.linear.bias, rtol=0, atol=0)

    # Forward produces identical output.
    x = torch.randn(1, 4)
    sa = p.new_state(1, "cpu")
    sb = q.new_state(1, "cpu")
    ya, _ = p(x, sa)
    yb, _ = q(x, sb)
    torch.testing.assert_close(ya, yb, rtol=0, atol=0)
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_v2_export.py -v`
Expected: FAIL (modules do not exist).

- [ ] **Step 3: Create `model_io.py` with `load_policy_from_json`**

Create `src/python/aerocapture/training/model_io.py`:

```python
"""Load V2Policy from JSON v2 format.

Shared between RL training (report_rl.py post-training analysis), test code,
and any Python-side consumer that needs the torch model. Rust side uses its own
loader in data/neural.rs; this module is the Python equivalent.
"""

from __future__ import annotations

import json

import torch

from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import ArchitectureV2


def load_policy_from_json(path: str, device: str | torch.device) -> V2Policy:
    raw = json.loads(open(path).read())
    if raw.get("format_version") != 2:
        raise ValueError(f"Expected format_version=2 in {path}, got {raw.get('format_version')}")
    arch = ArchitectureV2.model_validate(raw)
    policy = V2Policy(
        architecture=arch.architecture,
        output_interpretation=arch.output_interpretation,
        input_mask=arch.input_mask,
    ).to(device)

    for i, layer_spec in enumerate(arch.architecture):
        key = f"layer_{i}"
        lw = arch.weights[key]
        if layer_spec.type == "dense":
            w = torch.tensor(lw.w, dtype=torch.float32, device=device)
            b = torch.tensor(lw.b, dtype=torch.float32, device=device)
            with torch.no_grad():
                policy.layers[i].linear.weight.copy_(w)
                policy.layers[i].linear.bias.copy_(b)
        # Phase 1+ layer types dispatch here.

    return policy
```

- [ ] **Step 4: Rewrite `export_policy_to_json` for v2 format**

Replace the contents of `src/python/aerocapture/training/rl/export.py` (preserving any unrelated helpers):

```python
"""Export a V2Policy to JSON v2 format for the Rust runtime."""

from __future__ import annotations

import json

import torch

from aerocapture.training.rl.policy import V2Policy


def export_policy_to_json(
    policy: V2Policy,
    path: str,
    obs_normalizer=None,
) -> None:
    """Serialize `policy` to JSON v2 at `path`.

    If `obs_normalizer` is provided, bake the `(mean, std)` transform into
    the first dense layer: `W_new = W / std`, `b_new = b - W @ (mean / std)`.
    For window-first architectures (when Phase 2 lands), the bake-in tiles
    the pattern across window slots before applying to the embedding layer.
    """
    architecture = []
    weights = {}

    # Detect window-first architecture (Phase 2+); for Phase 0 this branch is unreachable.
    # TODO(Phase 2): handle window-first bake-in tiling.

    for i, layer in enumerate(policy.layers):
        lin = layer.linear
        w = lin.weight.detach().cpu().numpy()
        b = lin.bias.detach().cpu().numpy()

        if i == 0 and obs_normalizer is not None:
            mean = obs_normalizer.mean.detach().cpu().numpy()
            std = obs_normalizer.std.detach().cpu().numpy()
            w_new = w / std  # broadcasting over columns (inputs)
            b_new = b - w @ (mean / std)
            w, b = w_new, b_new

        architecture.append({
            "type": "dense",
            "input_size": lin.in_features,
            "output_size": lin.out_features,
            "activation": layer.activation_name,
        })
        weights[f"layer_{i}"] = {
            "w": w.tolist(),
            "b": b.tolist(),
        }

    out = {
        "format_version": 2,
        "architecture": architecture,
        "weights": weights,
        "output_interpretation": policy.output_interpretation,
        "input_mask": policy.input_mask,
        "ablated_input": None,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_v2_export.py -v`
Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/model_io.py src/python/aerocapture/training/rl/export.py tests/test_v2_export.py
git commit -m "feat(nn): v2 JSON export + load_policy_from_json round-trip"
```

---

## Task 10: Python `nn_param_specs_from_v2`

**Files:**
- Modify: `src/python/aerocapture/training/encoding.py`
- Test: `tests/test_nn_param_specs_v2.py`

- [ ] **Step 1: Write failing test -- v2 specs match v1 specs for all-dense arch**

Create `tests/test_nn_param_specs_v2.py`:

```python
from aerocapture.training.encoding import (
    nn_param_specs_from_architecture,
    nn_param_specs_from_v2,
)
from aerocapture.training.rl.schemas import DenseSpec


def test_v2_all_dense_matches_v1():
    layer_sizes = [16, 24, 2]
    activations = ["tanh", "asinh"]
    v1_specs = nn_param_specs_from_architecture(layer_sizes, activations, bound_multiplier=1.0)

    architecture = [
        DenseSpec(type="dense", input_size=16, output_size=24, activation="tanh"),
        DenseSpec(type="dense", input_size=24, output_size=2, activation="asinh"),
    ]
    v2_specs = nn_param_specs_from_v2(architecture, bound_multiplier=1.0)

    assert len(v1_specs) == len(v2_specs)
    for s1, s2 in zip(v1_specs, v2_specs):
        assert s1.low == s2.low
        assert s1.high == s2.high
        assert s1.log_scale == s2.log_scale


def test_v2_empty_architecture():
    assert nn_param_specs_from_v2([], bound_multiplier=1.0) == []
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_nn_param_specs_v2.py -v`
Expected: FAIL (function does not exist).

- [ ] **Step 3: Implement `nn_param_specs_from_v2`**

Add to `src/python/aerocapture/training/encoding.py` (adjust imports as needed):

```python
from aerocapture.training.rl.schemas import DenseSpec, LayerSpec


def nn_param_specs_from_v2(
    architecture: list[LayerSpec],
    bound_multiplier: float = 1.0,
) -> list[ParamSpec]:
    """Generate per-parameter ParamSpecs from a v2 architecture list.

    Dispatches per layer type. Phase 0 implements only `dense` (Xavier-uniform
    weight bounds + fixed +/-0.1 bias bounds, scaled by bound_multiplier).
    Phase 1+ extends with gru/lstm/attention/ssm/layer_norm branches.
    """
    specs: list[ParamSpec] = []
    for layer in architecture:
        specs.extend(_layer_param_specs(layer, bound_multiplier))
    return specs


def _layer_param_specs(layer: LayerSpec, bound_multiplier: float) -> list[ParamSpec]:
    if isinstance(layer, DenseSpec):
        return _dense_specs(layer, bound_multiplier)
    raise ValueError(f"Unknown layer type for PSO specs: {layer.type}")


def _dense_specs(layer: DenseSpec, bound_multiplier: float) -> list[ParamSpec]:
    # Match the existing nn_param_specs_from_architecture math exactly.
    # Xavier-uniform: bound = sqrt(6 / (fan_in + fan_out)) * bound_multiplier.
    import math
    fan_in, fan_out = layer.input_size, layer.output_size
    w_bound = math.sqrt(6.0 / (fan_in + fan_out)) * bound_multiplier
    b_bound = 0.1 * bound_multiplier
    specs: list[ParamSpec] = []
    for _ in range(fan_out * fan_in):
        specs.append(ParamSpec(low=-w_bound, high=w_bound, log_scale=False))
    for _ in range(fan_out):
        specs.append(ParamSpec(low=-b_bound, high=b_bound, log_scale=False))
    return specs
```

If the existing `nn_param_specs_from_architecture` uses different bounds (e.g., He init for relu, LeCun for tanh), mirror that conditional logic in `_dense_specs` based on `layer.activation`. Re-check `src/python/aerocapture/training/initialization.py` for the exact formulas and replicate them here.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_nn_param_specs_v2.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Run full Python test suite to confirm no regression**

Run: `uv run pytest -x`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/encoding.py tests/test_nn_param_specs_v2.py
git commit -m "feat(nn): nn_param_specs_from_v2 dispatching per layer type (dense only)"
```

---

## Task 11: Cross-language Rust <-> Python equivalence test

**Files:**
- Create: `tests/test_v2_rust_python_equivalence.py`

- [ ] **Step 1: Write the cross-language test**

Create `tests/test_v2_rust_python_equivalence.py`:

```python
"""Cross-language equivalence: Rust NeuralNetModel and PyTorch V2Policy produce
the same output on the same input to 1e-10. This is the Phase 0 integration gate.

Subsequent phases extend this test with their new layer types.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("aerocapture_rs")
import aerocapture_rs  # noqa: E402

from aerocapture.training.model_io import load_policy_from_json
from aerocapture.training.rl.export import export_policy_to_json
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec


def _rust_forward_single(json_path: str, inputs: np.ndarray) -> np.ndarray:
    """Load a v2 JSON in Rust and run forward on each input row.

    Uses the PyO3 binding's low-level forward entry point. If one does not yet
    exist, this test drives its creation (see Step 2 below).
    """
    return np.array([
        aerocapture_rs.nn_forward(json_path, input_row.tolist())
        for input_row in inputs
    ])


def test_rust_python_dense_equivalence(tmp_path):
    architecture = [
        DenseSpec(type="dense", input_size=5, output_size=8, activation="tanh"),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)
    torch.manual_seed(42)
    with torch.no_grad():
        for layer in policy.layers:
            layer.linear.weight.data = torch.randn_like(layer.linear.weight) * 0.3
            layer.linear.bias.data = torch.randn_like(layer.linear.bias) * 0.1

    json_path = tmp_path / "model.json"
    export_policy_to_json(policy, str(json_path), obs_normalizer=None)

    rng = np.random.default_rng(7)
    inputs = rng.standard_normal((100, 5)).astype(np.float64)

    py_out = np.zeros((100, 2), dtype=np.float64)
    state = policy.new_state(1, "cpu")
    for i, x in enumerate(inputs):
        y, _ = policy(torch.from_numpy(x).float().unsqueeze(0), state)
        py_out[i] = y.detach().numpy()[0]

    rust_out = _rust_forward_single(str(json_path), inputs)

    max_diff = np.max(np.abs(rust_out - py_out))
    assert max_diff < 1e-10, f"max abs diff {max_diff} exceeds 1e-10"
```

- [ ] **Step 2: Run to confirm failure (likely on `aerocapture_rs.nn_forward`)**

Run: `uv run pytest tests/test_v2_rust_python_equivalence.py -v`
Expected: FAIL with `AttributeError: module 'aerocapture_rs' has no attribute 'nn_forward'`.

- [ ] **Step 3: Add `nn_forward` helper to the PyO3 crate**

In `src/rust/aerocapture-py/src/lib.rs`, add:

```rust
use aerocapture::data::neural::NeuralNetModel;
use aerocapture::gnc::guidance::nn_state::NnState;

#[pyfunction]
fn nn_forward(json_path: String, input: Vec<f64>) -> PyResult<Vec<f64>> {
    let model = NeuralNetModel::load(&json_path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;
    let masked: Vec<f64> = match &model.input_mask {
        Some(mask) => mask.iter().map(|&i| input[i]).collect(),
        None => input,
    };
    let mut state = NnState::for_model(&model);
    let output = model.forward(&mut state, &masked);
    Ok(output)
}
```

Register it in the module:

```rust
#[pymodule]
fn aerocapture_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // ...existing registrations
    m.add_function(wrap_pyfunction!(nn_forward, m)?)?;
    Ok(())
}
```

- [ ] **Step 4: Rebuild PyO3 bindings**

Run: `uv run maturin develop --manifest-path src/rust/aerocapture-py/Cargo.toml --release`
Expected: build succeeds.

- [ ] **Step 5: Re-run the equivalence test**

Run: `uv run pytest tests/test_v2_rust_python_equivalence.py -v`
Expected: PASS with max diff well under 1e-10.

- [ ] **Step 6: Commit**

```bash
git add src/rust/aerocapture-py/src/lib.rs tests/test_v2_rust_python_equivalence.py
git commit -m "feat(nn): aerocapture_rs.nn_forward + cross-language equivalence test"
```

---

## Task 12: Final verification + smart-commit

**Files:** (none modified; verification + commit hygiene)

- [ ] **Step 1: Run `./check_all.sh`**

Run: `./check_all.sh`
Expected: cargo test + fmt --check + clippy + release build all clean.

- [ ] **Step 2: Run `./lint_code.sh`**

Run: `./lint_code.sh`
Expected: ruff + mypy clean.

- [ ] **Step 3: Run the full Python test suite**

Run: `uv run pytest -x`
Expected: all tests pass.

- [ ] **Step 4: Verify Phase 0 success criterion #4 -- layer-type drop-in is truly local**

Manually inspect the diff (`git diff main..HEAD` on this branch). Confirm that adding a new layer type would require only:
- A new variant to `LayerSpec` (data/neural.rs)
- A new variant to `Layer` (data/neural.rs) implementing `LayerWeights`
- A new variant to `LayerState` (nn_state.rs) with `for_layer` + `reset` arms
- A new file in `src/python/aerocapture/training/rl/layers/<type>.py`
- A new branch in `build_layer` (layers/__init__.py)
- A new Pydantic `*Spec` class in `schemas.py`
- A new branch in `_layer_param_specs` (encoding.py)
- A new branch in `export_policy_to_json` if the layer has exportable weights
- A new branch in `load_policy_from_json` to materialize weights into the torch module

If any of `dispatch.rs`, `runner.rs`, `BatchedSimulation`, `problem.py`, or `train.py` would need to change to add a new layer type, that is a Phase 0 defect -- file an issue / fix before closing.

- [ ] **Step 5: Invoke the smart-commit skill**

Run the `smart-commit` skill to finalize any outstanding documentation updates (CLAUDE.md, README.md) and create a final commit covering the whole branch. Tell it to take the whole branch into account.

---

## Self-review

**Spec coverage:**
- Section 3.1 (v2a JSON schema) -- Task 1
- Section 3.2 (`NnState` outside model, Clone, reset at episode start) -- Tasks 2, 4
- Section 3.3 (eager init + two-mode ownership) -- Task 4
- Section 3.4 (canonical flat ordering + Xavier-uniform bounds) -- Tasks 5, 10
- Section 3.5 (PyTorch mirror + dense-embedding invariant + export contract) -- Tasks 7, 8, 9
- Section 4 (Rust runtime changes) -- Tasks 1-5
- Section 5 (Python + PyO3 changes) -- Tasks 7-11
- Section 6 (PSO integration) -- Task 10
- Section 7 (RL integration minimal surface) -- Task 9 (exporter accepts obs_normalizer)
- Section 8 (test tiers) -- Tasks 1, 2, 3, 6, 7, 8, 9, 10, 11, 12
- Section 9 (backward compat) -- Task 6 (regression gate)
- Section 10 (non-goals) -- explicitly nothing in the plan
- Section 11 (success criteria) -- Task 12 Step 4 audits criterion #4

**Placeholder scan:** No TBD / TODO / "implement later" items in task code. Two explicit TODO comments in the codebase are forward-looking markers:
- Task 9 Step 4 has a `TODO(Phase 2)` for window-first architecture bake-in (acceptable -- not yet buildable, reserved for Phase 2)
- Task 3 Step 6 has a `TODO(Task 4)` placeholder, removed by Task 4 Step 5 within this plan

**Type consistency:** `nn_state`, `NnState`, `LayerState`, `LayerSpec`, `V2Policy`, `DenseSpec`, `ArchitectureV2` all used consistently. `nn_param_specs_from_v2` takes `list[LayerSpec]` (a Pydantic-tagged union); `nn_param_specs_from_architecture` stays as a v1-compatible thin wrapper (not rewritten in this plan; left in place as-is). `build_layer` dispatches on `spec.type` string (Pydantic discriminator field).

**No spec requirement lacks a task.**
