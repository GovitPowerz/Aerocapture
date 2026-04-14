# NN Input Expansion and Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand NN guidance from 16 to 23 candidate inputs (ref trajectory + exit signals), add a TOML-configurable input mask for pruning, and build an ablation analysis tool to rank input importance.

**Architecture:** The Rust NN guidance always computes a 23-element input vector. A configurable mask (stored in JSON model file) selects which inputs reach the network. An optional `ablated_input` field zeros out one input for importance analysis. The Python ablation script drives the analysis via PyO3.

**Tech Stack:** Rust (nalgebra, serde_json), Python (aerocapture_rs PyO3, numpy, matplotlib)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/rust/src/data/neural.rs` | Add `input_mask`, `ablated_input` to `NeuralNetModel`; JSON serialization; validation |
| Modify | `src/rust/src/gnc/guidance/neural.rs` | Expand to 23 inputs, apply mask + ablation, new function signature |
| Modify | `src/rust/src/gnc/guidance/dispatch.rs` | Pass `data` + `reference_velocity` to NN; speculative exit guidance |
| Create | `src/python/aerocapture/training/ablation.py` | Ablation analysis CLI tool |
| Create | `src/python/aerocapture/training/charts_ablation.py` | Ablation bar chart SVG |
| Modify | `configs/training/msr_aller_nn_train_consolidated.toml` | Add `input_mask` |
| Test | `src/rust/src/gnc/guidance/neural.rs` (inline tests) | 23-input vector, mask, ablation, backward compat |
| Test | `tests/test_ablation.py` | Ablation script output structure |

---

### Task 1: Add `input_mask` and `ablated_input` to `NeuralNetModel`

**Files:**
- Modify: `src/rust/src/data/neural.rs`

- [ ] **Step 1: Write failing tests for input_mask validation**

Add these tests in the existing `mod tests` block at the bottom of `src/rust/src/data/neural.rs`:

```rust
#[test]
fn input_mask_stored_on_model() {
    let model = NeuralNetModel {
        layer_sizes: vec![3, 2],
        layers: vec![Layer {
            w: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            b: vec![0.0, 0.0],
            activation: Activation::Linear,
        }],
        output_interpretation: "atan2".to_string(),
        input_mask: Some(vec![0, 5, 14]),
        ablated_input: None,
    };
    assert_eq!(model.input_mask, Some(vec![0, 5, 14]));
}

#[test]
fn input_mask_none_by_default() {
    let model = NeuralNetModel {
        layer_sizes: vec![16, 2],
        layers: vec![Layer {
            w: vec![vec![0.0; 16], vec![0.0; 16]],
            b: vec![0.0, 0.0],
            activation: Activation::Linear,
        }],
        output_interpretation: "atan2".to_string(),
        input_mask: None,
        ablated_input: None,
    };
    assert!(model.input_mask.is_none());
}

#[test]
fn validate_mask_length_mismatch() {
    let result = NeuralNetModel::validate_mask(&Some(vec![0, 1]), 3);
    assert!(result.is_err());
    assert!(result.unwrap_err().0.contains("length"));
}

#[test]
fn validate_mask_out_of_range() {
    let result = NeuralNetModel::validate_mask(&Some(vec![0, 25]), 2);
    assert!(result.is_err());
    assert!(result.unwrap_err().0.contains("out of range"));
}

#[test]
fn validate_mask_duplicates() {
    let result = NeuralNetModel::validate_mask(&Some(vec![0, 0]), 2);
    assert!(result.is_err());
    assert!(result.unwrap_err().0.contains("duplicate"));
}

#[test]
fn validate_mask_valid() {
    let result = NeuralNetModel::validate_mask(&Some(vec![0, 5, 14, 16, 20]), 5);
    assert!(result.is_ok());
}

#[test]
fn validate_mask_none_is_ok() {
    let result = NeuralNetModel::validate_mask(&None, 16);
    assert!(result.is_ok());
}

#[test]
fn validate_ablated_input_out_of_range() {
    let result = NeuralNetModel::validate_ablated_input(&Some(23));
    assert!(result.is_err());
}

#[test]
fn validate_ablated_input_valid() {
    let result = NeuralNetModel::validate_ablated_input(&Some(22));
    assert!(result.is_ok());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test --lib data::neural::tests -- --nocapture 2>&1 | tail -20`
Expected: compilation errors (fields/methods don't exist yet)

- [ ] **Step 3: Add fields and validation to `NeuralNetModel`**

In `src/rust/src/data/neural.rs`, add fields to the struct:

```rust
pub struct NeuralNetModel {
    pub layer_sizes: Vec<usize>,
    pub layers: Vec<Layer>,
    pub output_interpretation: String,
    /// Input mask: indices into the full 23-element input vector.
    /// When None, defaults to [0..16] for backward compatibility.
    pub input_mask: Option<Vec<usize>>,
    /// Ablation: if set, force this index to 0.0 in the full input vector.
    /// Used only during ablation analysis.
    pub ablated_input: Option<usize>,
}
```

Add a constant for the full input vector size:

```rust
/// Total number of candidate NN inputs (16 existing + 7 new).
pub const NN_FULL_INPUT_SIZE: usize = 23;
```

Add validation methods:

```rust
/// Validate input mask configuration.
pub fn validate_mask(mask: &Option<Vec<usize>>, expected_len: usize) -> Result<(), DataError> {
    if let Some(ref m) = mask {
        if m.len() != expected_len {
            return Err(DataError(format!(
                "input_mask length ({}) does not match layer_sizes[0] ({})",
                m.len(),
                expected_len
            )));
        }
        for &idx in m {
            if idx >= NN_FULL_INPUT_SIZE {
                return Err(DataError(format!(
                    "input_mask index {} out of range [0, {})",
                    idx, NN_FULL_INPUT_SIZE
                )));
            }
        }
        let mut seen = std::collections::HashSet::new();
        for &idx in m {
            if !seen.insert(idx) {
                return Err(DataError(format!(
                    "input_mask contains duplicate index {}",
                    idx
                )));
            }
        }
    }
    Ok(())
}

/// Validate ablated_input configuration.
pub fn validate_ablated_input(ablated: &Option<usize>) -> Result<(), DataError> {
    if let Some(idx) = ablated {
        if *idx >= NN_FULL_INPUT_SIZE {
            return Err(DataError(format!(
                "ablated_input index {} out of range [0, {})",
                idx, NN_FULL_INPUT_SIZE
            )));
        }
    }
    Ok(())
}
```

Update `from_json` to parse `input_mask` and `ablated_input` from the JSON:

Add optional fields to `NnJsonFile`:

```rust
struct NnJsonFile {
    format_version: u32,
    architecture: NnArchitecture,
    weights: std::collections::BTreeMap<String, NnLayerWeights>,
    output_interpretation: String,
    #[serde(default)]
    input_mask: Option<Vec<usize>>,
    #[serde(default)]
    ablated_input: Option<usize>,
}
```

In `from_json`, after building the model, validate and attach:

```rust
Self::validate_mask(&file.input_mask, file.architecture.layers[0])?;
Self::validate_ablated_input(&file.ablated_input)?;

Ok(NeuralNetModel {
    layer_sizes: file.architecture.layers,
    layers,
    output_interpretation: file.output_interpretation,
    input_mask: file.input_mask,
    ablated_input: file.ablated_input,
})
```

Update `save_json` to include the new fields in `NnJsonFile`:

```rust
let file = NnJsonFile {
    format_version: 1,
    architecture: NnArchitecture {
        layers: self.layer_sizes.clone(),
        activations,
    },
    weights,
    output_interpretation: self.output_interpretation.clone(),
    input_mask: self.input_mask.clone(),
    ablated_input: self.ablated_input,
};
```

Update `from_flat_weights` to accept optional mask:

```rust
Ok(NeuralNetModel {
    layer_sizes: layer_sizes.to_vec(),
    layers,
    output_interpretation: "atan2".to_string(),
    input_mask: None,
    ablated_input: None,
})
```

Fix all other places that construct `NeuralNetModel` (test helpers across the codebase) to add `input_mask: None, ablated_input: None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/rust && cargo test --lib data::neural 2>&1 | tail -20`
Expected: all tests pass including the new validation tests

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): add input_mask and ablated_input to NeuralNetModel with validation"
```

---

### Task 2: Expand `nn_bank_angle` to 23 inputs with mask application

**Files:**
- Modify: `src/rust/src/gnc/guidance/neural.rs`

- [ ] **Step 1: Write failing tests for 23-input vector and masking**

Add to the existing test module in `neural.rs`:

```rust
#[test]
fn full_23_input_vector_is_finite() {
    // Build a 23->2 network with all-zero weights and explicit full mask
    let nn = NeuralNetModel {
        layer_sizes: vec![23, 2],
        layers: vec![Layer {
            w: vec![vec![0.0; 23], vec![0.0; 23]],
            b: vec![1.0, 1.0],
            activation: Activation::Linear,
        }],
        output_interpretation: "atan2".to_string(),
        input_mask: Some((0..23).collect()),  // explicit full mask
        ablated_input: None,
    };
    let nav = test_nav();
    let planet = PlanetConfig::mars();
    let data = test_sim_data_with_ref_traj();

    let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);
    assert!(bank.is_finite(), "bank angle must be finite, got: {}", bank);
}

#[test]
fn mask_selects_correct_inputs() {
    // Network expects 3 inputs; mask selects indices [0, 8, 15]
    let nn = NeuralNetModel {
        layer_sizes: vec![3, 2],
        layers: vec![Layer {
            w: vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0]],
            b: vec![0.0, 0.0],
            activation: Activation::Linear,
        }],
        output_interpretation: "atan2".to_string(),
        input_mask: Some(vec![0, 8, 15]),
        ablated_input: None,
    };
    let nav = test_nav();
    let planet = PlanetConfig::mars();
    let data = test_sim_data_with_ref_traj();

    let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);
    assert!(bank.is_finite());
}

#[test]
fn ablation_zeros_target_input() {
    // Two networks: one with ablation on index 0, one without.
    // The ablated one should produce a different result (unless index 0 happens to be 0).
    let mut nn_normal = NeuralNetModel {
        layer_sizes: vec![23, 2],
        layers: vec![Layer {
            w: vec![
                (0..23).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect(),
                vec![0.0; 23],
            ],
            b: vec![0.0, 0.0],
            activation: Activation::Linear,
        }],
        output_interpretation: "atan2".to_string(),
        input_mask: None,
        ablated_input: None,
    };
    let nn_ablated = NeuralNetModel {
        ablated_input: Some(0),
        ..nn_normal.clone()
    };

    let nav = test_nav();
    let planet = PlanetConfig::mars();
    let data = test_sim_data_with_ref_traj();

    let bank_normal = nn_bank_angle(&nav, &nn_normal, &data, &planet, 50.0_f64.to_radians(), 0.0);
    let bank_ablated = nn_bank_angle(&nav, &nn_ablated, &data, &planet, 50.0_f64.to_radians(), 0.0);

    // Ablating index 0 (eccentricity excess) should change the output since
    // the first weight row depends solely on input 0.
    assert_ne!(bank_normal, bank_ablated, "ablation should change output");
}

#[test]
fn backward_compat_16_input_mask() {
    // A model with input_mask = [0..16] and 16-input architecture
    // should produce the same result as the legacy 16-input code path.
    let legacy_mask: Vec<usize> = (0..16).collect();
    let nn = NeuralNetModel {
        layer_sizes: vec![16, 2],
        layers: vec![Layer {
            w: vec![
                vec![0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.1, -0.1,
                     0.02, -0.03, 0.04, -0.01, 0.03, -0.02, 0.01, -0.04],
                vec![-0.1, 0.1, -0.05, 0.05, 0.15, -0.15, 0.05, -0.05,
                     -0.02, 0.03, -0.01, 0.04, -0.03, 0.02, -0.04, 0.01],
            ],
            b: vec![0.3, -0.2],
            activation: Activation::Tanh,
        }],
        output_interpretation: "atan2".to_string(),
        input_mask: Some(legacy_mask),
        ablated_input: None,
    };
    let nav = test_nav();
    let planet = PlanetConfig::mars();
    let data = test_sim_data_with_ref_traj();

    let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 0.0);
    assert!(bank.is_finite());
    // atan2 range
    assert!(bank > -std::f64::consts::PI - 1e-10 && bank <= std::f64::consts::PI + 1e-10);
}

#[test]
fn bounce_gated_inputs_zero_pre_bounce() {
    // With bounce_flag = 0, exit-related inputs (20, 21, 22) should be exactly 0.
    // Use a network with weights only on those indices to verify.
    let nn = NeuralNetModel {
        layer_sizes: vec![23, 2],
        layers: vec![Layer {
            w: vec![
                {
                    let mut w = vec![0.0; 23];
                    w[20] = 1.0; // exit_bank_angle
                    w[21] = 1.0; // density_exit
                    w[22] = 1.0; // ref_velocity_latched
                    w
                },
                vec![0.0; 23],
            ],
            b: vec![0.0, 1.0],
            activation: Activation::Linear,
        }],
        output_interpretation: "atan2".to_string(),
        input_mask: None,
        ablated_input: None,
    };

    let mut nav = test_nav();
    nav.bounce_flag = 0; // pre-bounce
    let planet = PlanetConfig::mars();
    let data = test_sim_data_with_ref_traj();

    let bank = nn_bank_angle(&nav, &nn, &data, &planet, 50.0_f64.to_radians(), 100.0);
    // output[0] = sum of exit inputs * weights = 0 (all gated)
    // output[1] = 1.0 (bias)
    // atan2(0, 1) = 0
    assert!((bank - 0.0).abs() < 1e-10, "exit inputs should be zero pre-bounce, got bank={}", bank);
}
```

Also add a helper `test_sim_data_with_ref_traj()` that returns a minimal `SimData` with a non-empty reference trajectory:

```rust
fn test_sim_data_with_ref_traj() -> SimData {
    use crate::data::guidance_params::{GuidanceParams, ReferenceTrajectory};
    // ... (same as test_sim_data from ftc.rs tests, but with a small ref_trajectory)
    let mut data = /* base SimData */;
    data.guidance.ref_trajectory = ReferenceTrajectory {
        n_points: 3,
        energy: vec![-8.0e6, -5.0e6, -2.0e6],
        pressure: vec![500.0, 2000.0, 500.0],
        radial_vel: vec![-200.0, 0.0, 100.0],
        altitude_rate: vec![-200.0, 0.0, 100.0],
        inclination: vec![0.87, 0.87, 0.87],
        time: vec![0.0, 300.0, 600.0],
        cos_bank: vec![0.4, 0.6, 0.8],
    };
    data
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test --lib gnc::guidance::neural 2>&1 | tail -20`
Expected: compilation errors (signature mismatch, missing SimData param)

- [ ] **Step 3: Implement the 23-input vector with mask and ablation**

Rewrite `nn_bank_angle` in `src/rust/src/gnc/guidance/neural.rs`:

```rust
use crate::data::SimData;
use crate::data::neural::NN_FULL_INPUT_SIZE;
use crate::gnc::guidance::exit;
use crate::gnc::navigation::coordinates::total_energy;

pub fn nn_bank_angle(
    nav: &NavigationOutput,
    nn: &NeuralNetModel,
    data: &SimData,
    planet: &PlanetConfig,
    target_inclination: f64,
    ref_velocity_latched: f64,
) -> f64 {
    let mu = planet.mu;

    // Radial velocity: V * sin(gamma)
    let velocity_radial = nav.velocity_estimated[0] * nav.velocity_estimated[1].sin();

    // Orbital elements
    let orbit = elements::from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    // Acceleration magnitude
    let accel_mag = (nav.acceleration_estimated[0].powi(2)
        + nav.acceleration_estimated[1].powi(2))
        .sqrt();

    // Altitude in km
    let altitude_km = (nav.position_estimated[0] - planet.equatorial_radius) / 1e3;

    // Energy for ref trajectory interpolation
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    // Reference trajectory values
    let ref_traj = &data.guidance.ref_trajectory;
    let cos_bank_nominal = ref_traj.interpolate(energy, &ref_traj.cos_bank);
    let pdyn_nominal = ref_traj.interpolate(energy, &ref_traj.pressure);
    let hdot_nominal = ref_traj.interpolate(energy, &ref_traj.radial_vel);

    // Current dynamic pressure
    let pdyn_current = 0.5 * nav.density_guidance
        * nav.velocity_estimated[0] * nav.velocity_estimated[0];
    let pdyn_error = pdyn_current - pdyn_nominal;

    // Bounce gating: exit-related inputs are exactly zero pre-bounce
    let bf = nav.bounce_flag as f64; // 0.0 or 1.0

    // Speculative exit guidance (stateless, cheap)
    let exit_bank = exit::exit_guidance(nav, data, planet, ref_velocity_latched);

    // Build full 23-element input vector
    let mut full_input = [0.0_f64; NN_FULL_INPUT_SIZE];

    // Existing 16 inputs (indices 0-15)
    full_input[0] = orbit.eccentricity - 1.0;
    full_input[1] = (orbit.inclination - target_inclination).to_degrees() * 3.0 / 5.0;
    full_input[2] = 2.0 * (velocity_radial / 1e3 + 1.2) / 1.5 - 1.0;
    full_input[3] = -mu / (2.0 * orbit.semi_major_axis) / 6e6;
    full_input[4] = (nav.velocity_estimated[0] / 3e3 - 1.5) * 2.0;
    full_input[5] = accel_mag / 20.0 - 1.0;
    full_input[6] = nav.heat_flux_fraction * 2.0 - 1.0;
    full_input[7] = nav.heat_load_fraction * 2.0 - 1.0;
    full_input[8] = (altitude_km - 65.0) / 65.0;
    full_input[9] = nav.velocity_estimated[1] / 0.3;
    full_input[10] = nav.position_estimated[2] / std::f64::consts::FRAC_PI_2;
    full_input[11] = nav.acceleration_estimated[0] / 50.0 - 1.0;
    full_input[12] = nav.acceleration_estimated[1] / 10.0;
    full_input[13] = nav.orbital_errors[0] / 5e5;
    full_input[14] = orbit.apoapsis_alt.clamp(-10e6, 10e6) / 1e6 - 1.0;
    full_input[15] = bf * 2.0 - 1.0;

    // New ref trajectory inputs (indices 16-19)
    full_input[16] = cos_bank_nominal;
    full_input[17] = pdyn_nominal / 2e3 - 1.0;
    full_input[18] = hdot_nominal / 500.0;
    full_input[19] = pdyn_error / 2e3;

    // New exit-related inputs (indices 20-22), gated by bounce_flag
    full_input[20] = (exit_bank / std::f64::consts::PI * 2.0 - 1.0) * bf;
    full_input[21] = ((nav.density_exit.max(1e-12).log10() + 7.0) / 5.0) * bf;
    full_input[22] = (ref_velocity_latched / 500.0) * bf;

    // Apply ablation (zero out one input for importance analysis)
    if let Some(idx) = nn.ablated_input {
        full_input[idx] = 0.0;
    }

    // Apply input mask
    let masked: Vec<f64> = match &nn.input_mask {
        Some(mask) => mask.iter().map(|&i| full_input[i]).collect(),
        None => full_input[..16].to_vec(), // backward compat: first 16 inputs
    };

    let output = nn.forward(&masked);
    output[0].atan2(output[1])
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/rust && cargo test --lib gnc::guidance::neural 2>&1 | tail -30`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/gnc/guidance/neural.rs
git commit -m "feat(nn): expand to 23-input vector with mask and ablation support"
```

---

### Task 3: Update `dispatch.rs` to pass new args to NN

**Files:**
- Modify: `src/rust/src/gnc/guidance/dispatch.rs`

- [ ] **Step 1: Update the NN call site in dispatch**

In `dispatch.rs`, find the NN branch (around line 168-171):

```rust
// Current code:
GuidanceType::NeuralNetwork => {
    let nn = data.neural_net.as_ref().expect("NN params not loaded");
    neural::nn_bank_angle(nav, nn, planet, data.target_orbit.inclination)
}
```

Replace with:

```rust
GuidanceType::NeuralNetwork => {
    let nn = data.neural_net.as_ref().expect("NN params not loaded");
    neural::nn_bank_angle(
        nav,
        nn,
        data,
        planet,
        data.target_orbit.inclination,
        state.reference_velocity,
    )
}
```

- [ ] **Step 2: Run the full test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: all tests pass (including existing dispatch tests)

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/gnc/guidance/dispatch.rs
git commit -m "feat(nn): pass SimData and reference_velocity to nn_bank_angle"
```

---

### Task 4: Fix all `NeuralNetModel` construction sites across the codebase

**Files:**
- Modify: multiple test files and source files that construct `NeuralNetModel`

The new fields `input_mask` and `ablated_input` need to be added everywhere `NeuralNetModel` is constructed. Since these are not `Default`-derived (the struct doesn't implement Default), every construction site needs updating.

- [ ] **Step 1: Find all construction sites**

Run: `cd src/rust && grep -rn "NeuralNetModel {" src/ tests/ 2>/dev/null`

For each site, add `input_mask: None, ablated_input: None` to the struct literal.

- [ ] **Step 2: Run full test suite to verify**

Run: `cd src/rust && cargo test 2>&1 | tail -30`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add -A src/rust/
git commit -m "fix(nn): add input_mask/ablated_input fields to all NeuralNetModel construction sites"
```

---

### Task 5: Update Python `evaluate.py` JSON writing to include `input_mask`

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py`

The Python side writes NN model JSON files during training. These need to include `input_mask` so Rust can read it back.

- [ ] **Step 1: Update `write_nn_json` to accept and write `input_mask`**

In `src/python/aerocapture/training/evaluate.py`, modify `write_nn_json`:

```python
def write_nn_json(
    weights: npt.NDArray[np.float64],
    network: NetworkConfig,
    filepath: str | Path,
    input_mask: list[int] | None = None,
) -> None:
```

Add `input_mask` to the JSON data dict:

```python
data = {
    "format_version": 1,
    "architecture": {
        "layers": network.layer_sizes,
        "activations": network.activations,
    },
    "weights": layer_weights,
    "output_interpretation": "atan2",
}
if input_mask is not None:
    data["input_mask"] = input_mask
```

- [ ] **Step 2: Thread `input_mask` through the training pipeline**

In `src/python/aerocapture/training/config.py`, add to `NetworkConfig`:

```python
@dataclass
class NetworkConfig:
    layer_sizes: list[int] = field(default_factory=lambda: [16, 24, 2])
    activations: list[str] = field(default_factory=lambda: ["tanh", "asinh"])
    input_mask: list[int] | None = None
```

Update the `__post_init__` validation to also check mask:

```python
def __post_init__(self) -> None:
    n_layers = len(self.layer_sizes) - 1
    if len(self.activations) != n_layers:
        msg = f"activations length ({len(self.activations)}) must equal len(layer_sizes)-1 ({n_layers})"
        raise ValueError(msg)
    if self.input_mask is not None:
        if len(self.input_mask) != self.layer_sizes[0]:
            msg = f"input_mask length ({len(self.input_mask)}) must equal layer_sizes[0] ({self.layer_sizes[0]})"
            raise ValueError(msg)
```

In `train.py`, parse `input_mask` from TOML (near line 779-783):

```python
_net = _toml_data.get("network", {})
if "layer_sizes" in _net:
    cfg.network.layer_sizes = _net["layer_sizes"]
if "activations" in _net:
    cfg.network.activations = _net["activations"]
if "input_mask" in _net:
    cfg.network.input_mask = _net["input_mask"]
```

In the training loop where `write_nn_json` is called, pass `input_mask=cfg.network.input_mask`.

- [ ] **Step 3: Run Python tests**

Run: `uv run pytest tests/ -x -q 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py src/python/aerocapture/training/config.py src/python/aerocapture/training/train.py
git commit -m "feat(nn): thread input_mask through Python training pipeline and JSON output"
```

---

### Task 6: Update NN training TOML config

**Files:**
- Modify: `configs/training/msr_aller_nn_train_consolidated.toml`

- [ ] **Step 1: Add `input_mask` for full 23-input training**

Update the `[network]` section:

```toml
[network]
layer_sizes = [23, 8, 32, 8, 2]
activations = ["tanh", "tanh", "tanh", "asinh"]
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
```

- [ ] **Step 2: Commit**

```bash
git add configs/training/msr_aller_nn_train_consolidated.toml
git commit -m "feat(nn): update NN training config for 23-input architecture"
```

---

### Task 7: Build the ablation analysis script

**Files:**
- Create: `src/python/aerocapture/training/ablation.py`
- Create: `src/python/aerocapture/training/charts_ablation.py`

- [ ] **Step 1: Define input names constant**

In `ablation.py`:

```python
"""Ablation analysis for NN input importance ranking.

Trains a full-input network, then measures cost degradation when each
input is individually zeroed out. Ranks inputs by importance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NN_INPUT_NAMES: list[str] = [
    "eccentricity_excess",      # 0
    "inclination_error",        # 1
    "radial_velocity",          # 2
    "orbital_energy",           # 3
    "velocity",                 # 4
    "accel_magnitude",          # 5
    "heat_flux_fraction",       # 6
    "heat_load_fraction",       # 7
    "altitude",                 # 8
    "fpa",                      # 9
    "latitude",                 # 10
    "drag_accel",               # 11
    "lift_accel",               # 12
    "sma_error",                # 13
    "apoapsis_alt",             # 14
    "bounce_flag",              # 15
    "cos_bank_nominal",         # 16
    "pdyn_nominal",             # 17
    "hdot_nominal",             # 18
    "pdyn_error",               # 19
    "exit_bank_angle",          # 20
    "density_exit",             # 21
    "ref_velocity_latched",     # 22
]
```

- [ ] **Step 2: Implement the ablation runner**

```python
def run_ablation(
    toml_path: str,
    training_dir: str,
    n_sims: int = 1000,
    sim_timeout_secs: float | None = None,
) -> dict:
    """Run ablation analysis on a trained NN model.

    Returns dict with keys: baseline_cost, results (list of per-input dicts),
    ranked (sorted by delta descending).
    """
    import aerocapture_rs

    # Run baseline (no ablation)
    baseline = aerocapture_rs.run_mc(toml_path, sim_timeout_secs=sim_timeout_secs)
    baseline_costs = _compute_costs(baseline)
    baseline_mean = float(np.mean(baseline_costs))

    results = []
    for idx in range(len(NN_INPUT_NAMES)):
        overrides = {"network.ablated_input": idx}
        ablated = aerocapture_rs.run_mc(toml_path, overrides=overrides, sim_timeout_secs=sim_timeout_secs)
        ablated_costs = _compute_costs(ablated)
        ablated_mean = float(np.mean(ablated_costs))
        delta = ablated_mean - baseline_mean

        results.append({
            "index": idx,
            "name": NN_INPUT_NAMES[idx],
            "baseline_cost": baseline_mean,
            "ablated_cost": ablated_mean,
            "delta": delta,
            "abs_delta": abs(delta),
        })

    ranked = sorted(results, key=lambda r: r["abs_delta"], reverse=True)
    for rank, r in enumerate(ranked):
        r["rank"] = rank + 1

    return {
        "baseline_cost": baseline_mean,
        "n_sims": n_sims,
        "results": results,
        "ranked": ranked,
    }


def _compute_costs(batch_results) -> np.ndarray:
    """Extract DV costs from batch results."""
    records = batch_results.final_records
    # Column index for dv_total in the 52-element final record
    # This matches the cost function: use the DV value directly
    dv_col = 6  # dv_total index in final_record
    return records[:, dv_col]
```

Note: The `_compute_costs` function needs to match the actual cost computation. Check `evaluate.py`'s `compute_cost` for the correct column index and formula. If the cost includes constraint penalties, replicate that logic here. The exact implementation should be verified during development by reading `evaluate.py:compute_cost`.

- [ ] **Step 3: Add CLI entry point**

```python
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NN input ablation analysis")
    parser.add_argument("training_dir", help="Path to training output directory")
    parser.add_argument("--toml", required=True, help="TOML config path")
    parser.add_argument("--n-sims", type=int, default=1000, help="MC sims per ablation run")
    parser.add_argument("--sim-timeout", type=float, default=None, help="Per-sim timeout (seconds)")
    args = parser.parse_args()

    print(f"Running ablation analysis with {args.n_sims} sims per input...")
    results = run_ablation(args.toml, args.training_dir, args.n_sims, args.sim_timeout)

    # Print table
    print(f"\nBaseline mean cost: {results['baseline_cost']:.4f}")
    print(f"{'Rank':<6}{'Index':<8}{'Name':<25}{'Delta':>12}{'Ablated Cost':>15}")
    print("-" * 66)
    for r in results["ranked"]:
        print(f"{r['rank']:<6}{r['index']:<8}{r['name']:<25}{r['delta']:>12.4f}{r['ablated_cost']:>15.4f}")

    # Save JSON
    out_path = Path(args.training_dir) / "ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    # Generate chart
    from aerocapture.training.charts_ablation import chart_ablation_bar
    svg_path = Path(args.training_dir) / "ablation_chart.svg"
    chart_ablation_bar(results["ranked"], str(svg_path))
    print(f"Chart saved to {svg_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create the ablation bar chart**

In `src/python/aerocapture/training/charts_ablation.py`:

```python
"""Ablation analysis chart."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def chart_ablation_bar(ranked: list[dict], output_path: str) -> None:
    """Bar chart of cost delta per input, ranked by importance."""
    sns.set_theme(style="whitegrid", palette="muted")

    names = [r["name"] for r in ranked]
    deltas = [r["delta"] for r in ranked]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#e74c3c" if d > 0 else "#3498db" for d in deltas]
    ax.barh(range(len(names)), deltas, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Cost Delta (ablated - baseline)")
    ax.set_title("NN Input Importance (Ablation Analysis)")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 5: Add `__main__` support**

Create `src/python/aerocapture/training/__main__ablation.py` -- actually, the `if __name__ == "__main__"` block is already in `ablation.py`. Just verify `python -m aerocapture.training.ablation --help` works by adding the module to the package.

Actually, `python -m aerocapture.training.ablation` will work directly since `ablation.py` has `if __name__ == "__main__": main()`.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/ablation.py src/python/aerocapture/training/charts_ablation.py
git commit -m "feat(nn): add ablation analysis script for input importance ranking"
```

---

### Task 8: Python tests for ablation module

**Files:**
- Create: `tests/test_ablation.py`

- [ ] **Step 1: Write tests for the ablation module**

```python
"""Tests for NN input ablation analysis."""

from __future__ import annotations

from aerocapture.training.ablation import NN_INPUT_NAMES


def test_input_names_length():
    """23 inputs in the full superset."""
    assert len(NN_INPUT_NAMES) == 23


def test_input_names_unique():
    """No duplicate input names."""
    assert len(set(NN_INPUT_NAMES)) == len(NN_INPUT_NAMES)


def test_input_names_no_empty():
    """No empty strings in input names."""
    for name in NN_INPUT_NAMES:
        assert name.strip(), f"Empty input name at index {NN_INPUT_NAMES.index(name)}"
```

- [ ] **Step 2: Write test for chart generation**

```python
import tempfile
from pathlib import Path

from aerocapture.training.charts_ablation import chart_ablation_bar


def test_ablation_chart_produces_svg():
    """Chart function produces an SVG file."""
    ranked = [
        {"name": f"input_{i}", "delta": 0.1 * (10 - i), "index": i, "rank": i + 1}
        for i in range(10)
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test.svg")
        chart_ablation_bar(ranked, path)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "<svg" in content
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_ablation.py -v 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_ablation.py
git commit -m "test(nn): add ablation module tests"
```

---

### Task 9: PyO3 bindings -- support `ablated_input` override

**Files:**
- Modify: `src/rust/aerocapture-py/src/config.rs`

The ablation script passes `{"network.ablated_input": 5}` as an override. The PyO3 config layer needs to handle this dot-path and inject it into the TOML before parsing.

- [ ] **Step 1: Check how overrides work in PyO3 config**

Read `src/rust/aerocapture-py/src/config.rs` to understand how dot-path overrides are merged. The existing mechanism for `guidance.ftc.exit_velocity_threshold` etc. should handle `network.ablated_input` automatically if the `[network]` section is parsed by Rust config.

If the Rust TOML parser doesn't have a `[network]` section (it currently doesn't -- `[network]` is Python-only), we need to either:
(a) Add `[network]` parsing to Rust config.rs, or
(b) Have the PyO3 layer merge `ablated_input` into the JSON model file instead

Option (a) is cleaner. Add a `TomlNetwork` struct to `config.rs`:

```rust
#[derive(Debug, Deserialize, Clone, Default)]
pub struct TomlNetwork {
    #[serde(default)]
    pub layer_sizes: Option<Vec<usize>>,
    #[serde(default)]
    pub activations: Option<Vec<String>>,
    #[serde(default)]
    pub input_mask: Option<Vec<usize>>,
    #[serde(default)]
    pub ablated_input: Option<usize>,
}
```

Add to `SimInput`:

```rust
pub struct SimInput {
    // ... existing fields ...
    #[serde(default)]
    pub network: Option<TomlNetwork>,
}
```

Then in `data/mod.rs` where the NN model is loaded, apply `input_mask` and `ablated_input` from the TOML network section (overriding the JSON values):

```rust
let mut neural_net = if config.guidance_type == GuidanceType::NeuralNetwork {
    if let Some(ref nn_path) = toml.data.neural_network {
        Some(neural::NeuralNetModel::load(nn_path)?)
    } else {
        None
    }
} else {
    None
};

// Apply TOML [network] overrides to loaded model
if let (Some(ref mut nn), Some(ref net_cfg)) = (&mut neural_net, &toml.network) {
    if let Some(ref mask) = net_cfg.input_mask {
        nn.input_mask = Some(mask.clone());
    }
    if let Some(ablated) = net_cfg.ablated_input {
        nn.ablated_input = Some(ablated);
    }
    // Re-validate after override
    NeuralNetModel::validate_mask(&nn.input_mask, nn.layer_sizes[0])?;
    NeuralNetModel::validate_ablated_input(&nn.ablated_input)?;
}
```

- [ ] **Step 2: Run the full Rust test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 3: Rebuild PyO3 bindings**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml 2>&1 | tail -5`
Expected: successful build

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/config.rs src/rust/src/data/mod.rs
git commit -m "feat(nn): add [network] TOML section with input_mask and ablated_input support"
```

---

### Task 10: Rebuild, run full test suites, and update golden files

**Files:**
- Various test/golden files

- [ ] **Step 1: Build Rust release binary**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./build.sh`
Expected: successful build of both Rust binary and PyO3 bindings

- [ ] **Step 2: Run Rust tests**

Run: `cd src/rust && cargo test 2>&1 | tail -30`
Expected: all tests pass. If NN golden files fail (because the NN function signature changed), regenerate them.

- [ ] **Step 3: Run Python tests**

Run: `uv run pytest tests/ -x -q 2>&1 | tail -20`
Expected: all tests pass

- [ ] **Step 4: Run linting**

Run: `./lint_code.sh 2>&1 | tail -20`
Expected: clean

- [ ] **Step 5: Regenerate NN golden file if needed**

If the neural golden test fails because the NN now computes 23 inputs instead of 16 (changing output even for the same weights due to ref trajectory access), regenerate:

Run: `./src/rust/target/release/aerocapture configs/test/test_ref_neural.toml`

Copy the output CSV to `tests/reference_data/rust_golden/` replacing the existing neural golden file.

- [ ] **Step 6: Commit any golden file updates**

```bash
git add tests/reference_data/rust_golden/
git commit -m "test(nn): regenerate NN golden file for 23-input architecture"
```

---

### Task 11: Smart commit

**Files:**
- `CLAUDE.md`, `README.md`

- [ ] **Step 1: Invoke `smart-commit` skill**

Use the `smart-commit` skill to update documentation and create a final commit covering the entire branch.

---

## Dependency Graph

```
Task 1 (NeuralNetModel fields)
  └─> Task 2 (23-input vector)
       └─> Task 3 (dispatch wiring)
            └─> Task 4 (fix construction sites)
                 └─> Task 9 (TOML [network] section + PyO3)
                      └─> Task 10 (full rebuild + golden files)
                           └─> Task 11 (smart commit)

Task 5 (Python evaluate.py) -- independent of Rust tasks 1-4
Task 6 (TOML config) -- independent, after Task 5
Task 7 (ablation script) -- independent of Rust tasks
Task 8 (ablation tests) -- after Task 7
```

Tasks 5-8 (Python) can be parallelized with Tasks 1-4 (Rust) since they're independent codebases.
