# NN Expanded Inputs & Full-Envelope Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the NN guidance from 8 to 16 inputs, enable full-envelope (capture + exit) operation, and refactor the guidance dispatcher into clean separation.

**Architecture:** Extract FTC-specific logic from the central guidance dispatcher, rename `FtcState`/`FtcOutput` to `GuidanceState`/`GuidanceOutput`, then expand the NN input vector with 8 new normalized inputs from navigation/orbital data. The NN remains a single phase-blind network using a bounce flag input to distinguish capture from exit.

**Tech Stack:** Rust (nalgebra, serde, proptest), Python (numpy, pytest)

**Spec:** `docs/superpowers/specs/2026-04-02-nn-expanded-inputs-design.md`

---

### Task 1: Extract FTC capture logic into dedicated `ftc.rs`

**Files:**
- Modify: `src/rust/src/gnc/guidance/ftc.rs` (lines 253-361 -- `capture_guidance()` and `compute_gains()`)

The current `ftc.rs` will become `dispatch.rs` in Task 2. Before that rename, extract the FTC-specific functions into a new file so git tracks the move cleanly.

This task creates the new `ftc.rs` file with the extracted functions. The old file keeps them temporarily -- removal happens in Task 2 when it becomes `dispatch.rs`.

- [ ] **Step 1: Create `src/rust/src/gnc/guidance/capture.rs` with FTC capture logic**

Create a new file with the extracted `capture_guidance` and `compute_gains` as a public `ftc_bank_angle` entry point:

```rust
//! FTC (Full Trajectory Control) capture-phase guidance.
//!
//! Altitude-gain predictor-corrector for the capture phase.
//! Separated from the central guidance dispatcher for consistency
//! with other scheme-specific modules.

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::gnc::navigation::estimator::NavigationOutput;

/// Persistent state for FTC capture guidance.
#[derive(Debug, Clone, Default)]
pub struct FtcCaptureState {
    pub securization_counters: [i32; 2],
    pub n_secur: i32,
}

/// Compute FTC capture-phase bank angle.
///
/// Returns unsigned bank angle magnitude in [0, pi] radians.
pub fn ftc_bank_angle(
    nav: &NavigationOutput,
    capture_state: &mut FtcCaptureState,
    data: &SimData,
    planet: &PlanetConfig,
) -> f64 {
    let (altitude, _) = geodetic_from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        planet,
    );
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    let ref_traj = &data.guidance.ref_trajectory;

    let velocity_relative = nav.velocity_estimated[0];
    let velocity_radial = velocity_relative * nav.velocity_estimated[1].sin();
    let dynamic_pressure_equilibrium =
        0.5 * nav.density_guidance * velocity_relative * velocity_relative;

    // Interpolate reference trajectory at current energy
    let cos_bank_nominal = ref_traj.interpolate(energy, &ref_traj.cos_bank);
    let dynamic_pressure_nominal = ref_traj.interpolate(energy, &ref_traj.pressure);
    let altitude_rate_nominal = ref_traj.interpolate(energy, &ref_traj.radial_vel);

    // Compute gains
    let (gain_altitude_rate, gain_dynamic_pressure) =
        compute_gains(altitude, &nav.aero_coefficients, data);

    // Predictor-corrector equation
    let dynamic_pressure_equilibrium_safe = if dynamic_pressure_equilibrium.abs() > 1e-10 {
        dynamic_pressure_equilibrium
    } else {
        1e-10
    };
    let mut cos_bank_commanded = cos_bank_nominal
        + gain_altitude_rate * (velocity_radial - altitude_rate_nominal)
            / dynamic_pressure_equilibrium_safe
        + gain_dynamic_pressure * (dynamic_pressure_equilibrium - dynamic_pressure_nominal)
            / dynamic_pressure_equilibrium_safe;

    // Securization: clamp cos to [-1, 1]
    let bank_angle_longitudinal;
    if cos_bank_commanded.abs() > 1.0 {
        cos_bank_commanded = cos_bank_commanded.signum();
        bank_angle_longitudinal = cos_bank_commanded.acos();
        capture_state.securization_counters[0] += 1;
        capture_state.n_secur += 1;
    } else {
        bank_angle_longitudinal = cos_bank_commanded.acos().abs();
    }

    bank_angle_longitudinal
}

/// Compute guidance gains from altitude-based Pdyn model.
fn compute_gains(altitude: f64, aero_coefficients: &[f64; 2], data: &SimData) -> (f64, f64) {
    let pdyn_table = &data.guidance.pdyn_table;
    let alt_km = altitude / 1e3;

    let mut found: Option<usize> = None;
    for i in 0..pdyn_table.len().saturating_sub(1) {
        if alt_km >= pdyn_table[i].altitude
            && alt_km < pdyn_table[i + 1].altitude
            && found.is_none()
        {
            found = Some(i);
        }
    }
    let table_index = found.unwrap_or_else(|| {
        if pdyn_table.is_empty() {
            0
        } else {
            pdyn_table.len() - 1
        }
    });

    let pressure_coeff = if table_index < pdyn_table.len() {
        pdyn_table[table_index].coeff_a
    } else {
        1.0
    };

    let damping_capture = data.guidance.capture_damping;
    let frequency_capture = data.guidance.capture_frequency;
    let reference_area = data.capsule.reference_area;
    let mass = data.capsule.mass;
    let cz = aero_coefficients[1];

    let gain_altitude_rate = if (reference_area * cz).abs() > 1e-30 {
        -2.0 * damping_capture * frequency_capture * mass / (reference_area * cz)
    } else {
        0.0
    };

    let gain_dynamic_pressure = if (pressure_coeff * reference_area * cz).abs() > 1e-30 {
        -frequency_capture * frequency_capture * mass / (pressure_coeff * reference_area * cz)
    } else {
        0.0
    };

    (gain_altitude_rate, gain_dynamic_pressure)
}
```

Note: We name this file `capture.rs` initially to avoid a naming conflict with the existing `ftc.rs`. In Task 2 we rename the old `ftc.rs` -> `dispatch.rs`, and then rename `capture.rs` -> `ftc.rs`.

- [ ] **Step 2: Add `capture` module to `mod.rs`**

In `src/rust/src/gnc/guidance/mod.rs`, add `pub mod capture;` after the existing `pub mod exit;` line:

```rust
pub mod capture;
```

- [ ] **Step 3: Verify it compiles**

Run: `cd src/rust && cargo check 2>&1 | tail -5`
Expected: compiles (the new module is defined but not yet called from anywhere)

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/capture.rs src/rust/src/gnc/guidance/mod.rs
git commit -m "refactor: extract FTC capture logic into dedicated capture.rs"
```

---

### Task 2: Rename `ftc.rs` -> `dispatch.rs`, then `capture.rs` -> `ftc.rs`

**Files:**
- Rename: `src/rust/src/gnc/guidance/ftc.rs` -> `src/rust/src/gnc/guidance/dispatch.rs`
- Rename: `src/rust/src/gnc/guidance/capture.rs` -> `src/rust/src/gnc/guidance/ftc.rs`
- Modify: `src/rust/src/gnc/guidance/mod.rs` (update module declarations)
- Modify: `src/rust/src/gnc/guidance/dispatch.rs` (rename structs, remove extracted functions, call new `ftc.rs`)
- Modify: `src/rust/src/simulation/runner.rs` (update imports)

- [ ] **Step 1: Rename files**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git mv src/rust/src/gnc/guidance/ftc.rs src/rust/src/gnc/guidance/dispatch.rs
git mv src/rust/src/gnc/guidance/capture.rs src/rust/src/gnc/guidance/ftc.rs
```

- [ ] **Step 2: Update `mod.rs`**

Replace the module declarations in `src/rust/src/gnc/guidance/mod.rs`:

Replace:
```rust
pub mod capture;
```
with:
```rust
pub mod dispatch;
```

And replace:
```rust
pub mod ftc;
```
with nothing (remove the line -- the old `ftc` module is now `dispatch`). Then the new `ftc.rs` (formerly `capture.rs`) keeps its module declaration implicitly via its filename.

Actually, we need both modules:
```rust
pub mod dispatch;
pub mod ftc;
```

Where `dispatch` is the renamed old `ftc` (contains `guidance_step`, `GuidanceState`, `GuidanceOutput`) and `ftc` is the renamed `capture.rs` (contains `ftc_bank_angle`).

The full `mod.rs` should be:

```rust
//! Guidance algorithms.

pub mod dispatch;
pub mod energy_controller;
pub mod equilibrium_glide;
pub mod exit;
pub mod fnpag;
pub mod ftc;
pub mod lateral;
pub mod neural;
pub mod piecewise_constant;
pub mod predguid;
pub mod reference;
pub mod thermal_limiter;

use crate::data::SphericalState;

/// Guidance command output
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GuidanceCommand {
    pub bank_angle: f64,
    pub aoa: f64,
}

/// Guidance algorithm trait
#[allow(dead_code)]
pub trait Guidance {
    fn compute(&mut self, state: &SphericalState, time: f64) -> GuidanceCommand;
}
```

- [ ] **Step 3: Rename structs in `dispatch.rs`**

In `src/rust/src/gnc/guidance/dispatch.rs`:

Rename `FtcState` -> `GuidanceState` (replace all occurrences in the file):
- Struct definition (line 17)
- `impl FtcState` block (line 45)
- All test usages

Rename `FtcOutput` -> `GuidanceOutput` (replace all occurrences in the file):
- Struct definition (line 68)
- Return type of `guidance_step` (line 89)
- `FtcOutput::default()` call (line 90)
- All test usages

- [ ] **Step 4: Remove `capture_guidance()` and `compute_gains()` from `dispatch.rs`**

Delete the `capture_guidance()` function (lines 253-310) and `compute_gains()` function (lines 312-361) from `dispatch.rs`. These now live in `ftc.rs`.

Also add `FtcCaptureState` to `GuidanceState` and add the import. In `dispatch.rs`:

Add import:
```rust
use crate::gnc::guidance::ftc::{self as ftc_capture, FtcCaptureState};
```

Add field to `GuidanceState`:
```rust
pub ftc_capture: FtcCaptureState,
```

In `GuidanceState::new()`, initialize it:
```rust
ftc_capture: FtcCaptureState::default(),
```

Update the FTC match arm in `guidance_step()` to call the new module:

Replace:
```rust
GuidanceType::Ftc => capture_guidance(nav, energy, altitude, state, data, planet),
```
with:
```rust
GuidanceType::Ftc => ftc_capture::ftc_bank_angle(nav, &mut state.ftc_capture, data, planet),
```

And remove the now-unused `energy` and `altitude` local variables from the `guidance_step` body IF they are only used by `capture_guidance`. Check: `energy` is also used for longitudinal activation check (line 106-122) and lateral guidance (line 215). `altitude` is used for AoA scheduling (line 102). So both stay -- they are NOT only used by `capture_guidance`.

- [ ] **Step 5: Update `runner.rs` imports**

In `src/rust/src/simulation/runner.rs`, line 10:

Replace:
```rust
use crate::gnc::guidance::ftc::{self, FtcState};
```
with:
```rust
use crate::gnc::guidance::dispatch::{self, GuidanceState};
```

Then update all usages in `runner.rs`:
- Line 503: `FtcState::new(` -> `GuidanceState::new(`
- Line 644: `ftc::guidance_step(` -> `dispatch::guidance_step(`
- Line 659: `ftc_out.bank_angle_commanded` stays (variable name, not type)
- The local variable `ftc_state` -> `guidance_state` and `ftc_out` -> `guidance_out` for clarity.

- [ ] **Step 6: Verify it compiles**

Run: `cd src/rust && cargo check 2>&1 | tail -10`
Expected: compiles with no errors.

- [ ] **Step 7: Run existing tests**

Run: `cd src/rust && cargo test 2>&1 | tail -20`
Expected: all tests pass (behavior unchanged, only names changed).

- [ ] **Step 8: Commit**

```bash
git add -A src/rust/src/gnc/guidance/ src/rust/src/simulation/runner.rs
git commit -m "refactor: rename ftc.rs -> dispatch.rs, extract FTC capture into ftc.rs

Rename FtcState -> GuidanceState, FtcOutput -> GuidanceOutput.
The central guidance dispatcher now lives in dispatch.rs,
and FTC-specific capture logic has its own ftc.rs module,
matching the pattern of other guidance scheme files."
```

---

### Task 3: Expand NN input vector from 8 to 16

**Files:**
- Modify: `src/rust/src/gnc/guidance/neural.rs` (lines 21-64)

- [ ] **Step 1: Write a failing test for 16-input NN**

Add a test in `src/rust/src/gnc/guidance/neural.rs` inside the `tests` module that constructs a 16-input network and verifies it produces a valid bank angle. This test should fail because `nn_bank_angle` still constructs an 8-element input.

Add this test:

```rust
#[test]
fn sixteen_input_network_produces_valid_output() {
    let nav = test_nav();
    let planet = PlanetConfig::mars();

    // 16-input network: 16 -> 24 -> 2
    let layer0 = Layer {
        w: vec![vec![0.01; 16]; 24],
        b: vec![0.0; 24],
        activation: Activation::Tanh,
    };
    let layer1 = Layer {
        w: vec![vec![0.1; 24], vec![-0.1; 24]],
        b: vec![0.0, 0.0],
        activation: Activation::Asinh,
    };
    let nn = NeuralNetModel {
        layer_sizes: vec![16, 24, 2],
        layers: vec![layer0, layer1],
        output_interpretation: "atan2".to_string(),
    };

    let bank = nn_bank_angle(&nav, &nn, &planet, 50.0_f64.to_radians());
    assert!(bank.is_finite(), "bank angle must be finite, got: {}", bank);
    assert!(
        bank > -std::f64::consts::PI - 1e-10 && bank <= std::f64::consts::PI + 1e-10,
        "bank angle out of atan2 range: {}",
        bank,
    );
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd src/rust && cargo test sixteen_input_network 2>&1 | tail -10`
Expected: FAIL -- the NN `forward()` will panic with "NN input length (8) does not match expected input size (16)".

- [ ] **Step 3: Expand `nn_bank_angle` to construct 16 inputs**

In `src/rust/src/gnc/guidance/neural.rs`, replace the entire `nn_bank_angle` function body. The function signature gains one parameter for the planet equatorial radius (already available via `planet`):

```rust
pub fn nn_bank_angle(
    nav: &NavigationOutput,
    nn: &NeuralNetModel,
    planet: &PlanetConfig,
    target_inclination: f64,
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

    // Acceleration magnitude: sqrt(drag^2 + lift^2)
    let accel_mag = (nav.acceleration_estimated[0] * nav.acceleration_estimated[0]
        + nav.acceleration_estimated[1] * nav.acceleration_estimated[1])
        .sqrt();

    // Altitude in km
    let altitude_km = (nav.position_estimated[0] - planet.equatorial_radius) / 1e3;

    // 16 normalized inputs
    let input = [
        // -- Existing 8 inputs (indices 0-7) --
        orbit.eccentricity - 1.0,                                        // 0: eccentricity excess
        (orbit.inclination - target_inclination).to_degrees() * 3.0 / 5.0, // 1: inclination error
        2.0 * (velocity_radial / 1e3 + 1.2) / 1.5 - 1.0,               // 2: radial velocity
        -mu / (2.0 * orbit.semi_major_axis) / 6e6,                      // 3: orbital energy
        (nav.velocity_estimated[0] / 3e3 - 1.5) * 2.0,                  // 4: velocity
        accel_mag / 20.0 - 1.0,                                         // 5: accel magnitude
        nav.heat_flux_fraction * 2.0 - 1.0,                             // 6: heat flux fraction
        nav.heat_load_fraction * 2.0 - 1.0,                             // 7: heat load fraction
        // -- New 8 inputs (indices 8-15) --
        (altitude_km - 65.0) / 65.0,                                    // 8: altitude
        nav.velocity_estimated[1] / 0.3,                                // 9: flight path angle
        nav.position_estimated[2] / std::f64::consts::FRAC_PI_2,        // 10: latitude
        nav.acceleration_estimated[0] / 50.0 - 1.0,                     // 11: drag acceleration
        nav.acceleration_estimated[1] / 10.0,                            // 12: lift acceleration
        nav.orbital_errors[0] / 5e5,                                    // 13: SMA error
        orbit.apoapsis_alt / 1e6 - 1.0,                                 // 14: apoapsis altitude
        nav.bounce_flag as f64 * 2.0 - 1.0,                             // 15: bounce flag
    ];

    let output = nn.forward(&input);

    // Bank angle from atan2
    output[0].atan2(output[1])
}
```

- [ ] **Step 4: Update the module-level doc comment**

At the top of `neural.rs`, update the doc comment:

Replace:
```rust
//! Feedforward network computing bank angle from navigation state.
//! Supports arbitrary layer architectures via NeuralNetModel.
//! Default: 8 inputs -> 12 hidden (tanh) -> 2 outputs (asinh) -> atan2 bank angle.
```
with:
```rust
//! Feedforward network computing bank angle from navigation state.
//! Supports arbitrary layer architectures via NeuralNetModel.
//! Default: 16 inputs -> 24 hidden (tanh) -> 2 outputs (asinh) -> atan2 bank angle.
//!
//! 16 inputs: eccentricity excess, inclination error, radial velocity, orbital energy,
//! velocity, acceleration magnitude, heat flux fraction, heat load fraction, altitude,
//! flight path angle, latitude, drag acceleration, lift acceleration, SMA error,
//! apoapsis altitude, bounce flag.
```

And update the function doc comment:

Replace:
```rust
/// - Normalizes 8 inputs from orbital/aerodynamic/thermal quantities
```
with:
```rust
/// - Normalizes 16 inputs from orbital/aerodynamic/thermal/geometric quantities
```

- [ ] **Step 5: Run the new test**

Run: `cd src/rust && cargo test sixteen_input_network 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/guidance/neural.rs
git commit -m "feat: expand NN guidance from 8 to 16 inputs

Add altitude, FPA, latitude, separate drag/lift accelerations,
SMA error, apoapsis altitude, and bounce flag as new normalized
inputs. Enables full-envelope capture+exit guidance."
```

---

### Task 4: Update existing `neural.rs` tests for 16 inputs

**Files:**
- Modify: `src/rust/src/gnc/guidance/neural.rs` (test module, lines 66-216)

- [ ] **Step 1: Update `test_nav()` fixture**

The existing `test_nav()` already sets the fields needed for the new inputs (`position_estimated`, `velocity_estimated`, `acceleration_estimated`, `bounce_flag` defaults to 0). Add explicit `orbital_errors` to make tests deterministic:

Replace:
```rust
fn test_nav() -> NavigationOutput {
    let r = 3_396_200.0 + 50_000.0; // Mars radius + 50 km
    let velocity = 5000.0;
    NavigationOutput {
        position_estimated: [r, 0.1, 0.05],
        velocity_estimated: [velocity, -0.10, 0.5],
        acceleration_estimated: [80.0, -12.0],
        aero_coefficients: [1.269, -0.205],
        density_guidance: 0.001,
        density_exit: 1e-6,
        dynamic_pressure_estimated: 0.5 * 0.001 * velocity * velocity,
        energy_estimated: -1e6,
        ..Default::default()
    }
}
```
with:
```rust
fn test_nav() -> NavigationOutput {
    let r = 3_396_200.0 + 50_000.0; // Mars radius + 50 km
    let velocity = 5000.0;
    NavigationOutput {
        position_estimated: [r, 0.1, 0.05],
        velocity_estimated: [velocity, -0.10, 0.5],
        acceleration_estimated: [80.0, -12.0],
        aero_coefficients: [1.269, -0.205],
        density_guidance: 0.001,
        density_exit: 1e-6,
        dynamic_pressure_estimated: 0.5 * 0.001 * velocity * velocity,
        energy_estimated: -1e6,
        orbital_errors: [1000.0, 0.01, 0.001, 0.002],
        ..Default::default()
    }
}
```

- [ ] **Step 2: Update `zero_weight_nn` helper**

Replace:
```rust
fn zero_weight_nn(bias0: f64, bias1: f64) -> NeuralNetModel {
    NeuralNetModel {
        layer_sizes: vec![8, 2],
        layers: vec![Layer {
            w: vec![vec![0.0; 8], vec![0.0; 8]],
            b: vec![bias0, bias1],
            activation: Activation::Linear,
        }],
        output_interpretation: "atan2".to_string(),
    }
}
```
with:
```rust
fn zero_weight_nn(bias0: f64, bias1: f64) -> NeuralNetModel {
    NeuralNetModel {
        layer_sizes: vec![16, 2],
        layers: vec![Layer {
            w: vec![vec![0.0; 16], vec![0.0; 16]],
            b: vec![bias0, bias1],
            activation: Activation::Linear,
        }],
        output_interpretation: "atan2".to_string(),
    }
}
```

- [ ] **Step 3: Update the 8->3->2 network in `output_in_valid_range`**

Replace:
```rust
let layer0 = Layer {
    w: vec![
        vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.05, -0.05],
        vec![-0.2, 0.1, -0.1, 0.3, -0.2, 0.1, 0.05, -0.05],
        vec![0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    ],
    b: vec![0.1, -0.1, 0.0],
    activation: Activation::Tanh,
};
let layer1 = Layer {
    w: vec![vec![0.5, -0.5, 0.2], vec![-0.3, 0.3, -0.1]],
    b: vec![0.0, 0.0],
    activation: Activation::Asinh,
};
let nn = NeuralNetModel {
    layer_sizes: vec![8, 3, 2],
    layers: vec![layer0, layer1],
    output_interpretation: "atan2".to_string(),
};
```
with:
```rust
let layer0 = Layer {
    w: vec![
        vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.05, -0.05, 0.1, -0.1, 0.05, 0.02, -0.02, 0.03, -0.03, 0.01],
        vec![-0.2, 0.1, -0.1, 0.3, -0.2, 0.1, 0.05, -0.05, -0.1, 0.1, -0.05, 0.03, -0.01, 0.02, -0.02, 0.01],
        vec![0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    ],
    b: vec![0.1, -0.1, 0.0],
    activation: Activation::Tanh,
};
let layer1 = Layer {
    w: vec![vec![0.5, -0.5, 0.2], vec![-0.3, 0.3, -0.1]],
    b: vec![0.0, 0.0],
    activation: Activation::Asinh,
};
let nn = NeuralNetModel {
    layer_sizes: vec![16, 3, 2],
    layers: vec![layer0, layer1],
    output_interpretation: "atan2".to_string(),
};
```

- [ ] **Step 4: Update `fixed_small_nn` in proptest module**

Replace:
```rust
fn fixed_small_nn() -> NeuralNetModel {
    NeuralNetModel {
        layer_sizes: vec![8, 2],
        layers: vec![Layer {
            w: vec![
                vec![0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.1, -0.1],
                vec![-0.1, 0.1, -0.05, 0.05, 0.15, -0.15, 0.05, -0.05],
            ],
            b: vec![0.3, -0.2],
            activation: Activation::Tanh,
        }],
        output_interpretation: "atan2".to_string(),
    }
}
```
with:
```rust
fn fixed_small_nn() -> NeuralNetModel {
    NeuralNetModel {
        layer_sizes: vec![16, 2],
        layers: vec![Layer {
            w: vec![
                vec![0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.1, -0.1, 0.02, -0.02, 0.03, -0.03, 0.04, -0.04, 0.01, -0.01],
                vec![-0.1, 0.1, -0.05, 0.05, 0.15, -0.15, 0.05, -0.05, -0.02, 0.02, -0.03, 0.03, -0.04, 0.04, -0.01, 0.01],
            ],
            b: vec![0.3, -0.2],
            activation: Activation::Tanh,
        }],
        output_interpretation: "atan2".to_string(),
    }
}
```

- [ ] **Step 5: Run all neural tests**

Run: `cd src/rust && cargo test neural 2>&1 | tail -15`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/guidance/neural.rs
git commit -m "test: update neural.rs tests for 16-input architecture"
```

---

### Task 5: Update golden NN test config and model

**Files:**
- Modify: `tests/reference_data/rust_golden/neural/nn_model_golden.json`
- Modify: `configs/test/test_neural_golden.toml` (if needed)

The golden NN model has an `[8, 12, 2]` architecture. It needs to become `[16, ...]` to match the new 16-input `nn_bank_angle`. The golden regression test (`src/rust/tests/guidance_regression.rs`) runs the simulator with this model and checks the output against reference data. After changing the model architecture, the reference outputs must be regenerated.

- [ ] **Step 1: Generate a new 16-input golden model**

Write a small script or use the existing Rust code to create a `[16, 12, 2]` model JSON with zero-padded weights (keeps the first 8 columns identical to the old model, adds 8 zero columns for the new inputs). This preserves the old model's behavior for the original 8 inputs while accepting the new 16-input format.

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run python -c "
import json
from pathlib import Path

model_path = Path('tests/reference_data/rust_golden/neural/nn_model_golden.json')
model = json.loads(model_path.read_text())

# Expand layer 0 weights from 8 to 16 columns (zero-pad new inputs)
old_layer0 = model['weights']['layer_0']
new_w = [row + [0.0] * 8 for row in old_layer0['w']]
old_layer0['w'] = new_w

# Update architecture
model['architecture']['layers'][0] = 16

model_path.write_text(json.dumps(model))
print('Updated golden model to 16 inputs')
"
```

- [ ] **Step 2: Regenerate golden reference data**

Build the Rust binary and re-run the golden test config to produce new reference output:

```bash
cd /Users/govit/Git/Govit/Aerocapture
cd src/rust && cargo build --release && cd ../..
./src/rust/target/release/aerocapture configs/test/test_neural_golden.toml
```

Then copy the new output files to replace the golden reference data. The exact output file names depend on the `results_suffix` (`.golden_neural`). Check what files the regression test compares against:

```bash
ls tests/reference_data/rust_golden/neural/
```

Copy the new simulation output to replace the golden reference files.

- [ ] **Step 3: Run the golden regression test**

Run: `cd src/rust && cargo test guidance_regression 2>&1 | tail -15`
Expected: all guidance regression tests pass (including the neural one with updated references).

- [ ] **Step 4: Commit**

```bash
git add tests/reference_data/rust_golden/neural/ configs/test/
git commit -m "test: regenerate golden NN model and reference data for 16 inputs"
```

---

### Task 6: Update training TOML config

**Files:**
- Modify: `configs/training/msr_aller_nn_train_consolidated.toml`

- [ ] **Step 1: Update the network layer sizes**

The current config has `layer_sizes = [8, 16, 64, 16, 2]`. Update the first element from 8 to 16:

Replace:
```toml
layer_sizes = [8, 16, 64, 16, 2]
```
with:
```toml
layer_sizes = [16, 16, 64, 16, 2]
```

- [ ] **Step 2: Commit**

```bash
git add configs/training/msr_aller_nn_train_consolidated.toml
git commit -m "config: update NN training TOML to 16 inputs"
```

---

### Task 7: Update Python `NetworkConfig` default

**Files:**
- Modify: `src/python/aerocapture/training/config.py` (line 23)

The default `layer_sizes` is `[6, 12, 2]` (legacy Fortran). While training TOMLs override this, the default should reflect the current architecture for consistency.

- [ ] **Step 1: Update the default**

Replace:
```python
layer_sizes: list[int] = field(default_factory=lambda: [6, 12, 2])
```
with:
```python
layer_sizes: list[int] = field(default_factory=lambda: [16, 24, 2])
```

And update the docstring:

Replace:
```python
"""Neural network architecture configuration.

Supports arbitrary layer configurations via `layer_sizes` and `activations`.
Default [6, 12, 2] with ["tanh", "asinh"] matches the legacy Fortran architecture.
"""
```
with:
```python
"""Neural network architecture configuration.

Supports arbitrary layer configurations via `layer_sizes` and `activations`.
Default [16, 24, 2] with ["tanh", "asinh"] matches the 16-input Rust architecture.
"""
```

- [ ] **Step 2: Update Python tests that hardcode `[6, 12, 2]`**

In `tests/test_config.py`, update all explicit `[6, 12, 2]` references. These tests use specific architectures in their assertions, so they need the arithmetic updated:

Line 32: `NetworkConfig(layer_sizes=[6, 12, 2])` -> `NetworkConfig(layer_sizes=[16, 24, 2])`
Line 35: comment `n_base_coef = (6*12 + 12) + (12*2 + 2) = 84 + 26 = 110` -> `n_base_coef = (16*24 + 24) + (24*2 + 2) = 408 + 50 = 458`
Line 36: `assert config.n_params == 110` -> `assert config.n_params == 458`

Line 42: Keep the parametrized architectures as-is -- they test generic computation, not specific defaults.

Line 79: `NetworkConfig(layer_sizes=[6, 12, 2])` -> `NetworkConfig(layer_sizes=[16, 24, 2])`
Line 83: `expected = config.network.n_base_coef * 16` -- this is generic, no change needed.

Line 90: `NetworkConfig(layer_sizes=[6, 12, 2])` -> `NetworkConfig(layer_sizes=[16, 24, 2])`
Line 92: `assert net.n_base_coef == 110` -> `assert net.n_base_coef == 458`

Line 96: `NetworkConfig(layer_sizes=[6, 12, 2])` -> `NetworkConfig(layer_sizes=[16, 24, 2])`

- [ ] **Step 3: Run Python tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_config.py -v 2>&1 | tail -20`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/config.py tests/test_config.py
git commit -m "feat: update Python NetworkConfig default to [16, 24, 2]"
```

---

### Task 8: Run full test suite and fix any remaining breakage

**Files:**
- Potentially any file with lingering `FtcState`/`FtcOutput` references or 8-input NN assumptions

- [ ] **Step 1: Run Rust tests**

Run: `cd src/rust && cargo test 2>&1 | tail -30`

Fix any failures. Common issues:
- Integration tests referencing `FtcState` or `ftc::guidance_step` (update imports)
- Golden reference data mismatches (re-run golden configs)

- [ ] **Step 2: Run Rust lints**

Run: `cd src/rust && cargo clippy 2>&1 | tail -20`

Fix any clippy warnings in modified code.

- [ ] **Step 3: Run Python tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -v 2>&1 | tail -30`

Fix any failures.

- [ ] **Step 4: Run full check suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./check_all.sh 2>&1 | tail -20`

Expected: all Rust checks pass (test, fmt, clippy, release build).

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh 2>&1 | tail -20`

Expected: all Python lints pass (ruff, mypy).

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: address remaining test and lint issues from NN expansion"
```

---

### Task 9: Smart commit (final)

- [ ] **Step 1: Invoke the `smart-commit` skill**

Use the `smart-commit` skill to update CLAUDE.md and README.md to reflect the changes across the whole branch, then commit everything.

Key documentation updates:
- CLAUDE.md: update the NN description (8 inputs -> 16 inputs, list all inputs, default architecture 16-24-2, full-envelope operation), update the `ftc.rs` entry to reflect the split into `dispatch.rs` + `ftc.rs`, update `GuidanceState`/`GuidanceOutput` names
- README.md: if it references NN architecture or inputs
