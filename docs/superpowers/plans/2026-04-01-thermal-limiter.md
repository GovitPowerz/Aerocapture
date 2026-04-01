# Thermal Safety Limiter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thermal safety limiter that overrides bank angle toward full lift-up when heat flux or heat load approach their constraint limits, with GA-tunable ramp parameters for unsigned-magnitude guidance schemes and thermal margin inputs for the NN scheme.

**Architecture:** New `thermal_limiter.rs` module with a pure ramp function, called from the central guidance dispatch after longitudinal bank angle computation. `NavigationOutput` extended with thermal fractions computed in the runner. NN input vector expanded from 6 to 8 with thermal margin fractions. Four GA-tunable limiter parameters added to the five unsigned-magnitude scheme param spaces.

**Tech Stack:** Rust (nalgebra, serde, proptest), Python (numpy, pytest)

---

### Task 1: Thermal limiter Rust module -- tests

**Files:**
- Create: `src/rust/src/gnc/guidance/thermal_limiter.rs`

- [ ] **Step 1: Create `thermal_limiter.rs` with the `ThermalLimiterParams` struct and stub function**

```rust
//! Thermal safety limiter -- bank angle override near heat flux / heat load limits.
//!
//! Smooth ramp from guidance-commanded bank angle toward full lift-up (cos_bank=1.0)
//! as thermal quantities approach constraint limits. GA-tunable activation thresholds
//! and ramp exponents per scheme. Applied to unsigned-magnitude schemes only.

/// GA-tunable thermal limiter parameters.
#[derive(Debug, Clone, Copy)]
pub struct ThermalLimiterParams {
    /// Fraction of max_heat_flux at which ramp begins (0.6--0.95).
    pub heat_flux_activation: f64,
    /// Fraction of max_heat_load at which ramp begins (0.6--0.95).
    pub heat_load_activation: f64,
    /// Ramp shape for heat flux (1.0=linear, 2.0=quadratic).
    pub heat_flux_ramp_exponent: f64,
    /// Ramp shape for heat load (1.0=linear, 2.0=quadratic).
    pub heat_load_ramp_exponent: f64,
}

impl Default for ThermalLimiterParams {
    fn default() -> Self {
        Self {
            heat_flux_activation: 1.0,   // 1.0 = never activates
            heat_load_activation: 1.0,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 1.0,
        }
    }
}

/// Apply thermal safety limit to a commanded cos(bank) value.
///
/// Blends `cos_bank_cmd` toward 1.0 (full lift-up) as thermal fractions
/// approach 1.0. Returns the limited cos(bank), always in [cos_bank_cmd, 1.0].
///
/// Both limiters are evaluated independently; the most restrictive wins.
/// If both fractions are below their activation thresholds, returns `cos_bank_cmd` unchanged.
pub fn apply_thermal_limit(
    cos_bank_cmd: f64,
    heat_flux_fraction: f64,
    heat_load_fraction: f64,
    params: &ThermalLimiterParams,
) -> f64 {
    let alpha_flux = compute_alpha(heat_flux_fraction, params.heat_flux_activation, params.heat_flux_ramp_exponent);
    let alpha_load = compute_alpha(heat_load_fraction, params.heat_load_activation, params.heat_load_ramp_exponent);
    let alpha = alpha_flux.max(alpha_load);
    (1.0 - alpha) * cos_bank_cmd + alpha * 1.0
}

/// Compute ramp blending factor alpha for a single thermal quantity.
///
/// Returns 0.0 below activation, 1.0 at or above 100%, smooth ramp in between.
fn compute_alpha(fraction: f64, activation: f64, exponent: f64) -> f64 {
    if fraction <= activation {
        0.0
    } else if fraction >= 1.0 {
        1.0
    } else {
        ((fraction - activation) / (1.0 - activation)).powf(exponent)
    }
}
```

- [ ] **Step 2: Add unit tests below the implementation**

Append this `#[cfg(test)]` module at the bottom of `thermal_limiter.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn active_params() -> ThermalLimiterParams {
        ThermalLimiterParams {
            heat_flux_activation: 0.8,
            heat_load_activation: 0.85,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 2.0,
        }
    }

    #[test]
    fn below_activation_returns_unchanged() {
        let p = active_params();
        let cos_cmd = 0.3;
        let result = apply_thermal_limit(cos_cmd, 0.5, 0.5, &p);
        assert_relative_eq!(result, cos_cmd, epsilon = 1e-12);
    }

    #[test]
    fn at_limit_returns_full_lift_up() {
        let p = active_params();
        let result = apply_thermal_limit(-0.5, 1.0, 0.0, &p);
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn heat_load_at_limit_returns_full_lift_up() {
        let p = active_params();
        let result = apply_thermal_limit(-0.5, 0.0, 1.0, &p);
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn above_limit_returns_full_lift_up() {
        let p = active_params();
        let result = apply_thermal_limit(-0.5, 1.5, 0.0, &p);
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn mid_ramp_linear() {
        let p = ThermalLimiterParams {
            heat_flux_activation: 0.8,
            heat_load_activation: 1.0, // disabled
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 1.0,
        };
        let cos_cmd = 0.0;
        // fraction=0.9, activation=0.8 => alpha = (0.9-0.8)/(1.0-0.8) = 0.5
        // result = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        let result = apply_thermal_limit(cos_cmd, 0.9, 0.0, &p);
        assert_relative_eq!(result, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn mid_ramp_quadratic() {
        let p = ThermalLimiterParams {
            heat_flux_activation: 1.0, // disabled
            heat_load_activation: 0.8,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 2.0,
        };
        let cos_cmd = 0.0;
        // fraction=0.9, activation=0.8, exponent=2.0
        // alpha = ((0.9-0.8)/(1.0-0.8))^2 = 0.5^2 = 0.25
        // result = 0.75 * 0.0 + 0.25 * 1.0 = 0.25
        let result = apply_thermal_limit(cos_cmd, 0.0, 0.9, &p);
        assert_relative_eq!(result, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn most_restrictive_wins() {
        let p = ThermalLimiterParams {
            heat_flux_activation: 0.8,
            heat_load_activation: 0.8,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 1.0,
        };
        // flux fraction = 0.85 => alpha_flux = (0.85-0.8)/0.2 = 0.25
        // load fraction = 0.95 => alpha_load = (0.95-0.8)/0.2 = 0.75
        // alpha_max = 0.75
        let cos_cmd = 0.0;
        let result = apply_thermal_limit(cos_cmd, 0.85, 0.95, &p);
        let expected = 0.25 * 0.0 + 0.75 * 1.0;
        assert_relative_eq!(result, expected, epsilon = 1e-12);
    }

    #[test]
    fn zero_fractions_no_intervention() {
        let p = active_params();
        let cos_cmd = -0.7;
        let result = apply_thermal_limit(cos_cmd, 0.0, 0.0, &p);
        assert_relative_eq!(result, cos_cmd, epsilon = 1e-12);
    }

    #[test]
    fn default_params_never_activate() {
        let p = ThermalLimiterParams::default();
        let cos_cmd = -1.0;
        // activation=1.0 means fraction must exceed 1.0 to trigger
        let result = apply_thermal_limit(cos_cmd, 0.99, 0.99, &p);
        assert_relative_eq!(result, cos_cmd, epsilon = 1e-12);
    }

    #[test]
    fn negative_cos_bank_pushed_toward_one() {
        let p = ThermalLimiterParams {
            heat_flux_activation: 0.5,
            heat_load_activation: 1.0,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 1.0,
        };
        let cos_cmd = -1.0; // full lift-down
        // fraction=1.0 => alpha=1.0
        let result = apply_thermal_limit(cos_cmd, 1.0, 0.0, &p);
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn output_between_cmd_and_one(
                cos_cmd in -1.0..=1.0_f64,
                flux_frac in 0.0..2.0_f64,
                load_frac in 0.0..2.0_f64,
                flux_act in 0.5..1.0_f64,
                load_act in 0.5..1.0_f64,
                flux_exp in 0.5..3.0_f64,
                load_exp in 0.5..3.0_f64,
            ) {
                let p = ThermalLimiterParams {
                    heat_flux_activation: flux_act,
                    heat_load_activation: load_act,
                    heat_flux_ramp_exponent: flux_exp,
                    heat_load_ramp_exponent: load_exp,
                };
                let result = apply_thermal_limit(cos_cmd, flux_frac, load_frac, &p);
                prop_assert!(result.is_finite(), "result not finite: {}", result);
                prop_assert!(result >= cos_cmd - 1e-12, "result {} < cos_cmd {}", result, cos_cmd);
                prop_assert!(result <= 1.0 + 1e-12, "result {} > 1.0", result);
            }

            #[test]
            fn monotonic_in_fraction(
                cos_cmd in -1.0..=1.0_f64,
                frac_lo in 0.0..1.0_f64,
                frac_hi in 0.0..1.0_f64,
                activation in 0.5..0.99_f64,
                exponent in 0.5..3.0_f64,
            ) {
                let p = ThermalLimiterParams {
                    heat_flux_activation: activation,
                    heat_load_activation: 1.0,
                    heat_flux_ramp_exponent: exponent,
                    heat_load_ramp_exponent: 1.0,
                };
                let lo = frac_lo.min(frac_hi);
                let hi = frac_lo.max(frac_hi);
                let r_lo = apply_thermal_limit(cos_cmd, lo, 0.0, &p);
                let r_hi = apply_thermal_limit(cos_cmd, hi, 0.0, &p);
                // Higher fraction => more intervention => result closer to 1.0
                prop_assert!(r_hi >= r_lo - 1e-12, "not monotonic: r_hi={} < r_lo={}", r_hi, r_lo);
            }
        }
    }
}
```

- [ ] **Step 3: Register the module in `mod.rs`**

In `src/rust/src/gnc/guidance/mod.rs`, add after the `pub mod reference;` line:

```rust
pub mod thermal_limiter;
```

- [ ] **Step 4: Run the tests**

Run: `cd src/rust && cargo test thermal_limiter -- --nocapture`
Expected: All tests pass (unit + proptest).

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/gnc/guidance/thermal_limiter.rs src/rust/src/gnc/guidance/mod.rs
git commit -m "add thermal limiter module with tests"
```

---

### Task 2: Add `max_heat_load` to Constraints struct and TOML parsing

**Files:**
- Modify: `src/rust/src/data/mod.rs:129-136` (Constraints struct)
- Modify: `src/rust/src/config.rs:506-511` (TomlConstraints struct)
- Modify: `src/rust/src/data/mod.rs:284-288` (Constraints construction)

- [ ] **Step 1: Add `max_heat_load` to the `Constraints` struct**

In `src/rust/src/data/mod.rs`, edit the `Constraints` struct:

```rust
#[derive(Debug, Clone, Copy, Default)]
pub struct Constraints {
    pub max_heat_flux: f64,        // W/m^2 (from kW/m^2)
    pub max_load_factor: f64,      // m/s^2 (from g, multiplied by g0=9.81)
    pub max_dynamic_pressure: f64, // Pa (from kPa)
    pub max_heat_load: f64,        // J/m^2 (from kJ/m^2)
}
```

- [ ] **Step 2: Add `max_heat_load` to `TomlConstraints`**

In `src/rust/src/config.rs`, edit the `TomlConstraints` struct:

```rust
#[derive(Debug, Deserialize, Clone)]
pub struct TomlConstraints {
    pub max_heat_flux: f64,        // kW/m^2
    pub max_load_factor: f64,      // g
    pub max_dynamic_pressure: f64, // kPa
    #[serde(default)]
    pub max_heat_load: f64,        // kJ/m^2
}
```

- [ ] **Step 3: Wire `max_heat_load` into the Constraints construction**

In `src/rust/src/data/mod.rs`, edit the constraints construction (around line 284):

```rust
        let constraints = Constraints {
            max_heat_flux: f.constraints.max_heat_flux * 1e3,
            max_load_factor: f.constraints.max_load_factor * G0,
            max_dynamic_pressure: f.constraints.max_dynamic_pressure * 1e3,
            max_heat_load: f.constraints.max_heat_load * 1e3, // kJ/m^2 -> J/m^2
        };
```

- [ ] **Step 4: Run Rust tests to verify nothing broke**

Run: `cd src/rust && cargo test`
Expected: All existing tests pass. The `Default` derive on `Constraints` gives `max_heat_load: 0.0` for test configs that don't set it.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/mod.rs src/rust/src/config.rs
git commit -m "add max_heat_load to Constraints struct and TOML parsing"
```

---

### Task 3: Add `ThermalLimiterParams` to `GuidanceParams` and TOML parsing

**Files:**
- Modify: `src/rust/src/data/guidance_params.rs:124-168` (GuidanceParams struct)
- Modify: `src/rust/src/config.rs:312-335` (TomlGuidance struct)
- Modify: `src/rust/src/data/mod.rs:431-473` (GuidanceParams construction with FTC)
- Modify: `src/rust/src/data/mod.rs:485-521` (GuidanceParams construction without FTC)

- [ ] **Step 1: Add `thermal_limiter` field to `GuidanceParams`**

In `src/rust/src/data/guidance_params.rs`, add `use crate::gnc::guidance::thermal_limiter::ThermalLimiterParams;` at the top (after the existing `use` statement), then add the field to `GuidanceParams`:

```rust
    pub piecewise_constant: PiecewiseConstantParams,

    // Thermal safety limiter (shared by unsigned-magnitude schemes)
    pub thermal_limiter: ThermalLimiterParams,
}
```

And in the `Default` impl for `GuidanceParams`, add:

```rust
                piecewise_constant: PiecewiseConstantParams::default(),
                thermal_limiter: ThermalLimiterParams::default(),
```

- [ ] **Step 2: Add `TomlThermalLimiterParams` to config.rs and wire into `TomlGuidance`**

In `src/rust/src/config.rs`, add the TOML struct (near the other guidance param structs, e.g., after `TomlLateralParams`):

```rust
#[derive(Debug, Deserialize, Clone)]
pub struct TomlThermalLimiterParams {
    #[serde(default = "default_one")]
    pub heat_flux_activation: f64,
    #[serde(default = "default_one")]
    pub heat_load_activation: f64,
    #[serde(default = "default_one")]
    pub heat_flux_ramp_exponent: f64,
    #[serde(default = "default_one")]
    pub heat_load_ramp_exponent: f64,
}
```

Then add the field to `TomlGuidance`:

```rust
    /// Lateral guidance parameters (shared by unsigned-magnitude schemes)
    #[serde(default)]
    pub lateral: Option<TomlLateralParams>,
    /// Thermal safety limiter parameters (shared by unsigned-magnitude schemes)
    #[serde(default)]
    pub thermal_limiter: Option<TomlThermalLimiterParams>,
}
```

- [ ] **Step 3: Wire `thermal_limiter` into `GuidanceParams` construction**

In `src/rust/src/data/mod.rs`, in both the "with FTC params" and "without FTC params" branches of `GuidanceParams` construction, add the `thermal_limiter` field. After the `piecewise_constant` line:

```rust
                piecewise_constant: piecewise_constant_params.clone(),
                thermal_limiter: if let Some(ref tl) = toml.guidance.thermal_limiter {
                    ThermalLimiterParams {
                        heat_flux_activation: tl.heat_flux_activation,
                        heat_load_activation: tl.heat_load_activation,
                        heat_flux_ramp_exponent: tl.heat_flux_ramp_exponent,
                        heat_load_ramp_exponent: tl.heat_load_ramp_exponent,
                    }
                } else {
                    ThermalLimiterParams::default()
                },
```

Add the import at the top of `mod.rs`:

```rust
use crate::gnc::guidance::thermal_limiter::ThermalLimiterParams;
```

Apply this same pattern to **both** branches of the GuidanceParams construction (the "with FTC" branch around line 468-473 and the "without FTC" branch around line 516-521).

- [ ] **Step 4: Run Rust tests**

Run: `cd src/rust && cargo test`
Expected: All tests pass. Default params (activation=1.0) mean limiter is inert.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/guidance_params.rs src/rust/src/config.rs src/rust/src/data/mod.rs
git commit -m "add ThermalLimiterParams to GuidanceParams and TOML parsing"
```

---

### Task 4: Extend `NavigationOutput` with thermal fractions

**Files:**
- Modify: `src/rust/src/gnc/navigation/estimator.rs:52-73` (NavigationOutput struct)
- Modify: `src/rust/src/simulation/runner.rs` (compute fractions before guidance call)

- [ ] **Step 1: Add thermal fraction fields to `NavigationOutput`**

In `src/rust/src/gnc/navigation/estimator.rs`, add two fields at the end of `NavigationOutput` (before the closing brace):

```rust
    pub capture_time: f64,          // capture duration
    // Thermal state (for guidance limiter and NN inputs)
    pub heat_flux_fraction: f64,    // current_heat_flux / max_heat_flux (0.0 if no limit)
    pub heat_load_fraction: f64,    // cumulative_heat_load / max_heat_load (0.0 if no limit)
}
```

Since `NavigationOutput` derives `Default`, these will be `0.0` by default -- limiter sees no threat.

- [ ] **Step 2: Compute thermal fractions in runner.rs before guidance call**

In `src/rust/src/simulation/runner.rs`, after the navigation output is computed (after line 565: `};`) and before the guidance call (line 576: `let ftc_out = ftc::guidance_step(`), add:

```rust
            // Compute thermal fractions for guidance limiter + NN inputs.
            // Instantaneous heat flux uses the same formula as track_peak_values.
            {
                let (alt_for_thermal, _) = geodetic_from_spherical(
                    sim.state[0], sim.state[1], sim.state[2], planet,
                );
                let rho_thermal = data.atmosphere.density_at(alt_for_thermal)
                    * (1.0 + run_state.density_bias);
                let v_eff_thermal = effective_airspeed(
                    sim.state[3], sim.state[4], sim.state[5], sim.state[2],
                    alt_for_thermal, data, run_state,
                );
                let heat_flux_now = data.capsule.cq * rho_thermal.sqrt()
                    * v_eff_thermal.powf(3.05);

                nav_out.heat_flux_fraction = if data.constraints.max_heat_flux > 0.0 {
                    heat_flux_now / data.constraints.max_heat_flux
                } else {
                    0.0
                };
                nav_out.heat_load_fraction = if data.constraints.max_heat_load > 0.0 {
                    sim.state[6] / data.constraints.max_heat_load
                } else {
                    0.0
                };
            }
```

Note: `nav_out` must be declared as `let mut nav_out` for this to work. Check the existing code -- if `nav_out` is not `mut`, add `mut` to the declaration (the `let nav_out = match &mut nav_filter {` on line 518).

- [ ] **Step 3: Run Rust tests**

Run: `cd src/rust && cargo test`
Expected: All tests pass. Fractions default to 0.0, so behavior is unchanged.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/navigation/estimator.rs src/rust/src/simulation/runner.rs
git commit -m "extend NavigationOutput with thermal fractions, compute in runner"
```

---

### Task 5: Call thermal limiter from guidance dispatch

**Files:**
- Modify: `src/rust/src/gnc/guidance/ftc.rs:1-10` (imports)
- Modify: `src/rust/src/gnc/guidance/ftc.rs:173-176` (after longitudinal bank angle, before skip_lateral)

- [ ] **Step 1: Add thermal_limiter import**

In `src/rust/src/gnc/guidance/ftc.rs`, add to the existing imports at the top:

```rust
use crate::gnc::guidance::thermal_limiter;
```

- [ ] **Step 2: Call the limiter after longitudinal bank computation**

In `src/rust/src/gnc/guidance/ftc.rs`, after line 173 (`state.n_active += 1;`) and before line 176 (`let skip_lateral = matches!(`), insert:

```rust
    // === Thermal safety limiter (unsigned-magnitude schemes only) ===
    let uses_thermal_limiter = !matches!(
        guidance_type,
        GuidanceType::PiecewiseConstant | GuidanceType::NeuralNetwork
    );
    if uses_thermal_limiter && longitudinal_active == 1 && !is_reference {
        let cos_bank = bank_angle_longitudinal.cos();
        let cos_limited = thermal_limiter::apply_thermal_limit(
            cos_bank,
            nav.heat_flux_fraction,
            nav.heat_load_fraction,
            &data.guidance.thermal_limiter,
        );
        bank_angle_longitudinal = cos_limited.acos();
    }
```

- [ ] **Step 3: Run Rust tests**

Run: `cd src/rust && cargo test`
Expected: All tests pass. Default params (activation=1.0) mean the limiter never activates.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/ftc.rs
git commit -m "call thermal limiter from guidance dispatch for unsigned-magnitude schemes"
```

---

### Task 6: Extend NN inputs with thermal fractions

**Files:**
- Modify: `src/rust/src/gnc/guidance/neural.rs:21-62` (nn_bank_angle function)
- Modify: `src/rust/src/gnc/guidance/neural.rs:64-214` (tests)

- [ ] **Step 1: Extend NN input vector from 6 to 8**

In `src/rust/src/gnc/guidance/neural.rs`, replace the input array (lines 49-56):

```rust
    // 8 normalized inputs (6 orbital/aero + 2 thermal margins)
    let input = [
        orbit.eccentricity - 1.0,
        (orbit.inclination - target_inclination).to_degrees() * 3.0 / 5.0,
        2.0 * (velocity_radial / 1e3 + 1.2) / 1.5 - 1.0,
        -mu / (2.0 * orbit.semi_major_axis) / 6e6,
        (nav.velocity_estimated[0] / 3e3 - 1.5) * 2.0,
        accel_mag / 20.0 - 1.0,
        nav.heat_flux_fraction * 2.0 - 1.0,
        nav.heat_load_fraction * 2.0 - 1.0,
    ];
```

- [ ] **Step 2: Update test helper `zero_weight_nn` from 6 to 8 inputs**

In the test module, update `zero_weight_nn`:

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

- [ ] **Step 3: Update test `output_in_valid_range` network from 6 to 8 inputs**

Update `layer0` in the `output_in_valid_range` test:

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
```

And update `layer1` input count:

```rust
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

- [ ] **Step 4: Update proptest `fixed_small_nn` from 6 to 8 inputs**

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

- [ ] **Step 5: Run tests**

Run: `cd src/rust && cargo test neural -- --nocapture`
Expected: All NN tests pass with 8-input networks.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/guidance/neural.rs
git commit -m "extend NN guidance inputs from 6 to 8 with thermal fractions"
```

---

### Task 7: Update NN training TOML config

**Files:**
- Modify: `configs/training/msr_aller_nn_train_consolidated.toml`

- [ ] **Step 1: Update `layer_sizes` from 6 to 8 inputs**

In `configs/training/msr_aller_nn_train_consolidated.toml`, change:

```toml
[network]
layer_sizes = [8, 16, 64, 16, 2]
activations = ["asinh", "asinh", "asinh", "asinh"]
```

- [ ] **Step 2: Commit**

```bash
git add configs/training/msr_aller_nn_train_consolidated.toml
git commit -m "update NN training config to 8-input architecture for thermal inputs"
```

---

### Task 8: Python GA param space -- thermal limiter params

**Files:**
- Modify: `src/python/aerocapture/training/param_spaces.py`

- [ ] **Step 1: Add `_THERMAL_LIMITER_PARAMS` shared list**

In `src/python/aerocapture/training/param_spaces.py`, after the `_LATERAL_PARAMS` definition (line 40) and before the `PARAM_SPACES` dict (line 43), add:

```python
# Thermal safety limiter params shared by all unsigned-magnitude schemes.
# Prefixed with "thermal." so evaluate.py routes them to [guidance.thermal_limiter] in TOML.
_THERMAL_LIMITER_PARAMS: list[ParamSpec] = [
    ParamSpec("thermal.heat_flux_activation", 0.6, 0.95, 1.0),
    ParamSpec("thermal.heat_load_activation", 0.6, 0.95, 1.0),
    ParamSpec("thermal.heat_flux_ramp_exponent", 0.5, 3.0, 1.0),
    ParamSpec("thermal.heat_load_ramp_exponent", 0.5, 3.0, 1.0),
]
```

- [ ] **Step 2: Add `_THERMAL_LIMITER_PARAMS` to the 5 unsigned-magnitude schemes**

In the `PARAM_SPACES` dict, append `*_THERMAL_LIMITER_PARAMS` to each of the five scheme lists. For each scheme (`equilibrium_glide`, `energy_controller`, `pred_guid`, `fnpag`, `ftc`), add `*_THERMAL_LIMITER_PARAMS` after `*_EXIT_PARAMS`:

For example, `equilibrium_glide` becomes:

```python
    "equilibrium_glide": [
        ParamSpec("k_hdot_scale", 0.05, 1.0, 0.3),
        ParamSpec("v_ratio_threshold", 0.9, 1.5, 1.1),
        ParamSpec("velocity_bias_high", 0.0, 0.5, 0.15),
        ParamSpec("velocity_bias_low", 0.0, 1.0, 0.3),
        ParamSpec("alt_bias_threshold", 20.0, 80.0, 40.0),
        ParamSpec("cos_bank_min", -1.0, 0.0, -0.5),
        ParamSpec("cos_bank_max", 0.5, 1.0, 0.95),
        *_LATERAL_PARAMS,
        *_EXIT_PARAMS,
        *_THERMAL_LIMITER_PARAMS,
    ],
```

Apply the same pattern to `energy_controller`, `pred_guid`, `fnpag`, and `ftc`. Do NOT add to `piecewise_constant`.

- [ ] **Step 3: Run linter**

Run: `uv run ruff check src/python/aerocapture/training/param_spaces.py`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/param_spaces.py
git commit -m "add thermal limiter GA params to unsigned-magnitude scheme param spaces"
```

---

### Task 9: Python evaluate.py -- route thermal overrides to TOML

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py:361-380` (write_guidance_toml)

- [ ] **Step 1: Add `thermal.*` routing in `write_guidance_toml`**

In `src/python/aerocapture/training/evaluate.py`, in `write_guidance_toml` (around line 362-364), extend the param splitting to handle `thermal.*`. Replace the existing split block:

```python
    # Split lateral, exit, and thermal params from scheme-specific params
    lateral_params = {k.removeprefix("lateral."): v for k, v in params.items() if k.startswith("lateral.")}
    exit_params = {k.removeprefix("exit."): v for k, v in params.items() if k.startswith("exit.")}
    thermal_params = {k.removeprefix("thermal."): v for k, v in params.items() if k.startswith("thermal.")}
    scheme_params = {
        k: v for k, v in params.items()
        if not k.startswith("lateral.") and not k.startswith("exit.") and not k.startswith("thermal.")
    }
```

Then after the exit params merge block (after line 380: `toml_data["guidance"].setdefault("ftc", {}).update(exit_params)`), add:

```python
    # Merge thermal limiter params into [guidance.thermal_limiter]
    if thermal_params:
        toml_data["guidance"].setdefault("thermal_limiter", {}).update(thermal_params)
```

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/python/aerocapture/training/evaluate.py`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py
git commit -m "route thermal.* GA params to [guidance.thermal_limiter] in TOML"
```

---

### Task 10: Python tests -- param spaces and TOML routing

**Files:**
- Modify: `tests/test_toml_patching.py` (add thermal routing test)

- [ ] **Step 1: Add test for thermal params in param spaces**

Add a new test to `tests/test_toml_patching.py`:

```python
class TestThermalLimiterParams:
    """Thermal limiter params present in unsigned-magnitude schemes and route correctly."""

    UNSIGNED_SCHEMES = ["equilibrium_glide", "energy_controller", "pred_guid", "fnpag", "ftc"]

    @pytest.mark.parametrize("scheme", UNSIGNED_SCHEMES)
    def test_thermal_params_in_param_space(self, scheme: str) -> None:
        """All unsigned-magnitude schemes include thermal limiter params."""
        specs = PARAM_SPACES[scheme]
        thermal_names = {s.name for s in specs if s.name.startswith("thermal.")}
        expected = {
            "thermal.heat_flux_activation",
            "thermal.heat_load_activation",
            "thermal.heat_flux_ramp_exponent",
            "thermal.heat_load_ramp_exponent",
        }
        assert thermal_names == expected, f"scheme={scheme}: thermal params mismatch: {thermal_names}"

    def test_piecewise_constant_has_no_thermal_params(self) -> None:
        """Piecewise constant should NOT have thermal limiter params."""
        specs = PARAM_SPACES["piecewise_constant"]
        thermal_names = [s.name for s in specs if s.name.startswith("thermal.")]
        assert thermal_names == [], f"piecewise_constant should not have thermal params: {thermal_names}"

    @pytest.mark.parametrize("scheme", UNSIGNED_SCHEMES)
    def test_thermal_params_route_to_toml_section(self, scheme: str, tmp_path: Path) -> None:
        """thermal.* params end up in [guidance.thermal_limiter] in the patched TOML."""
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        chrom_len = len(specs) * config.ga.n_bit
        chrom = make_chromosome(chrom_len, strategy="mid")

        params = decode_params_from_chromosome(chrom, config)
        base_toml = TRAINING_CONFIGS[scheme]
        out_path = tmp_path / f"{scheme}_thermal.toml"
        written = write_guidance_toml(base_toml, scheme, params, output_path=out_path)

        with open(written, "rb") as f:
            parsed = tomllib.load(f)

        thermal_section = parsed.get("guidance", {}).get("thermal_limiter", {})
        assert "heat_flux_activation" in thermal_section, f"scheme={scheme}: heat_flux_activation missing"
        assert "heat_load_activation" in thermal_section, f"scheme={scheme}: heat_load_activation missing"
        assert "heat_flux_ramp_exponent" in thermal_section, f"scheme={scheme}: heat_flux_ramp_exponent missing"
        assert "heat_load_ramp_exponent" in thermal_section, f"scheme={scheme}: heat_load_ramp_exponent missing"
```

- [ ] **Step 2: Run the new tests**

Run: `uv run pytest tests/test_toml_patching.py -v`
Expected: All tests pass, including the new thermal tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_toml_patching.py
git commit -m "add Python tests for thermal limiter param spaces and TOML routing"
```

---

### Task 11: Full Rust build and test suite

**Files:** None (verification only)

- [ ] **Step 1: Run full Rust test suite**

Run: `cd src/rust && cargo test`
Expected: All tests pass (existing + new thermal_limiter tests).

- [ ] **Step 2: Run Rust lints**

Run: `cd src/rust && cargo clippy -- -D warnings && cargo fmt --check`
Expected: No warnings, no formatting issues.

- [ ] **Step 3: Run full Python test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass. Some tests that depend on `PARAM_SPACES` lengths (like `test_config.py`) will naturally adapt because they read `len(PARAM_SPACES[scheme])` dynamically.

- [ ] **Step 4: Run Python lints**

Run: `./lint_code.sh`
Expected: Clean (ruff + mypy).

---

### Task 12: Invoke smart-commit skill

- [ ] **Step 1: Invoke the `smart-commit` skill**

Take the whole `feature/exit-phase-guidance` branch into account and sync CLAUDE.md/README.md with the new thermal limiter functionality.
