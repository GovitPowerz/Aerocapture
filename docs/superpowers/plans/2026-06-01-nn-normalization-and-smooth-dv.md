# NN Input Pipeline v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify NN input normalization into a single per-input `{transform, scale, center}` schema (model-JSON-embedded, TOML-overridable, baked defaults) and redefine the 3 DV inputs to be defined + smooth across `e=1`, removing the sentinel.

**Architecture:** `build_nn_input` is split into *extraction* (input-specific raw scalars) + a uniform *normalization* loop driven by a per-input spec list resolved from the loaded model (or a baked default table). The DV inputs become plain `asinh` entries fed by a new smooth `maneuver::predicted_dv_for_nn`. Calibration writes the normalization block into the model instead of pasting Rust consts.

**Tech Stack:** Rust (nalgebra, serde, neural.rs/maneuver.rs), PyO3 (`aerocapture_rs`), Python (numpy), pytest, cargo test.

**Spec:** `docs/superpowers/specs/2026-06-01-nn-normalization-and-smooth-dv-design.md`

---

## File Structure

- `src/rust/src/data/neural.rs` — `NormTransform` enum + `NormSpec` struct + `apply_norm`; `DEFAULT_NORMALIZATION` const; `normalization` field on `NeuralNetModel` (resolved Vec) + `NnJsonFile` (Option); default-fill in all constructors; serialize on the write path.
- `src/rust/src/gnc/guidance/neural.rs` — refactor `build_nn_input` into extract-then-normalize; resolve specs from `data.neural_net` or `DEFAULT_NORMALIZATION`; swap in the smooth DV; delete sentinel + the old `S_*`/`C_*`/`H_*` const block (values move into `DEFAULT_NORMALIZATION`).
- `src/rust/src/orbit/maneuver.rs` — new `predicted_dv_for_nn`; `compute_deltav` untouched.
- `src/rust/src/config.rs` — parse `[network.normalization]` TOML override; overlay onto the model.
- `src/rust/aerocapture-py/src/lib.rs` — `flat_weights_to_json` embeds the normalization block.
- `src/python/aerocapture/training/calibrate_inputs.py` — emit/write a `{transform, scale, center}` block; drop `CURRENT_TRANSFORMS`/`drop_sentinel`/`_*_CONST_NAME`; read the current transform from the model/default to invert.
- `src/python/aerocapture/training/{model_io.py, rl/export.py}` — round-trip the `normalization` block.
- `tests/test_nn_scale_parity.py` — retire (single source of truth).

---

## Task 1: Normalization primitive — `NormTransform` + `NormSpec` + `apply_norm`

**Files:**
- Modify: `src/rust/src/data/neural.rs`
- Test: inline `#[cfg(test)]` in `src/rust/src/data/neural.rs`

- [ ] **Step 1: Write the failing test**

Add to the test module in `src/rust/src/data/neural.rs`:

```rust
#[test]
fn apply_norm_divisor_forms() {
    // none: (raw - center)/scale
    assert!((apply_norm(50.0, &NormSpec { transform: NormTransform::None, scale: 0.5, center: 0.5 }) - 99.0).abs() < 1e-12);
    // asinh: asinh((raw - center)/scale)
    let got = apply_norm(880.0, &NormSpec { transform: NormTransform::Asinh, scale: 880.0, center: 0.0 });
    assert!((got - 1.0_f64.sinh().asinh()).abs() < 1e-12); // asinh(1.0)
    // tanh: tanh((raw - center)/scale)
    let got = apply_norm(30.0, &NormSpec { transform: NormTransform::Tanh, scale: 30.0, center: 0.0 });
    assert!((got - 1.0_f64.tanh()).abs() < 1e-12);
    // raw passthrough
    assert!((apply_norm(0.3, &NormSpec { transform: NormTransform::None, scale: 1.0, center: 0.0 }) - 0.3).abs() < 1e-12);
}

#[test]
fn default_normalization_has_full_width() {
    assert_eq!(DEFAULT_NORMALIZATION.len(), NN_FULL_INPUT_SIZE);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd src/rust && cargo test --lib apply_norm_divisor default_normalization_has -- --nocapture`
Expected: FAIL — types/const undefined.

- [ ] **Step 3: Add the types + helper (DEFAULT_NORMALIZATION added in Task 4 — for now a stub sized to the const)**

In `src/rust/src/data/neural.rs`, add near the top (after imports):

```rust
/// Per-input normalization transform applied after the affine `(raw - center)/scale`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormTransform {
    #[default]
    None,
    Asinh,
    Tanh,
}

/// Uniform per-input normalization: `norm = transform((raw - center) / scale)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormSpec {
    pub transform: NormTransform,
    pub scale: f64,
    pub center: f64,
}

impl Default for NormSpec {
    fn default() -> Self {
        Self { transform: NormTransform::None, scale: 1.0, center: 0.0 }
    }
}

#[inline]
pub fn apply_norm(raw: f64, spec: &NormSpec) -> f64 {
    let v = (raw - spec.center) / spec.scale;
    match spec.transform {
        NormTransform::None => v,
        NormTransform::Asinh => v.asinh(),
        NormTransform::Tanh => v.tanh(),
    }
}
```

For `default_normalization_has_full_width` to pass now, add a temporary all-default table (real values land in Task 4):

```rust
pub const DEFAULT_NORMALIZATION: [NormSpec; NN_FULL_INPUT_SIZE] =
    [NormSpec { transform: NormTransform::None, scale: 1.0, center: 0.0 }; NN_FULL_INPUT_SIZE];
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd src/rust && cargo test --lib apply_norm_divisor default_normalization_has && cargo clippy --lib -- -D warnings`
Expected: PASS, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): add NormSpec/NormTransform + apply_norm (divisor form)"
```

---

## Task 2: Smooth, always-defined DV — `maneuver::predicted_dv_for_nn`

**Files:**
- Modify: `src/rust/src/orbit/maneuver.rs`
- Test: inline `#[cfg(test)]` in `src/rust/src/orbit/maneuver.rs`

- [ ] **Step 1: Write the failing tests**

Add to (or create) the `#[cfg(test)]` module in `src/rust/src/orbit/maneuver.rs`. These use `OrbitalElements`, `OrbitalTarget`, `ParkingOrbit`, `PlanetConfig::mars()` — mirror how existing maneuver tests (if any) or `elements` tests construct these; build a minimal `OrbitalElements` by hand.

```rust
use crate::config::PlanetConfig;
use crate::data::{OrbitalElements, OrbitalTarget, ParkingOrbit};

fn mk_orbit(sma: f64, ecc: f64, incl: f64, planet: &PlanetConfig) -> OrbitalElements {
    let a = sma;
    let rp = a * (1.0 - ecc);
    let ra = a * (1.0 + ecc);
    OrbitalElements {
        semi_major_axis: a,
        eccentricity: ecc,
        inclination: incl,
        periapsis_alt: rp - planet.equatorial_radius,
        apoapsis_alt: ra - planet.equatorial_radius,
        arg_periapsis: 0.0,
        ..Default::default()
    }
}

fn parking() -> ParkingOrbit {
    ParkingOrbit { apoapsis: 500_000.0, periapsis: 300_000.0, ..Default::default() }
}
fn target() -> OrbitalTarget {
    OrbitalTarget { semi_major_axis: 3.796e6 + 400_000.0, eccentricity: 0.05, inclination: 0.9, ..Default::default() }
}

#[test]
fn predicted_dv_finite_for_elliptical_and_hyperbolic() {
    let p = PlanetConfig::mars();
    for ecc in [0.2_f64, 0.8, 1.2, 2.0] {
        let sma = if ecc < 1.0 { 5.0e6 } else { -5.0e6 }; // a<0 for hyperbolic
        let o = mk_orbit(sma, ecc, 0.8, &p);
        let dv = predicted_dv_for_nn(&o, &target(), &parking(), &p);
        assert!(dv[0].is_finite() && dv[1].is_finite() && dv[2].is_finite(), "ecc={ecc} -> {dv:?}");
    }
}

#[test]
fn predicted_dv2_is_zero_when_hyperbolic() {
    let p = PlanetConfig::mars();
    let o = mk_orbit(-5.0e6, 1.5, 0.8, &p);
    let dv = predicted_dv_for_nn(&o, &target(), &parking(), &p);
    assert_eq!(dv[1], 0.0, "dv2 must be 0 for hyperbolic, got {}", dv[1]);
}

#[test]
fn predicted_dv1_continuous_across_e1() {
    // Sweep a fixed-periapsis family through e=1; dv1 must not jump.
    let p = PlanetConfig::mars();
    let rp = 3.796e6 + 50_000.0; // fixed periapsis radius
    let mut prev: Option<f64> = None;
    // a = rp/(1-e); step e from 0.98 to 1.02 around the boundary.
    for k in 0..=40 {
        let e = 0.98 + 0.001 * k as f64;
        let a = rp / (1.0 - e); // a>0 for e<1, a<0 for e>1, |a|->inf at e=1
        let o = mk_orbit(a, e, 0.8, &p);
        let dv1 = predicted_dv_for_nn(&o, &target(), &parking(), &p)[0];
        assert!(dv1.is_finite(), "e={e} dv1 not finite");
        if let Some(pv) = prev {
            assert!((dv1 - pv).abs() < 50.0, "dv1 jump at e={e}: {pv} -> {dv1}");
        }
        prev = Some(dv1);
    }
}
```

NOTE: the `OrbitalElements`/`OrbitalTarget`/`ParkingOrbit` field names + `Default` derive must match the real structs — read `src/rust/src/orbit/elements.rs` and `src/rust/src/data/mod.rs` and adjust the constructors (e.g. if `Default` isn't derived, fill every field). The `50.0` jump tolerance assumes m/s; loosen only if a real continuous function legitimately exceeds it (it should not).

- [ ] **Step 2: Run to verify it fails**

Run: `cd src/rust && cargo test --lib predicted_dv -- --nocapture`
Expected: FAIL — `predicted_dv_for_nn` undefined.

- [ ] **Step 3: Implement `predicted_dv_for_nn`**

Add to `src/rust/src/orbit/maneuver.rs`:

```rust
/// NN-input correction-DV: signed components, defined + smooth across e=1.
/// Distinct from `compute_deltav` (which is the terminal maneuver plan).
/// - dv1: energy-closing burn at current periapsis (vis-viva) -> "Δv to close the orbit".
/// - dv2: periapsis-correction at apoapsis; 0 when hyperbolic (continuous limit).
/// - dv3: inclination plane change (same as compute_deltav).
pub fn predicted_dv_for_nn(
    orbit: &OrbitalElements,
    target: &OrbitalTarget,
    parking: &ParkingOrbit,
    planet: &PlanetConfig,
) -> [f64; 3] {
    let mu = planet.mu;
    let req = planet.equatorial_radius;
    let a = orbit.semi_major_axis;
    let rp = req + orbit.periapsis_alt;
    let ra_t = req + parking.apoapsis;
    let rp_t = req + parking.periapsis;

    // dv1: burn at current periapsis to bring apoapsis (energy) to target apoapsis.
    // vis-viva v = sqrt(mu (2/r - 1/a)); a<0 (hyperbolic) -> higher speed.
    let dv1 = if rp > 0.0 && a.abs() > 0.0 {
        let v_cur = (mu * (2.0 / rp - 1.0 / a)).max(0.0).sqrt();
        let a_t1 = (rp + ra_t) / 2.0;
        let v_tgt = (mu * (2.0 / rp - 1.0 / a_t1)).max(0.0).sqrt();
        v_cur - v_tgt
    } else {
        0.0
    };

    // dv2: periapsis-correction at apoapsis (apoapsis-referenced). 0 for hyperbolic.
    let rapoge = req + orbit.apoapsis_alt;
    let dv2 = if orbit.eccentricity < 1.0 && rapoge.is_finite() && rapoge > 0.0 {
        let vitfin1 = (2.0 * mu * rp_t / (rapoge * (rapoge + rp_t))).sqrt();
        let vitini1 = (2.0 * mu * rp / (rapoge * (rapoge + rp))).sqrt();
        vitfin1 - vitini1
    } else {
        0.0
    };

    // dv3: inclination plane change (mirror compute_deltav's dv3 exactly).
    let target_sma = target.semi_major_axis;
    let target_ecc = target.eccentricity;
    let pi = std::f64::consts::PI;
    let anoneu = [2.0 * pi - orbit.arg_periapsis, pi - orbit.arg_periapsis];
    let mut vitneu = [0.0_f64; 2];
    for i in 0..2 {
        let rayneu = target_sma * (1.0 - target_ecc * target_ecc) / (1.0 + target_ecc * anoneu[i].cos());
        vitneu[i] = (2.0 * mu * (1.0 / rayneu - 1.0 / (2.0 * target_sma))).sqrt();
    }
    let dincli = (target.inclination - orbit.inclination).abs();
    let dv3 = 2.0 * vitneu[0].min(vitneu[1]) * (dincli / 2.0).sin();

    [dv1, dv2, dv3]
}
```

(The `.max(0.0)` guards on the vis-viva radicands defend against tiny-negative-from-roundoff; for valid geometry they're no-ops.)

- [ ] **Step 4: Run to verify it passes**

Run: `cd src/rust && cargo test --lib predicted_dv -- --nocapture && cargo clippy --lib -- -D warnings`
Expected: PASS (3 tests), clippy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/orbit/maneuver.rs
git commit -m "feat(nn): smooth always-defined predicted_dv_for_nn (energy-closing dv1, dv2->0 hyperbolic)"
```

---

## Task 3: Characterization test — lock current `build_nn_input` output for indices 0-31

**Files:**
- Test: inline `#[cfg(test)]` in `src/rust/src/gnc/guidance/neural.rs`

This snapshots current behavior BEFORE the refactor so Task 4 can prove the normalization rewrite is bit-identical for the non-DV indices.

- [ ] **Step 1: Write the characterization test against CURRENT code**

Add to the test module in `src/rust/src/gnc/guidance/neural.rs`. Build the input two ways for two fixtures (the existing `test_nav()`/`test_sim_data_with_ref_traj()` and a reduced-velocity elliptical variant) and store the first 32 values as the golden snapshot:

```rust
#[test]
fn build_nn_input_characterization_0_to_31() {
    let planet = PlanetConfig::mars();
    let data = test_sim_data_with_ref_traj();
    let full_mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
    // Two fixtures: default (hyperbolic-ish) and reduced-velocity (elliptical).
    for scale_v in [1.0_f64, 0.45] {
        let mut nav = test_nav();
        nav.velocity_estimated[0] *= scale_v;
        let inp = build_nn_input(
            &nav, Some(&full_mask), None, 0.0, &data, &planet,
            50.0_f64.to_radians(), 0.0, Some(0.01), 0.2, 12.0, 3.0, 0.15,
        );
        // Print the snapshot so Task 4 can paste the expected array if needed.
        eprintln!("CHAR scale_v={scale_v}: {:?}", &inp[0..32]);
        // Assert finite + in a sane band (real lock happens in Task 4 via the
        // refactor reproducing these exact values).
        for (i, v) in inp[0..32].iter().enumerate() {
            assert!(v.is_finite(), "input[{i}] not finite");
        }
    }
}
```

- [ ] **Step 2: Run it green against current code + capture the snapshot**

Run: `cd src/rust && cargo test --lib build_nn_input_characterization -- --nocapture`
Expected: PASS. **Copy the two printed `CHAR …` arrays** — these are the bit-exact current values for indices 0-31. In Task 4, after the refactor, the same call must reproduce them.

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/gnc/guidance/neural.rs
git commit -m "test(nn): characterization snapshot of build_nn_input[0..31] pre-refactor"
```

---

## Task 4: `DEFAULT_NORMALIZATION` table + model field + refactor `build_nn_input`

**Files:**
- Modify: `src/rust/src/data/neural.rs` (real `DEFAULT_NORMALIZATION`, `normalization` field on `NeuralNetModel` + `NnJsonFile`, default-fill in constructors)
- Modify: `src/rust/src/gnc/guidance/neural.rs` (extract-then-normalize refactor, swap DV, delete sentinel + const block)
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Replace the stub `DEFAULT_NORMALIZATION` with the real table**

In `src/rust/src/data/neural.rs`, replace the Task-1 stub with the exact encoding of today's transforms (divisor form `{transform, scale, center}`). DV entries (32-34) use the current `S_DV1/2/3` as provisional scales (recalibrated in Task 6):

```rust
pub const DEFAULT_NORMALIZATION: [NormSpec; NN_FULL_INPUT_SIZE] = {
    use NormTransform::{Asinh, None as N, Tanh};
    const A: fn(f64) -> NormSpec = |s| NormSpec { transform: Asinh, scale: s, center: 0.0 };
    const AF: fn(f64, f64) -> NormSpec = |scale, center| NormSpec { transform: N, scale, center };
    const ID: NormSpec = NormSpec { transform: N, scale: 1.0, center: 0.0 };
    [
        AF(0.8754754, 0.9125593),   // 0 ecc_excess
        AF(1.443277, -1.167222),    // 1 inclination_error (deg)
        A(8.794982e2),              // 2 radial_velocity
        A(5.180226e6),              // 3 orbital_energy
        AF(1178.859, 4534.045),     // 4 velocity
        A(2.494108e1),              // 5 accel_magnitude
        AF(0.4524197, 0.4533209),   // 6 heat_flux_fraction
        AF(0.4363704, 0.4366122),   // 7 heat_load_fraction
        AF(43.24290, 82.93086),     // 8 altitude
        AF(0.1246266, -0.05801090), // 9 fpa
        AF(0.2803614, 0.2875094),   // 10 latitude
        A(2.367649e1),              // 11 drag_accel
        A(7.841004e0),              // 12 lift_accel
        A(2.396120e7),              // 13 sma_error
        A(4.752185e7),              // 14 apoapsis_alt
        AF(0.5, 0.5),               // 15 bounce_flag (flag*2-1)
        ID,                         // 16 cos_bank_nominal (raw)
        AF(808.8315, 812.3864),     // 17 pdyn_nominal
        A(7.416992e2),              // 18 hdot_nominal
        A(3.373053e2),              // 19 pdyn_error
        AF(std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2), // 20 exit_bank_teacher
        AF(0.1, 0.0),               // 21 inclination_err_rate (deg/s * 10)
        AF(std::f64::consts::PI, 0.0), // 22 prev_bank_signed
        NormSpec { transform: Tanh, scale: 30.0, center: 0.0 },  // 23 time_since_sign_flip
        NormSpec { transform: Tanh, scale: 100.0, center: 0.0 }, // 24 inclination_err_integral (deg·s)
        ID, ID, ID, ID, ID, ID,     // 25-30 sin/cos pairs (identity after extraction)
        A(3.750782e4),              // 31 periapsis_alt
        A(1.052305e2),              // 32 predicted_dv1 (provisional; recalibrated Task 6)
        A(1.046783e3),              // 33 predicted_dv2 (provisional)
        A(1.254637e2),              // 34 predicted_dv3 (provisional)
    ]
};
```

(If `const fn`/closures in const aren't accepted by the toolchain, expand each entry to a literal `NormSpec { … }`. Verify it compiles.)

- [ ] **Step 2: Add `normalization` to the model + JSON struct + constructors**

In `NeuralNetModel` (line ~1160) add:
```rust
    /// Per-candidate-input normalization (length NN_FULL_INPUT_SIZE). Defaults to
    /// DEFAULT_NORMALIZATION when the JSON omits the block.
    pub normalization: Vec<NormSpec>,
```
In `NnJsonFile` (line ~1002) add:
```rust
    #[serde(default)]
    normalization: Option<Vec<NormSpec>>,
```
In EVERY `NeuralNetModel` constructor (`from_v2_json`, the v1 path in `from_json_str`, `from_flat_weights_v2`) set the field via a shared helper:
```rust
fn resolve_normalization(block: Option<Vec<NormSpec>>) -> Vec<NormSpec> {
    match block {
        Some(v) if v.len() == NN_FULL_INPUT_SIZE => v,
        _ => DEFAULT_NORMALIZATION.to_vec(),
    }
}
```
`from_v2_json`/v1: `normalization: resolve_normalization(file.normalization)`. `from_flat_weights_v2` (PSO path, no JSON block): `normalization: DEFAULT_NORMALIZATION.to_vec()`.

- [ ] **Step 3: Write the failing refactor test (backward-compat + override)**

Add to the test module in `src/rust/src/gnc/guidance/neural.rs`:

```rust
#[test]
fn refactor_preserves_inputs_0_to_31() {
    // The refactored build_nn_input (using DEFAULT_NORMALIZATION when no model)
    // must reproduce the Task-3 characterization values bit-for-bit for 0..31.
    let planet = PlanetConfig::mars();
    let data = test_sim_data_with_ref_traj();
    let full_mask: Vec<usize> = (0..NN_FULL_INPUT_SIZE).collect();
    let mut nav = test_nav();
    nav.velocity_estimated[0] *= 0.45;
    let inp = build_nn_input(
        &nav, Some(&full_mask), None, 0.0, &data, &planet,
        50.0_f64.to_radians(), 0.0, Some(0.01), 0.2, 12.0, 3.0, 0.15,
    );
    // Spot-check a few transformed indices against the explicit formula:
    let orbit = elements::from_spherical(
        nav.position_estimated[0], nav.position_estimated[1], nav.position_estimated[2],
        nav.velocity_estimated[0], nav.velocity_estimated[1], nav.velocity_estimated[2], &planet);
    assert!((inp[0] - (orbit.eccentricity - 0.9125593) / 0.8754754).abs() < 1e-9, "ecc affine");
    let drag = nav.acceleration_estimated[0];
    assert!((inp[11] - (drag / 2.367649e1).asinh()).abs() < 1e-9, "drag asinh");
}
```
(Also keep the existing `radial_velocity_input_is_asinh_of_raw` / `ecc_excess_input_is_calibrated_affine` / `pdyn_error_input_is_asinh_of_raw` tests — they must still pass through the refactor.)

- [ ] **Step 4: Run to verify it fails (pre-refactor build_nn_input still uses inline transforms — should still pass actually; the real gate is Step 5 not breaking it)**

Run: `cd src/rust && cargo test --lib refactor_preserves_inputs -- --nocapture`
Expected: PASS already (the formula matches current inline code). This test is the invariant the refactor must keep green.

- [ ] **Step 5: Refactor `build_nn_input` — extract then normalize, swap DV, delete sentinel**

In `src/rust/src/gnc/guidance/neural.rs`:
1. Keep all the *extraction* logic that computes raw physical quantities, but write each raw value into a `let mut raw = [0.0_f64; NN_FULL_INPUT_SIZE];` instead of the normalized `full_input`. Specifically the raw values are:
   - `raw[0]=orbit.eccentricity`, `raw[1]=(orbit.inclination - target_inclination).to_degrees()`, `raw[2]=velocity_radial`, `raw[3]=-mu/(2.0*orbit.semi_major_axis)`, `raw[4]=nav.velocity_estimated[0]`, `raw[5]=accel_mag`, `raw[6]=nav.heat_flux_fraction`, `raw[7]=nav.heat_load_fraction`, `raw[8]=altitude_km`, `raw[9]=nav.velocity_estimated[1]`, `raw[10]=nav.position_estimated[2]`, `raw[11]=nav.acceleration_estimated[0]`, `raw[12]=nav.acceleration_estimated[1]`, `raw[13]=nav.orbital_errors[0]`, `raw[14]=orbit.apoapsis_alt`, `raw[15]=nav.bounce_flag as f64`, `raw[16]=cos_bank_nominal`, `raw[17]=pdyn_nominal`, `raw[18]=hdot_nominal`, `raw[19]=pdyn_error`, `raw[20]=exit_bank`, `raw[21]=di_err_dt.to_degrees()` (with the `None=>0.0` branch preserved), `raw[22]=prev_bank_signed`, `raw[23]=time_since_last_sign_flip`, `raw[24]=inclination_error_integral.to_degrees()`, `raw[25]=exit_bank.sin()`, `raw[26]=exit_bank.cos()`, `raw[27]=prev_bank_signed.sin()`, `raw[28]=prev_bank_signed.cos()`, `raw[29]=prev_realized_bank.sin()`, `raw[30]=prev_realized_bank.cos()`, `raw[31]=orbit.periapsis_alt`.
   - **DV (32-34):** replace the entire sentinel block with `let dv = maneuver::predicted_dv_for_nn(&orbit, &data.target_orbit, &data.parking_orbit, planet); raw[32]=dv[0]; raw[33]=dv[1]; raw[34]=dv[2];`
2. Resolve the spec list and normalize uniformly:
   ```rust
   let norm: &[NormSpec] = data
       .neural_net
       .as_ref()
       .map(|m| m.normalization.as_slice())
       .unwrap_or(&crate::data::neural::DEFAULT_NORMALIZATION);
   let mut full_input = [0.0_f64; NN_FULL_INPUT_SIZE];
   for i in 0..NN_FULL_INPUT_SIZE {
       full_input[i] = crate::data::neural::apply_norm(raw[i], &norm[i]);
   }
   ```
   (Bring `apply_norm`, `NormSpec`, `DEFAULT_NORMALIZATION` into scope via `use`.)
3. Keep the ablation logic (`full_input[idx] = ablated_value`) and the mask-select/return logic AFTER normalization, unchanged.
4. DELETE the now-unused `const S_RADIAL_VELOCITY … S_DV3`, `const C_*`, `const H_*`, `const DV_SENTINEL_NORM` block (their values now live in `DEFAULT_NORMALIZATION`). Also remove the now-unused per-index inline transform expressions.

- [ ] **Step 6: Run the full neural suite + the DV tests + characterization**

Run: `cd src/rust && cargo test --lib neural -- --nocapture 2>&1 | tail -20`
Expected: PASS. `refactor_preserves_inputs_0_to_31`, the radial/ecc/pdyn_error value tests, and the existing `dv_inputs_*` tests must pass. The two `dv_inputs_sentinel_when_hyperbolic` / `dv_inputs_live_when_elliptical` tests reference the deleted sentinel/`S_DV*`; UPDATE them: hyperbolic now expects `inp[33] (dv2) == apply_norm(0.0, &DEFAULT_NORMALIZATION[33])` and `inp[32]`/`inp[34]` = the asinh of the (now defined) dv1/dv3; or simpler, assert finiteness + that dv2's raw is 0 when hyperbolic (via `predicted_dv_for_nn`). Rewrite them to match the new semantics (no sentinel).

- [ ] **Step 7: clippy + fmt + verify neural golden unchanged**

Run:
```bash
cd src/rust && cargo fmt && cargo clippy --lib --all-targets -- -D warnings
cargo test --test guidance_regression 2>&1 | tail -8
```
Expected: clippy clean; guidance_regression 6/6 PASS. The `neural` golden uses indices 0-15 only (input_mask=None) and the refactor preserves those bit-for-bit → golden unchanged. If it changed, STOP — the default table or extraction has a discrepancy; diff against the Task-3 snapshot.

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/data/neural.rs src/rust/src/gnc/guidance/neural.rs
git commit -m "feat(nn): extract-then-normalize build_nn_input via NormSpec + smooth DV, remove sentinel"
```

---

## Task 5: TOML `[network.normalization]` override

**Files:**
- Modify: `src/rust/src/config.rs`
- Test: inline `#[cfg(test)]` in `src/rust/src/config.rs`

- [ ] **Step 1: Write the failing test**

In `src/rust/src/config.rs` test module, add a test that a TOML `[network]` with a `normalization` array parses into `Vec<NormSpec>` and overrides the model's block. Mirror the existing `input_mask`/`output_parameterization` override tests (find one with `rg -n "output_parameterization|input_mask" src/rust/src/config.rs` and copy its structure). Assert e.g. a normalization entry `{transform="asinh", scale=10.0, center=0.0}` round-trips.

```rust
#[test]
fn toml_normalization_override_parses() {
    let toml_str = r#"
        [network]
        normalization = [ { transform = "asinh", scale = 10.0, center = 0.0 } ]
    "#;
    let parsed: TomlNetwork = toml::from_str(toml_str).unwrap();
    let n = parsed.normalization.unwrap();
    assert_eq!(n.len(), 1);
    assert_eq!(n[0].scale, 10.0);
}
```
(Use the actual TOML network struct name — find it via `rg -n "struct TomlNetwork|input_mask" src/rust/src/config.rs`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd src/rust && cargo test --lib toml_normalization_override -- --nocapture`
Expected: FAIL — field absent.

- [ ] **Step 3: Add the field + overlay**

In the TOML network struct in `config.rs`, add `#[serde(default)] pub normalization: Option<Vec<crate::data::neural::NormSpec>>` (NormSpec already derives Deserialize). Where the loaded model gets TOML overrides applied (near the `input_mask`/`output_parameterization` overlay, ~`src/rust/src/data/mod.rs:678` area — find it), add: if `toml.network.normalization` is `Some(v)` and `v.len() == NN_FULL_INPUT_SIZE`, set `model.normalization = v` (else error with a clear length message).

- [ ] **Step 4: Run to verify it passes + full config tests**

Run: `cd src/rust && cargo test --lib config -- --nocapture && cargo clippy --lib -- -D warnings`
Expected: PASS, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/config.rs src/rust/src/data/mod.rs
git commit -m "feat(nn): TOML [network.normalization] override onto model"
```

---

## Task 6: Recalibrate DV scales + Python write path + retire dual-maintenance

**Files:**
- Modify: `src/rust/aerocapture-py/src/lib.rs` (`flat_weights_to_json` embeds normalization)
- Modify: `src/python/aerocapture/training/calibrate_inputs.py`
- Modify: `src/python/aerocapture/training/model_io.py`, `src/python/aerocapture/training/rl/export.py`
- Delete: `tests/test_nn_scale_parity.py`

- [ ] **Step 1: Rebuild PyO3 + recalibrate the DV scales for the redefined DV**

The DV inputs were redefined (Task 2), so their distributions changed — the provisional `S_DV1/2/3` in `DEFAULT_NORMALIZATION` are stale. Rebuild and recalibrate:
```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
uv run python -m aerocapture.training.calibrate_inputs \
  --toml configs/training/msr_aller_nn_train_consolidated.toml --n-sims 500 --output /tmp/dvcal.txt
grep -E "predicted_dv|S_DV" /tmp/dvcal.txt
```
Paste the new `S_DV1/2/3` (the asinh scales for the 3 DV inputs) into `DEFAULT_NORMALIZATION[32..35]` in `neural.rs`, rebuild PyO3, and re-run to confirm the DV inputs now report low out-of-range in `nn_input_report` (no more 21% sentinel saturation — the DV is meaningful throughout). NOTE: `calibrate_inputs.py` still has the OLD `CURRENT_TRANSFORMS` at this point; the DV entries there must temporarily match `DEFAULT_NORMALIZATION` (asinh, the new scales) for the inversion to be correct, OR do this recalibration AFTER Step 2 rewires calibration to read the default table. Prefer reordering: do Step 2 first, then this recalibration.

- [ ] **Step 2: Rewire `calibrate_inputs.py` to the unified schema (single source of truth)**

Rewrite `calibrate_inputs.py` so it:
1. Reads the CURRENT per-input transform from a single source — export `DEFAULT_NORMALIZATION` from Rust via a new `aerocapture_rs.default_normalization()` PyO3 helper (returns a list of `{transform, scale, center}` dicts), and use THAT to invert normalized→raw (replacing the hand-maintained `CURRENT_TRANSFORMS`). This closes the dual-maintenance loop.
2. Emits the calibrated block as `{transform, scale, center}` entries (the model-JSON form), not Rust consts. Provide `--write-model PATH` to write the `normalization` array directly into a model JSON's top-level `normalization` field.
3. Deletes `CURRENT_TRANSFORMS`, `_ASINH_CONST_NAME`, `_AFFINE_CONST_NAME`, `drop_sentinel`, `_DV_INDICES` (no sentinel anymore — DV calibrates over its full, now-meaningful distribution).
4. Add a PyO3 helper in `src/rust/aerocapture-py/src/lib.rs`: `default_normalization() -> Vec<PyDict>` reading `aerocapture::data::neural::DEFAULT_NORMALIZATION`.

Update `tests/test_calibrate_inputs.py`: keep `derive_asinh_scale`/`derive_affine` tests; drop `invert_transform`/`drop_sentinel`/`affine_ch` tests that referenced the removed machinery (or repoint `invert_transform` to use the default-normalization-derived transform). Add a test that `default_normalization()` returns `NN_FULL_INPUT_SIZE` entries matching the expected transforms for a few indices (e.g. idx 11 asinh, idx 0 none).

- [ ] **Step 3: `flat_weights_to_json` embeds the normalization block**

In `src/rust/aerocapture-py/src/lib.rs::flat_weights_to_json`, after building the model JSON, add a top-level `"normalization"` array = `DEFAULT_NORMALIZATION` serialized (so PSO-deployed `best_model.json` is self-describing). If the function already takes the source model/config, prefer threading any override; else default-embed `DEFAULT_NORMALIZATION`.

- [ ] **Step 4: model_io / export round-trip**

In `src/python/aerocapture/training/model_io.py` and `rl/export.py`, ensure the `normalization` field is preserved on load and written on export (pass-through; no transformation). Add a round-trip test: load a model JSON with a `normalization` block, export, reload, assert the block is byte-stable.

- [ ] **Step 5: Retire the parity test + run Python suite**

```bash
git rm tests/test_nn_scale_parity.py
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
uv run pytest tests/test_calibrate_inputs.py tests/test_pyo3.py tests/test_collect_nn_inputs.py -q
./lint_code.sh
```
Expected: PASS, ruff+mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/rust/aerocapture-py/src/lib.rs src/python/aerocapture/training/calibrate_inputs.py src/python/aerocapture/training/model_io.py src/python/aerocapture/training/rl/export.py tests/test_calibrate_inputs.py src/rust/src/data/neural.rs
git rm tests/test_nn_scale_parity.py
git commit -m "feat(nn): normalization single-source-of-truth (model JSON), recalibrate DV scales, retire parity test"
```

---

## Task 7: Full verification + golden/equivalence sweep

**Files:** tests + goldens as needed.

- [ ] **Step 1: Cross-language equivalence still holds**

Run: `uv run pytest tests/test_v2_rust_python_equivalence.py -q`
Expected: PASS — these feed explicit input vectors to `nn_forward` (layer math), unaffected by `build_nn_input`/normalization changes. If any asserts a candidate width or normalization, update it.

- [ ] **Step 2: Full Rust + Python suites**

Run:
```bash
./check_all.sh
uv run pytest tests -q
```
Expected: Rust green (test/fmt/clippy/build); Python green except the pre-existing `test_save_dir_resolution` (the user's deploy-dir choice, unrelated). The `neural` golden + 5 non-NN goldens stay bit-identical (DV redefinition doesn't touch the 16-input golden). If a golden changed unexpectedly, STOP and diff — it signals a refactor bug.

- [ ] **Step 3: Confirm models need retrain (handoff note)**

The DV inputs changed semantics, so the three deployed models (atan2/scaled_pi/delta) must be retrained against the new DV. Document in the handoff. (Existing models load fine via backward-compat — no `normalization` block → `DEFAULT_NORMALIZATION` — but their learned weights expect the old sentinel-based DV, so they'll underperform until retrained.)

- [ ] **Step 4: Commit any golden/test updates**

```bash
git add -A -- tests src/rust/tests
git commit -m "test(nn): verify goldens + equivalence after normalization/DV refactor"
```
(Only if there were changes; skip if clean.)

---

## Task 8: Sync docs + commit the branch

- [ ] **Step 1: smart-commit the whole branch**

Invoke the `smart-commit` skill, telling it to take the whole branch into account: CLAUDE.md needs the unified `{transform, scale, center}` normalization (model-JSON-embedded + TOML override + `DEFAULT_NORMALIZATION`), the redefined smooth `predicted_dv_for_nn` (no sentinel), the retired `CURRENT_TRANSFORMS`/parity test, and `calibrate_inputs.py` writing the block. README's NN row may need a touch (normalization now self-describing in the model).

---

## Self-Review Notes

- **Spec coverage:** unified schema (T1/T4/T5), divisor form `{transform,scale,center}` (T1), model-JSON-embedded + TOML override + defaults (T4/T5), smooth DV dv1/dv2/dv3 (T2), sentinel removal cascade (T4/T6), calibration writes block + single source of truth + parity retired (T6), build_nn_input extract-then-normalize (T4), golden bit-identical (T4/T7), retrain note (T7). Covered.
- **DV-scale recalibration** (T6 Step 1) is data-derived (a calibration run), not a hardcoded value — same pattern as the v2 plan; the characterization test + golden guard the non-DV indices.
- **Ordering caveat** flagged in T6S1: do the calibration-rewire (T6S2) before/with the DV recalibration so inversion uses the right transforms.
- **Retrain is the user's step**; plan ends at a tested, committed, backward-compatible pipeline.
