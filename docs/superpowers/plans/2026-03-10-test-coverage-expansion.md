# Test Coverage Expansion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand test coverage with risk-weighted unit tests and property-based testing across both Rust simulator and Python GA training pipeline.

**Architecture:** Infrastructure-first approach — build shared test helpers (Rust `tests/common/` modules, Python `conftest.py` + fixtures), then add guidance algorithm unit tests (Rust), training pipeline tests (Python), and cross-cutting error-path tests. Property-based testing via `proptest` (Rust) and `hypothesis` (Python).

**Tech Stack:** Rust (`proptest`, `approx`, `rstest`), Python (`hypothesis`, `pytest`)

**Critical codebase conventions (read before implementing):**
- `Planet` is an **enum** (`Planet::Mars`), not a struct. Constants come from methods: `planet.equatorial_radius()`, `planet.mu()`, `planet.omega()`, `planet.j2()`.
- `NavigationOutput` uses **Fortran-style field names**: `positn: [f64; 3]`, `vitesn: [f64; 3]`, `acceln: [f64; 2]`, `coefan: [f64; 2]`, `roguid`, `pdynan`, `energn`, etc.
- `NavigationBiases` uses `pos`/`vel`, not `position`/`velocity`.
- `AtmosphereModel` fields are **plural**: `altitudes`, `densities`, plus `ref_density`, `scale_factor`, `ref_altitude`, `gas_constant`, `density_profile`.
- `SimData` has `final_conditions: FinalConditions`, `parking_orbit: ParkingOrbit`, and `periods: TimePeriods` at top level.
- Existing tests in `equilibrium_glide.rs`, `energy_controller.rs`, `predguid.rs` already cover basic cases — add **proptests only** to those modules, don't duplicate.

---

## Chunk 1: Dependencies & Test Infrastructure

### Task 1: Add proptest and hypothesis dependencies

**Files:**
- Modify: `src/rust/Cargo.toml`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add proptest to Rust dev-dependencies**

In `src/rust/Cargo.toml`, add to `[dev-dependencies]`:

```toml
[dev-dependencies]
approx = "0.5"
proptest = "1.6"
rstest = "0.26"
```

- [ ] **Step 2: Add hypothesis to Python dev dependencies**

In `pyproject.toml`, add `"hypothesis>=6.100"` to the dev group:

```toml
[dependency-groups]
dev = [
    "hypothesis>=6.100",
    "pytest>=8.0",
    "ruff>=0.15.5",
    # ... rest unchanged
]
```

- [ ] **Step 3: Install and verify**

Run:
```bash
cd src/rust && cargo check --tests
cd ../.. && uv sync --group dev
```
Expected: Both succeed without errors.

- [ ] **Step 4: Commit**

```bash
git add src/rust/Cargo.toml pyproject.toml uv.lock
git commit -m "chore: add proptest and hypothesis for property-based testing"
```

---

### Task 2: Rust shared test fixtures module

**Files:**
- Modify: `src/rust/tests/common/mod.rs`
- Create: `src/rust/tests/common/fixtures.rs`

- [ ] **Step 1: Create fixtures.rs with shared builders**

Create `src/rust/tests/common/fixtures.rs`. Follow the existing pattern from `equilibrium_glide.rs:107-194` (the working `test_nav()` and `test_sim_data()` builders). Key: use `Planet::Mars` (enum), Fortran-style field names for `NavigationOutput`, plural field names for `AtmosphereModel`.

```rust
//! Shared test fixtures for integration tests.
//!
//! Consolidates the patterns from equilibrium_glide.rs, estimator.rs, init.rs.

use aerocapture::config::Planet;
use aerocapture::data::aerodynamics::AeroTables;
use aerocapture::data::atmosphere::{AtmosphereModel, DensityProfile};
use aerocapture::data::capsule::Capsule;
use aerocapture::data::guidance_params::GuidanceParams;
use aerocapture::data::incidence::IncidenceProfile;
use aerocapture::data::pilot::{PilotModel, PilotType};
use aerocapture::data::{
    Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit, SimData,
    SphericalState, SuccessCriteria, TimePeriods,
};
use aerocapture::gnc::navigation::estimator::{NavigationBiases, NavigationOutput};

/// Build a NavigationOutput from key flight parameters.
/// Mirrors the `test_nav()` pattern in equilibrium_glide.rs:107-120.
pub fn nav_from_state(
    altitude: f64,
    velocity: f64,
    flight_path: f64,
    density: f64,
    drag_accel: f64,
    lift_accel: f64,
) -> NavigationOutput {
    let planet = Planet::Mars;
    let r = planet.equatorial_radius() + altitude;
    NavigationOutput {
        positn: [r, 0.0, 0.0],
        vitesn: [velocity, flight_path, std::f64::consts::FRAC_PI_2],
        acceln: [drag_accel, lift_accel],
        coefan: [1.269, -0.205],
        roguid: density,
        roexit: 1e-6,
        pdynan: 0.5 * density * velocity * velocity,
        energn: velocity * velocity / 2.0 - planet.mu() / r,
        ..Default::default()
    }
}

/// Minimal SimData with Mars defaults.
/// Mirrors equilibrium_glide.rs:122-194 pattern.
pub fn minimal_sim_data() -> SimData {
    SimData {
        capsule: Capsule {
            mass: 1089.0,
            reference_area: 14.7,
            cq: 0.00008242,
            max_bank_rate: 15.0_f64.to_radians(),
            periods: TimePeriods::default(),
        },
        aero: AeroTables {
            n_points: 2,
            incidence: vec![-0.5, 0.0],
            cx: vec![1.269, 1.269],
            cz: vec![-0.205, -0.205],
            equilibrium_aoa: -0.48,
            ..Default::default()
        },
        atmosphere: AtmosphereModel {
            n_points: 3,
            altitudes: vec![0.0, 50_000.0, 130_000.0],
            densities: vec![0.02, 0.001, 1e-8],
            ref_density: 1e-8,
            scale_factor: 1e-4,
            ref_altitude: 130_000.0,
            gas_constant: 1.3,
            density_profile: DensityProfile::default(),
        },
        entry: EntryConditions {
            state: SphericalState {
                altitude: 130_000.0,
                velocity: 5687.0,
                flight_path: -10.8_f64.to_radians(),
                ..Default::default()
            },
            initial_bank: 64.77_f64.to_radians(),
            initial_aoa: -27.5_f64.to_radians(),
            initial_date: 0.0,
        },
        guidance: GuidanceParams {
            density_filter_gain: 0.8,
            exit_velocity_threshold: 4400.0,
            exit_altitude_threshold: 60_000.0,
            ..Default::default()
        },
        incidence: IncidenceProfile {
            n_points: 2,
            altitudes: vec![-10_000.0, 150_000.0],
            incidences: vec![-0.48, -0.48],
        },
        periods: TimePeriods::default(),
        pilot: PilotModel {
            pilot_type: PilotType::Perfect,
            time_constant: 0.0,
            damping: 0.0,
            frequency: 0.0,
        },
        target_orbit: OrbitalTarget {
            semi_major_axis: 3_649_622.0,
            eccentricity: 0.067,
            inclination: 50.0_f64.to_radians(),
            raan: -7.612_f64.to_radians(),
            apoapsis: 500_130.0,
            periapsis: 11_233.0,
        },
        final_conditions: FinalConditions::default(),
        parking_orbit: ParkingOrbit::default(),
        constraints: Constraints::default(),
        success: SuccessCriteria::default(),
        wind_enabled: false,
        neural_net: None,
        dispersion_config: None,
    }
}

/// Zero navigation biases (no measurement errors).
pub fn zero_nav_biases() -> NavigationBiases {
    NavigationBiases {
        pos: [0.0; 3],
        vel: [0.0; 3],
        drag: 0.0,
    }
}
```

- [ ] **Step 2: Update common/mod.rs to export fixtures**

Add `pub mod fixtures;` to `src/rust/tests/common/mod.rs` below the existing helpers.

- [ ] **Step 3: Verify compilation**

Run: `cd src/rust && cargo test --no-run`
Expected: Compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add src/rust/tests/common/
git commit -m "test(rust): add shared test fixtures module with builders"
```

---

### Task 3: Rust domain-specific assertion helpers

**Files:**
- Create: `src/rust/tests/common/assertions.rs`
- Modify: `src/rust/tests/common/mod.rs`

- [ ] **Step 1: Create assertions.rs**

```rust
//! Domain-specific test assertions for aerocapture simulation.

/// Assert a value is finite and within [min, max].
pub fn assert_finite_bounded(value: f64, min: f64, max: f64, context: &str) {
    assert!(value.is_finite(), "{context}: expected finite, got {value}");
    assert!(
        value >= min && value <= max,
        "{context}: expected [{min}, {max}], got {value}"
    );
}

/// Assert a bank angle is finite and within [0, pi] radians.
pub fn assert_bank_angle_valid(angle_rad: f64, context: &str) {
    assert_finite_bounded(
        angle_rad,
        0.0,
        std::f64::consts::PI,
        &format!("{context} (bank angle)"),
    );
}

/// Assert all components of a 3-vector are finite.
pub fn assert_vector_finite(v: &[f64; 3], context: &str) {
    for (i, &x) in v.iter().enumerate() {
        assert!(
            x.is_finite(),
            "{context}[{i}]: expected finite, got {x}"
        );
    }
}
```

- [ ] **Step 2: Export from mod.rs**

Add `pub mod assertions;` to `src/rust/tests/common/mod.rs`.

- [ ] **Step 3: Verify compilation**

Run: `cd src/rust && cargo test --no-run`
Expected: Compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add src/rust/tests/common/
git commit -m "test(rust): add domain-specific assertion helpers"
```

---

### Task 4: Python conftest.py and shared fixtures

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/factories.py`
- Modify: `tests/test_regression.py` (remove duplicated `_build_rust`)
- Modify: `tests/test_mc_domain.py` (remove duplicated `_build_rust`)

- [ ] **Step 1: Create conftest.py with shared fixtures**

```python
"""Shared pytest fixtures for the Aerocapture test suite."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "src" / "rust" / "target" / "release" / "aerocapture"


@pytest.fixture(scope="session")
def rust_binary() -> Path:
    """Build the Rust simulator once per session, return the binary path."""
    if not BINARY.exists():
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=ROOT / "src" / "rust",
            check=True,
        )
    return BINARY


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Per-test temporary directory for simulation outputs."""
    return tmp_path
```

Note: `autouse=False` (default) — only tests that need the binary request it explicitly.

- [ ] **Step 2: Create fixtures/__init__.py**

```python
"""Test fixture factories."""
```

- [ ] **Step 3: Create fixtures/factories.py**

```python
"""Factory functions for building test objects with sensible defaults."""

from __future__ import annotations

import numpy as np

from aerocapture.training.config import GAConfig, NetworkConfig, SimConfig, TrainingConfig
from aerocapture.training.param_spaces import PARAM_SPACES


def make_training_config(guidance_type: str = "equilibrium_glide") -> TrainingConfig:
    """Build a minimal TrainingConfig for the given guidance type."""
    return TrainingConfig(
        network=NetworkConfig(),
        ga=GAConfig(n_bit=16, p_min=-3.0, p_max=3.0, direct_encoding=True),
        sim=SimConfig(
            executable="dummy",
            nn_param_file="dummy.json",
            final_file="final.csv",
        ),
        save_dir="dummy",
        guidance_type=guidance_type,
    )


def make_chromosome(length: int, *, strategy: str = "mid") -> np.ndarray:
    """Generate a binary chromosome.

    Strategies:
        mid   — alternating 0/1 (mid-range parameter values)
        zeros — all zeros (minimum parameter values)
        ones  — all ones (maximum parameter values)
        random — uniformly random bits (seed=42)
    """
    if strategy == "mid":
        return np.array([i % 2 for i in range(length)], dtype=np.int8)
    if strategy == "zeros":
        return np.zeros(length, dtype=np.int8)
    if strategy == "ones":
        return np.ones(length, dtype=np.int8)
    if strategy == "random":
        return np.random.default_rng(42).integers(0, 2, size=length, dtype=np.int8)
    msg = f"Unknown strategy: {strategy}"
    raise ValueError(msg)
```

- [ ] **Step 4: Remove _build_rust from test_regression.py and test_mc_domain.py**

In `tests/test_regression.py`, remove the `_build_rust` fixture and `BINARY` constant. Update to use the session-scoped `rust_binary` fixture from conftest.py. Add `rust_binary` as a parameter to tests that need it, or keep `autouse` locally if preferred.

In `tests/test_mc_domain.py`, same refactor — remove the identical `_build_rust` fixture.

- [ ] **Step 5: Run existing tests to verify refactor**

Run: `uv run pytest tests/ -v`
Expected: All 14 existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/fixtures/ tests/test_regression.py tests/test_mc_domain.py
git commit -m "test(python): add conftest.py with shared fixtures, deduplicate _build_rust"
```

---

## Chunk 2: Rust Guidance Algorithm Unit Tests

**Important:** `equilibrium_glide.rs`, `energy_controller.rs`, and `predguid.rs` already have working unit tests. For these modules, **only add proptest property tests** to the existing `#[cfg(test)]` modules. For `ftc.rs`, `fnpag.rs`, `neural.rs`, and `reference.rs` — these have **no tests** and need full test modules.

### Task 5: Add proptest to equilibrium glide (existing tests)

**Files:**
- Modify: `src/rust/src/gnc/guidance/equilibrium_glide.rs` (expand existing `#[cfg(test)]` at line 90)

- [ ] **Step 1: Add proptest module to existing tests**

The existing test module (lines 90-249) already has `test_nav()`, `test_sim_data()`, and 3 deterministic tests. Add a proptest submodule at the end of the existing `mod tests`:

```rust
    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn output_always_finite_and_bounded(
                alt in 10_000.0..130_000.0_f64,
                vel in 2000.0..7000.0_f64,
                fpa in -0.2..0.05_f64,
                rho in 1e-6..0.05_f64,
            ) {
                let mut nav = test_nav(vel);
                let r = Planet::Mars.equatorial_radius() + alt;
                nav.positn[0] = r;
                nav.vitesn[1] = fpa;
                nav.roguid = rho;
                nav.pdynan = 0.5 * rho * vel * vel;

                let data = test_sim_data();
                let planet = Planet::Mars;
                let bank = equilibrium_glide_bank(&nav, &data, &planet);

                let min_bank = 15.0_f64.to_radians();
                let max_bank = 120.0_f64.to_radians();
                prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
                prop_assert!(bank >= min_bank - 1e-10, "bank below min: {}", bank);
                prop_assert!(bank <= max_bank + 1e-10, "bank above max: {}", bank);
            }
        }
    }
```

- [ ] **Step 2: Run all equilibrium glide tests**

Run: `cd src/rust && cargo test equilibrium_glide::tests -- --nocapture`
Expected: 4 tests pass (3 existing deterministic + 1 proptest).

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/gnc/guidance/equilibrium_glide.rs
git commit -m "test(rust): add proptest to equilibrium glide guidance"
```

---

### Task 6: Add proptest to energy controller (existing tests)

**Files:**
- Modify: `src/rust/src/gnc/guidance/energy_controller.rs` (expand existing `#[cfg(test)]` at line 90)

- [ ] **Step 1: Read the existing tests**

Read `energy_controller.rs` from line 90 onward. Note the existing `test_nav()` and `test_sim_data()` helpers and test names.

- [ ] **Step 2: Add proptest submodule**

Same pattern as Task 5: add a `mod prop` with proptest that verifies output is always finite and in [0, π] for valid flight states.

- [ ] **Step 3: Run and verify**

Run: `cd src/rust && cargo test energy_controller::tests -- --nocapture`
Expected: All pass (existing + 1 proptest).

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/energy_controller.rs
git commit -m "test(rust): add proptest to energy controller guidance"
```

---

### Task 7: Add proptest to predguid (existing tests)

**Files:**
- Modify: `src/rust/src/gnc/guidance/predguid.rs` (expand existing `#[cfg(test)]` at line 117)

- [ ] **Step 1: Read existing tests, add proptest**

Same pattern as Tasks 5-6.

- [ ] **Step 2: Run and verify**

Run: `cd src/rust && cargo test predguid::tests -- --nocapture`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/gnc/guidance/predguid.rs
git commit -m "test(rust): add proptest to predguid guidance"
```

---

### Task 8: FNPAG unit tests (new)

**Files:**
- Modify: `src/rust/src/gnc/guidance/fnpag.rs` (add `#[cfg(test)]` module)

- [ ] **Step 1: Read fnpag.rs public API**

Read `fnpag.rs` fully. Note `fnpag_bank()` signature, `FnpagState` struct, the bisection/secant algorithm, `predict_exit_energy()` helper, and the `rho < 1e-10` early exit.

- [ ] **Step 2: Write test module**

Add at the bottom of `fnpag.rs`. Follow the existing pattern from `equilibrium_glide.rs` — create local `test_nav()` and `test_sim_data()` using the same struct field names (`positn`, `vitesn`, `acceln`, `coefan`, `roguid`, `pdynan`, `energn`, `Planet::Mars`).

Key deterministic tests:
- `low_density_returns_previous_bank`: set `roguid < 1e-10` → returns `bank_prev`
- `first_call_initializes`: fresh `FnpagState` → after call, state changes
- `output_bounded`: typical MSR state → bank is finite and in reasonable range

Key proptest:
- For valid flight states (alt 20k-100k, vel 3k-6k, fpa -0.15..0.0, rho 1e-5..0.01), output is finite

- [ ] **Step 3: Run and verify**

Run: `cd src/rust && cargo test fnpag::tests -- --nocapture`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/fnpag.rs
git commit -m "test(rust): add FNPAG unit tests with proptest"
```

---

### Task 9: FTC guidance unit tests (new)

**Files:**
- Modify: `src/rust/src/gnc/guidance/ftc.rs` (add `#[cfg(test)]` module)

This is the most complex guidance module — dispatches to all 6 schemes, has lateral guidance, phase management.

- [ ] **Step 1: Read ftc.rs thoroughly**

Read the full file. Note `guidance_step()` signature, `FtcState`, `FtcOutput`, `guicap()`, `tbgain()`, `guilat()`. All internal functions are private — test via the public `guidance_step()` API.

- [ ] **Step 2: Write test module**

Create `#[cfg(test)] mod tests` with local `test_nav()`, `test_sim_data()` (same pattern as other guidance modules). Test via `guidance_step()`:

Key deterministic tests:
- `guidance_step_returns_finite_output`: typical MSR state → FtcOutput bank angle is finite
- `reference_mode_returns_initial_bank`: when `is_reference=true` → output tracks reference bank
- `ftc_output_bank_bounded`: output bank in [0, π]

Key proptest:
- For valid states, `guidance_step()` always produces finite, bounded output

- [ ] **Step 3: Run and verify**

Run: `cd src/rust && cargo test ftc::tests -- --nocapture`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/ftc.rs
git commit -m "test(rust): add FTC guidance unit tests with proptest"
```

---

### Task 10: Neural network guidance unit tests (new)

**Files:**
- Modify: `src/rust/src/gnc/guidance/neural.rs` (add `#[cfg(test)]` module)
- Read: `src/rust/src/data/neural.rs` for `NeuralNetModel` struct

- [ ] **Step 1: Read neural.rs and data/neural.rs**

Note `nn_bank_angle()` signature, input normalization, output mapping (`atan2`), and how `NeuralNetModel` is constructed.

- [ ] **Step 2: Write tests**

Key tests:
- `zero_weights_known_output`: construct a tiny NN (e.g., 2 inputs, 2 hidden, 2 outputs) with all-zero weights → output is `atan2(activation(bias[0]), activation(bias[1]))` — verify it matches hand-computed value
- `output_in_valid_range`: for typical MSR state with a small random NN → bank ∈ [0, π]
- Proptest: for valid states + small NN, output is always finite

- [ ] **Step 3: Run and verify**

Run: `cd src/rust && cargo test neural::tests -- --nocapture`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/neural.rs
git commit -m "test(rust): add neural network guidance unit tests"
```

---

### Task 11: Reference guidance unit test (new)

**Files:**
- Modify: `src/rust/src/gnc/guidance/reference.rs` (add `#[cfg(test)]` module)

- [ ] **Step 1: Read reference.rs**

Note the `ReferenceGuidance` struct and its `Guidance` trait implementation. Check if `compute()` takes `&mut self` or `&self`.

- [ ] **Step 2: Write test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_configured_bank_angle() {
        let bank = 1.13;  // ~64.77°
        let aoa = 0.175;  // ~10°
        let mut guidance = ReferenceGuidance { bank_angle: bank, aoa };
        let cmd = guidance.compute(&Default::default(), 0.0);
        assert_eq!(cmd.bank_angle, bank);
        assert_eq!(cmd.aoa, aoa);
    }
}
```

Note: Use `let mut guidance` if the trait method requires `&mut self`.

- [ ] **Step 3: Run and verify**

Run: `cd src/rust && cargo test reference::tests -- --nocapture`
Expected: 1 test passes.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/gnc/guidance/reference.rs
git commit -m "test(rust): add reference guidance unit test"
```

---

## Chunk 3: Rust Other Coverage Gaps

### Task 12: Wind model tests

**Files:**
- Modify: `src/rust/src/physics/winds.rs` (add `#[cfg(test)]` module)

- [ ] **Step 1: Read winds.rs**

Read the full file (~29 lines). Note the `wind_velocity()` signature and `WindVelocity` struct.

- [ ] **Step 2: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_returns_zero() {
        let w = wind_velocity(50_000.0, 0.1, 0.5, false);
        assert_eq!(w.north, 0.0);
        assert_eq!(w.east, 0.0);
        assert_eq!(w.up, 0.0);
    }

    #[test]
    fn enabled_returns_zero_for_stub() {
        // Document that the current implementation is a stub
        let w = wind_velocity(50_000.0, 0.1, 0.5, true);
        assert_eq!(w.north, 0.0);
        assert_eq!(w.east, 0.0);
        assert_eq!(w.up, 0.0);
    }
}
```

Adapt field names to match the actual `WindVelocity` struct — read `winds.rs` first.

- [ ] **Step 3: Run and verify**

Run: `cd src/rust && cargo test winds::tests -- --nocapture`
Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/physics/winds.rs
git commit -m "test(rust): add wind model unit tests"
```

---

### Task 13: Navigation estimator expanded tests

**Files:**
- Modify: `src/rust/src/gnc/navigation/estimator.rs` (expand existing `#[cfg(test)]` module)

- [ ] **Step 1: Read existing tests**

Read the existing `#[cfg(test)]` module (starts at line ~245). Note the existing `test_sim_data()`, `call_navigate()` helpers and the 6 existing tests.

- [ ] **Step 2: Add density filter stability test**

Feed adversarial density ratios (oscillating high/low) for 100 steps → verify `coefro` stays bounded. Use the existing `call_navigate()` helper pattern.

```rust
#[test]
fn density_filter_stable_under_adversarial_input() {
    let data = test_sim_data();
    let planet = Planet::Mars;
    let biases = NavigationBiases::default();
    let mut nav_state = NavigationState::new();
    let dt = 1.0;

    for step in 0..100 {
        // Oscillate density between 10x and 0.1x nominal
        let alt = 50_000.0;
        let r = planet.equatorial_radius() + alt;
        let positr = [r, 0.0, 0.0];
        let vitesn = [5000.0, -0.1, std::f64::consts::FRAC_PI_2];
        let time = step as f64 * dt;

        let _out = navigate(
            &positr, &vitesn, -0.48, time, &biases, &mut nav_state,
            &data, &planet, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        );

        assert!(
            nav_state.coefro.is_finite(),
            "coefro not finite at step {step}: {}",
            nav_state.coefro
        );
        assert!(
            nav_state.coefro > 0.0,
            "coefro non-positive at step {step}: {}",
            nav_state.coefro
        );
    }
}
```

- [ ] **Step 3: Add phase transition tests**

Test that `iphase` transitions correctly:
- Start at `iphase=1` (capture), feed states until conditions trigger transition to `iphase=2` (exit) or `iphase=3` (emergency)
- Test bounce detection: gamma goes positive → `ibounc` flag set

- [ ] **Step 4: Add proptest for finite outputs**

For any valid state (bounded altitude, velocity, biases), `navigate()` outputs are finite.

- [ ] **Step 5: Run and verify**

Run: `cd src/rust && cargo test estimator::tests -- --nocapture`
Expected: All tests pass (existing 6 + new ~4).

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/navigation/estimator.rs
git commit -m "test(rust): expand navigation estimator tests (filter stability, phase transitions)"
```

---

### Task 14: Simulation init expanded tests

**Files:**
- Modify: `src/rust/src/simulation/init.rs` (expand existing `#[cfg(test)]` module)

- [ ] **Step 1: Read existing tests**

Read the existing `#[cfg(test)]` module. Note the 3 existing tests and `test_sim_data()` helper.

- [ ] **Step 2: Add proptest for dispersion bounds**

For any dispersion draw within ±3σ, the dispersed entry state should stay physically valid (altitude > 0, velocity > 0, etc.).

- [ ] **Step 3: Run and verify**

Run: `cd src/rust && cargo test init::tests -- --nocapture`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/simulation/init.rs
git commit -m "test(rust): add proptest for dispersion bounds in simulation init"
```

---

### Task 15: Cross-cutting NaN/Inf error-path tests

**Files:**
- Create: `src/rust/tests/error_paths.rs`

Integration-level tests verifying guidance schemes handle degenerate inputs without panicking.

- [ ] **Step 1: Write error-path integration tests**

```rust
//! Tests that guidance schemes handle degenerate inputs gracefully.
//! Not testing correctness — testing that nothing panics or produces NaN/Inf.

mod common;

use aerocapture::config::Planet;
use aerocapture::gnc::guidance::equilibrium_glide::equilibrium_glide_bank;
use aerocapture::gnc::guidance::fnpag::{fnpag_bank, FnpagState};
// ... import other guidance functions as needed

use common::fixtures::{minimal_sim_data, nav_from_state};

#[test]
fn equilibrium_glide_zero_velocity_no_panic() {
    let nav = nav_from_state(60_000.0, 0.0, 0.0, 0.001, 0.0, 0.0);
    let data = minimal_sim_data();
    let planet = Planet::Mars;
    let bank = equilibrium_glide_bank(&nav, &data, &planet);
    assert!(bank.is_finite(), "expected finite, got {bank}");
}

#[test]
fn fnpag_zero_density_no_panic() {
    let nav = nav_from_state(130_000.0, 5687.0, -0.2, 0.0, 0.0, 0.0);
    let data = minimal_sim_data();
    let planet = Planet::Mars;
    let mut state = FnpagState::default();
    let bank = fnpag_bank(&nav, &mut state, &data, &planet);
    assert!(bank.is_finite(), "expected finite, got {bank}");
}

// Add similar tests for each guidance scheme with:
// - Zero velocity
// - Zero density
// - Extreme flight path angle (±90°)
// - Very high altitude (above atmosphere table)
```

- [ ] **Step 2: Run and verify**

Run: `cd src/rust && cargo test --test error_paths -- --nocapture`
Expected: All pass (or expose real bugs to fix).

- [ ] **Step 3: Commit**

```bash
git add src/rust/tests/error_paths.rs
git commit -m "test(rust): add error-path tests for degenerate guidance inputs"
```

---

## Chunk 4: Python GA Training Pipeline Tests

### Task 16: Chromosome encode/decode roundtrip tests

**Files:**
- Create: `tests/test_chromosome.py`

- [ ] **Step 1: Write roundtrip tests**

```python
"""Tests for chromosome encoding/decoding roundtrips."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from aerocapture.training.evaluate import decode_params_from_chromosome
from aerocapture.training.param_spaces import PARAM_SPACES
from aerocapture.training.population import encode_params_to_chromosome

from tests.fixtures.factories import make_training_config


@pytest.mark.parametrize("scheme", list(PARAM_SPACES.keys()))
class TestChromosomeRoundtrip:
    """Encode -> decode roundtrip for all guidance schemes."""

    def test_roundtrip_preserves_values(self, scheme: str) -> None:
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        params = {s.name: s.default for s in specs}
        chromosome = encode_params_to_chromosome(params, config)
        decoded = decode_params_from_chromosome(chromosome, config)
        for s in specs:
            assert decoded[s.name] == pytest.approx(params[s.name], rel=0.01), (
                f"{s.name}: expected {params[s.name]}, got {decoded[s.name]}"
            )

    def test_all_zeros_gives_minimum(self, scheme: str) -> None:
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        chromosome = np.zeros(config.chrom_length, dtype=np.int8)
        decoded = decode_params_from_chromosome(chromosome, config)
        for s in specs:
            if s.log_scale:
                assert decoded[s.name] == pytest.approx(10**s.p_min, rel=0.1)
            else:
                assert decoded[s.name] == pytest.approx(s.p_min, rel=0.01)

    def test_all_ones_gives_maximum(self, scheme: str) -> None:
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        chromosome = np.ones(config.chrom_length, dtype=np.int8)
        decoded = decode_params_from_chromosome(chromosome, config)
        for s in specs:
            if s.log_scale:
                assert decoded[s.name] == pytest.approx(10**s.p_max, rel=0.1)
            else:
                assert decoded[s.name] == pytest.approx(s.p_max, rel=0.01)


@pytest.mark.parametrize("scheme", list(PARAM_SPACES.keys()))
class TestChromosomeProperties:
    """Property-based tests for chromosome decoding."""

    @given(data=st.data())
    @settings(max_examples=50)
    def test_decoded_params_respect_bounds(self, scheme: str, data: st.DataObject) -> None:
        config = make_training_config(scheme)
        specs = PARAM_SPACES[scheme]
        chrom = data.draw(
            arrays(dtype=np.int8, shape=config.chrom_length, elements=st.integers(0, 1))
        )
        decoded = decode_params_from_chromosome(chrom, config)
        for s in specs:
            val = decoded[s.name]
            if s.log_scale:
                lo, hi = 10**s.p_min, 10**s.p_max
            else:
                lo, hi = s.p_min, s.p_max
            assert lo - 1e-6 <= val <= hi + 1e-6, (
                f"{s.name}: {val} not in [{lo}, {hi}]"
            )
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_chromosome.py -v`
Expected: 20 tests pass (5 schemes × 4 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_chromosome.py
git commit -m "test(python): add chromosome encode/decode roundtrip + property tests"
```

---

### Task 17: Cost function edge case tests

**Files:**
- Create: `tests/test_cost.py`

- [ ] **Step 1: Create cost function test file**

Test `compute_cost()` from `aerocapture.training.evaluate`. Column indices for the 53-column legacy array: 8=energy, 10=eccentricity, 28=sim_time, 30=periapsis_err, 31=apoapsis_err, 42=dv_total.

```python
"""Tests for the GA cost function edge cases."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aerocapture.training.evaluate import compute_cost


class TestComputeCostEdgeCases:
    def test_nan_energy_returns_penalty(self) -> None:
        row = np.zeros((1, 53))
        row[0, 8] = float("nan")
        cost = compute_cost(row)
        assert cost > 1e5, f"NaN energy should penalize, got {cost}"
        assert np.isfinite(cost), f"Cost should be finite, got {cost}"

    def test_parabolic_energy_zero(self) -> None:
        row = np.zeros((1, 53))
        row[0, 8] = 0.0   # energy = 0 (parabolic)
        row[0, 10] = 1.0   # eccentricity = 1
        row[0, 28] = 500.0
        cost = compute_cost(row)
        assert np.isfinite(cost)
        assert cost > 1e3  # should be penalized

    def test_captured_orbit_reasonable_cost(self) -> None:
        row = np.zeros((1, 53))
        row[0, 8] = -0.5     # captured
        row[0, 10] = 0.3     # elliptical
        row[0, 28] = 1200.0
        row[0, 30] = 5.0     # periapsis error km
        row[0, 31] = 10.0    # apoapsis error km
        row[0, 42] = 150.0   # dv_total m/s
        cost = compute_cost(row)
        assert cost < 1e4, f"Captured orbit cost too high: {cost}"

    def test_cost_always_nonnegative(self) -> None:
        row = np.zeros((1, 53))
        row[0, 8] = -1.0
        row[0, 10] = 0.1
        cost = compute_cost(row)
        assert cost >= 0.0

    def test_multi_sim_rms(self) -> None:
        """Multi-sim cost is RMS of individual costs."""
        row = np.zeros((1, 53))
        row[0, 8] = -0.5
        row[0, 10] = 0.3
        row[0, 30] = 5.0
        row[0, 31] = 10.0
        row[0, 42] = 150.0
        rows = np.vstack([row, row])
        cost_single = compute_cost(row)
        cost_double = compute_cost(rows)
        assert cost_double == pytest.approx(cost_single, rel=0.01)


class TestComputeCostProperties:
    @given(
        energy=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        ecc=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_cost_always_finite_for_finite_inputs(self, energy: float, ecc: float) -> None:
        row = np.zeros((1, 53))
        row[0, 8] = energy
        row[0, 10] = ecc
        row[0, 28] = 1000.0
        row[0, 30] = 10.0
        row[0, 31] = 20.0
        row[0, 42] = 200.0
        cost = compute_cost(row)
        assert np.isfinite(cost), f"Non-finite cost: energy={energy}, ecc={ecc}, cost={cost}"
        assert cost >= 0.0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_cost.py -v`
Expected: All pass (or expose real NaN-handling bugs to fix).

- [ ] **Step 3: Commit**

```bash
git add tests/test_cost.py
git commit -m "test(python): add cost function edge case + property tests"
```

---

### Task 18: TOML patching roundtrip tests

**Files:**
- Create: `tests/test_toml_patching.py`

- [ ] **Step 1: Write TOML roundtrip tests**

```python
"""Tests for TOML patching (decode chromosome -> write TOML -> parse back)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

from aerocapture.training.evaluate import decode_params_from_chromosome, write_guidance_toml
from aerocapture.training.param_spaces import PARAM_SPACES

from tests.fixtures.factories import make_training_config

ROOT = Path(__file__).resolve().parent.parent
TRAINING_CONFIGS = {
    "equilibrium_glide": ROOT / "configs" / "training" / "msr_aller_eqglide_train.toml",
    "energy_controller": ROOT / "configs" / "training" / "msr_aller_energy_controller_train.toml",
    "pred_guid": ROOT / "configs" / "training" / "msr_aller_pred_guid_train.toml",
    "fnpag": ROOT / "configs" / "training" / "msr_aller_fnpag_train.toml",
    "ftc": ROOT / "configs" / "training" / "msr_aller_ftc_train.toml",
}
# Note: neural_network uses write_nn_json(), not write_guidance_toml(), so excluded here.


@pytest.mark.parametrize("scheme", list(TRAINING_CONFIGS.keys()))
def test_toml_roundtrip(scheme: str, tmp_path: Path) -> None:
    """Decode mid-range chromosome -> write TOML -> verify TOML parses."""
    config = make_training_config(scheme)
    chrom = np.array([i % 2 for i in range(config.chrom_length)], dtype=np.int8)
    params = decode_params_from_chromosome(chrom, config)
    base_toml = TRAINING_CONFIGS[scheme]
    output_path = tmp_path / f"{scheme}_patched.toml"
    write_guidance_toml(base_toml, scheme, params, output_path)

    with open(output_path, "rb") as f:
        data = tomllib.load(f)
    assert "guidance" in data, "Patched TOML missing [guidance] section"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_toml_patching.py -v`
Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_toml_patching.py
git commit -m "test(python): add TOML patching roundtrip tests"
```

---

### Task 19: Config validation tests

**Files:**
- Create: `tests/test_config.py`

- [ ] **Step 1: Write config tests**

```python
"""Tests for TrainingConfig validation and consistency."""

from __future__ import annotations

import pytest

from aerocapture.training.config import GAConfig, NetworkConfig, SimConfig, TrainingConfig
from aerocapture.training.param_spaces import PARAM_SPACES


class TestTrainingConfig:
    @pytest.mark.parametrize("scheme", list(PARAM_SPACES.keys()))
    def test_n_params_matches_param_space(self, scheme: str) -> None:
        config = TrainingConfig(
            network=NetworkConfig(),
            ga=GAConfig(n_bit=16, direct_encoding=True),
            sim=SimConfig(executable="x", nn_param_file="x", final_file="x"),
            save_dir="x",
            guidance_type=scheme,
        )
        assert config.n_params == len(PARAM_SPACES[scheme])

    @pytest.mark.parametrize("scheme", list(PARAM_SPACES.keys()))
    def test_chrom_length_is_n_params_times_n_bit(self, scheme: str) -> None:
        n_bit = 16
        config = TrainingConfig(
            network=NetworkConfig(),
            ga=GAConfig(n_bit=n_bit, direct_encoding=True),
            sim=SimConfig(executable="x", nn_param_file="x", final_file="x"),
            save_dir="x",
            guidance_type=scheme,
        )
        assert config.chrom_length == config.n_params * n_bit

    def test_nn_n_params_uses_network_config(self) -> None:
        config = TrainingConfig(
            network=NetworkConfig(layer_sizes=[6, 12, 2], activations=["tanh", "asinh"]),
            ga=GAConfig(n_bit=16, direct_encoding=True),
            sim=SimConfig(executable="x", nn_param_file="x", final_file="x"),
            save_dir="x",
            guidance_type="neural_network",
        )
        assert config.n_params == config.network.n_base_coef
        assert config.n_params > 0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_config.py
git commit -m "test(python): add TrainingConfig validation tests"
```

---

### Task 20: GA operator tests

**Files:**
- Create: `tests/test_ga_operators.py`

- [ ] **Step 1: Write GA operator tests**

```python
"""Tests for GA selection, crossover, and mutation operators."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from aerocapture.training.train import crossover_and_mutate, roulette_selection

from tests.fixtures.factories import make_training_config


class TestRouletteSelection:
    def test_returns_valid_index(self) -> None:
        costs = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        rng = np.random.default_rng(42)
        for _ in range(50):
            idx = roulette_selection(costs, rng)
            assert 0 <= idx < len(costs)

    def test_lower_cost_selected_more_often(self) -> None:
        costs = np.array([1.0, 1000.0, 1000.0, 1000.0, 1000.0])
        rng = np.random.default_rng(42)
        selections = [roulette_selection(costs, rng) for _ in range(200)]
        count_best = selections.count(0)
        assert count_best > 50, f"Best selected only {count_best}/200 times"

    def test_equal_costs_uniform_selection(self) -> None:
        costs = np.array([100.0] * 5)
        rng = np.random.default_rng(42)
        selections = [roulette_selection(costs, rng) for _ in range(500)]
        for i in range(5):
            count = selections.count(i)
            assert count > 50, f"Index {i}: {count}/500 (expected ~100)"


class TestCrossoverAndMutate:
    def test_output_shape_matches_input(self) -> None:
        config = make_training_config("equilibrium_glide")
        config.ga.n_pop = 6
        chrom_len = config.chrom_length
        pop = np.random.default_rng(42).integers(0, 2, size=(6, chrom_len), dtype=np.int8)
        costs = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        rng = np.random.default_rng(42)
        offspring = crossover_and_mutate(pop, costs, config, rng)
        assert offspring.shape == pop.shape

    def test_output_is_binary(self) -> None:
        config = make_training_config("equilibrium_glide")
        config.ga.n_pop = 6
        chrom_len = config.chrom_length
        pop = np.random.default_rng(42).integers(0, 2, size=(6, chrom_len), dtype=np.int8)
        costs = np.ones(6) * 100.0
        rng = np.random.default_rng(42)
        offspring = crossover_and_mutate(pop, costs, config, rng)
        assert set(np.unique(offspring)).issubset({0, 1})

    @given(data=st.data())
    @settings(max_examples=20)
    def test_offspring_always_valid(self, data: st.DataObject) -> None:
        n_pop = 6
        config = make_training_config("equilibrium_glide")
        config.ga.n_pop = n_pop
        chrom_len = config.chrom_length
        pop = data.draw(
            arrays(dtype=np.int8, shape=(n_pop, chrom_len), elements=st.integers(0, 1))
        )
        costs = data.draw(
            arrays(dtype=np.float64, shape=n_pop, elements=st.floats(1.0, 1e6))
        )
        rng = np.random.default_rng(42)
        offspring = crossover_and_mutate(pop, costs, config, rng)
        assert offspring.shape == (n_pop, chrom_len)
        assert np.all((offspring == 0) | (offspring == 1))
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_ga_operators.py -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ga_operators.py
git commit -m "test(python): add GA operator tests (selection, crossover, mutation)"
```

---

### Task 21: Run full test suite, lint, and smart-commit

**Files:** None (verification + commit)

- [ ] **Step 1: Run all Rust tests**

Run: `cd src/rust && cargo test`
Expected: All tests pass (existing ~113 + new ~25-35).

- [ ] **Step 2: Run all Python tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (existing 14 + new ~30-40).

- [ ] **Step 3: Run linters**

Run: `./lint_code.sh && ./check_all.sh`
Expected: No lint errors, no clippy warnings.

- [ ] **Step 4: Smart-commit**

Use the `/smart-commit` skill to sync CLAUDE.md and README.md with the new test infrastructure, then commit everything.
