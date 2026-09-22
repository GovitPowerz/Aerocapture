# Rust Simulator Test Suite Design

## Motivation

The Rust simulator was validated against the Fortran reference (22/24 photo columns bit-identical across 725 timesteps), but has only 14 inline unit tests (dispersions + initialization). The physics core, GNC chain, integrator, and sim loop have zero Rust-side tests. All regression testing goes through Python scripts comparing against Fortran golden references.

This design establishes a standalone Rust test suite so the simulator can be tested, refactored, and extended without depending on the legacy Fortran codebase.

## Approach: Hybrid Testing Pyramid

Three tiers built in parallel:

1. **Unit tests** — analytical ground truth for physics/math modules
2. **Integration tests** — snapshot-based tests for composed subsystems
3. **E2E tests** — full sim runs asserting on physical invariants

## Dependencies

Add to `Cargo.toml` as dev-dependencies:

- `approx` — float comparison macros (`assert_relative_eq!`, `assert_abs_diff_eq!`)
- `rstest` — parameterized tests (`#[rstest]` + `#[case]`)

## Tier 1: Unit Tests

Inline `#[cfg(test)]` blocks in each module. Analytical/hand-computed expected values.

### Physics Layer

| Module | Test Strategy | Example Cases |
|--------|--------------|---------------|
| `physics/gravity.rs` | Closed-form J2 gravity | Equator vs pole (J2 effect), known altitude, zero-J2 reduces to spherical |
| `physics/atmosphere.rs` | Table lookup | Interpolation between entries, boundary handling (below min/above max), exact table hit |
| `physics/aerodynamics.rs` | F = Cx * q * S | Known Cx/AoA lookup, force at known dynamic pressure, zero-velocity = zero force |
| `physics/winds.rs` | Simple profiles | Zero wind, known wind at known altitude |

### GNC Layer

| Module | Test Strategy | Example Cases |
|--------|--------------|---------------|
| `navigation/coordinates.rs` | Roundtrip + analytical | Spherical <-> Cartesian identity, geodetic at equator/pole, E_circular = -mu/2a |
| `navigation/estimator.rs` | Filter dynamics | Zero bias = passthrough, exponential decay, step response |
| `guidance/ftc.rs` | Gain response | Zero deviation = zero correction, known error -> expected bank delta |
| `guidance/reference.rs` | Trivial | Returns constant bank angle |
| `guidance/equilibrium_glide.rs` | Equilibrium condition | At equilibrium (L*cos(bank) = g - V^2/r), correction ~ zero |
| `guidance/energy_controller.rs` | Energy tracking | Known dissipation rate -> expected bank response |
| `guidance/predguid.rs` | Drag tracking | Reference drag matched = zero correction |
| `guidance/fnpag.rs` | Predictor-corrector | Convergence on simple test case |
| `guidance/neural.rs` | Forward pass | Known weights -> known output (deterministic matrix math) |
| `control/pilot.rs` | Rate limiting | Within limit = passthrough, exceeding limit = clamped |

### Integration / Orbit Layer

| Module | Test Strategy | Example Cases |
|--------|--------------|---------------|
| `integration/rk4.rs` | Solve known ODE | dy/dx = x with Gill's variant -> y = x^2/2, simple harmonic oscillator |
| `integration/sequencer.rs` | Cadence logic | Modules fire at correct timesteps |
| `orbit/elements.rs` | Roundtrip + textbook | Known state -> Keplerian elements (circular, elliptical, hyperbolic) |
| `orbit/maneuver.rs` | Analytical | Hohmann transfer delta-V, circular orbit delta-V = 0 |

### Data Layer

| Module | Test Strategy | Notes |
|--------|--------------|-------|
| `data/dispersions.rs` | Already has 11 tests | Keep, extend if needed |
| `data/*` (other) | TOML parsing | Load minimal config, verify fields parsed correctly |

Estimated: ~50-60 tests across 18+ modules.

## Tier 2: Integration Tests

Located in `src/rust/tests/integration/`. Snapshot-based where analytical verification is impractical.

### Snapshot Generation

A one-time helper runs the validated simulator and dumps intermediate state to JSON at key boundaries. Snapshots committed to `tests/snapshots/`.

### Test Cases

| Test | Exercises | Snapshot Points |
|------|----------|-----------------|
| Single timestep pipeline | nav -> guidance -> control -> RK4 | State vector after each stage |
| GNC chain per scheme | All 6 schemes, parameterized with `#[rstest]` | Post-guidance bank command, post-pilot bank |
| RK4 with real physics | Gravity + atmosphere + aero for multiple steps | State after 1, 5, 10 steps |
| Nav filter convergence | Density filter over ~50 steps | Filter estimate vs truth trajectory |
| Phase transitions | Bounce, capture, atmospheric exit detection | Phase flags at expected timesteps |
| MC dispersion pipeline | Apply dispersions -> run -> check spread | Final elements mean/std within bounds |
| TOML config round-trip | Load full config -> verify SimData | All config fields |

### File Structure

```
src/rust/tests/
  integration/
    mod.rs
    single_timestep.rs
    gnc_chain.rs
    rk4_physics.rs
    nav_filter.rs
    phase_transitions.rs
    monte_carlo.rs
    config_loading.rs
  snapshots/
    single_timestep.json
    gnc_chain_ftc.json
    gnc_chain_eqglide.json
    ...
```

### Shared Test Utilities

`tests/test_helpers.rs` providing:

- `make_test_sim_data()` — minimal but complete SimData
- `make_nav_state(overrides)` — navigation state with sensible defaults
- `load_snapshot(name)` — deserialize snapshot JSON
- `assert_state_approx_eq(actual, expected, tol)` — vector comparison with `approx`

Estimated: ~15-20 tests.

## Tier 3: End-to-End Tests

Located in `src/rust/tests/e2e/`. Full sim runs from TOML config, asserting on physical invariants.

### Test Cases

| Test | Config | Assertions |
|------|--------|------------|
| Reference trajectory | Constant bank, no guidance | Final altitude/velocity/FPA in range, no panics |
| FTC capture | FTC, single sim | Orbit captured (e < 1), apoapsis in target band, delta-V below threshold |
| Each guidance scheme | 6 schemes, `#[rstest]` | Capture criteria: orbit achieved, cost reasonable |
| Monte Carlo | Any scheme, n_sims=50, seeded | All runs produce output, mean apoapsis in band, no NaN, deterministic |
| Edge cases | Near-corridor-boundary entries | Handles gracefully — capture or clean exit, no divergence |
| Output format | Any config | CSV parses, correct columns, headers match |

### Key Decisions

- **No Fortran references** — assertions on physical invariants, not bit-exact matching
- **Test-specific TOML configs** — minimal, committed to repo, independent of training configs
- **Physically meaningful assertions** — "apoapsis is 250 +/- 50 km" not "column 7 line 300 = 3.456789D+05"
- **Determinism** — all MC tests use fixed seed for reproducibility

### File Structure

```
src/rust/tests/
  e2e/
    mod.rs
    reference_trajectory.rs
    guidance_schemes.rs
    monte_carlo.rs
    edge_cases.rs
    output_format.rs
  configs/
    test_reference.toml
    test_ftc.toml
    test_eqglide.toml
    ...
```

Estimated: ~10-15 tests.

## What Gets Retired

Once the Rust test suite is in place:

1. `tests/test_regression.py` -> replaced by Rust E2E
2. `tests/test_mc_domain.py` -> replaced by Rust MC tests
3. `tests/reference_data/` -> archived (27 MB freed from repo)
4. `tests/test_parsers.py` stays (Python analysis tools still need it)

## What Doesn't Change

- `check_all.sh` already runs `cargo test`, picks up new tests automatically
- Python analysis tools and their tests remain independent
- `lint_code.sh`, `setup_env.sh` unaffected

## Estimated Total

~75-95 Rust tests across all three tiers, covering physics, GNC, integration, orbit, data loading, and full simulation runs.
