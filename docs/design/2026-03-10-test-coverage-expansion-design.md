# Test Coverage Expansion Design

**Date**: 2026-03-10
**Approach**: Risk-Weighted Coverage (Approach B)
**Balance**: ~50/50 Rust/Python
**Testing styles**: Deterministic hand-written + property-based (proptest/hypothesis)
**Infrastructure**: Invest upfront in shared fixtures, builders, conftest.py

---

## Current State

### Rust (~113 tests)
- **Well-covered**: Physics (gravity, atmosphere, aerodynamics, coordinates), integration (RK4, sequencer), orbital mechanics, control (pilot, attitude), dispersions, data loading
- **Golden regression**: 6 guidance schemes with bit-level deterministic validation
- **E2E**: 9 subprocess tests covering all configs + MC determinism

### Python (~14 tests)
- **Covered**: IO parsers (CSV detection, column mapping, empty files), MC integration (determinism, seed variation), regression (3 golden trajectories)
- **Essentially untested**: GA training pipeline (~1,000+ lines), plotting, config validation

### Key Gaps (both sides)
- Guidance algorithms have zero unit tests (Rust)
- GA chromosome encode/decode, TOML patching, GA operators untested (Python)
- No error-path or NaN/Inf tests anywhere
- No shared test infrastructure (duplicated fixtures, no conftest.py)
- `physics/winds.rs` completely untested
- Navigation estimator under-tested (5 tests for ~400 lines)

---

## Section 1: Test Infrastructure

### Rust — Shared Test Helpers

Consolidate scattered fixtures in `src/rust/tests/common/`:

- **`fixtures.rs`**: `TestSimBuilder` with builder pattern — `.with_velocity(5687.0).with_guidance(Guidance::Ftc).build()`. Consolidates `test_sim_data()`, `make_aero()`, `make_capsule()`, `mars_test_fixtures()`.
- **`assertions.rs`**: Domain-specific asserts — `assert_finite_bounded(value, min, max)`, `assert_bank_angle_valid(angle)`, `assert_orbital_elements_physical(elements)`. Wraps `approx` with meaningful error messages.
- **`proptest_strategies.rs`**: `proptest` strategies for valid flight states (altitude 0–300km, velocity 1000–8000 m/s, gamma -10° to +5°), guidance params within bounds, aerodynamic tables.

### Python — conftest.py + Shared Fixtures

- **`tests/conftest.py`**: Session-scoped Rust build fixture (deduplicated from test_regression.py/test_mc_domain.py), temp directory fixture, common `run_sim()` helper.
- **`tests/fixtures/factories.py`**: Factory functions — `make_training_config(**overrides)`, `make_network_config(**overrides)`, `make_chromosome(scheme, strategy="mid")`, `make_final_row(**overrides)`.

### Dependencies
- Add `hypothesis` to `[dependency-groups] dev` in `pyproject.toml`
- Add `proptest` to `[dev-dependencies]` in `Cargo.toml`

---

## Section 2: Rust — Guidance Algorithm Unit Tests

### FTC (`gnc/guidance/ftc.rs`)
- **Deterministic cases**: Known entry state → expected bank angle. Test each phase (capture, glide, exit) independently.
- **Property tests**: For any valid flight state, output bank angle is finite and within [0°, 180°]. Gains × zero error → zero correction.
- **Edge cases**: Zero velocity, zero dynamic pressure, altitude at phase transition boundaries.

### Neural Network (`gnc/guidance/neural.rs`)
- **JSON loading**: Valid architecture parses correctly, malformed JSON returns error.
- **Forward pass math**: Hand-computed 2-layer network (tiny weights) → verify output matches manual calculation.
- **Property tests**: For any valid input vector, output is finite. All-zero weights → known output (biases through activations).

### FNPAG (`gnc/guidance/fnpag.rs`)
- **Convergence**: Known drag profile → predictor-corrector converges within max iterations.
- **Property tests**: Output bank angle always finite and bounded for valid states.
- **Edge cases**: Max iterations reached (doesn't hang), near-zero lift.

### Reference (`gnc/guidance/reference.rs`)
- Returns configured constant bank angle. Quick win.

### Guidance Dispatcher (`gnc/guidance/mod.rs`)
- Correct scheme selected based on config enum.
- Phase transitions fire at expected conditions.

---

## Section 3: Rust — Other Coverage Gaps

### Wind Model (`physics/winds.rs`)
- Zero wind config → zero wind vector. Non-zero wind → correct direction/magnitude.
- Property tests: wind output always finite for valid altitudes.

### Navigation Estimator (`gnc/navigation/estimator.rs`)
- **Density filter stability**: Adversarial density ratios (spikes, near-zero drops) → filter gain stays bounded (historical lambda corruption regression test).
- **Property tests**: For any valid nav state + biases, outputs are finite. `coefro` stays within [0, clamp_max].
- **Branch coverage**: High altitude reset, bounce detection edge (gamma exactly 0), first-call initialization.

### Simulation Init (`simulation/init.rs`)
- Reference trajectory loading: energy/pdyn/hdot tables load and interpolate correctly.
- Property test: dispersed state within expected sigma bounds of nominal.

### Error Path Tests (cross-cutting)
- **NaN/Inf propagation**: Feed NaN into each guidance scheme → doesn't silently produce garbage (safe default or property test catches non-finite output).
- **Out-of-range inputs**: Negative altitude, velocity = 0, gamma = ±90° — no panics, no division by zero.

---

## Section 4: Python — GA Training Pipeline Tests

### Chromosome Encode/Decode (`population.py`, `evaluate.py`)
- **Roundtrip**: `encode → decode → original params` within tolerance. Same for NN weights.
- **Property tests (hypothesis)**: For any chromosome in [0, 1], decoded params respect bounds from `param_spaces.py`. Log-scale decoding produces correct order of magnitude.
- **Edge cases**: All-zeros, all-ones, single-element chromosome.

### Cost Function (`evaluate.py` — `compute_cost`)
- **NaN/Inf inputs**: Energy = NaN, eccentricity = Inf → penalty, not crash.
- **Boundary cases**: Energy exactly 0 (parabolic), eccentricity exactly 1.0, negative SMA.
- **Property tests**: Cost always non-negative and finite for physically plausible inputs.

### TOML Patching (`evaluate.py` — `write_guidance_toml`)
- **Roundtrip**: Decode chromosome → write TOML → parse back → values match.
- **All 6 schemes**: Each scheme's param space produces a valid TOML the Rust parser accepts.

### Config Validation (`config.py`)
- Each training TOML in `configs/training/` loads without error.
- Missing required fields → clear error, not KeyError.
- Wrong types → caught early.

### GA Operators (`train.py`)
- **Roulette selection**: Probability proportional to fitness. Zero-fitness population doesn't crash.
- **Crossover/mutation**: Output chromosome same length as input. Values stay within [0, 1].
- **Property tests**: Valid population → one generation → valid population.

---

## Section 5: What We're NOT Testing

- **Plotting modules** (`plotting/`): Low bug risk, high effort, visual by nature.
- **`compare_guidance.py`**: Orchestration — covered by its components.
- **Struct-only data modules** (Rust `data/*.rs`): Definitions + `Default` impls — covered by integration tests.
- **`mod.rs` re-exports**: Zero logic, zero risk.
