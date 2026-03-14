# PyO3 Rust-Python Interface for Aerocapture

**Date:** 2026-03-14
**Status:** Draft

## Motivation

The GA training pipeline currently invokes the Rust simulator as a subprocess per evaluation: write TOML config to disk, fork+exec the binary, parse output files. This introduces per-evaluation overhead (process fork ~2-5ms, file I/O ~0.6ms) and limits parallelism to sequential subprocess calls.

A PyO3 interface eliminates this overhead and enables:

1. **Performance** — no subprocess/file I/O overhead; Rayon-based batch parallelism across all CPU cores (~8x speedup on 8-core machine for population evaluation).
2. **Ergonomics** — structured results as numpy arrays and Python dicts instead of file parsing; in-memory config patching instead of TOML file rewriting; enables richer analysis like seed rotation strategies that need per-sim cost distributions.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Crate structure | Separate `aerocapture-py` workspace member | Clean separation; CLI builds unaffected; standard Rust-PyO3 pattern |
| API granularity | Run-level (not step-level) | Sufficient for GA training and analysis; avoids exposing internal sim state across FFI |
| Config passing | Hybrid: base TOML + Python dict overrides | Maps to existing GA workflow (patch params per individual); avoids duplicating full config as Python dataclasses |
| Python callbacks | Not supported | No need for Python-defined guidance laws; simplifies FFI boundary |
| CLI coexistence | Both CLI and PyO3 module | CLI for quick runs and CI; PyO3 for training/analysis |
| Migration | Gradual with subprocess fallback | Existing tests keep passing; CI can run without PyO3 module |

## Crate Structure

```
src/rust/
  Cargo.toml              <- workspace root (adds workspace.members)
  src/                    <- existing core crate (unchanged except SimResult return)
  aerocapture-py/
    Cargo.toml            <- PyO3 cdylib, depends on aerocapture
    src/
      lib.rs              <- #[pymodule] entry point
      config.rs           <- TOML loading + dict override merging
      results.rs          <- SimResult -> numpy/dict conversion
      batch.rs            <- Rayon parallel batch runner
    pyproject.toml        <- maturin build backend config
```

**Build:**
```bash
cargo build --release                        # CLI (unchanged)
cd src/rust/aerocapture-py && maturin develop --release  # Python module
```

## Core Crate Change

The single change to the existing `aerocapture` crate: `run_simulation()` in `runner.rs` returns a `SimResult` struct in addition to writing files (CLI path unchanged).

```rust
pub struct SimResult {
    pub trajectory: Vec<[f64; 8]>,  // per-timestep: [r, lon, lat, V, gamma, psi, flux, time]
    pub final_state: FinalState,     // orbital elements, delta-v, cost
    pub captured: bool,
}

pub struct FinalState {
    pub apoapsis: f64,
    pub periapsis: f64,
    pub inclination: f64,
    pub delta_v: f64,
    pub cost: f64,
    pub sma: f64,
    pub ecc: f64,
    pub raan: f64,
    pub aop: f64,
    pub true_anomaly: f64,
}
```

## Python API

### Single run

```python
import aerocapture_rs as aero

result = aero.run(
    toml_path="configs/training/msr_aller_eqglide_train.toml",
    overrides={
        "guidance.equilibrium_glide.k_hdot": 0.15,
        "montecarlo.seed": 42,
    }
)

result.trajectory   # numpy array, shape (N, 8)
result.final        # dict: {apoapsis, periapsis, inclination, delta_v, cost, captured}
result.orbital      # dict: {sma, ecc, inc, raan, aop, ta}
```

### Batch run (parallel)

```python
results = aero.run_batch(
    toml_path="configs/training/msr_aller_eqglide_train.toml",
    overrides_list=[
        {"montecarlo.seed": i, "guidance.equilibrium_glide.k_hdot": 0.15}
        for i in range(100)
    ],
    n_threads=8,  # defaults to num_cpus
)

results.costs        # numpy array, shape (100,)
results.captured     # numpy array, shape (100,), bool
results.finals       # list of dicts
results.trajectories # list of numpy arrays (optional, off by default)
```

Rayon work-stealing on the Rust side — true parallelism, no GIL.

### Config inspection

```python
config = aero.load_config("configs/training/msr_aller_eqglide_train.toml")
config.guidance_scheme    # "equilibrium_glide"
config.monte_carlo_seeds  # 50
config.to_dict()          # full config as nested Python dict
```

## Config Override Mechanism

Overrides use dot-separated key paths mapping to TOML table structure:

1. Parse TOML file into `toml::Value` (generic tree)
2. Walk the overrides dict: `"guidance.equilibrium_glide.k_hdot"` -> nested key path
3. Patch the tree in-memory
4. Deserialize patched tree into `SimInput` (existing config struct)

No TOML file rewriting in the hot path.

## Python-Side Migration

Gradual migration with subprocess fallback:

| File | Change |
|------|--------|
| `evaluate.py` | `run_simulation()` calls `aero.run()` instead of `subprocess.run()` + file parse. Falls back to subprocess if `aerocapture_rs` not importable. |
| `train.py` | GA evaluation loop uses `aero.run_batch()` for population-level parallelism. |
| `compare_guidance.py` | `aero.run_batch()` replaces subprocess loop. |
| `config.py` | `executable` field becomes optional/deprecated. New field: `use_pyo3: bool = True`. |
| `final_report.py` | Uses `run_batch()` for 1000-sim MC evaluation. |

The subprocess path remains as a fallback for environments without the PyO3 module (CI, quick testing).

## Testing Strategy

### Rust (aerocapture-py crate)

- Config override merging: dot-path keys -> TOML tree patching
- `SimResult` -> numpy array conversion: shape, dtype correctness
- Integration: golden TOML -> PyO3 run -> compare against `tests/reference_data/`

### Python

- Existing ~180 tests pass unchanged (subprocess fallback)
- New PyO3 tests:
  - `test_pyo3_single_run`: golden config, results match subprocess output
  - `test_pyo3_batch`: N sims, shape and value consistency
  - `test_pyo3_overrides`: patched params actually take effect
  - `test_pyo3_fallback`: mock `aerocapture_rs` unimportable, subprocess works
- **Bit-identical regression**: PyO3 results == subprocess results for identical configs (same Rust code, same inputs)

### CI

- Separate CI job for PyO3 tests (maturin build + test), keeps existing fast Rust/Python jobs unaffected

## Performance Expectations

Typical GA run: 50 generations x 20 population = 1000 evaluations, ~20ms avg sim.

| Metric | Sequential subprocess | PyO3 + Rayon (8 cores) |
|--------|----------------------|------------------------|
| Per-eval overhead | ~5ms | ~0.01ms |
| Parallelism | 1 core | 8 cores |
| Total time | ~25s | ~3-4s |

PyO3 does NOT speed up the sim physics (already compiled Rust). Gains are from eliminated overhead + parallel evaluation.

## Dependencies

### Rust (aerocapture-py)
- `pyo3` with `extension-module` feature
- `numpy` (PyO3 numpy integration)
- `rayon` for batch parallelism
- `toml` (already used in core crate)

### Python
- `maturin` as build backend
- `aerocapture-rs` as optional dependency in `pyproject.toml`

### Build tool
- `maturin` (installed via `uv pip install maturin`)
