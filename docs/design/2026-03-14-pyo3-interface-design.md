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
| Cost computation | Stays in Python | Cost function logic is GA-specific and evolves with training experiments; Rust returns raw sim data, Python computes cost |

## Crate Structure

```
src/rust/
  Cargo.toml              <- workspace root (adds workspace.members)
  src/                    <- existing core crate (unchanged except public RunOutput)
  aerocapture-py/
    Cargo.toml            <- PyO3 cdylib, depends on aerocapture
    src/
      lib.rs              <- #[pymodule] entry point
      config.rs           <- TOML loading + dict override merging
      results.rs          <- RunOutput -> numpy/dict conversion
      batch.rs            <- Rayon parallel batch runner
    pyproject.toml        <- maturin build backend config
```

**Build:**
```bash
cargo build --release                        # CLI (unchanged)
cd src/rust/aerocapture-py && maturin develop --release  # Python module
```

## Core Crate Change

The existing private `SimResult` in `runner.rs` (containing `sim_idx`, `final_line: [f64; 52]`, `photo_lines: Vec<[f64; 24]>`) is internal to the file writer. It stays as-is.

A new **public** struct `RunOutput` is added to `lib.rs`, populated from the same data that currently fills `final_line` and `photo_lines`:

```rust
/// Public output from a single simulation run, for use by PyO3 and tests.
pub struct RunOutput {
    pub trajectory: Vec<[f64; 8]>,   // per-timestep: [r, lon, lat, V, gamma, psi, flux, time]
    pub final_record: [f64; 52],     // full 52-column final record (same layout as file output)
    pub captured: bool,              // true if orbit is bound (ecc < 1 && energy < 0)
}
```

**Key design choice**: `final_record` preserves the existing 52-column layout verbatim. This means:
- The Python cost function (`compute_cost`) works unchanged — same column indices
- All 52 values are available (orbital elements, delta-V components, max heat flux, bounce alt, etc.)
- No risk of omitting a field that some analysis script depends on
- Named field accessors can be added as convenience methods in the PyO3 wrapper without changing the core struct

The `run()` function in `runner.rs` gains a second return path: in addition to the existing `SimResult` (used by file writers), it also builds `RunOutput`. The file-writing path (CLI) remains unchanged.

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

result.trajectory    # numpy array, shape (N, 8)
result.final_record  # numpy array, shape (52,) — same column layout as file output
result.captured      # bool

# Convenience accessors (thin wrappers over final_record columns)
result.energy        # final_record[7], MJ/kg
result.ecc           # final_record[9]
result.periapsis_alt # final_record[14], km
result.apoapsis_alt  # final_record[15], km
result.delta_v       # final_record[41], m/s total
result.peri_err      # final_record[29], km
result.apo_err       # final_record[30], km
```

### Batch run (parallel)

```python
results = aero.run_batch(
    toml_path="configs/training/msr_aller_eqglide_train.toml",
    overrides_list=[
        {"montecarlo.seed": i, "guidance.equilibrium_glide.k_hdot": 0.15}
        for i in range(100)
    ],
    n_threads=8,              # defaults to num_cpus; uses a scoped Rayon thread pool
    include_trajectories=False,  # set True to include per-sim trajectory arrays
)

results.final_records  # numpy array, shape (100, 52) — directly compatible with compute_cost()
results.captured       # numpy array, shape (100,), bool
results.trajectories   # list of numpy arrays, only populated if include_trajectories=True
```

`run_batch()` uses a **scoped Rayon thread pool** (`rayon::ThreadPoolBuilder::new().num_threads(n).build_scoped()`), separate from the global pool, to avoid conflicts with any internal Rayon usage. A new scoped pool is created per `run_batch()` call — acceptable overhead for typical GA workloads (~50 calls).

The base config and data tables (atmosphere, aerodynamics, reference trajectory) are **loaded once** and shared across all batch items. Only the overridden fields differ per run.

### Config inspection

```python
config = aero.load_config("configs/training/msr_aller_eqglide_train.toml")
config.guidance_scheme    # "equilibrium_glide"
config.monte_carlo_seeds  # 50
config.to_dict()          # full config as nested Python dict
```

`load_config()` returns a `#[pyclass]` wrapper around the parsed `toml::Value` tree. It exposes a handful of convenience properties (`.guidance_scheme`, `.monte_carlo_seeds`) and `.to_dict()` for full access. It does NOT wrap `SimInput` — that stays internal to Rust.

## Config Override Mechanism

Overrides use dot-separated key paths mapping to TOML table structure:

1. Parse TOML file into `toml::Value` (generic tree)
2. Walk the overrides dict: `"guidance.equilibrium_glide.k_hdot"` -> nested key path
3. Patch the tree in-memory
4. Deserialize patched tree into `SimInput` (existing config struct)

No TOML file rewriting in the hot path.

**Type coercion rules:**
- Python `int` auto-promotes to TOML `Float` if the existing key holds a float
- Python `float` stays as TOML `Float`
- Python `str` stays as TOML `String`
- Python `bool` stays as TOML `Boolean`
- Python `list` maps to TOML `Array` (elements coerced recursively)
- Type mismatch (e.g., `str` for a `Float` key) raises `ValueError` with the key path and expected type

## Python-Side Migration

Gradual migration with subprocess fallback:

| File | Change |
|------|--------|
| `evaluate.py` | `run_simulation()` calls `aero.run()`. `compute_cost()` updated to use 0-based 52-column indices (see below). Falls back to subprocess if `aerocapture_rs` not importable. |
| `train.py` | GA evaluation loop uses `aero.run_batch()` for population-level parallelism. `results.final_records` (shape `(N, 52)`) feeds directly into updated `compute_cost()`. |
| `compare_guidance.py` | `aero.run_batch()` replaces subprocess loop. |
| `config.py` | `executable` field becomes optional/deprecated. New field: `use_pyo3: bool = True`. |
| `final_report.py` | Uses `run_batch()` for 1000-sim MC evaluation. |

**Cost function stays in Python; column indices updated.** The existing `compute_cost()` uses a legacy 53-column layout where column 0 is `sim_number` and data starts at column 1 (e.g., `energy` at `[:, 8]` = data index 7). The PyO3 API returns a clean 52-column `final_record` with no `sim_number` prefix. `compute_cost()` column indices are updated to 0-based:

| Field | Legacy index (53-col) | New index (52-col) |
|-------|----------------------|-------------------|
| energy | 8 | 7 |
| ecc | 10 | 9 |
| sim_time | 28 | 27 |
| peri_err | 30 | 29 |
| apo_err | 31 | 30 |
| dv_total | 42 | 41 |

The subprocess fallback path in `run_simulation()` strips the `sim_number` column before returning, so both paths produce `(N, 52)` arrays with the same layout.

The subprocess path remains as a fallback for environments without the PyO3 module (CI, quick testing).

## Testing Strategy

### Rust (aerocapture-py crate)

- Config override merging: dot-path keys -> TOML tree patching, including type coercion and error cases
- `RunOutput` -> numpy array conversion: shape, dtype, column layout correctness
- Integration: golden TOML -> PyO3 run -> compare `final_record` against existing golden reference data in `tests/reference_data/`

### Python

- Existing ~180 tests pass unchanged (subprocess fallback)
- New PyO3 tests:
  - `test_pyo3_single_run`: golden config, `result.final_record` matches subprocess-parsed output bit-for-bit
  - `test_pyo3_batch`: N sims, `results.final_records` shape and value consistency
  - `test_pyo3_overrides`: patched params actually take effect (different gain -> different cost)
  - `test_pyo3_cost_compat`: feed PyO3 `final_records` into existing `compute_cost()`, verify identical output to subprocess path
  - `test_pyo3_fallback`: mock `aerocapture_rs` unimportable, subprocess path still works
  - `test_pyo3_type_coercion`: int-to-float promotion, type mismatch errors
- **Bit-identical regression**: PyO3 results == subprocess results for identical configs (same Rust code, same inputs)

### CI

- Separate CI job for PyO3 tests (maturin build + test), keeps existing fast Rust/Python jobs unaffected
- The PyO3 job installs with `uv pip install maturin && cd src/rust/aerocapture-py && maturin develop --release`, then runs `pytest tests/ -k pyo3`

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
- `numpy` (PyO3 numpy integration crate, `pyo3-numpy` on crates.io)
- `rayon` for batch parallelism
- `toml` (already used in core crate)

### Python
- `maturin` as build backend (in `pyproject.toml` `[build-system]`)
- `aerocapture-rs` installed via `maturin develop` during development
- For distribution: `maturin build` produces a wheel

### Build tool
- `maturin` added to the dev dependency group in `pyproject.toml` (canonical install path: `uv sync --group dev`)
- CI uses the same `uv sync --group dev` path, then `cd src/rust/aerocapture-py && maturin develop --release`
