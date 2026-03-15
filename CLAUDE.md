# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aerocapture is a trajectory simulation tool for aerocapture maneuvers (primarily Mars Sample Return). The project was modernized from a legacy Fortran 77 codebase into a **Rust simulator** with **Python analysis tools**. The Rust simulator was **validated against the Fortran reference** (now removed from the working tree but preserved in git history) — FTC guided trajectories matched to bit-level precision across all 725 timesteps (22/24 photo columns exact; the remaining 2 were Fortran uninitialized variable artifacts).

The simulation models a spacecraft entering a planet's atmosphere at hyperbolic velocity, using aerodynamic forces and bank angle modulation to capture into a target orbit. The GNC chain is: Navigation (state estimation + density filter) -> Guidance (one of 6 algorithms: FTC, NN, Equilibrium Glide, Energy Controller, PredGuid, FNPAG) -> Control (pilot dynamics + roll reversal). All guidance schemes have TOML-configurable parameters and can be GA-optimized.

## Build & Development Commands

```bash
# ── Rust Simulator ──
cd src/rust
cargo build --release              # Build optimized binary
# Run from repo root:
./src/rust/target/release/aerocapture configs/test/test_ref_orig.toml

# ── PyO3 Bindings ──
cd src/rust/aerocapture-py
maturin develop --release          # Build + install aerocapture_rs module
# Or via uv:
uv run maturin develop --release

# ── Python Analysis ──
uv sync                            # Install dependencies (Python >=3.14)
uv sync --group dev                # Include dev tools (pytest, ruff, mypy, maturin)
pytest tests                       # Run all tests
pytest tests/test_foo.py::test_bar -v

# ── Utility Scripts (from repo root) ──
./build.sh                         # Build Rust binary + PyO3 bindings (-c to clean artifacts)
./setup_env.sh                     # Create fresh .venv + install deps
./lint_code.sh                     # Run ruff (imports, format, lint) + mypy
./check_all.sh                     # Rust: test + fmt --check + clippy + release build
./upgrade_dependencies.sh          # uv sync --upgrade
```

## Architecture

### Rust Simulator (`src/rust/`)

The Rust code is a reimplementation of the original Fortran algorithms with all variable names modernized to explicit English (no French/Fortran legacy names remain). The crate has both `lib.rs` (public API: `RunOutput` struct + `run_for_api()`) and `main.rs` (CLI entry). A Cargo workspace contains two members: the core `aerocapture` crate and the `aerocapture-py` PyO3 binding crate. TOML config as a CLI argument (`./aerocapture config.toml`) is the only supported input format. TOML supports all 6 guidance schemes and inline vehicle/mission data.

```
src/rust/src/
  main.rs                          — CLI entry, TOML config loading
  config.rs                        — TOML parser (Planet, MissionType, SimInput)
  data/
    mod.rs, SimData                — Top-level data container
    atmosphere.rs                  — Atmosphere density table
    aerodynamics.rs                — Cx/Cz vs AoA tables
    capsule.rs                     — Vehicle: mass, reference area, max bank rate
    guidance_params.rs             — Guidance law config: FTC gains, EqGlide, EnergyCtrl, PredGuid, FNPAG params
    dispersions.rs                 — Monte Carlo dispersion profiles
    navigation.rs                  — Navigation error profiles
    incidence.rs                   — AoA profile tables
    pilot.rs                       — Pilot dynamics parameters
  physics/
    gravity.rs                     — J2 oblate gravity
    atmosphere.rs                  — Density lookup
    aerodynamics.rs                — Force computation
    winds.rs                       — Wind model
  gnc/
    navigation/
      estimator.rs                 — State estimation + density filter
      coordinates.rs               — Spherical<>Cartesian, geodetic, total energy
    guidance/
      ftc.rs                       — FTC capture-phase guidance
      reference.rs                 — Constant bank angle mode
      neural.rs                    — NN guidance (modular JSON architecture, GA-trained)
      equilibrium_glide.rs         — Equilibrium glide with hdot damping + velocity bias
      energy_controller.rs         — Energy dissipation tracking via pdyn/hdot feedback
      predguid.rs                  — Apollo/Shuttle-heritage drag tracking guidance
      fnpag.rs                     — Lu's numerical predictor-corrector (FNPAG)
    control/
      pilot.rs                     — Pilot dynamics
      attitude.rs                  — Attitude command realization
  integration/
    rk4.rs                         — Gill-variant RK4
    sequencer.rs                   — Module cadence scheduling
  orbit/
    elements.rs                    — Orbital elements from state vector
    maneuver.rs                    — Delta-V cost computation
  simulation/
    runner.rs                      — Main sim loop: run() for CLI, run_for_api() for PyO3
    init.rs                        — Per-run initialization
    output.rs                      — File writers (photo, final, CSV)
```

Key Rust dependency: `nalgebra` for vector/matrix ops.

### PyO3 Bindings (`src/rust/aerocapture-py/`)

Separate workspace member crate providing Python bindings via PyO3. Built with `maturin develop --release`. Imports as `aerocapture_rs` in Python.

```
src/rust/aerocapture-py/src/
  lib.rs         — Module entry: run(), run_mc(), run_batch(), load_config()
  config.rs      — TOML loading with dot-path override merging
  results.rs     — SimResult/BatchResults pyclasses with numpy getters
  batch.rs       — Rayon parallel batch execution
```

Key API:
- `aerocapture_rs.run(toml_path, overrides=None)` → `SimResult` with `.final_record` (52,), `.captured`, `.energy`, `.ecc`, etc. Returns first result only (use `run_mc` for multi-sim).
- `aerocapture_rs.run_mc(toml_path, overrides=None, include_trajectories=False)` → `BatchResults` with all n_sims results. Use for MC evaluations needing the full distribution.
- `aerocapture_rs.run_batch(toml_path, overrides_list, n_threads=None, include_trajectories=False)` → `BatchResults` with `.final_records` (N, 52)
- `aerocapture_rs.load_config(toml_path)` → Python dict

The training pipeline (`evaluate.py`) auto-detects PyO3 availability and falls back to subprocess if not installed. Override dict uses dot-separated TOML key paths with type coercion (int→float when existing field is float).

### Data Files (`data/`)

- `data/atmosphere/mars.dat` — Mars density vs altitude table (tabulated MarsGram 3.8)
- `data/atmosphere/earth.dat` — Earth atmosphere table
- `data/reference_trajectory/msr_aller.dat` — MSR reference trajectory (energy vs pdyn/hdot/cos_bank)
- `data/reference_trajectory/esr_aller.dat` — ESR reference trajectory

### Input Configuration

TOML config files in `configs/` are the only supported input format, organized into subdirectories: `configs/nominal/` (simulation configs), `configs/training/` (GA training configs), `configs/test/` (golden test configs). Each config specifies mission, guidance scheme, vehicle, entry conditions, aerodynamics, Monte Carlo settings, and data file paths. The NN weight file path (`[data] neural_network`) and optional architecture override (`[network] layer_sizes`, `activations`) are read from TOML at training time. The `[simulation]` section supports `max_time` (default: 3000.0 s) as a hard wall to prevent runaway simulations.

### Python Tools (`src/python/`, `pyproject.toml`)

Python analysis package (numpy, pandas, matplotlib, deap, scipy) for:

- Output file parsers (photo, final, CSV files)
- Visualization (corridor plots, MC ensembles, CDF of correction cost)
- GA training pipeline: optimizes any guidance scheme's parameters (not just NN weights)
  - `train.py` — Main GA loop with checkpoint save/resume (`--guidance <scheme> --toml <config> [--no-tui] [--rotate-seeds | --adaptive-seeds] [--seed-pool-cap N] [--cost-alpha F] [--cvar-percentile P] [--skip-final-report] [--final-n-sims N]`). Graceful KeyboardInterrupt handling: Ctrl+C saves checkpoint and returns cleanly with `interrupted: True`.
  - `param_spaces.py` — Per-scheme parameter bounds (with optional log-scale encoding)
  - `evaluate.py` — Decode chromosome -> write params (NN JSON or patched TOML) -> run sim -> cost. Uses PyO3 direct call when `aerocapture_rs` is available, subprocess fallback otherwise.
  - `compare_guidance.py` — Fair head-to-head comparison on identical MC scenarios
  - `initialization.py` — Activation-aware weight init (Xavier/He/LeCun uniform) for NN population seeding
  - `seed_pool.py` — Adaptive seed pool for MC dispersions: rolling pool of seeds scored by population-relative difficulty (CVaR-blended fitness), with incremental growth and redundancy eviction. `SeedPool` class with `evaluate_population()` (supports PyO3 batch), `score_difficulty()`, `evict_redundant()`, and checkpoint serialization.
  - `weight_stats.py` — Per-layer weight statistics (min/max/mean/std) for training instrumentation
- Training visualization:
  - `metrics.py` — Pure metric functions: cost stats, diversity, capture rate, convergence speed, stagnation
  - `logger.py` — `TrainingLogger`: writes one JSONL line per generation; in-memory buffer for live display
  - `display.py` — `LiveDisplay`: Rich TUI with sparklines, ETA, progress bar (degrades to `NoopDisplay` when `--no-tui` or non-interactive)
  - `report.py` — Plotly self-contained HTML convergence reports (single-run and cross-scheme comparison); auto-generated at end of training, also standalone CLI: `python -m aerocapture.training.report`
  - `final_report.py` — Post-training final evaluation: runs 1000-sim MC re-evaluation via `run_mc()`, generates Plotly HTML with delta-V distributions, orbital error distributions, entry conditions scatter, and summary statistics; auto-generated at end of training, also standalone CLI: `python -m aerocapture.training.final_report`

## GA Training & Comparison

```bash
# ── Optimize a guidance scheme (with Rich TUI) ──
uv run python -m aerocapture.training.train \
    --guidance equilibrium_glide \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20

# ── Disable TUI (e.g. in CI or when piping output) ──
uv run python -m aerocapture.training.train \
    --guidance equilibrium_glide \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20 --no-tui

# ── Adaptive seed pool (curates MC seeds by difficulty) ──
uv run python -m aerocapture.training.train \
    --guidance equilibrium_glide \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20 --adaptive-seeds

# ── Resume from checkpoint ──
uv run python -m aerocapture.training.train \
    --guidance equilibrium_glide \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --resume training_output/equilibrium_glide

# ── Compare all schemes on identical MC scenarios ──
uv run python -m aerocapture.training.compare_guidance \
    --base-toml configs/training/msr_aller_eqglide_train.toml \
    --n-sims 100 \
    --schemes equilibrium_glide energy_controller pred_guid fnpag ftc neural_network

# ── Generate post-training convergence report ──
uv run python -m aerocapture.training.report training_output/equilibrium_glide/
uv run python -m aerocapture.training.report --compare training_output/

# ── Generate final evaluation report (1000-sim MC re-evaluation) ──
# Automatically runs at end of training; also available standalone:
uv run python -m aerocapture.training.final_report \
    training_output/equilibrium_glide/ \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --n-sims 1000 --seed 42
```

Guidance schemes and their TOML training configs:
- `neural_network` -> `configs/training/msr_aller_nn_train_consolidated.toml`
- `equilibrium_glide` -> `configs/training/msr_aller_eqglide_train.toml`
- `energy_controller` -> `configs/training/msr_aller_energy_controller_train.toml`
- `pred_guid` -> `configs/training/msr_aller_pred_guid_train.toml`
- `fnpag` -> `configs/training/msr_aller_fnpag_train.toml`
- `ftc` -> `configs/training/msr_aller_ftc_train.toml`

Optimized params saved to `training_output/<scheme>/best_params.json` (or `best_model.json` for NN).

## Key Lessons & Pitfalls

### Historical: Fortran Common Block Size Mismatch (Root Cause of Density Explosion)

*Context: explains why the Rust density filter code has careful gain-clamping logic.*

The original Fortran had a density filter instability at step ~40, caused by a **common block size mismatch** in `guilat.f`. The `/reftab/` common block was declared with only 4 arrays (64,000 bytes) in `guilat.f`, while other files declared it with 6 arrays (96,000 bytes). The gfortran linker allocated the smaller size and placed `/estiro/` (containing `lambda`, the density filter gain) in overlapping memory. Writing `refdates(57)` corrupted lambda from 0.8 to 56.0, causing the filter equation to amplify errors by 55x per step.

### Historical: Fortran Uninitialized Variables in photra.f

*Context: explains why Rust validation tolerates 2/24 column mismatches in regression tests.*

- `xrayon` (planet radius) never assigned -> 0 at runtime
- `romver` uninitialized at first call -> col 22 garbage at timestep 0
- `xphoto(24)` retains stale `numsuc` from another subroutine via stack reuse

### Energy Computation

Energy must use **absolute (inertial) velocity**, not relative velocity. The Rust `total_energy()` converts relative->absolute via `to_absolute_cartesian` before computing E = V_abs^2/2 - mu/r.

## Conventions

- **Rust**: Edition 2024, nalgebra for linear algebra, release profile with LTO
- **Python**: Python >=3.14, Ruff (line-length 160, target py314), uv package manager, pytest, mypy strict mode. Dev tools in `[dependency-groups]` (not `[project.optional-dependencies]`). Training deps (deap, scipy) are core dependencies.
- **Testing (Python)**: pytest, hypothesis (property-based). Golden reference files under `tests/reference_data/`. Shared fixtures in `tests/conftest.py` (session-scoped Rust build) and `tests/fixtures/factories.py` (config/chromosome factories). ~218 tests covering parsers, regression, MC, GA pipeline (chromosome, cost, TOML patching, config, operators), training visualization (metrics, logger, display, integration, report, final evaluation), NN weight initialization, seed rotation, adaptive seed pool (CVaR, aggregation, growth, eviction, scoring, checkpoint, evaluation, integration), graceful interrupt handling, PyO3 integration (bit-identical regression against subprocess path).
- **Testing (Rust)**: Three-tier pyramid — unit tests (inline `#[cfg(test)]` modules with proptest property tests), integration tests (`src/rust/tests/`), E2E subprocess tests. Shared test infrastructure in `tests/common/` (fixtures.rs, assertions.rs). Dev-dependencies: `approx` (float comparison), `rstest` (parameterized tests), `proptest` (property-based testing). ~176 tests covering physics, GNC, guidance (all 6 schemes), navigation, error paths, `run_for_api()`. Run with `cargo test` or `./check_all.sh`.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) — Rust (fmt, clippy, test), Python (ruff lint, ruff format, mypy, pytest), and PyO3 (maturin build + pytest test_pyo3.py) run on PRs to `main` and manual dispatch (`workflow_dispatch`).
- **Validation**: Rust vs Fortran comparison complete — 22/24 photo columns bit-identical across 725 timesteps.

## Tone

Be a **quirky friendly but critical peer reviewer**. Think of yourself as a quirky senior developer doing a code review: helpful, but holding me to high standards. Always **Challenge inefficiencies**: if I'm doing something the hard way, call it out.
