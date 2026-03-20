# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aerocapture is a trajectory simulation tool for aerocapture maneuvers (primarily Mars Sample Return). The project was modernized from a legacy Fortran 77 codebase into a **Rust simulator** with **Python analysis tools**. The Rust simulator was **validated against the Fortran reference** (now removed from the working tree but preserved in git history) — FTC guided trajectories matched to bit-level precision across all 725 timesteps (22/24 photo columns exact; the remaining 2 were Fortran uninitialized variable artifacts).

The simulation models a spacecraft entering a planet's atmosphere at hyperbolic velocity, using aerodynamic forces and bank angle modulation to capture into a target orbit. The GNC chain is: Navigation (state estimation + density filter) -> Guidance (one of 7 algorithms: FTC, NN, Equilibrium Glide, Energy Controller, PredGuid, FNPAG, Piecewise Constant) -> Control (pilot dynamics + roll reversal). Schemes providing signed bank angles (NN, Piecewise Constant) bypass lateral guidance — they control roll direction directly. All guidance schemes have TOML-configurable parameters and can be GA-optimized.

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
  config.rs                        — TOML parser (Planet, MissionType, SimInput) + base inheritance (deep_merge, resolve_toml_bases, from_toml_file)
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
      ftc.rs                       — FTC capture-phase guidance + central guidance dispatch
      reference.rs                 — Constant bank angle mode
      neural.rs                    — NN guidance (modular JSON architecture, GA-trained, signed bank via atan2)
      equilibrium_glide.rs         — Equilibrium glide with hdot damping + velocity bias
      energy_controller.rs         — Energy dissipation tracking via pdyn/hdot feedback
      predguid.rs                  — Apollo/Shuttle-heritage drag tracking guidance
      fnpag.rs                     — Lu's numerical predictor-corrector (FNPAG)
      piecewise_constant.rs        — 10-segment bank angle profile (GA-optimized, produces ref trajectory + corridor)
    control/
      pilot.rs                     — Pilot dynamics
      attitude.rs                  — Attitude command realization
  integration/
    rk4.rs                         — Gill-variant RK4
    sequencer.rs                   — Module cadence scheduling
  orbit/
    elements.rs                    — Orbital elements from state vector
    maneuver.rs                    — Delta-V cost computation (only called for confirmed captures)
  simulation/
    runner.rs                      — Main sim loop: run() for CLI, run_for_api() for PyO3; tracks peak heat flux, g-load, dynamic pressure; pending crash detection (ifinal=4); virtual DV for all termination outcomes
    init.rs                        — Per-run initialization
    output.rs                      — File writers (photo, final, CSV)
```

Key Rust dependency: `nalgebra` for vector/matrix ops.

### PyO3 Bindings (`src/rust/aerocapture-py/`)

Separate workspace member crate providing Python bindings via PyO3. Built with `maturin develop --release`. Imports as `aerocapture_rs` in Python.

```
src/rust/aerocapture-py/src/
  lib.rs         — Module entry: run(), run_mc(), run_batch(), load_config()
  config.rs      — TOML loading with base inheritance resolution + dot-path override merging
  results.rs     — SimResult/BatchResults pyclasses with numpy getters
  batch.rs       — Rayon parallel batch execution
```

Key API:
- `aerocapture_rs.run(toml_path, overrides=None)` → `SimResult` with `.final_record` (52,), `.captured`, `.energy`, `.ecc`, `.dispersions` (24,), etc. Returns first result only (use `run_mc` for multi-sim).
- `aerocapture_rs.run_mc(toml_path, overrides=None, include_trajectories=False)` → `BatchResults` with all n_sims results. When `include_trajectories=True`, populates per-timestep trajectory data (N, 12) for corridor plots. `.dispersions` (N, 24) always populated.
- `aerocapture_rs.run_batch(toml_path, overrides_list, n_threads=None, include_trajectories=False)` → `BatchResults` with `.final_records` (N, 52), `.dispersions` (N, 24)
- `aerocapture_rs.load_config(toml_path)` → Python dict

The training pipeline (`evaluate.py`) auto-detects PyO3 availability and falls back to subprocess if not installed. Override dict uses dot-separated TOML key paths with type coercion (int→float when existing field is float).

### Data Files (`data/`)

- `data/atmosphere/mars.dat` — Mars density vs altitude table (tabulated MarsGram 3.8)
- `data/atmosphere/earth.dat` — Earth atmosphere table
- `data/reference_trajectory/msr_aller.dat` — MSR reference trajectory (energy vs pdyn/hdot/cos_bank)
- `data/reference_trajectory/esr_aller.dat` — ESR reference trajectory

### Input Configuration

TOML config files in `configs/` are the only supported input format, organized into subdirectories: `configs/missions/` (shared per-planet base configs), `configs/nominal/` (simulation configs), `configs/training/` (GA training configs), `configs/test/` (golden test configs).

**Base inheritance:** Configs support a `base` key (string or array of strings) that references parent TOML files, resolved relative to the declaring file. The loader deep-merges bases left-to-right, then overlays the child's own keys. This eliminates duplication — mission-level content (entry, vehicle, aero, flight, orbit, success, incidence, atmosphere paths) lives in `configs/missions/mars.toml` or `earth.toml`, common training settings (MC dispersions, cost function) live in `configs/training/common.toml`, and each leaf config only specifies its overrides (guidance type, n_sims, results_suffix). Both Rust (`resolve_toml_bases()` in `config.rs`) and Python (`load_toml_with_bases()` in `toml_utils.py`) implement the same resolution logic.

Each config specifies mission, guidance scheme, vehicle, entry conditions, aerodynamics, Monte Carlo settings, and data file paths. Mission TOMLs include a `[corridor]` section with asymmetric restricted corridor bounds (`delta_za_restricted_low`, `delta_za_restricted_high` in km). The NN weight file path (`[data] neural_network`) and optional architecture override (`[network] layer_sizes`, `activations`) are read from TOML at training time. The `[simulation]` section supports `max_time` (default: 3000.0 s) as a hard wall to prevent runaway simulations. Training TOMLs include a `[cost_function]` section with configurable thresholds and penalty weights for g-load (`g_load_limit`, `g_load_weight`), heat flux (`heat_flux_limit`, `heat_flux_weight`), and the log-cap DV compression threshold (`dv_threshold`, default 1000.0 m/s).

### Python Tools (`src/python/`, `pyproject.toml`)

Python analysis package (numpy, pandas, matplotlib, deap, scipy) for:

- Output file parsers (photo, final, CSV files)
- Visualization (corridor plots, MC ensembles, CDF of correction cost)
- GA training pipeline: optimizes any guidance scheme's parameters (not just NN weights)
  - `train.py` — Main GA loop with checkpoint save/resume (`--guidance <scheme> --toml <config> [--no-tui] [--rotate-seeds | --adaptive-seeds] [--seed-pool-cap N] [--cost-alpha F] [--cvar-percentile P] [--skip-final-report] [--final-n-sims N]`). Auto-resumes from existing checkpoint when output dir exists (no `--resume` needed); `--resume` only needed to specify a non-default directory. On resume, `--n-gen` means "N additional generations" (not total). A checkpoint is always saved at end of training (not just at interval multiples). Graceful KeyboardInterrupt handling: Ctrl+C saves checkpoint and returns cleanly with `interrupted: True`. Final evaluation prints capture rate, delta-V, and orbital error percentiles (p50/p95/mean) to stdout.
  - `param_spaces.py` — Per-scheme parameter bounds (with optional log-scale encoding)
  - `evaluate.py` — Decode chromosome -> write params (NN JSON or patched TOML) -> run sim -> cost. Uses PyO3 direct call when `aerocapture_rs` is available, subprocess fallback otherwise. Cost function uses `log_cap(dv)` — a C1-continuous log-capped function (linear below `dv_threshold`, logarithmic above) — as primary objective, with TOML-configurable normalized soft constraint penalties for g-load and heat flux exceedances. All termination outcomes (captured, hyperbolic, crash, pending crash, timeout) produce meaningful DV values from Rust, so no branching on capture status is needed.
  - `compare_guidance.py` — Fair head-to-head comparison on identical MC scenarios
  - `initialization.py` — Activation-aware weight init (Xavier/He/LeCun uniform) for NN population seeding
  - `seed_pool.py` — Adaptive seed pool for MC dispersions: rolling pool of seeds scored by population-relative difficulty (CVaR-blended fitness), with incremental growth and redundancy eviction. `SeedPool` class with `evaluate_population()` (supports PyO3 batch), `score_difficulty()`, `evict_redundant()`, and checkpoint serialization.
  - `toml_utils.py` — `load_toml_with_bases()`: TOML loading with `base` inheritance resolution (mirrors Rust `resolve_toml_bases`)
  - `weight_stats.py` — Per-layer weight statistics (min/max/mean/std) for training instrumentation
- Training visualization:
  - `metrics.py` — Pure metric functions: cost stats, diversity, capture rate, convergence speed, stagnation
  - `logger.py` — `TrainingLogger`: writes one JSONL line per generation; in-memory buffer for live display
  - `display.py` — `LiveDisplay`: Rich TUI with sparklines, ETA, progress bar (degrades to `NoopDisplay` when `--no-tui` or non-interactive)
  - `report.py` — Plotly self-contained HTML convergence reports (single-run and cross-scheme comparison) with dynamic grid layout; conditionally shows seed pool evolution panel (adaptive seeds) and MC seed trace panel (rotate seeds) when relevant JSONL fields are present; detects resume points from JSONL file boundaries and renders vertical markers on all panels. Auto-generated at end of training, also standalone CLI: `python -m aerocapture.training.report`
  - `final_report.py` — Post-training final evaluation: runs 1000-sim MC re-evaluation via `run_mc(include_trajectories=True)`, returns `FinalEvalData(final_array, trajectories, dispersions)`. Produces three output files: (1) Plotly HTML with delta-V/orbital error distributions (dv1=periapsis, dv2=apoapsis, dv3=inclination), entry/exit conditions scatter, performance summary table (g-load, heat flux, bank consumption, orbital errors, DV with Mean/Std/Min/p5-p95/Max); (2) matplotlib PNG with energy corridor panels (pdyn, inclination, bank angle vs energy — MC spaghetti + 4-layer zone fill: red crash/hyperbolic, grey transition, white restricted corridor + three nominals: red piecewise-constant reference, orange undispersed guidance, green best-case MC); (3) separate Plotly HTML with dispersion correlation grid (~24 scatter subplots with linear regression R²/p-value). Auto-generated at end of training, also standalone CLI: `python -m aerocapture.training.final_report`
  - `corridor.py` — Corridor boundary computation via `CorridorAccumulator`. During `piecewise_constant` GA training, each generation's trajectories (plus 11 constant-bank-angle sentinel chromosomes from 0° to 180° in 18° steps) are classified (`classify_trajectories` with asymmetric bounds `delta_za_low`/`delta_za_high`; recognizes `ifinal=4` pending crash) and their pdyn envelopes updated incrementally (running max/min per energy bin). Sentinel trajectories improve corridor boundary resolution by tracing the full lift-up (hyperbolic boundary) to full lift-down (crash boundary) range. Produces schema-v4 `.npz` cache with 4 envelopes (crash, restricted upper/lower, capture), nominal trajectory, and DV. Gaussian smoothing applied at save time. Cached per mission in `training_output/<mission>/corridor_boundaries.npz`. Also produces `ref_trajectory.dat` (7-column format) for schemes that track a reference trajectory.

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

# ── Resume training (auto-detects checkpoint; --n-gen means "N additional") ──
uv run python -m aerocapture.training.train \
    --guidance equilibrium_glide \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50

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
- `piecewise_constant` -> `configs/training/msr_aller_piecewise_constant_train.toml` **(train first — produces ref trajectory + corridor)**
- `neural_network` -> `configs/training/msr_aller_nn_train_consolidated.toml`
- `equilibrium_glide` -> `configs/training/msr_aller_eqglide_train.toml`
- `energy_controller` -> `configs/training/msr_aller_energy_controller_train.toml` *(requires ref trajectory)*
- `pred_guid` -> `configs/training/msr_aller_pred_guid_train.toml` *(requires ref trajectory)*
- `fnpag` -> `configs/training/msr_aller_fnpag_train.toml` *(requires ref trajectory)*
- `ftc` -> `configs/training/msr_aller_ftc_train.toml` *(requires ref trajectory)*

**Training order:** Run `piecewise_constant` first — it produces `training_output/<mission>/ref_trajectory.dat` (optimized reference for other schemes) and `corridor_boundaries.npz` (4-layer corridor envelopes from GA population history). Schemes marked *(requires ref trajectory)* will error at startup if the ref trajectory is missing. Schemes without the marker (`neural_network`, `equilibrium_glide`) can be trained independently.

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
- **Testing (Python)**: pytest, hypothesis (property-based). Golden reference files under `tests/reference_data/`. Shared fixtures in `tests/conftest.py` (session-scoped Rust build) and `tests/fixtures/factories.py` (config/chromosome factories). ~276 tests covering parsers, regression, MC, GA pipeline (chromosome, cost, TOML patching, config, operators), training visualization (metrics, logger, display, integration, report, final evaluation), NN weight initialization, seed rotation, adaptive seed pool (CVaR, aggregation, growth, eviction, scoring, checkpoint, evaluation, integration), graceful interrupt handling, TOML base inheritance resolution, PyO3 integration (bit-identical regression against subprocess path), report resume detection and conditional panel rendering, final report corridor/dispersion/entry-exit panels, corridor accumulator (incremental envelope building, checkpoint roundtrip, asymmetric bounds, ifinal=4 pending crash classification), unified cost function (log_cap C0/C1 continuity, monotonicity, cost ordering).
- **Testing (Rust)**: Three-tier pyramid — unit tests (inline `#[cfg(test)]` modules with proptest property tests), integration tests (`src/rust/tests/`), E2E subprocess tests. Shared test infrastructure in `tests/common/` (fixtures.rs, assertions.rs). Dev-dependencies: `approx` (float comparison), `rstest` (parameterized tests), `proptest` (property-based testing), `tempfile` (temp dirs for base inheritance tests). ~201 tests covering physics, GNC, guidance (all 7 schemes including piecewise_constant), navigation, error paths, `run_for_api()`, peak value tracking, TOML base inheritance (deep_merge, resolve_toml_bases, cycle detection), virtual DV ranges (proptest: crash DV in [10k,20k], hyperbolic DV >= 10k, cost ordering invariant). Run with `cargo test` or `./check_all.sh`.
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) — Rust (fmt, clippy, test), Python (ruff lint, ruff format, mypy, pytest), and PyO3 (maturin build + pytest test_pyo3.py) run on PRs to `main` and manual dispatch (`workflow_dispatch`).
- **Validation**: Rust vs Fortran comparison complete — 22/24 photo columns bit-identical across 725 timesteps.

## Tone

Be a **quirky friendly but critical peer reviewer**. Think of yourself as a quirky senior developer doing a code review: helpful, but holding me to high standards. Always **Challenge inefficiencies**: if I'm doing something the hard way, call it out.
