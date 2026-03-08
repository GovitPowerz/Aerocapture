# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aerocapture is a trajectory simulation tool for aerocapture maneuvers (primarily Mars Sample Return). The project is being modernized from a legacy Fortran 77 codebase (~10,675 lines across ~65 Fortran files) into a **Rust simulator** with **Python analysis tools**. The Rust simulator has been **validated against the Fortran reference** — FTC guided trajectories match to bit-level precision across all 725 timesteps (22/24 photo columns exact; the remaining 2 are Fortran uninitialized variable artifacts).

The simulation models a spacecraft entering a planet's atmosphere at hyperbolic velocity, using aerodynamic forces and bank angle modulation to capture into a target orbit. The GNC chain is: Navigation (state estimation + density filter) -> Guidance (one of 6 algorithms: FTC, NN, Equilibrium Glide, Energy Controller, PredGuid, FNPAG) -> Control (pilot dynamics + roll reversal). All guidance schemes have TOML-configurable parameters and can be GA-optimized.

## Build & Development Commands

```bash
# ── Rust Simulator ──
cd src/rust
cargo build --release              # Build optimized binary
# Run from old_codebase/exec/ (CWD matters — reads ../donnees/, writes ../sorties/)
cd ../../old_codebase/exec
../../src/rust/target/release/aerocapture < test_input.in

# ── Legacy Fortran (from old_codebase/exec/) ──
cd old_codebase/exec
make clean && make                 # ALWAYS clean before rebuild (stale .o corruption)
./aerocap < test_input.in          # Run legacy simulation
# Output: ../sorties/photo.*, ../sorties/final.*, fort.201-204

# ── Python Analysis ──
uv sync                            # Install dependencies (Python >=3.14)
uv sync --group dev                # Include dev tools (pytest, ruff, mypy)
pytest tests                       # Run all tests
pytest tests/test_foo.py::test_bar -v

# ── Utility Scripts (from repo root) ──
./setup_env.sh                     # Create fresh .venv + install deps
./lint_code.sh                     # Run ruff (imports, format, lint) + mypy
./check_all.sh                     # Rust: test + fmt --check + clippy
./upgrade_dependencies.sh          # uv sync --upgrade
```

## Architecture

### Rust Simulator (`src/rust/`)

The Rust code is a faithful line-by-line reimplementation of the Fortran. Each module maps directly to one or more Fortran subroutines. The entry point supports both legacy `.in` config from stdin (`./aerocap < config.in`) and TOML config as a CLI argument (`./aerocap config.toml`). TOML is preferred for new work as it supports all 6 guidance schemes and inline vehicle/mission data.

```
src/rust/src/
  main.rs                          — CLI entry, stdin parsing, data loading
  config.rs                        — .in file parser (Planet, MissionType, SimInput)
  data/
    mod.rs, SimData                — Top-level data container
    atmosphere.rs                  — Atmosphere density table (from fatmos.*)
    aerodynamics.rs                — Cx/Cz vs AoA tables (from aerodyn.*)
    capsule.rs                     — Vehicle: mass, reference area, max bank rate
    guidance_params.rs             — Guidance law config: FTC gains, EqGlide, EnergyCtrl, PredGuid, FNPAG params
    dispersions.rs                 — Monte Carlo dispersion profiles
    navigation.rs                  — Navigation error gabarits
    incidence.rs                   — AoA profile tables
    pilot.rs                       — Pilot dynamics parameters
  physics/
    gravity.rs                     — J2 oblate gravity (← fgravi.f)
    atmosphere.rs                  — Density lookup (← fatmos.f)
    aerodynamics.rs                — Force computation (← faeros.f)
    winds.rs                       — Wind model (← fvents.f)
  gnc/
    navigation/
      estimator.rs                 — State estimation + density filter (← naviga.f)
      coordinates.rs               — Spherical↔Cartesian, geodetic, total energy (← xvabsl.f, frayon.f, energi.f)
    guidance/
      ftc.rs                       — FTC capture-phase guidance (← guidag.f + guicap.f + guilon.f + guilat.f + vigite.f + guialf.f)
      reference.rs                 — Constant bank angle mode
      neural.rs                    — NN guidance (modular JSON architecture, GA-trained)
      equilibrium_glide.rs         — Equilibrium glide with hdot damping + velocity bias
      energy_controller.rs         — Energy dissipation tracking via pdyn/hdot feedback
      predguid.rs                  — Apollo/Shuttle-heritage drag tracking guidance
      fnpag.rs                     — Lu's numerical predictor-corrector (FNPAG)
    control/
      pilot.rs                     — Pilot dynamics (← pilote.f)
      attitude.rs                  — Attitude command realization
  integration/
    rk4.rs                         — Gill-variant RK4 (← rkutta.f)
    sequencer.rs                   — Module cadence scheduling (← sequen.f)
  orbit/
    elements.rs                    — Orbital elements from state vector (← orbito.f)
    maneuver.rs                    — Delta-V cost computation (← ergols.f)
  simulation/
    runner.rs                      — Main sim loop (← simmsr.f + realit.f)
    init.rs                        — Per-run initialization (← inimsr.f + etaini.f)
    output.rs                      — File writers (photo, final, fort.*)
```

Key Rust dependency: `nalgebra` for vector/matrix ops.

### Legacy Fortran (`old_codebase/`)

Fortran 77 fixed-form code. Two variants:

- `fortran_original/` — FTC predictor-corrector guidance (65 source files)
- `fortran_neural/` — adds neural network guidance variant (guidnn.f, lecgnn.f)

**Simulation flow** (simmsr.f main loop, one timestep):

1. `naviga` — Add navigation biases to true state, estimate density via exponential filter, detect bounce/phase transitions
2. `guidag` → `guilon` → `guicap` — Compute bank angle command from reference trajectory deviations
3. `guilat` + `vigite` — Roll sign management via inclination error
4. `guialf` — AoA command
5. `pilote` — Apply pilot dynamics to bank angle command
6. `realit` — RK4 integration of equations of motion (4 sub-steps with fgravi, fatmos, faeros at each)
7. `photra`/`sortie` — Write output

**Data sharing**: Fortran common blocks. Notable ones:

- `/capsul/` — mass, reference area, max bank rate
- `/period/` — module cadences (tnavig, tguida, tpilot, tinteg)
- `/modatm/` — atmosphere model coefficients
- `/geoide/` — J2, mu, eccentricity
- `/trigon/` — pi, deg-to-rad conversion
- `/gainmu/` — FTC gain filter parameters (amorft, pulsft)
- `/modpdn/` — pdyn table segment count (nzapd)
- `/estiro/` — density filter lambda
- `/tabgit/` — precomputed bank angle lookup table (500x500x2)

### Data Files (`old_codebase/donnees/`)

Configuration files selected via suffix (e.g., `.msr_aller64`):

- `atmosphere.*` — Density vs altitude table (tabulated MarsGram 3.8)
- `aerodyn.*` — Cx, Cz vs AoA
- `capsule.*` — Vehicle properties
- `guidage.*` — Guidance parameters (filter lambda, pdyn table, gains)
- `tables_energie_gains.*` — Reference trajectory (energy vs pdyn/hdot/cos_bank)
- `dispersions.*` — Monte Carlo dispersion profiles
- Mission variants: MSR (Mars Sample Return), ESR, AFE, Demo

### Input Configuration

Two formats supported:
- **TOML** (preferred): `./aerocapture config.toml` — supports all 6 guidance schemes, inline vehicle data, domain-based MC dispersions, per-scheme parameter tuning. Configs in `configs/`.
- **Legacy `.in`**: `./aerocapture < config.in` — planet ID, guidance type (0=reference/1=FTC), number of sims, reference bank angle, initial orbital elements, file suffixes for data variant selection, output options.

### Python Tools (`src/python/`, `pyproject.toml`)

Python analysis package (numpy, pandas, matplotlib, deap, scipy) for:

- Output file parsers (photo, final, fort.\* files with Fortran D-notation floats)
- Visualization (corridor plots, MC ensembles, CDF of correction cost)
- GA training pipeline: optimizes any guidance scheme's parameters (not just NN weights)
  - `train.py` — Main GA loop with checkpoint save/resume (`--guidance <scheme> --toml <config>`)
  - `param_spaces.py` — Per-scheme parameter bounds (with optional log-scale encoding)
  - `evaluate.py` — Decode chromosome → write params (NN JSON or patched TOML) → run sim → cost
  - `compare_guidance.py` — Fair head-to-head comparison on identical MC scenarios

## GA Training & Comparison

```bash
# ── Optimize a guidance scheme ──
uv run python -m aerocapture.training.train \
    --guidance equilibrium_glide \
    --toml configs/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20

# ── Resume from checkpoint ──
uv run python -m aerocapture.training.train \
    --guidance equilibrium_glide \
    --toml configs/msr_aller_eqglide_train.toml \
    --resume save_net/equilibrium_glide

# ── Compare all schemes on identical MC scenarios ──
uv run python -m aerocapture.training.compare_guidance \
    --base-toml configs/msr_aller_eqglide_train.toml \
    --n-sims 100 \
    --schemes equilibrium_glide energy_controller pred_guid fnpag ftc neural_network
```

Guidance schemes and their TOML training configs:
- `neural_network` → `configs/msr_aller_nn_train_consolidated.toml`
- `equilibrium_glide` → `configs/msr_aller_eqglide_train.toml`
- `energy_controller` → `configs/msr_aller_energy_controller_train.toml`
- `pred_guid` → `configs/msr_aller_pred_guid_train.toml`
- `fnpag` → `configs/msr_aller_fnpag_train.toml`
- `ftc` → `configs/msr_aller_ftc_train.toml`

Optimized params saved to `save_net/<scheme>/best_params.json` (or `best_model.json` for NN).

## Key Lessons & Pitfalls

### Fortran Common Block Size Mismatch (Root Cause of Density Explosion)

The density filter instability at step ~40 was caused by a **common block size mismatch** in `guilat.f`. The `/reftab/` common block was declared with only 4 arrays (64,000 bytes) in `guilat.f`, while `lectci.f`, `guicap.f`, `initia.f`, and `integr.f` declared it with 6 arrays (96,000 bytes). The gfortran linker allocated the smaller size and placed `/estiro/` (containing `lambda`, the density filter gain) in memory overlapping with `refdates(57)`. When `lectci.f` wrote `refdates(57)`, it corrupted lambda from 0.8 to 56.0, causing the filter equation `coefro = (1-lambda)*coefro + lambda*(roesti/rorefr)` to amplify errors by 55x per step. **Fixed** in both `fortran_original/guilat.f` and `fortran_neural/guilat.f` by adding the missing `refdates` and `refcmu` arrays to the common block declaration.

**Always do `make clean_orig && make original` after any Fortran source change** — stale `.o` files cause silent corruption with `gfortran -O3 -ffast-math`.

### Fortran Uninitialized Variables in photra.f

- `xrayon` (planet radius for post-bounce phase check) is declared but never assigned — defaults to 0, making `positr(1) - xrayon` always > 80km, so iphase is always 3 after bounce
- `romver` uninitialized at first call → col 22 garbage at timestep 0
- `xphoto(24)` retains stale `numsuc` value from `etafin.f` via stack reuse between calls

### Input File Format Variants

- **Original variant** (`fortran_original/entree.f`): **30 reads** from stdin
- **Neural variant** (`fortran_neural/entree.f`): **32 reads** (adds `natgnn` + `sufgnn` lines)
- All `.in` files in the repo were originally for the neural variant. `test_input.in` has been rewritten for the original variant.

### Energy Computation

Energy must use **absolute (inertial) velocity**, not relative velocity. Both Fortran `enrtot()`/`energi()` and Rust `total_energy()` convert relative→absolute via `xvabsl`/`to_absolute_cartesian` before computing E = V_abs²/2 - mu/r. Photo output column 19 should use this absolute energy.

### Fortran Output Format

Photo files use `format(24(1x,d12.5))` — 24 columns of Fortran D-notation floats per line. The Rust output writer must match this format exactly for comparison.

## Conventions

- **Rust**: Edition 2024, nalgebra for linear algebra, release profile with LTO
- **Python**: Python >=3.14, Ruff (line-length 160, target py314), uv package manager, pytest, mypy strict mode. Dev tools in `[dependency-groups]` (not `[project.optional-dependencies]`). Training deps (deap, scipy) are core dependencies.
- **Testing**: pytest for Python, Fortran golden reference files under `tests/reference_data/`
- **Validation**: Rust vs Fortran comparison complete — 22/24 photo columns bit-identical across 725 timesteps. See `tests/compare_results.py` for the comparison framework.

## Tone

Be a **quirky friendly but critical peer reviewer**. Think of yourself as a quirky senior developer doing a code review: helpful, but holding me to high standards. Always **Challenge inefficiencies**: if I'm doing something the hard way, call it out.
