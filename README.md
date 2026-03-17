# Aerocapture

Trajectory simulation tool for aerocapture maneuvers, primarily targeting Mars Sample Return (MSR). Models a spacecraft entering a planet's atmosphere at hyperbolic velocity, using aerodynamic forces and bank angle modulation to capture into a target orbit.

Modernized from a legacy Fortran 77 codebase (~10,675 lines, ~65 files) into a **Rust simulator** with **Python analysis tools**. The Rust simulator has been validated against the Fortran reference to bit-level precision. The legacy Fortran code has been removed (preserved in git history).

## Quick Start

```bash
# Build the Rust simulator
cd src/rust
cargo build --release

# Run with TOML config
./target/release/aerocapture ../../configs/nominal/msr_aller_ftc_nominal.toml
```

## Project Structure

```
src/
  rust/                    Rust simulator (validated reimplementation)
    aerocapture-py/        PyO3 Python bindings (aerocapture_rs module)
  python/                  Python analysis package (parsing, plotting, training)
configs/
  missions/                Shared per-planet base configs (Mars, Earth)
  nominal/                 Nominal simulation configurations
  training/                GA training configs (per scheme) + common.toml (shared MC/cost)
  test/                    Golden test configurations (regression tests)
data/
  atmosphere/              Atmosphere density tables (Mars, Earth)
  reference_trajectory/    Reference trajectories for guided schemes
tests/                     Test framework and golden reference data
```

## Guidance Schemes

Six guidance algorithms are implemented, all GA-optimizable:

| Scheme | Description | Params |
|---|---|---|
| **FTC** | Predictor-corrector with reference trajectory tracking | 8 |
| **Neural Network** | Trained NN maps nav state to bank angle command | 110 |
| **Equilibrium Glide** | Balances gravity, centrifugal, and lift forces | 7 |
| **Energy Controller** | Tracks reference energy dissipation profile | 3 |
| **PredGuid** | Apollo/Shuttle-heritage drag tracking | 3 |
| **FNPAG** | Lu's numerical predictor-corrector | 5 |

## GNC Architecture

The simulation implements a full closed-loop GNC chain:

1. **Navigation** — State estimation with density filter (exponential filter on atmospheric density ratio)
2. **Guidance** — One of 6 algorithms computes bank angle command (see table above)
3. **Lateral guidance** — Roll sign management via inclination error with deadband
4. **Control** — Pilot dynamics model applies rate limits and first/second-order lag to bank angle commands
5. **Integration** — Gill-variant RK4 propagates equations of motion with J2 gravity, tabulated atmosphere, and aerodynamic forces

## GA Optimization

All guidance schemes can be optimized via genetic algorithm. The GA tunes each scheme's parameters to minimize correction delta-V across Monte Carlo dispersions, with TOML-configurable soft constraint penalties for g-load and heat flux exceedances. Training auto-resumes from existing checkpoints (use `-fs` to start fresh). On resume, `--n-gen` means "N additional generations." Training supports graceful Ctrl+C interruption (saves checkpoint and returns cleanly).

```bash
# Optimize any guidance scheme (Rich TUI with sparklines and ETA)
uv run python -m aerocapture.training.train \
    --guidance equilibrium_glide \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20

# Disable TUI (CI / piped output)
uv run python -m aerocapture.training.train ... --no-tui

# Rotate MC dispersion seeds each generation (prevents overfitting)
uv run python -m aerocapture.training.train ... --rotate-seeds

# Adaptive seed pool (curates seeds by difficulty, CVaR-blended fitness)
uv run python -m aerocapture.training.train ... --adaptive-seeds

# Compare all schemes on identical MC scenarios
uv run python -m aerocapture.training.compare_guidance \
    --base-toml configs/training/msr_aller_eqglide_train.toml \
    --n-sims 100

# Convergence report (dynamic layout with resume markers and seed panels)
uv run python -m aerocapture.training.report training_output/equilibrium_glide/
uv run python -m aerocapture.training.report --compare training_output/

# Pre-compute corridor boundaries (cached per mission, shared across schemes)
# Reads [corridor] section from mission TOML for delta_za and n_sims defaults
uv run python -m aerocapture.training.corridor \
    --toml configs/missions/mars.toml

# Final evaluation report (1000-sim MC re-evaluation)
# Includes: delta-V/orbital error distributions, entry/exit conditions,
# performance summary table, energy corridor PNG (pdyn with 4-layer zones:
# crash/undershoot/corridor/overshoot/hyperbolic, inclination, bank angle,
# DV distribution), and dispersion correlation grid (~24 scatter plots).
# Auto-generated at end of training; also standalone:
uv run python -m aerocapture.training.final_report \
    training_output/equilibrium_glide/ \
    --toml configs/training/msr_aller_eqglide_train.toml \
    --corridor training_output/mars/corridor_boundaries.npz \
    --n-sims 1000 --seed 42
```

## Validation

The Rust simulator matches the Fortran reference across all 725 timesteps of a guided FTC trajectory:
- **22 of 24** photo output columns are bit-identical
- The remaining 2 differ only at the first timestep due to Fortran uninitialized variable artifacts (`romver`, `numsuc`)

## PyO3 Python Bindings

The `aerocapture_rs` Python module provides direct access to the Rust simulator, eliminating subprocess overhead for GA training.

```python
import aerocapture_rs as aero

# Single run
result = aero.run("configs/test/test_ref_orig.toml")
print(f"Captured: {result.captured}, dV: {result.delta_v:.1f} m/s")

# Batch run with per-sim overrides (parallel via Rayon)
overrides = [{"simulation.random_seed": float(i) / 10} for i in range(100)]
batch = aero.run_batch("configs/training/msr_aller_ftc_train.toml", overrides)
print(f"Final records: {batch.final_records.shape}")  # (100, 52)
```

Build with: `cd src/rust/aerocapture-py && maturin develop --release`

The training pipeline auto-detects PyO3 and falls back to subprocess if not installed.

## CI

GitHub Actions runs on PRs to `main` and manual dispatch:

- **Rust**: `cargo fmt --check`, `cargo clippy`, `cargo test --release`
- **Python**: `ruff check`, `ruff format --check`, `mypy`, `pytest`
- **PyO3**: `maturin develop --release`, `pytest tests/test_pyo3.py`

## Build Commands

```bash
# Build everything (Rust binary + PyO3 bindings)
./build.sh

# Build and clean intermediate artifacts
./build.sh -c

# Python dependencies
uv sync && pytest tests
```

## Roadmap

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for the physics, GNC, and software improvement roadmap.
