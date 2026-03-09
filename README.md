# Aerocapture

Trajectory simulation tool for aerocapture maneuvers, primarily targeting Mars Sample Return (MSR). Models a spacecraft entering a planet's atmosphere at hyperbolic velocity, using aerodynamic forces and bank angle modulation to capture into a target orbit.

Modernized from a legacy Fortran 77 codebase (~10,675 lines, ~65 files) into a **Rust simulator** with **Python analysis tools**. The Rust simulator has been validated against the Fortran reference to bit-level precision. The legacy Fortran code has been removed (preserved in git history).

## Quick Start

```bash
# Build the Rust simulator
cd src/rust
cargo build --release

# Run with TOML config
./target/release/aerocapture ../../configs/msr_aller_ftc_nominal.toml
```

## Project Structure

```
src/
  rust/                    Rust simulator (validated reimplementation)
  python/                  Python analysis package (parsing, plotting, training)
configs/                   TOML configuration files (per guidance scheme)
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

All guidance schemes can be optimized via genetic algorithm. The GA tunes each scheme's parameters to minimize orbit insertion error across Monte Carlo dispersions.

```bash
# Optimize any guidance scheme
uv run python -m aerocapture.training.train \
    --guidance equilibrium_glide \
    --toml configs/msr_aller_eqglide_train.toml \
    --n-gen 50 --n-pop 20

# Compare all schemes on identical MC scenarios
uv run python -m aerocapture.training.compare_guidance \
    --base-toml configs/msr_aller_eqglide_train.toml \
    --n-sims 100
```

## Validation

The Rust simulator matches the Fortran reference across all 725 timesteps of a guided FTC trajectory:
- **22 of 24** photo output columns are bit-identical
- The remaining 2 differ only at the first timestep due to Fortran uninitialized variable artifacts (`romver`, `numsuc`)

## CI

GitHub Actions runs on every push and PR:

- **Rust**: `cargo fmt --check`, `cargo clippy`, `cargo test --release`
- **Python**: `ruff check`, `ruff format --check`, `mypy`, `pytest`

## Build Commands

```bash
# Rust
cd src/rust && cargo build --release

# Python
uv sync && pytest tests
```

## Roadmap

See [improvements.md](improvements.md) for the physics, GNC, and software improvement roadmap.
