# Aerocapture

Trajectory simulation tool for aerocapture maneuvers, primarily targeting Mars Sample Return (MSR). Models a spacecraft entering a planet's atmosphere at hyperbolic velocity, using aerodynamic forces and bank angle modulation to capture into a target orbit.

Modernized from a legacy Fortran 77 codebase (~10,675 lines, ~65 files) into a **Rust simulator** with **Python analysis tools**. The Rust simulator has been validated against the Fortran reference to bit-level precision.

## Quick Start

```bash
# Build the Rust simulator
cd src/rust
cargo build --release

# Run a guided FTC simulation (from old_codebase/exec/)
cd ../../old_codebase/exec
../../src/rust/target/release/aerocapture < test_input.in

# Output written to ../sorties/photo.test
```

## Project Structure

```
src/
  rust/                    Rust simulator (validated reimplementation)
  python/                  Python analysis package (parsing, plotting, NN training)
old_codebase/
  fortran_original/        Legacy Fortran 77 — FTC predictor-corrector guidance
  fortran_neural/          Fortran variant with neural network guidance
  donnees/                 Data files (atmosphere, aerodynamics, capsule, guidance)
  exec/                    Input configs, makefiles, MATLAB scripts
tests/                     Test framework and Fortran reference data
```

## GNC Architecture

The simulation implements a full closed-loop GNC chain:

1. **Navigation** — State estimation with density filter (exponential filter on atmospheric density ratio)
2. **Guidance** — FTC predictor-corrector computes bank angle command from reference trajectory deviations
3. **Lateral guidance** — Roll sign management via inclination error with deadband
4. **Control** — Pilot dynamics model applies rate limits and first/second-order lag to bank angle commands
5. **Integration** — Gill-variant RK4 propagates equations of motion with J2 gravity, tabulated atmosphere, and aerodynamic forces

## Validation

The Rust simulator matches the Fortran reference across all 725 timesteps of a guided FTC trajectory:
- **22 of 24** photo output columns are bit-identical
- The remaining 2 differ only at the first timestep due to Fortran uninitialized variable artifacts (`romver`, `numsuc`)

## Build Commands

```bash
# Rust
cd src/rust && cargo build --release

# Fortran (always clean first)
cd old_codebase/exec && make clean && make

# Python
uv sync && pytest tests
```

## Roadmap

See [improvements.md](improvements.md) for the physics, GNC, and software improvement roadmap.
