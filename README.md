# Aerocapture

Trajectory simulation tool for aerocapture maneuvers, primarily targeting Mars Sample Return (MSR). Models a spacecraft entering a planet's atmosphere at hyperbolic velocity, using aerodynamic forces and bank angle modulation to capture into a target orbit.

Built as a **Rust simulator** with **Python analysis tools**. Validated against a legacy reference implementation to bit-level precision.

## Quick Start

```bash
# Build the Rust simulator
cd src/rust && cargo build --release && cd ../..

# Run a simulation with TOML config
./src/rust/target/release/aerocapture configs/nominal/msr_aller_ftc_nominal.toml

# Build PyO3 bindings (optional, speeds up training ~10x)
cd src/rust/aerocapture-py && maturin develop --release && cd ../../..

# Set up Python environment
uv sync --group dev

# Run tests
cargo test --release --manifest-path src/rust/Cargo.toml
uv run pytest tests/
```

## Project Structure

```
src/
  rust/                    Rust simulator (core crate + CLI entry)
    aerocapture-py/        PyO3 Python bindings (aerocapture_rs module)
  python/                  Python analysis package (parsing, plotting, training)
  typst/                   PDF report templates (compiled by typst)
configs/
  planets/                 Planet physical constants (mu, radii, omega, J2/J3/J4)
  missions/                Shared per-planet base configs (inherit from planets/)
  nominal/                 Nominal simulation configurations
  training/                GA training configs (per scheme) + common.toml (shared MC/cost)
  test/                    Golden test configurations (regression tests)
data/
  atmosphere/              Atmosphere density + wind tables (Mars, Earth)
  reference_trajectory/    Reference trajectories for guided schemes
training_output/           GA training output (checkpoints, logs, reports, animations)
tests/                     Python test suite + golden reference data
```

## TOML Configuration

Configs are the **only input format** — no command-line flags for mission parameters. The system uses a **base inheritance** mechanism: each config can reference parent files via a `base` key, resolved relative to the declaring file. The loader deep-merges bases left-to-right, then overlays the child's own keys.

```
configs/planets/mars.toml          Planet constants (mu, radii, J2/J3/J4)
         ^
configs/missions/mars.toml         Entry conditions, vehicle, aero, atmosphere, constraints
         ^
configs/training/common.toml       MC dispersions, cost function, simulation settings
         ^
configs/training/msr_aller_eqglide_train.toml   Just: guidance type + results suffix
```

A typical training config is only 9 lines because everything else is inherited:

```toml
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "equilibrium_glide"

[data]
results_suffix = ".train_eqglide"
```

Adding a new planet requires only a new TOML preset file in `configs/planets/` — no Rust changes.

## Physical Models

| Model | Description |
|---|---|
| **Gravity** | J2/J3/J4 zonal harmonic gravity (J3/J4 default to 0 if omitted) |
| **Atmosphere** | Tabulated density vs altitude (MarsGram 3.8 for Mars). Separate piecewise-exponential onboard model for nav/guidance (auto-fitted or explicit segments) |
| **Winds** | Altitude-dependent zonal/meridional wind profiles with MC dispersions (based on Forget et al. 1999) |
| **Aerodynamics** | Cx/Cz vs angle-of-attack tables, configurable vehicle (mass, reference area, max bank rate) |
| **Integration** | Fixed-step Gill-variant RK4 (default, validated) or adaptive Dormand-Prince 4(5) with PI step-size control |

## GNC Architecture

The simulation implements a full closed-loop GNC chain:

1. **Navigation** — Two modes: legacy bias-only, or 13-state EKF (IMU sensor model + star tracker with atmospheric blackout + lift-corrected drag-derived density estimation). Legacy filter includes rate-of-change limiting and gain saturation [0.1, 10.0]. Configurable via `[navigation] mode = "bias"` or `"ekf"`.
2. **Guidance** — One of 7 algorithms computes a bank angle command (see table below). After the trajectory nadir (bounce), FTC + 4 unsigned-magnitude schemes automatically switch to a shared **exit phase controller** (dynamic pressure feedback with radial velocity damping) for apoapsis targeting on the ascending leg.
3. **Lateral guidance** — Roll sign management via predictive first-order inclination projection (projects error forward by configurable tau seconds; reverses only when projected error exceeds threshold). Shared by all unsigned-magnitude schemes (FTC, EqGlide, EnergyCtrl, PredGuid, FNPAG). Remains active during exit phase for inclination correction. NN and PiecewiseConstant produce signed bank angles and bypass both lateral and exit guidance entirely.
4. **Control** — Pilot dynamics model applies rate limits and first/second-order lag to bank angle commands
5. **Integration** — Propagates equations of motion with all physical models above. Adaptive mode sub-steps within each GNC tick — guidance/navigation cadences are unchanged.

## Guidance Schemes

Seven guidance algorithms, all GA-optimizable:

| Scheme | Description | Params | Notes |
|---|---|---|---|
| **Piecewise Constant** | 10-segment bank angle profile | 10 | Train first — produces ref trajectory + corridor |
| **FTC** | Predictor-corrector with reference trajectory tracking | 8 | Requires ref trajectory |
| **Neural Network** | Trained NN maps 16- or 23-input nav state to signed bank angle (atan2). v1 dense-only arch (`layer_sizes`/`activations`) or v2 heterogeneous arch (`[[network.architecture]]`, supports `dense` + `gru`). Trainable via GA/PSO (pymoo) or RL (PPO with chunked truncated BPTT for recurrent policies, experimental SAC). | arch-dependent | Independent, signed bank, full-envelope (capture + exit) |
| **Equilibrium Glide** | Balances gravity, centrifugal, and lift forces | 7 | Independent |
| **Energy Controller** | Tracks reference energy dissipation profile | 3 | Requires ref trajectory |
| **PredGuid** | Apollo/Shuttle-heritage drag tracking | 3 | Requires ref trajectory |
| **FNPAG** | Lu's numerical predictor-corrector (3D predictor with J2 gravity, RK4) | 5 | Requires ref trajectory |

**Training order:** Run `piecewise_constant` first — it produces `ref_trajectory.dat` (optimized reference) and `corridor_boundaries.npz` (4-layer corridor envelopes from GA population history). Schemes marked "Requires ref trajectory" will error at startup if it's missing.

## GA Optimization

All guidance schemes can be optimized via genetic algorithm. The GA tunes each scheme's parameters to minimize correction delta-V across Monte Carlo dispersions, with TOML-configurable soft constraint penalties for g-load, heat flux, and integrated heat load exceedances.

The cost function uses a C1-continuous log-capped compression (`log_cap`) that smoothly transitions from linear to logarithmic above a configurable threshold (default 1000 m/s), preventing outliers from dominating the RMS. The simulator returns meaningful DV values for all termination outcomes (captured, hyperbolic, crash, pending crash, timeout), so no branching on capture status is needed.

Training features:
- Auto-resumes from existing checkpoints (use `-fs` to start fresh)
- `--n-gen` means "N additional generations" when resuming
- Graceful Ctrl+C (saves checkpoint and returns cleanly)
- Rich TUI with sparklines, ETA, progress bar
- Adaptive MC dispersion seeds (prevents overfitting)
- Supports GA (SBX + polynomial mutation), CMA-ES, DE, PSO via `--algorithm` or TOML `[optimizer]`
- PDF report auto-generated at end of training

```bash
# Train all schemes with optimized settings (piecewise_constant first for ref trajectory)
./train_all.sh                     # all schemes in dependency order (incl. nn_rl)
./train_all.sh eqglide fnpag       # specific schemes only

# Optimize a single guidance scheme
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_eqglide_train.toml \
    --n-gen 2500 --n-pop 60

# Disable TUI (CI / piped output)
uv run python -m aerocapture.training.train <config.toml> --no-tui
```

### RL Training (PPO)

Parallel track to the GA for the `neural_network` guidance scheme. PPO-trained policies export to the same `best_model.json` format the GA produces and deploy via the Rust `neural_network` runtime -- `compare_guidance` treats RL as just another scheme (`neural_network_rl`). Supports warm-starting from GA-trained weights (`--data-neural-network`), potential-based phase-aware reward shaping (`r = gamma*Phi(s') - Phi(s)`: corridor tracking + constraint proximity during capture, apoapsis targeting + eccentricity reduction during exit; optimum-preserving per Ng/Harada/Russell 1999), running return and observation normalization (obs normalization baked into exported weights for zero Rust changes), truncation-aware value bootstrap (PPO uses `V(terminal_obs)` on `max_time` timeouts instead of masking them as terminations), and SAC with replay buffer persisted across checkpoint resumes.

```bash
# Train PPO from scratch
uv run python -m aerocapture.training.rl.train \
    configs/training/msr_aller_rl_train.toml \
    --algorithm ppo --from-scratch

# Warm-start from GA-trained model (recommended)
uv run python -m aerocapture.training.rl.train \
    configs/training/msr_aller_rl_train.toml \
    --algorithm ppo --data-neural-network training_output/neural_network/best_model.json

# Fine-tune with conservative hyperparameters
uv run python -m aerocapture.training.rl.train \
    configs/training/msr_aller_rl_train.toml \
    --algorithm ppo --data-neural-network training_output/neural_network/best_model.json \
    --learning-rate 3e-5 --clip-range 0.1 --entropy-coef 0.001 --min-log-std -4.0

# Head-to-head RL vs GA on identical MC scenarios
uv run python -m aerocapture.training.compare_guidance \
    --n-sims 500 --schemes neural_network neural_network_rl

# train_all.sh alias
./train_all.sh nn_rl
```

Architecture: step-able `BatchedSimulation` pyclass (Rayon-parallel per-tick advance over N SimStates, GIL released via `py.detach()`, auto-reset on episode end, `info["truncated"]` surfaced so GAE/SAC distinguish timeouts from terminations). `step()` returns `(obs, reward, done, info, aux)` where `aux` provides `(energy, pdyn)` per env for the capture-phase energy component. CleanRL-style PPO/SAC in `src/python/aerocapture/training/rl/`: PyTorch MLP with GA warm-start via `load_weights_from_json()`, `StepRewardCalculator` for potential-based phase-aware shaping, `ReturnNormalizer` + `ObsNormalizer` (vectorized Chan's parallel Welford), reserved-seed validation gate, graceful Ctrl+C, final MC evaluation summary, three-part PDF report. PPO supports `target_kl` early-stop and per-step return normalization for stable advantages. SAC's critic operates on the 2D Gaussian latent (`atan2(raw[0], raw[1])` still drives the env), with the replay buffer included in `checkpoint.pt` for full-state resume.

CLI flags: `--algorithm {ppo|sac}`, `--total-steps`, `--n-envs`, `--rollout-steps`, `--validation-n-sims`, `--validation-interval-updates`, `--data-neural-network`, `--from-scratch`, `--learning-rate`, `--clip-range`, `--entropy-coef`, `--min-log-std`, `--update-epochs`, `--lr-anneal-start`, `--target-kl`, `--no-tui`, `--skip-report`, `--resume`, `--output-dir`. `--from-scratch` and `--data-neural-network` are mutually exclusive. Full spec at `docs/superpowers/specs/2026-04-15-rl-nn-guidance-design.md`.

### Training seed strategies

The `[optimizer] seed_strategy` key (required) controls how Monte Carlo seeds are picked across generations. All three strategies use the same `training_n_sims` size knob.

| Strategy    | What it does                                                                           | When to use |
| ----------- | -------------------------------------------------------------------------------------- | ----------- |
| `"fixed"`   | Deterministic `[mc_seed + 0, ..., mc_seed + (n_sims-1)]`; seeds never change.          | Debugging, A/B comparisons where the cost landscape must be identical across runs. |
| `"rotating"`| Fresh random seeds drawn every generation, disjoint from reserved sets.                | Production default candidate: landscape shifts each gen so the optimizer can't overfit to a fixed scenario set. |
| `"adaptive"`| Random bootstrap, then curated-CDF: refreshed on validated-best or every `seed_pool_interval` gens. Each curation draws `curation_sample_size` probes, runs the top `curation_top_k` individuals, and picks one seed per cost quantile bin. | When you want a lower-variance fitness signal than rotating; pairs well with a strong `validation_n_sims`. |

Typical TOML snippet:

```toml
[optimizer]
algorithm = "ga"
seed_strategy = "adaptive"
training_n_sims = 20
seed_pool_interval = 50
curation_top_k = 5
curation_sample_size = 1000
```

Override per-scheme by adding `seed_strategy = "..."` in a leaf training TOML. See `CLAUDE.md` for full details.

## Reports and Visualization

### PDF Reports (Typst)

Auto-generated at end of training, or standalone:

```bash
# Single-scheme report (training convergence + final MC evaluation)
uv run python -m aerocapture.training.report \
    training_output/equilibrium_glide/ \
    --toml configs/training/msr_aller_eqglide_train.toml

# Cross-scheme comparison PDF
uv run python -m aerocapture.training.report --compare training_output/
```

Reports include: cost convergence curves, population diversity, corridor plots with zone fills, altitude/heat flux/g-load/bank angle vs time spaghetti with constraint limit lines, DV distributions, entry/exit conditions, performance summary tables, dispersion correlation grids with three-way trajectory classification. Compiled via `typst` (install with `brew install typst`). Degrades gracefully if Typst is not installed -- charts are still generated as SVGs. A `final_eval.parquet` file (65 columns: 39 final-record + 26 dispersions, with embedded config metadata) is auto-written alongside the PDF when `pyarrow` is available.

### Training Animation

Replay training checkpoints as a GIF showing how corridors and trajectories evolve over generations:

```bash
uv run python -m aerocapture.training.animate \
    training_output/piecewise_constant/ \
    --toml configs/training/msr_aller_piecewise_constant_train.toml \
    --n-sims 100 --fps 4 --every 5
```

Produces a 2x2 animation (corridor with envelope fills, inclination, bank angle, cost CDF) by re-running MC simulations at each checkpoint via PyO3.

### Scheme Comparison

Fair head-to-head comparison on identical MC scenarios. Each scheme uses its own training TOML config (so network architecture, navigation params, etc. are preserved):

```bash
uv run python -m aerocapture.training.compare_guidance \
    --n-sims 500 \
    --schemes equilibrium_glide energy_controller pred_guid fnpag ftc neural_network piecewise_constant
```

### Sensitivity Analysis

Variance-based sensitivity analysis to rank which MC dispersion parameters most influence DV cost. Uses SALib (Morris elementary effects + Sobol indices) via the `run_with_draws()` PyO3 API.

```bash
# Full analysis: Morris to rank top-10, then Sobol on those 10
uv run python -m aerocapture.training.sensitivity \
    configs/training/msr_aller_eqglide_train.toml \
    --morris-n 1000 --sobol-n 1024 --top-k 10 \
    --output-dir training_output/sensitivity/

# Morris only (faster screening pass)
uv run python -m aerocapture.training.sensitivity \
    configs/training/msr_aller_eqglide_train.toml \
    --morris-n 500 --morris-only

# Sobol only on all 26 parameters
uv run python -m aerocapture.training.sensitivity \
    configs/training/msr_aller_eqglide_train.toml \
    --sobol-n 1024 --sobol-only
```

Results saved to `output_dir/sensitivity_results.json` with mu_star/sigma (Morris) and S1/ST indices (Sobol).

## PyO3 Python Bindings

The `aerocapture_rs` Python module provides direct access to the Rust simulator, eliminating subprocess overhead for GA training.

```python
import aerocapture_rs as aero

# Single run
result = aero.run("configs/test/test_ref_orig.toml")
print(f"Captured: {result.captured}, dV: {result.delta_v:.1f} m/s")

# Monte Carlo with trajectory data
mc = aero.run_mc("config.toml", overrides={"simulation.n_sims": 1000},
                 include_trajectories=True)
print(f"Final records: {mc.final_records.shape}")       # (1000, 52)
print(f"Trajectories: {len(mc.trajectories)} arrays")   # list of (N, 17)

# Batch run with per-sim overrides (parallel via Rayon)
overrides = [{"guidance.equilibrium_glide.gain_kp": v} for v in [0.1, 0.5, 1.0]]
batch = aero.run_batch("config.toml", overrides)

# Run with pre-computed draws (e.g. SALib sensitivity matrices)
draws = np.zeros((100, 26), dtype=np.float64)  # shape (N, 26)
result = aero.run_with_draws("config.toml", draws)
print(f"Dispersions roundtrip: {result.dispersions.shape}")  # (100, 26)
```

Build with: `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`

The training pipeline auto-detects PyO3 and falls back to subprocess if not installed.

## Validation

The Rust simulator has been validated against a reference implementation across all 725 timesteps of a guided FTC trajectory:
- **22 of 24** photo output columns are bit-identical
- The remaining 2 differ only at the first timestep due to uninitialized variable artifacts in the reference

## Testing

```bash
# Rust tests
cargo test --release --manifest-path src/rust/Cargo.toml

# Python tests (~505 tests)
uv run pytest tests/

# Linting + type checking
./lint_code.sh        # ruff (imports, format, lint) + mypy

# Full Rust check (test + fmt + clippy + release build)
./check_all.sh
```

**Rust tests** cover: physics (J2/J3/J4 gravity with proptest), all 7 guidance schemes, exit phase guidance (pdyn feedback with proptest), phase dispatch, lateral guidance, navigation (bias + EKF, SimPhase gating), wind model, control (pilot dynamics, angle utils), DOPRI45 adaptive integrator, TOML base inheritance, virtual DV ranges, trajectory heat load, density perturbation (OU config presets, step function statistics, TOML parsing, E2E backward compat).

**Python tests** cover: parsers, regression, GA pipeline, training visualization, training animation, NN weight initialization, curated-CDF seed framework (stratified picking, curation probe, checkpoint roundtrip), graceful interrupt, TOML base inheritance, PyO3 integration (bit-identical regression), corridor accumulator, unified cost function, sensitivity analysis (build_problem structure + Morris/Sobol pipeline shape/correctness), Parquet output (write/read roundtrip, schema, metadata, data integrity), RL training (GaussianPolicy / ValueNetwork, PyTorch→JSON export roundtrip, AerocaptureVecEnv wrapper, PBRS telescoping identity + terminal cost parity with GA, PPO GAE/update rule, SAC update rule, config parser with nested ppo overrides, RL-flavored PDF report charts, end-to-end PPO smoke test).

## CI

GitHub Actions runs on PRs to `main` and manual dispatch:

- **Rust**: `cargo fmt --check`, `cargo clippy`, `cargo test --release`
- **Python**: `ruff check`, `ruff format --check`, `mypy`, `pytest`
- **PyO3**: `maturin develop --release`, `pytest tests/test_pyo3.py tests/test_v2_rust_python_equivalence.py tests/test_gru_pso_smoke.py tests/test_gru_ppo_smoke.py tests/test_lstm_pso_smoke.py tests/test_lstm_ppo_smoke.py tests/test_flat_weights_to_json_lstm.py`

## Build Commands

```bash
./build.sh              # Build Rust binary + PyO3 bindings (-c to clean artifacts)
./setup_env.sh          # Create fresh .venv + install deps
./lint_code.sh          # Run ruff (imports, format, lint) + mypy
./check_all.sh          # Rust: test + fmt --check + clippy + release build
./upgrade_dependencies.sh   # uv sync --upgrade
```

## Roadmap

See [TODO.md](TODO.md) for the prioritized task list and [IMPROVEMENTS.md](IMPROVEMENTS.md) for the detailed physics, GNC, and software improvement roadmap.
