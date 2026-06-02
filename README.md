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
3. **Lateral guidance** — Roll sign management via predictive first-order inclination projection (projects error forward by configurable tau seconds; reverses only when projected error exceeds threshold). Shared by all unsigned-magnitude schemes (FTC, EqGlide, EnergyCtrl, PredGuid, FNPAG). Remains active during exit phase for inclination correction. PiecewiseConstant always produces a signed bank angle and bypasses lateral and exit guidance entirely; NN does the same in `mode = "full_neural"` (default), but `mode = "magnitude_only"` (set under `[guidance.neural_network]`) reduces the NN output to its absolute value and routes it through the same unsigned-magnitude pipeline as FTC.
4. **Control** — Pilot dynamics model applies rate limits and first/second-order lag to bank angle commands
5. **Integration** — Propagates equations of motion with all physical models above. Adaptive mode sub-steps within each GNC tick — guidance/navigation cadences are unchanged.

## Guidance Schemes

Seven guidance algorithms, all GA-optimizable:

| Scheme | Description | Params | Notes |
|---|---|---|---|
| **Piecewise Constant** | N-segment bank angle profile (N tunable via TOML `n_segments` / `bank_angles = [...]`, default 10) | N | Train first — produces ref trajectory + corridor |
| **FTC** | Predictor-corrector with reference trajectory tracking | 8 | Requires ref trajectory |
| **Neural Network** | Trained NN maps a configurable subset of 35 candidate inputs (incl. seam-free `(sin,cos)` bank-history pairs + data-driven asinh/affine-normalized orbit/energy/acceleration signals + periapsis altitude + 3 live correction-DV components from per-tick `predicted_dv_for_nn` on the current osculating orbit, defined + smooth across capture, no sentinel) to a bank angle. Per-input normalization (`{transform, scale, center}`) is embedded in the model JSON / overridable via TOML `[network.normalization]`, data-driven by `calibrate_inputs.py`. Bank decoders (`output_parameterization`): `atan2_signed` (2-output), `scaled_pi` (`n·π·tanh`), `delta` (bounded increment on prev realized bank) for `full_neural`; `acos_tanh` magnitude for `magnitude_only`. v1 dense-only arch (`layer_sizes`/`activations`) or v2 heterogeneous arch (`[[network.architecture]]`, supports `dense` + `gru` + `lstm` + `window` + `transformer` + `mamba`). Trainable via GA/PSO (pymoo) or RL (PPO with chunked truncated BPTT for recurrent policies, experimental SAC). | arch-dependent | Independent, signed bank, full-envelope (capture + exit) |
| **Equilibrium Glide** | Balances gravity, centrifugal, and lift forces | 7 | Independent |
| **Energy Controller** | Tracks reference energy dissipation profile | 3 | Requires ref trajectory |
| **PredGuid** | Apollo/Shuttle-heritage drag tracking | 3 | Requires ref trajectory |
| **FNPAG** | Lu's numerical predictor-corrector (3D predictor with J2 gravity, RK4) | 5 | Requires ref trajectory |

**Training order:** Run `piecewise_constant` first — it produces `ref_trajectory.dat` (optimized reference) and `corridor_boundaries.npz` (4-layer corridor envelopes from GA population history). Schemes marked "Requires ref trajectory" will error at startup if it's missing.

### NN-vs-FTC Parity Bundle (`nn_joint`)

A separate NN training mode (`./train_all.sh nn_joint`) flips three TOML opt-in knobs under `[guidance.neural_network]` to close the structural gap with FTC's joint-optimization advantage:

- `scaffolding = "full"` extends the PSO chromosome with FTC's 17 scaffolding params (lateral / exit / nav / thermal / shaping), seeded at FTC's GA optimum + jitter, so the NN co-adapts the actuator pipeline rather than driving FTC-tuned frozen values. The knob is three-valued: `"off"` (default) optimizes NN weights only; `"live"` appends just the 3 params that are live in `full_neural` (nav density filter ×2 + command shaping), seeded from defaults with no FTC dependency; `"full"` is the parity-bundle setting here. It is declared per-leaf config (not in the shared `nn_common.toml` base) and the active mode is printed at training start.
- `output_parameterization = "acos_tanh"` swaps the `atan2(out[0], out[1]).abs()` decoder (which wastes half the output range under `magnitude_only`) for `bank = acos(tanh(out[0]))` — single output, smooth `[0, π]` mapping that aligns with FTC's internal `cos_bank` representation. Validated at config load (requires `mode = "magnitude_only"`, last-layer `output_size = 1`, `activation = "tanh"`).
- Warm-start: either the legacy `warm_start_from = "training_output/ftc/best_params.json"` (single supervisor) OR a `[warm_start]` TOML block (multi-supervisor BPTT for recurrent architectures). Both encode the cloned weights into the PSO initial population. Reserved seed offset `4_000_000` keeps the supervised data disjoint from validation / final-eval / RL pools.

All three knobs default off; existing trained NNs and existing configs are bit-identical. Requires FTC training output (`./train_all.sh ftc` first). Spec: `docs/superpowers/specs/2026-05-07-nn-ftc-parity-bundle-design.md` (parity bundle); `docs/superpowers/specs/2026-05-22-warm-start-all-archs-design.md` (multi-supervisor BPTT for Dense/GRU/LSTM/Window/Transformer/Mamba).

### Multi-Supervisor BPTT Warm-Start (`[warm_start]`)

For recurrent NN architectures (GRU/LSTM/Mamba/Transformer), the warm-start path collects supervised traces from multiple non-NN schemes simultaneously, picks the best teacher per Monte Carlo seed (lowest-DV captured trajectory), and runs chunked truncated-BPTT supervised pre-training against the per-seed winners. Configured via a `[warm_start]` TOML block — presence of the block enables warm-start (no separate `warm_start_from` needed):

```toml
[guidance.neural_network]
mode = "magnitude_only"               # required for warm-start
output_parameterization = "acos_tanh" # recommended (atan2_signed only uses half the codomain under .abs())

[warm_start]
supervisor_schemes = ["ftc", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag"]
bptt_length = 32
n_warm_seeds = 200
n_epochs = 10
bound_multiplier = 4.0
jitter = 0.02
cmaes_sigma0 = 0.1
```

Pipeline: each supervisor scheme runs over the same `n_warm_seeds` reserved seed pool via `aerocapture_rs.collect_supervised` (Rust per-trajectory return); the unsigned bank target is `guidance_out.pre_lateral_magnitude` (in [0, π]); `_select_best_teacher_per_seed` picks the captured trajectory with lowest DV per seed; trajectories are split into `bptt_length` windows and forwarded through `V2Policy.forward_seq_means` (autograd-friendly mirror of `evaluate`); Adam MSE with reproducible `torch.manual_seed` for `n_epochs`. Mamba layers get HiPPO + LSTM forget-bias-1 init via `_seed_policy_init` before Adam runs (zero-init would start training at a degenerate fixed point). The cached chromosome is keyed on architecture + supervisor mtimes + `base_mc_seed`, so rerunning with a different `monte_carlo.seed` invalidates the cache. A gen-0 validation MC baseline is auto-written to `warm_start_baseline.json` via `run_mc` (honors `simulation.n_sims`, threads `sim_timeout_secs`) so you have a "did warm-start help?" signal before generation 0.

After warm-start, `aerocapture.training.warm_start_compare.render_trajectory_comparison` runs the supervisor (primary scheme from `supervisor_schemes[0]`) and the warm-started NN on BOTH the training pool (`n_warm_seeds`) and the validation pool (`optimizer.validation_n_sims`), writes 20 SVG panels (5 quantities × 2 sides × 2 pools: corridor pdyn/inclination/bank, altitude vs time, heat flux vs time) under `<save_dir>/warm_start_report/compare_*.svg`, and the `warm_start_report.pdf` includes a side-by-side "Trajectory comparison" section so you can visually compare supervisor vs warm-started NN behaviour on identical dispersion draws -- before PSO even starts. Compute cost is ~2 × (`n_warm_seeds` + `validation_n_sims`) MC sims (~2-3 min for n_warm_seeds=5000, validation_n_sims=1000), best-effort: failure in any (pool, side) records the error in the manifest and the rest of the report still renders. The NN candidate input vector includes four "lateral-state telemetry" inputs (indices 21-24: inclination-error rate, previous bank command, time since last sign flip, integrated inclination error) that make the supervisor's signed-bank decision Markovian -- without them, post-reversal near-duplicate states collapse the supervised MSE target under bimodal sign disagreement (FTC measured ~20% sign-disagree on near-duplicates within radius 0.10).

## GA Optimization

All guidance schemes can be optimized via genetic algorithm. The GA tunes each scheme's parameters to minimize correction delta-V across Monte Carlo dispersions, with TOML-configurable soft constraint penalties for g-load, heat flux, and integrated heat load exceedances.

The cost function is a C-infinity softplus-quadratic DV penalty (`dv_cost`) with a smooth knee at `dv_threshold` (common-default 1000 m/s, code-default 500 m/s), plus TOML-configurable soft constraint penalties, optionally wrapped in a monotonic `cost_transform` (`"linear"` | `"sqrt"` | `"log"` | `"squared"` | `"cubed"`; `configs/training/common.toml` ships `"log"`) to reshape the landscape -- `"log"` (np.log1p) compresses the tail more aggressively than `"sqrt"` while preserving the zero-cost identity. The simulator returns meaningful DV values for all termination outcomes: captured -> real orbital-correction DV; hyperbolic -> `10000 + v_excess`; crash/pending-crash/timeout -> energy-proportional virtual DV `3000 + 1000 * min(|E_orb - E_target|_MJkg, 50) - 500 * t/t_max` (softened near the capture boundary so the optimizer explores closer to the crash limit, with a time-survival term for cold-start gradient).

Training features:
- Auto-resumes from existing checkpoints (use `-fs` to start fresh)
- `--n-gen` means "N additional generations" when resuming
- Resume with a larger/smaller `[optimizer] n_pop`: the resumed population is grown (keep originals + `grow_fresh_fraction` fresh-random + clone+jitter) or shrunk (best-N), in both the single-algorithm and islands paths
- `cost_transform` is recorded in checkpoints; changing it on resume re-validates the best under the new metric (single-algo and per-island)
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

# 3-island PSO / GA / DE with periodic top-3 / worst-6 migration (algorithm = "islands")
# Targets PSO premature swarm-convergence: GA / DE migrants inject fresh search points every
# k_period gens. Per-island n_pop=64 -> 192 total individuals / gen (3x single-algorithm cost).
# Winning island's best_model.json / best_params.json drop into compare_guidance unchanged.
uv run python -m aerocapture.training.train \
    configs/training/msr_aller_islands_train.toml \
    --n-gen 2500
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

### Checkpoint retention

Stateful NN architectures write 10-15 MB per `checkpoint_g{NNNNN}.npz`, so a long PSO run easily fills several GB. Only the latest checkpoint is needed for resume; older ones are useful only for rollback or animation playback.

**Opt in to auto-pruning** by setting `keep_last` in the TOML:

```toml
[checkpoints]
keep_last = 10   # keep only the 10 most recent checkpoint pairs; null = keep all (default)
```

The per-generation JSONL log, `best_model.json`, `warm_start_*` cache, and PDF report are NOT touched, so post-training analysis works unchanged.

**Clean up existing output dirs a posteriori:**

```bash
# Dry-run on one scheme
uv run python -m aerocapture.training.cleanup_checkpoints \
    training_output/neural_network_gru_pso --keep-last 10 --dry-run

# Apply across every scheme directory at once
uv run python -m aerocapture.training.cleanup_checkpoints \
    training_output/ --recursive --keep-last 10
```

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
- **PyO3**: `maturin develop --release`, `pytest tests/test_pyo3.py tests/test_v2_rust_python_equivalence.py tests/test_gru_pso_smoke.py tests/test_gru_ppo_smoke.py tests/test_lstm_pso_smoke.py tests/test_lstm_ppo_smoke.py tests/test_flat_weights_to_json_lstm.py tests/test_rust_python_window_equivalence.py tests/test_window_pso_smoke.py tests/test_flat_weights_to_json_window.py`

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
