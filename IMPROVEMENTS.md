# Improvements Roadmap

This document lists physics, GNC, and software improvements for the aerocapture simulator. Items are grouped by domain and roughly prioritized within each section (high-impact first).

---

## 1. Atmosphere Model

*(No open items -- see Done section below.)*

---

## 2. Thermal Environment

*(No open items -- see Done section below.)*

---

## 3. Navigation

*(No open items -- see Done section below.)*

---

## 4. Guidance

*(4.1 FTC gain discontinuity fix -- see Done section below.)*

*(4.2 FNPAG predictor fidelity -- see Done section below.)*

*(4.3 Bank angle rate/acceleration limits -- see Done section below.)*

---

## 5. Lateral Guidance

*(5.1 Predictive roll reversal -- see Done section below.)*

---

## 6. Monte Carlo Framework

*(No open items -- see Done section below.)*

---

## 7. Integration

*(7.1 Event detection -- see Done section below.)*

---

## 8. Output & Analysis

*(8.1 Output formats -- see Done section below.)*

---

## 9. Training & ML

### 9.1 Alternative optimization algorithms and real-valued encoding

*(Done -- see Done section below.)*

### 9.2 Representative seed curation

When a new best is found, run a large MC campaign (e.g. 1000 sims) and select the N training seeds whose cost distribution best represents the full 1000-sim distribution. Gives coverage of 1000 sims at the cost of N.

- **Pro:** Efficient -- full difficulty spectrum in few seeds.
- **Con:** Seeds that are representative for the current best may not be representative for a different individual ("representative for whom?" problem). Expensive trigger early in training. Non-trivial to define "representative" (match CDF? percentiles? tail weight?).
- **When:** After epoch rotation proves its value as a refinement.

### 9.3 Multi-objective optimization (mean vs CVaR)

Replace the scalar blended fitness with a 2-objective Pareto front: minimize mean DV and minimize CVaR (worst-tail DV). pymoo's NSGA-II handles this natively.

- **Pro:** No arbitrary alpha blending. Maintains population diversity naturally via Pareto pressure. Gives a menu of solutions from "best average" to "most robust."
- **Con:** Harder to pick "the" best solution afterward (need a selection criterion on the Pareto front). More complex reporting.

### 9.4 Population restarts / island model

Periodically reinitialize part of the population to escape local basins. Or run multiple independent populations that occasionally exchange best individuals (island model with migration).

- **Pro:** Directly combats convergence to a single basin. Island model adds diversity without losing exploitation.
- **Con:** Slower overall convergence. Could layer on top of epoch rotation if basin trapping is still observed after the landscape is made dynamic.

### 9.5 Validation-gated best selection

Instead of log-only validation, use validation as a filter: reject a "new best" if its validation cost is worse than the previous best's validation cost.

- **Pro:** Stronger generalization guarantee.
- **Con:** Adds coupling between validation and training signals. Could slow convergence if validation is noisy. Revisit if overfitting is observed in validation logs.

### 9.6 Validation charts in PDF reports

*(Done -- see Done section below.)*

### 9.7 Adaptive training_n_sims

Start with fewer seeds per generation (e.g. 5) and increase as training progresses (e.g. ramp to 20 by gen 100). Early gens benefit from faster iteration; later gens benefit from more robust evaluation when differences between individuals are smaller.

### 9.8 Bayesian optimization for low-dimensional schemes

Surrogate-model-based optimization using Gaussian Processes or Random Forests as a surrogate for the expensive MC fitness function. Promising for guidance parameter schemes (10-26 params) where each evaluation is costly.

- **Investigation**: Evaluate BoTorch (PyTorch-based, state of the art) or scikit-optimize as a pymoo-compatible backend. Key challenge: noisy fitness from MC evaluation requires noise-aware acquisition functions (e.g., noisy Expected Improvement).
- **Impact**: Could dramatically reduce the number of evaluations needed for convergence on smooth parameter landscapes, at the cost of surrogate model overhead.

### 9.9 Reinforcement learning for neural network guidance

Train the NN guidance controller as an RL policy rather than optimizing static weights via evolutionary algorithms. The simulator is already step-able (state -> action -> next state).

- **Investigation**: Wrap the Rust simulator as a Gym-compatible environment via a PyO3 step API (expose per-timestep state/action interface, not just full-trajectory evaluation). Evaluate PPO, SAC, or TD3 for continuous bank-angle control.
- **Impact**: Fundamentally different paradigm from weight optimization -- RL can learn temporal strategies that static weight optimization cannot express. Separate effort from the pymoo framework.

### 9.10 Recurrent and transformer architectures

The neural network guidance uses a feedforward architecture (`gnc/guidance/neural.rs`). This cannot exploit temporal correlations in the trajectory.

- **Improvement**: Implement LSTM or Transformer-based guidance that conditions on trajectory history (previous states, density estimates, bank angle commands). Requires backpropagation through time for training but can also be trained with GA.
- **Impact**: Could learn more sophisticated strategies that adapt to evolving conditions during the pass, rather than reacting to instantaneous state only.

### 9.11 Neural navigation and control

Only guidance uses neural networks. Navigation and control use classical algorithms.

- **Improvement**: Train neural counterparts for the density estimator (replacing the exponential filter) and the pilot model (replacing the first/second-order dynamics). Compare against classical algorithms on identical MC scenarios.
- **Impact**: Benchmarks classical vs learned components; may discover better density estimation strategies.

---

## 10. Mission Extensions

### 10.1 Earth return (ESR) mission profiles

The simulator has ESR reference trajectory data (`data/reference_trajectory/esr_aller.dat`) but ESR-specific configurations and validation are limited.

- **Improvement**: Develop and validate ESR entry profiles, including Earth-specific atmosphere dispersions, higher entry velocities (~12 km/s), and appropriate thermal constraints.
- **Impact**: Supports Mars Sample Return Earth entry phase analysis.

---

## Done

Items completed and merged.

| Item | When | Details |
|------|------|---------|
| 1.1 Time-varying density perturbations | 2026-04-01 | Gauss-Markov (OU) process with Off/Low/Medium/High/Custom presets. `[monte_carlo.density_perturbation]` TOML section. Multiplicative on static bias. Also: wind dispersions refactored to use same level preset pattern; `common.toml` wind direction_bias reduced from 30 to 10 deg (medium preset). |
| 2.1 Heat rate/load as guidance constraints | 2026-04-01 | Thermal safety limiter: smooth bank-to-lift-up ramp near heat flux/load limits (PR #22) |
| 4.2 Exit phase guidance | 2026-04-01 | Shared pdyn-feedback exit controller for ascending leg after trajectory nadir (PR #22) |
| 3.1 Density filter hardening | 2026-04-02 | Legacy bias-mode filter: rate-of-change limiting (configurable `density_gain_max_delta`, default 0.1) + gain saturation [0.1, 10.0] matching EKF bounds. GA-optimizable for all unsigned-magnitude schemes via `nav.` prefix routing. |
| 3.2 Lift-corrected drag extraction | 2026-04-02 | Both nav modes use `Cx*cos(alpha) + Cz*sin(alpha)` denominator instead of `Cx`-only. Corrects ~4% density estimation error at AoA=10 deg. Activates previously unused `run_cz_bias` MC dispersion. |
| 4.1 FTC gain discontinuity fix | 2026-04-03 | Replaced altitude-dependent gain table (`compute_gains()`) with analytical exponential decay model + cosine fade. Gains smoothly taper to zero between `gain_fade_start_km` (80) and `gain_fade_end_km` (100). GA-optimizable params: `pressure_coeff_base`, `pressure_coeff_scale_height`, `gain_fade_start_km`, `gain_fade_end_km`. |
| 4.2 FNPAG predictor fidelity | 2026-04-04 | Replaced planar 3-state Euler predictor with full 3D 6-DOF RK4 predictor. Adds J2/J3/J4 gravity via `gravity::gravity()`, Coriolis/centrifugal terms via `planet.omega`, and correct inertial exit energy via `total_energy()` (fixes ~30% systematic bias from using relative velocity). Uses onboard atmosphere model (no cheating). Zero lateral lift (roll sign unknown to predictor). Existing `FnpagParams` unchanged (GA-tunable `prediction_dt`, `energy_tol`, bank limits). |
| 4.3 Bank angle rate/acceleration limits | 2026-04-04 | Dispatch-layer S-curve command shaper: uses pilot-realized bank angle as feedback baseline (not last commanded), applies acceleration-limited rate shaping (`max_bank_acceleration` in deg/s^2) producing trapezoidal rate profiles. Falls back to legacy hard-clamp when `[guidance.command_shaping]` absent. GA-optimizable via `shaping.` prefix. Also fixed pre-existing `nav.`/`thermal.` routing bug in `train.py` `_batch_eval`, and added checkpoint chromosome padding for backward compatibility when param space grows. |
| 5.1 Predictive roll reversal | 2026-04-05 | Replaced reactive corridor-based lateral guidance (`(V/slope)^4 + intercept`) with first-order inclination projection. Algorithm projects inclination error forward by `tau` seconds using finite-difference rate estimation and reverses only when projected error exceeds `threshold`. Anti-chatter `min_reversal_interval` prevents rapid re-triggering. `[guidance.lateral]` params: `tau` (s), `threshold` (deg), `min_reversal_interval` (s), `lateral_activation`/`lateral_inhibition` (MJ/kg), `max_reversals`. GA-optimizable for all 5 unsigned-magnitude schemes. |
| 6.1 Advanced sampling methods | 2026-04-07 | LHS (Latin Hypercube Sampling) and Sobol quasi-random sequences added to `dispersions.rs`. 26-dim `DimTransform` system maps unit samples to per-dimension distributions (Gaussian/Uniform/UniformRange/Fixed). `[monte_carlo] sampling = "lhs"/"sobol"/"random"` TOML key. Sobol uses Owen-scrambled `sobol_burley` crate (max 65536 samples). LHS uses stratified Fisher-Yates shuffle. Training configs default to LHS. PyO3 `run_with_draws()` API accepts external (N, 26) draw matrices for SALib integration. |
| 6.2 Sensitivity analysis | 2026-04-07 | SALib-based sensitivity analysis: Morris elementary effects (screening all 26 dispersion dims) + Sobol variance decomposition (S1/ST/S2 on top-k parameters). `build_problem()` converts `[monte_carlo]` config to SALib problem dict with per-dimension distribution types and SI-unit bounds mirroring Rust `build_dim_transforms()`. CLI: `python -m aerocapture.training.sensitivity`. Report Part 3 integration (Morris bar chart + ranked table, Sobol bars, S2 heatmap, convergence plot). |
| 7.1 Event detection | 2026-04-07 | DOPRI45 dense output (Hermite continuous extension) + Brent's root-finding locates bounce, atmosphere exit, crash, and phase transition events to ~1 ms precision within adaptive substeps. Fixed RK4 path unchanged. Event records interleaved into trajectory output. See `integration/events.rs` and `integration/dopri45.rs`. |
| 8.1 Output formats | 2026-04-12 | Parquet output module (`parquet_output.py`): 65-column files (39 final-record + 26 dispersion) with full resolved config metadata. Auto-written alongside PDF reports. Dispersion grid chart now uses three-way trajectory classification (blue/orange/red) with regression on captured only. CSV unchanged. |
| 9.1 Real-valued optimization | 2026-04-12 | Replaced binary GA (16-bit chromosomes, roulette selection, bit-flip mutation) with pymoo-based real-valued optimization. All algorithms work on normalized [0,1] float arrays. Four algorithms: GA (SBX + polynomial mutation), CMA-ES, DE, PSO -- selectable via `[optimizer] algorithm` in TOML or `--algorithm` CLI. Hybrid loop: pymoo `algorithm.next()` stepping with custom outer loop for adaptive seed pool (K-generation checkpoint updates with full re-evaluation), stress testing, Rich TUI, JSONL logging, and corridor accumulation. `AerocaptureProblem(pymoo.Problem)` subclass handles batch PyO3 evaluation with per-seed cost aggregation (RMS). NN path writes temp JSON weight files per individual. Deleted: `local_search.py`, `migration.py`, binary encoding in `evaluate.py`, `GAConfig`. |
| 9.1b Epoch seed rotation + validation gate | 2026-04-13 | Each generation draws `training_n_sims` (default 20) fresh random MC seeds. Full population re-evaluated via `_run_batch()` after `algorithm.next()` (pymoo `Evaluator` skips existing F). Validation gate: fixed 1000-seed set (separate RNG) fires periodically + on new best, logs mean/median/std/p95/worst cost + capture rate to JSONL, shown persistently in TUI. Validation cost curves (mean + p95) overlaid on convergence chart in PDF report. `[optimizer]` TOML keys: `training_n_sims`, `validation_n_sims`, `validation_interval`. |
| 9.1c C-infinity cost function | 2026-04-13 | Replaced log_cap (flat gradient on non-captures: slope 0.1 at dv=10000) with softplus-quadratic `dv_cost`: `cost = dv + sp(dv-T, k=0.01) + sp^2/(2S)`. C-infinity, slope 2.9 at dv=10000 (29x stronger), 34000 cost spread on non-captures (vs 694). Constraint penalties also use softplus (k=100) instead of hard max(0,x) kink. Event photo rows now carry GNC context from enclosing tick (was zeroed, causing trajectory discontinuities). |
| 9.6 Validation charts | 2026-04-13 | Validation mean (solid green) and p95 (dashed green) overlaid on training convergence chart. Sparse points at gens where validation fired. |

---

## Not For Now

Items deferred because they require unavailable data, external dependencies, or are out of scope for the current simulator.

| Item | Reason |
|------|--------|
| MCD v6+ atmosphere interface | External dependency, out of scope |
| Horizontal atmosphere variation | Depends on MCD or gravity wave model |
| Third-body perturbations | Low impact for single-pass aerocapture |
| Mach-dependent Cx/Cz tables | No data available |
| Dynamic aero uncertainty model | Needs regime-specific uncertainty data |
| Ablation coupling | Needs TPS data, low impact for single-pass |
| Radiative heat flux correlation | Borderline relevance at Mars entry speeds |
| RCS actuator dynamics | Not modeling propulsion |
| AoA modulation | Needs multi-AoA/Mach aero data |
| Coupled longitudinal-lateral guidance | Ambitious, lower priority |
| Real-time visualization | Nice-to-have, significant plumbing |
| Multi-pass aerocapture | Large scope expansion |
| Drag modulation | No drag device data |
| Venus/Titan applications | Atmosphere data not available |
