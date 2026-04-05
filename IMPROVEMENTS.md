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

### 6.1 Advanced sampling methods

Current Monte Carlo uses Gaussian (initial state, navigation) and uniform (atmosphere, aero, pilot) random draws (`data/dispersions.rs`).

- **Improvement**: Latin Hypercube Sampling (LHS), Sobol quasi-random sequences, or importance sampling for rare-event analysis (e.g., TPS failure probability).
- **Impact**: Better coverage of the dispersion space with fewer runs (10x efficiency improvement typical for LHS vs random).

### 6.2 Sensitivity analysis

No built-in sensitivity analysis (which dispersions matter most?).

- **Improvement**: Add Sobol indices computation, tornado diagrams, or Morris method screening to identify dominant uncertainty contributors.
- **Impact**: Focuses engineering effort on the uncertainties that actually matter.

---

## 7. Integration

### 7.1 Event detection

No proper event detection (atmosphere entry/exit, bounce, crash). Currently uses altitude threshold checks at fixed intervals.

- **Improvement**: Implement root-finding-based event detection (e.g., Brent's method on altitude - threshold = 0) to precisely locate atmosphere boundaries and extrema.
- **Impact**: More accurate entry/exit timing, cleaner phase transitions.

---

## 8. Output & Analysis

### 8.1 Output formats

Output is clean CSV with named column headers — photo CSV and final CSV with 10 significant figures. Remaining improvements:

- Add HDF5 or Parquet output for large MC campaigns (smaller files, faster I/O)
- Embed config/dispersions/random seed metadata in the output file for reproducibility
- Add per-run dispersion values to the final CSV

---

## 9. Training & ML

### 9.1 Alternative optimization algorithms and real-valued encoding

The current GA uses binary-encoded chromosomes with roulette wheel selection, uniform crossover, and bit-flip mutation. This is functional but has known limitations: scale-blind bit-flip mutation doesn't respect parameter sensitivity, and binary encoding wastes resolution.

- **Improvement**: Switch from binary GA to real-valued GA: SBX crossover + polynomial mutation (DEAP built-in), normalize all parameters to [0,1] internally, adaptive mutation rates per parameter. Also consider CMA-ES (excellent for continuous optimization), PSO, differential evolution, or Bayesian optimization. Investigate Reinforcement Learning for NN training.
- **Impact**: Potentially faster convergence, especially for smooth parameter landscapes. Real-valued encoding eliminates the scale-blind bit-flip problem entirely.

### 9.2 Recurrent and transformer architectures

The neural network guidance uses a feedforward architecture (`gnc/guidance/neural.rs`). This cannot exploit temporal correlations in the trajectory.

- **Improvement**: Implement LSTM or Transformer-based guidance that conditions on trajectory history (previous states, density estimates, bank angle commands). Requires backpropagation through time for training but can also be trained with GA.
- **Impact**: Could learn more sophisticated strategies that adapt to evolving conditions during the pass, rather than reacting to instantaneous state only.

### 9.3 Neural navigation and control

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
