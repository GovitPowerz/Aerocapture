# Improvements Roadmap

This document lists physics, GNC, and software improvements for the aerocapture simulator. Items are grouped by domain and roughly prioritized within each section (high-impact first).

---

## 1. Atmosphere Model

### 1.1 Time-varying density perturbations

Current dispersions are static per-run (piecewise linear altitude profile applied as a constant multiplier). Real atmospheric variability includes gravity waves, dust storms, and diurnal cycles.

- **Improvement**: Add stochastic density perturbations that evolve during a run (e.g., Gauss-Markov process, or Dryden-like turbulence model applied to density).
- **Impact**: Tests guidance robustness to transient density features, not just static biases.

---

## 2. Thermal Environment

### 2.1 Heat rate and heat load as guidance constraints

Heat flux is tracked but not used as a guidance constraint.

- **Improvement**: Add heat rate and integrated heat load to the constraint set that guidance can actively manage (e.g., bank-up to reduce heating when approaching limits).

---

## 3. Navigation

### 3.1 Improve density estimation filter

The exponential filter `gain = (1-lambda)*gain_prev + lambda*(rho_est/rho_model)` is functional with lambda clamped to [0.01, 0.99] (`estimator.rs`), preventing a legacy instability. The EKF includes a density correction state, but the legacy bias-mode filter could still benefit from:

- Add gain saturation bounds (e.g., 0.1 < gain < 10) as a safety net
- Add outlier rejection (if |rho_est/rho_model - gain| > threshold, hold previous value)

### 3.2 Drag acceleration extraction

Currently `rho_est = 2*|a_drag|*m / (Cx*S*V^2)` assumes all measured acceleration is drag. This ignores the lift component and gravity projection.

- **Improvement**: Decompose the total measured acceleration into drag and lift components using the known bank angle and estimated AoA, or use a full acceleration model inversion.
- **Impact**: More accurate density estimation, especially at high bank angles where lift-to-drag ratio contributes to the total acceleration.

---

## 4. Guidance

### 4.1 Fix gain discontinuity at altitude table boundary

The altitude-dependent gain table (`compute_gains()` in `ftc.rs`) has entries up to a maximum altitude. Above this ceiling, extrapolation can cause gain discontinuities.

- **Improvement**: Extend the gain table to cover the full altitude range, or implement smooth gain scheduling that fades to zero above the sensible atmosphere.
- **Impact**: Prevents transient bank angle spikes during initial entry and final exit.

### 4.2 Exit phase guidance

The navigation state tracks `guidance_phase` (capture=1, exit=2) but the phase transition logic is currently hardcoded to phase 1. Exit phase guidance is not active.

- **Improvement**: Implement exit phase guidance that targets the final orbit parameters (apoapsis, periapsis, inclination) using the remaining atmospheric pass. Enable phase transition when radial velocity becomes positive after the trajectory nadir.
- **Impact**: Better orbit insertion accuracy, especially for inclination and RAAN corrections.

### 4.3 FNPAG predictor fidelity

FNPAG uses simplified dynamics for prediction (planar, no J2, constant bank, exponential atmosphere). This limits accuracy for long atmospheric passes or high-latitude entries.

- **Improvement**: Add J2 gravity and 3D trajectory propagation to the predictor. Use the actual atmosphere table instead of an exponential fit. Consider adaptive prediction horizon.
- **Impact**: Better convergence and accuracy for challenging entry conditions.

### 4.4 Bank angle rate and acceleration limits

Guidance computes bank angle without considering how fast the vehicle can actually rotate. The pilot model (`gnc/control/pilot.rs`) enforces rate limits, but guidance doesn't anticipate this.

- **Improvement**: Add bank angle rate and acceleration constraints to guidance command generation. Use rate-limited command shaping.
- **Impact**: Reduces guidance-pilot lag, improves tracking during rapid maneuvers.

---

## 5. Lateral Guidance

### 5.1 Improved roll reversal logic

The current roll sign management uses a velocity-dependent inclination error corridor: `i_max(V) = (V/corridor_slope)^4 + corridor_intercept`. This can cause unnecessary reversals.

- **Improvement**: Predictive roll reversal that accounts for the remaining trajectory and total delta-inclination needed. Bank-angle-weighted heading error integration.
- **Impact**: Fewer roll reversals = less propellant, less thermal exposure.

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
