# Improvements Roadmap

This document lists physics, GNC, and software improvements for the aerocapture simulator. The Rust simulator is validated against the original Fortran reference (22/24 photo columns bit-identical across 725 timesteps) and all 6 guidance schemes are operational with GA-optimizable parameters. Items are grouped by domain and roughly prioritized within each section (high-impact first).

**Legend**: [DONE] = implemented, [PARTIAL] = partially addressed, [NEW] = added since last revision

---

## 1. Atmosphere Model

### 1.1 Upgrade to Mars Climate Database (MCD)

The current model uses tabulated MarsGram 3.8 density vs altitude (`data/atmosphere/mars.dat`) with linear interpolation and exponential extrapolation above the table ceiling (`physics/atmosphere.rs`).

- **Improvement**: Interface with MCD v6+ (LMD/CNRS), which provides density, temperature, pressure, and winds as functions of (altitude, latitude, longitude, solar longitude, local time, dust scenario).
- **Impact**: More realistic atmospheric variability, especially important for Monte Carlo campaigns.

### 1.2 Separate truth vs onboard atmosphere models

Currently both truth and onboard models use the same density table. The only difference is that the truth model applies a dispersion multiplier via the `DensityProfile` (5 altitude breakpoints in `data/dispersions.rs`). The onboard model is unrealistically close to truth.

- **Improvement**: Use a separate, degraded onboard atmosphere model (e.g., exponential with fitted scale height) distinct from the truth model. The density estimator should then correct the onboard model, not just track a known table.
- **Impact**: More realistic navigation filter behavior and guidance robustness assessment.

### 1.3 Time-varying density perturbations

Current dispersions are static per-run (piecewise linear altitude profile applied as a constant multiplier). Real atmospheric variability includes gravity waves, dust storms, and diurnal cycles.

- **Improvement**: Add stochastic density perturbations that evolve during a run (e.g., Gauss-Markov process, or Dryden-like turbulence model applied to density).
- **Impact**: Tests guidance robustness to transient density features, not just static biases.

### 1.4 [NEW] Wind model

The wind model is currently a stub returning zero velocity (`physics/winds.rs`). Mars entry encounters zonal winds up to ~100 m/s.

- **Improvement**: Implement altitude-dependent zonal/meridional wind profiles (e.g., from MCD or parametric models). Add wind dispersions for Monte Carlo.
- **Impact**: Affects relative velocity, dynamic pressure, and guidance accuracy — particularly important for low-L/D vehicles.

### 1.5 Horizontal atmosphere variation

The original Fortran had a crude sinusoidal horizontal variation model. The Rust code does not implement this.

- **Improvement**: Use MCD's native 3D fields, or implement a proper gravity wave model with configurable wavelength/amplitude spectra.

---

## 2. Gravity Model

### 2.1 Higher-order gravity harmonics

Current model (`physics/gravity.rs`) uses J2 only: radial and lateral components from a single zonal harmonic.

- **Improvement**: Add J3, J4 (significant for Mars), or implement full spherical harmonics up to degree N using a standard gravity field (e.g., GMM-3 for Mars).
- **Impact**: Minor for aerocapture (aero forces dominate during atmospheric pass), but important for coasting phases and orbit determination after exit.

### 2.2 Third-body perturbations

No Sun or Phobos/Deimos gravitational perturbation.

- **Improvement**: Add point-mass third-body accelerations for long-duration coasting arcs.
- **Impact**: Low during atmospheric pass, relevant for multi-orbit or multi-pass scenarios.

---

## 3. Aerodynamics

### 3.1 Mach-dependent aero coefficients

Mach number is not used — Cx and Cz are interpolated from 1D tables indexed by AoA only (`data/aerodynamics.rs`).

- **Improvement**: Implement 2D aero tables Cx(AoA, Mach) and Cz(AoA, Mach). At Mars entry speeds (>5 km/s), real-gas and rarefied flow effects change the coefficients significantly across the trajectory.
- **Impact**: High — the vehicle transitions from free-molecular to hypersonic continuum flow during entry, with Cx/Cz variations of 10-30%.

### 3.2 Dynamic aero uncertainty model

Current aero dispersions are constant uniform multipliers across the entire trajectory (`data/dispersions.rs`, aerodynamics section).

- **Improvement**: Altitude/Mach-dependent aero uncertainty profiles. Different uncertainty levels for different flow regimes (free-molecular, transitional, continuum).
- **Impact**: Better Monte Carlo fidelity.

### 3.3 Ablation coupling

No coupling between thermal protection system ablation and aerodynamic coefficients. Mass loss and shape change are ignored.

- **Improvement**: Simple mass-decrement model coupled with AoA trim shift.
- **Impact**: Low for short aerocapture passes (~5 min), potentially significant for multi-pass scenarios.

---

## 4. Thermal Environment

### 4.1 Improved heat flux correlation

Current model: `q = Cq * sqrt(rho) * V^3` (Sutton-Graves convective correlation) in `physics/aerodynamics.rs`. No radiative heating.

- **Improvement**: Add radiative heating component (significant above ~6 km/s), possibly via Tauber-Sutton or tabulated CFD-based correlations. Add stagnation point vs acreage distribution.
- **Impact**: More accurate TPS sizing and thermal constraint evaluation.

### 4.2 [NEW] Integrated heat load tracking

Instantaneous heat flux is computed and max values tracked (`runner.rs`), but total integrated heat load is not accumulated. The state vector has a `flux` component but it stores instantaneous, not cumulative.

- **Improvement**: Accumulate `sum(q * dt)` during the atmospheric pass and output it in final CSV. Add integrated heat load as a cost function component for GA training.
- **Impact**: Enables TPS mass estimation and thermal margin assessment.

### 4.3 Heat rate and heat load as guidance constraints

Heat flux is tracked but not used as a guidance constraint.

- **Improvement**: Add heat rate and integrated heat load to the constraint set that guidance can actively manage (e.g., bank-up to reduce heating when approaching limits).

---

## 5. Navigation

### 5.1 Replace bias-only navigation with a Kalman filter

The current navigation model (`gnc/navigation/estimator.rs`) adds constant biases to the true state. There is no Kalman filter, no IMU model, no star tracker update, and no actual state estimation.

- **Improvement**: Implement an Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF) with:
  - IMU propagation (accelerometers + gyros with bias, scale factor, noise models)
  - Star tracker updates (periodic, with blackout during atmospheric pass)
  - Drag-derived altitude updates
- **Impact**: Critical for realistic closed-loop GNC assessment. The current model is too optimistic (perfect knowledge with small bias).

### 5.2 [PARTIAL] Improve density estimation filter

The exponential filter `gain = (1-lambda)*gain_prev + lambda*(rho_est/rho_model)` is functional with lambda clamped to [0.01, 0.99] (`estimator.rs`), preventing the legacy Fortran instability. However it remains a simple exponential smoother with no state covariance.

- **Remaining improvements**:
  - Add gain saturation bounds (e.g., 0.1 < gain < 10) as a safety net
  - Replace with a proper density estimation state in the Kalman filter (see 5.1)
  - Add outlier rejection (if |rho_est/rho_model - gain| > threshold, hold previous value)
- **Impact**: Improves guidance robustness for edge cases where the density ratio spikes transiently.

### 5.3 Drag acceleration extraction

Currently `rho_est = 2*|a_drag|*m / (Cx*S*V^2)` assumes all measured acceleration is drag. This ignores the lift component and gravity projection.

- **Improvement**: Decompose the total measured acceleration into drag and lift components using the known bank angle and estimated AoA, or use a full acceleration model inversion.
- **Impact**: More accurate density estimation, especially at high bank angles where lift-to-drag ratio contributes to the total acceleration.

---

## 6. Guidance

### 6.1 [DONE] Predictor-corrector guidance

FNPAG (`gnc/guidance/fnpag.rs`) implements a full numerical predictor-corrector (Ping Lu's algorithm):

- Forward trajectory prediction via 2000-step Euler integration with simplified dynamics
- Secant method root-finding to match target exit orbital energy
- Re-plans every guidance step — inherently robust to dispersions

The original FTC scheme (`gnc/guidance/ftc.rs`) remains a proportional feedback law on reference trajectory deviations (not a true predictor-corrector), but FNPAG fills this gap.

### 6.2 Fix gain discontinuity at altitude table boundary

The altitude-dependent gain table (`compute_gains()` in `ftc.rs`) has entries up to a maximum altitude. Above this ceiling, extrapolation can cause gain discontinuities.

- **Improvement**: Extend the gain table to cover the full altitude range, or implement smooth gain scheduling that fades to zero above the sensible atmosphere.
- **Impact**: Prevents transient bank angle spikes during initial entry and final exit.

### 6.3 [NEW] Exit phase guidance

The navigation state tracks `guidance_phase` (capture=1, exit=2) but the phase transition logic is currently hardcoded to phase 1. Exit phase guidance is not active.

- **Improvement**: Implement exit phase guidance that targets the final orbit parameters (apoapsis, periapsis, inclination) using the remaining atmospheric pass. Enable phase transition when radial velocity becomes positive after the trajectory nadir.
- **Impact**: Better orbit insertion accuracy, especially for inclination and RAAN corrections.

### 6.4 [NEW] FNPAG predictor fidelity

FNPAG uses simplified dynamics for prediction (planar, no J2, constant bank, exponential atmosphere). This limits accuracy for long atmospheric passes or high-latitude entries.

- **Improvement**: Add J2 gravity and 3D trajectory propagation to the predictor. Use the actual atmosphere table instead of an exponential fit. Consider adaptive prediction horizon.
- **Impact**: Better convergence and accuracy for challenging entry conditions.

### 6.5 Bank angle rate and acceleration limits

Guidance computes bank angle without considering how fast the vehicle can actually rotate. The pilot model (`gnc/control/pilot.rs`) enforces rate limits, but guidance doesn't anticipate this.

- **Improvement**: Add bank angle rate and acceleration constraints to guidance command generation. Use rate-limited command shaping.
- **Impact**: Reduces guidance-pilot lag, improves tracking during rapid maneuvers.

---

## 7. Control (Pilot Model)

### 7.1 Actuator dynamics

The pilot model (`gnc/control/pilot.rs`) supports Perfect, FirstOrder, and SecondOrder dynamics with bank rate saturation. No actuator saturation, no thruster on/off logic, no fuel consumption.

- **Improvement**: Model RCS thruster pulse-width modulation, fuel mass tracking, thruster failure modes.
- **Impact**: More realistic bank angle response and enables propellant budget analysis.

### 7.2 AoA modulation

AoA is set from an altitude-dependent incidence profile lookup (`data/incidence.rs`). No dynamic AoA control during the pass.

- **Improvement**: Implement AoA trim control that adjusts L/D as a function of altitude or energy to expand the flyable corridor.
- **Impact**: Additional control authority for corridor management.

---

## 8. Lateral Guidance & Roll Reversal

### 8.1 Improved roll reversal logic

The current roll sign management (`lateral_guidance()` in `ftc.rs`) uses a velocity-dependent inclination error corridor: `i_max(V) = (V/corridor_slope)^4 + corridor_intercept`. This can cause unnecessary reversals.

- **Improvement**: Predictive roll reversal that accounts for the remaining trajectory and total delta-inclination needed. Bank-angle-weighted heading error integration.
- **Impact**: Fewer roll reversals = less propellant, less thermal exposure.

### 8.2 Coupled longitudinal-lateral guidance

Longitudinal (bank magnitude) and lateral (bank sign) guidance are currently decoupled. The bank magnitude is computed ignoring the sign constraint, then the sign is chosen independently.

- **Improvement**: Joint optimization of bank magnitude and sign, accounting for both energy management and plane change simultaneously.
- **Impact**: Better overall trajectory optimization, especially for missions requiring large plane changes.

### 8.3 [NEW] NN-driven roll reversal

The neural network guidance currently outputs a signed bank angle via `atan2(out[0], out[1])` but still relies on the FTC lateral guidance for roll reversal decisions.

- **Improvement**: Let the NN handle roll reversal directly by training it to output the sign as part of its bank angle command. This removes the dependency on the classical corridor-based reversal logic.
- **Impact**: Potentially smoother trajectories and fewer reversals if the NN learns a better policy.

---

## 9. Monte Carlo Framework

### 9.1 Advanced sampling methods

Current Monte Carlo uses Gaussian (initial state, navigation) and uniform (atmosphere, aero, pilot) random draws (`data/dispersions.rs`).

- **Improvement**: Latin Hypercube Sampling (LHS), Sobol quasi-random sequences, or importance sampling for rare-event analysis (e.g., TPS failure probability).
- **Impact**: Better coverage of the dispersion space with fewer runs (10x efficiency improvement typical for LHS vs random).

### 9.2 [DONE] Parallel execution

Monte Carlo runs are parallelized with Rayon (`runner.rs`). Each run is independent, using `par_iter().map()` for linear speedup with core count.

### 9.3 Sensitivity analysis

No built-in sensitivity analysis (which dispersions matter most?).

- **Improvement**: Add Sobol indices computation, tornado diagrams, or Morris method screening to identify dominant uncertainty contributors.
- **Impact**: Focuses engineering effort on the uncertainties that actually matter.

---

## 10. Integration

### 10.1 Adaptive step sizing

The Gill-variant RK4 integrator (`integration/rk4.rs`) uses a fixed timestep (configurable via TOML `periods.integration`, typically 1 s). This is fine during the atmospheric pass but wasteful during coasting.

- **Improvement**: Implement RK4(5) Dormand-Prince with adaptive step control, or at minimum a two-phase scheme (large steps during coast, small steps during atmospheric pass).
- **Impact**: Faster simulation for multi-pass or long-coast scenarios, better accuracy during rapid dynamics.

### 10.2 Event detection

No proper event detection (atmosphere entry/exit, bounce, crash). Currently uses altitude threshold checks at fixed intervals.

- **Improvement**: Implement root-finding-based event detection (e.g., Brent's method on altitude - threshold = 0) to precisely locate atmosphere boundaries and extrema.
- **Impact**: More accurate entry/exit timing, cleaner phase transitions.

---

## 11. Output & Analysis

### 11.1 [PARTIAL] Output formats

Output is now clean CSV with named column headers — photo CSV (21 columns) and final CSV (39 columns) with 10 significant figures (`simulation/output.rs`). This is a major improvement over the legacy fixed-width Fortran format.

- **Remaining improvements**:
  - Add HDF5 or Parquet output for large MC campaigns (smaller files, faster I/O)
  - Embed config/dispersions/random seed metadata in the output file for reproducibility
  - Add per-run dispersion values to the final CSV

### 11.2 Real-time visualization

No real-time feedback during simulation.

- **Improvement**: Optional WebSocket or shared-memory interface for live trajectory plotting during long MC campaigns.

### 11.3 [NEW] Training visualization

The GA training pipeline saves checkpoints but has limited visualization of training progress.

- **Improvement**: Live cost function plots, population diversity metrics, convergence diagnostics. Optionally integrate with TensorBoard or Weights & Biases for experiment tracking.
- **Impact**: Faster debugging of training runs, better hyperparameter tuning.

---

## 12. Training & ML

### 12.1 [NEW] Alternative optimization algorithms

The current GA uses binary-encoded chromosomes with roulette wheel selection, uniform crossover, and bit-flip mutation (`training/train.py`). This is functional but may not be the most efficient optimizer for all parameter spaces.

- **Improvement**: Add CMA-ES (excellent for continuous optimization), PSO (particle swarm), or differential evolution as alternative optimizers. Consider Bayesian optimization for expensive-to-evaluate configurations. Investigate the use of Reinforcement Learning for NN (and others?).
- **Impact**: Potentially faster convergence, especially for smooth parameter landscapes (equilibrium glide, energy controller).

### 12.2 [NEW] Recurrent and transformer architectures

The neural network guidance uses a feedforward architecture (`gnc/guidance/neural.rs`). This cannot exploit temporal correlations in the trajectory.

- **Improvement**: Implement LSTM or Transformer-based guidance that conditions on trajectory history (previous states, density estimates, bank angle commands). Requires backpropagation through time for training but can also be trained with GA and equivalent.
- **Impact**: Could learn more sophisticated strategies that adapt to evolving conditions during the pass, rather than reacting to instantaneous state only.

### 12.3 [NEW] Neural navigation and control

Only guidance uses neural networks. Navigation and control use classical algorithms.

- **Improvement**: Train neural counterparts for the density estimator (replacing the exponential filter) and the pilot model (replacing the first/second-order dynamics). Compare against classical algorithms on identical MC scenarios.
- **Impact**: Benchmarks classical vs learned components; may discover better density estimation strategies.

### 12.4 [NEW] Cost function design

The current cost function penalizes energy error, inclination error, max g-load, and heat load. The relative weighting and functional form significantly affect what the optimizer converges to.

- **Improvement**: Investigate multi-objective optimization (Pareto fronts of correction cost vs constraints). Add delta-V cost as a primary objective. Explore curriculum learning (easy scenarios first, then harder dispersions).
- **Impact**: Better-tuned guidance parameters that balance performance across the dispersion space.

---

## 13. Mission Extensions

### 13.1 Multi-pass aerocapture

Current code assumes single-pass aerocapture (enter atmosphere, exit, done).

- **Improvement**: Support multi-pass scenarios where the vehicle performs several atmospheric dips to gradually lower the orbit. Requires inter-pass coast propagation and re-entry targeting.
- **Impact**: Enables lower-L/D vehicle designs and reduces peak heating.

### 13.2 Drag modulation

Current control is bank angle only (lift vector orientation). Some aerocapture concepts use drag modulation (jettisoning ballast, deploying drag devices) instead of or in addition to bank angle.

- **Improvement**: Add drag modulation as an alternative control mode. Model discrete drag events (ballast jettison) or continuous modulation (deployable surfaces).
- **Impact**: Enables simulation of a broader class of aerocapture vehicles.

### 13.3 Venus and Titan applications

The code currently supports Moon, Earth, Mars, Jupiter (`config.rs` Planet enum). Venus and Titan are prime aerocapture targets.

- **Improvement**: Add Venus (CO2 atmosphere, ~90 bar surface, extreme heating) and Titan (N2/CH4 atmosphere, low gravity, thick atmosphere) with appropriate atmosphere tables and gravity models.
- **Impact**: Broadens the simulator's applicability to other planetary missions.

### 13.4 [NEW] Earth return (ESR) mission profiles

The simulator has ESR reference trajectory data (`data/reference_trajectory/esr_aller.dat`) but ESR-specific configurations and validation are limited.

- **Improvement**: Develop and validate ESR entry profiles, including Earth-specific atmosphere dispersions, higher entry velocities (~12 km/s), and appropriate thermal constraints.
- **Impact**: Supports Mars Sample Return Earth entry phase analysis.
