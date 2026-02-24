# Improvements Roadmap

This document lists physics, GNC, and software improvements to pursue **after** the Rust simulator exactly replicates the Fortran reference. Items are grouped by domain and roughly prioritized within each section (high-impact first).

---

## 1. Atmosphere Model

### 1.1 Upgrade to Mars Climate Database (MCD)
The current model uses tabulated MarsGram 3.8 density vs altitude (fatmos.f → intrmo.f interpolation). MarsGram 3.8 dates from the late 1990s.
- **Improvement**: Interface with MCD v6+ (LMD/CNRS), which provides density, temperature, pressure, and winds as functions of (altitude, latitude, longitude, solar longitude, local time, dust scenario).
- **Impact**: More realistic atmospheric variability, especially important for Monte Carlo campaigns.

### 1.2 Separate truth vs onboard atmosphere models
Currently both `imodel=0` (truth) and `imodel=1` (onboard) call the same `intrmo()` with the same table. The only difference is that `imodel=0` applies a dispersion multiplier via the `profro` profile. This means the onboard model is unrealistically close to truth.
- **Improvement**: Use a separate, degraded onboard atmosphere model (e.g., exponential with fitted scale height) distinct from the truth model. The density estimator should then correct the onboard model, not just track a known table.
- **Impact**: More realistic navigation filter behavior and guidance robustness assessment.

### 1.3 Time-varying density perturbations
Current dispersions are static per-run (piecewise linear altitude profile applied as a constant multiplier). Real atmospheric variability includes gravity waves, dust storms, and diurnal cycles.
- **Improvement**: Add stochastic density perturbations that evolve during a run (e.g., Gauss-Markov process, or Dryden-like turbulence model applied to density).
- **Impact**: Tests guidance robustness to transient density features, not just static biases.

### 1.4 Horizontal atmosphere variation
The code has a `atmvar` flag and sinusoidal horizontal variation model (based on Cartesian distance from entry point). This is crude.
- **Improvement**: Use MCD's native 3D fields, or implement a proper gravity wave model with configurable wavelength/amplitude spectra.

---

## 2. Gravity Model

### 2.1 Higher-order gravity harmonics
Current model (fgravi.f) uses J2 only: radial and lateral components from a single zonal harmonic.
- **Improvement**: Add J3, J4 (significant for Mars), or implement full spherical harmonics up to degree N using a standard gravity field (e.g., GMM-3 for Mars).
- **Impact**: Minor for aerocapture (aero forces dominate during atmospheric pass), but important for the coasting phases and orbit determination after exit.

### 2.2 Third-body perturbations
No Sun or Phobos/Deimos gravitational perturbation.
- **Improvement**: Add point-mass third-body accelerations for long-duration coasting arcs.
- **Impact**: Low during atmospheric pass, relevant for multi-orbit or multi-pass scenarios.

---

## 3. Aerodynamics

### 3.1 Mach-dependent aero coefficients
The code computes Mach number (`vitmac = vitrel/vitson`) but never uses it — Cx and Cz are looked up by AoA only (faeros.f).
- **Improvement**: Implement 2D aero tables Cx(AoA, Mach) and Cz(AoA, Mach). At Mars entry speeds (>5 km/s), real-gas and rarefied flow effects change the coefficients significantly across the trajectory.
- **Impact**: High — the vehicle transitions from free-molecular to hypersonic continuum flow during entry, with Cx/Cz variations of 10-30%.

### 3.2 Dynamic aero uncertainty model
Current aero dispersions are constant multipliers (dadrag, dnlift in `/mecaer/`). Applied uniformly across the entire trajectory.
- **Improvement**: Altitude/Mach-dependent aero uncertainty profiles. Different uncertainty levels for different flow regimes (free-molecular, transitional, continuum).
- **Impact**: Better Monte Carlo fidelity.

### 3.3 Ablation coupling
No coupling between thermal protection system ablation and aerodynamic coefficients. Mass loss and shape change are ignored.
- **Improvement**: Simple mass-decrement model coupled with AoA trim shift.
- **Impact**: Low for short aerocapture passes (~5 min), potentially significant for multi-pass scenarios.

---

## 4. Thermal Environment

### 4.1 Improved heat flux correlation
Current model: `q = Cq * sqrt(rho) * V^3.05` (Sutton-Graves convective correlation). No radiative heating.
- **Improvement**: Add radiative heating component (significant above ~6 km/s), possibly via Tauber-Sutton or tabulated CFD-based correlations. Add stagnation point vs acreage distribution.
- **Impact**: More accurate TPS sizing and thermal constraint evaluation.

### 4.2 Integrated heat load as constraint
Heat load is integrated (`somflu` in state vector) but not used as a guidance constraint.
- **Improvement**: Add heat rate and integrated heat load to the constraint set that guidance can actively manage.

---

## 5. Navigation

### 5.1 Replace bias-only navigation with a Kalman filter
The current navigation model (naviga.f) simply adds constant biases to the true state: `positn(i) = positr(i) + dispos(i)`. There is no Kalman filter, no IMU model, no star tracker update, and no actual state estimation.
- **Improvement**: Implement an Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF) with:
  - IMU propagation (accelerometers + gyros with bias, scale factor, noise models)
  - Star tracker updates (periodic, with blackout during atmospheric pass)
  - Drag-derived altitude updates
- **Impact**: Critical for realistic closed-loop GNC assessment. The current model is too optimistic (perfect knowledge with small bias).

### 5.2 Improve density estimation filter
The exponential filter `coefro = (1-lambda)*coefro + lambda*(roesti/rorefr)` with lambda=0.8 is:
- **Fragile**: Previously observed to explode at step ~40 in Fortran — this was traced to a **common block size mismatch bug** in `guilat.f` that corrupted lambda from 0.8 to 56.0 (see CLAUDE.md). The bug has been **fixed**; the filter now works correctly with lambda=0.8.
- **Naive**: No state covariance, no outlier rejection.
- **Improvement options** (for robustness beyond the bug fix):
  - Add coefro clamping/saturation (e.g., 0.1 < coefro < 10) as a safety net
  - Replace with a proper density estimation state in the Kalman filter
  - Add outlier rejection (if |roesti/rorefr - coefro| > threshold, hold previous value)
- **Impact**: Improves guidance robustness for edge cases where the density ratio spikes transiently.

### 5.3 Drag acceleration extraction
Currently `roesti = 2*|acdram|*m / (Cx*S*V^2)` assumes all measured acceleration is drag. This ignores the lift component and gravity projection.
- **Improvement**: Decompose the total measured acceleration into drag and lift components using the known bank angle and estimated AoA, or use a full acceleration model inversion.
- **Impact**: More accurate density estimation, especially at high bank angles where lift-to-drag ratio contributes to the total acceleration.

---

## 6. Guidance

### 6.1 Implement actual predictor-corrector
The current "FTC predictor-corrector" has no predictor. The bank angle command is computed from a corrector formula only:
```
cosmuc = cmunom + gaindh*(vitrad - hdtnom)/pdyneq + gainpd*(pdyneq - prenom)/pdyneq
```
This is a proportional feedback law on reference trajectory deviations, not a predictor-corrector.
- **Improvement**: Implement a numerical predictor that propagates the trajectory forward (using simplified dynamics) to predict the exit conditions, then iterates on bank angle to hit the target orbit. This is the standard approach in modern aerocapture guidance (e.g., FNPEG, HYPAS).
- **Impact**: Dramatically better targeting accuracy, especially for large dispersions where the linear corrector saturates.

### 6.2 Fix gain discontinuity at altitude table boundary
The altitude-dependent gain table (tbgain.f) has 26 entries from 0 to 96.25 km. Above 96.25 km, the code uses `coefpd = -1e-6` (an extrapolation artifact), causing `gainpd` to jump from ~936 to ~950,305 — a factor of 1000x discontinuity.
- **Improvement**: Extend the gain table to cover the full altitude range, or implement smooth gain scheduling that fades to zero above the sensible atmosphere.
- **Impact**: Prevents transient bank angle spikes during initial entry and final exit.

### 6.3 Exit phase guidance
The code has `iphase=2` (exit phase) detection logic in naviga.f, but guilon.f only calls `guicap` (capture phase). The exit phase guidance (`guiext`) is referenced but not connected.
- **Improvement**: Implement exit phase guidance that targets the final orbit parameters (apoapsis, periapsis, inclination) using the remaining atmospheric pass.
- **Impact**: Better orbit insertion accuracy, especially for the inclination and RAAN corrections.

### 6.4 Table-based command interpolation
The precomputed bank angle table `tabepd(2, 500, 500)` uses nearest-neighbor lookup (find closest energy/pdyn grid point). No interpolation between grid points.
- **Improvement**: Bilinear interpolation on the (energy, pdyn) grid.
- **Impact**: Smoother bank angle commands, reduced chattering.

### 6.5 Bank angle rate and acceleration limits
The guidance computes a bank angle without considering how fast the vehicle can actually rotate. The pilot model (pilote.f) enforces rate limits, but the guidance doesn't anticipate this.
- **Improvement**: Add bank angle rate and acceleration constraints to the guidance command generation. Use rate-limited command shaping.
- **Impact**: Reduces guidance-pilot lag, improves tracking during rapid maneuvers.

---

## 7. Control (Pilot Model)

### 7.1 Actuator dynamics
The pilot model (pilote.f) uses simple first/second-order dynamics. No actuator saturation, no thruster on/off logic, no fuel consumption.
- **Improvement**: Model RCS thruster pulse-width modulation, fuel mass tracking, thruster failure modes.
- **Impact**: More realistic bank angle response and enables propellant budget analysis.

### 7.2 AoA modulation
The AoA is currently constant throughout the pass (guialf.f just returns the commanded value). In practice, AoA could be modulated to control L/D and manage the entry corridor.
- **Improvement**: Implement AoA trim control that adjusts L/D as a function of altitude or energy to expand the flyable corridor.
- **Impact**: Additional control authority for corridor management.

---

## 8. Lateral Guidance & Roll Reversal

### 8.1 Improved roll reversal logic
The current roll sign management (vigite.f) uses simple inclination error thresholding with a deadband. This can cause unnecessary reversals.
- **Improvement**: Predictive roll reversal that accounts for the remaining trajectory and total delta-inclination needed. Bank-angle-weighted heading error integration.
- **Impact**: Fewer roll reversals = less propellant, less thermal exposure.

### 8.2 Coupled longitudinal-lateral guidance
Longitudinal (bank magnitude) and lateral (bank sign) guidance are currently decoupled. The bank magnitude is computed ignoring the sign constraint, then the sign is chosen independently.
- **Improvement**: Joint optimization of bank magnitude and sign, accounting for both energy management and plane change simultaneously.
- **Impact**: Better overall trajectory optimization, especially for missions requiring large plane changes.

---

## 9. Monte Carlo Framework

### 9.1 Advanced sampling methods
Current Monte Carlo uses simple uniform/Gaussian random draws (bunifo.f, bgauss.f).
- **Improvement**: Latin Hypercube Sampling (LHS), Sobol quasi-random sequences, or importance sampling for rare-event analysis (e.g., TPS failure probability).
- **Impact**: Better coverage of the dispersion space with fewer runs (10x efficiency improvement typical for LHS vs random).

### 9.2 Parallel execution
The Fortran loop is sequential (one sim at a time in simmsr.f).
- **Improvement**: Rayon-based parallel MC in Rust. Each run is independent, trivially parallelizable.
- **Impact**: Linear speedup with core count. Critical for large MC campaigns (10,000+ runs).

### 9.3 Sensitivity analysis
No built-in sensitivity analysis (which dispersions matter most?).
- **Improvement**: Add Sobol indices computation, tornado diagrams, or Morris method screening to identify dominant uncertainty contributors.
- **Impact**: Focuses engineering effort on the uncertainties that actually matter.

---

## 10. Integration

### 10.1 Adaptive step sizing
The RK4 integrator uses a fixed timestep (tinteg, typically 1s). This is fine during the atmospheric pass but wasteful during coasting.
- **Improvement**: Implement RK4(5) Dormand-Prince with adaptive step control, or at minimum a two-phase scheme (large steps during coast, small steps during atmospheric pass).
- **Impact**: Faster simulation for multi-pass or long-coast scenarios, better accuracy during rapid dynamics.

### 10.2 Event detection
No proper event detection (atmosphere entry/exit, bounce, crash). Currently uses altitude threshold checks at fixed intervals.
- **Improvement**: Implement root-finding-based event detection (e.g., Brent's method on altitude - threshold = 0) to precisely locate atmosphere boundaries and extrema.
- **Impact**: More accurate entry/exit timing, cleaner phase transitions.

---

## 11. Output & Analysis

### 11.1 Modern output formats
Current output: fixed-width Fortran-format text files. Difficult to parse, lossy precision.
- **Improvement**: Add HDF5 or Parquet output alongside legacy format. Include metadata (config, dispersions, random seeds) in the output file.
- **Impact**: Easier post-processing, reproducibility, smaller file sizes for large MC.

### 11.2 Real-time visualization
No real-time feedback during simulation.
- **Improvement**: Optional WebSocket or shared-memory interface for live trajectory plotting during long MC campaigns.

---

## 12. Mission Extensions

### 12.1 Multi-pass aerocapture
Current code assumes single-pass aerocapture (enter atmosphere, bounce, exit, done).
- **Improvement**: Support multi-pass scenarios where the vehicle performs several atmospheric dips to gradually lower the orbit.
- **Impact**: Enables lower-L/D vehicle designs and reduces peak heating.

### 12.2 Drag modulation
Current control is bank angle only (lift vector orientation). Some aerocapture concepts use drag modulation (jettisoning ballast, deploying drag devices) instead of or in addition to bank angle.
- **Improvement**: Add drag modulation as an alternative control mode.
- **Impact**: Enables simulation of a broader class of aerocapture vehicles.

### 12.3 Venus and Titan applications
The code currently supports Moon, Earth, Mars, Jupiter. Venus and Titan are prime aerocapture targets.
- **Improvement**: Add Venus (CO2 atmosphere, ~90 bar surface, extreme heating) and Titan (N2/CH4 atmosphere, low gravity, thick atmosphere) with appropriate atmosphere and gravity models.
