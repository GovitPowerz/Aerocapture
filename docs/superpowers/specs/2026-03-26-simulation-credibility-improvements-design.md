# Simulation Credibility Improvements — Design Spec

**Date:** 2026-03-26
**Goal:** Close the most impactful physics and GNC credibility gaps for engineering review readiness and internal confidence in MC results.
**Scope:** Three items, sequenced bottom-up by effort: integrated heat load → wind model → EKF navigation.

---

## 1. Integrated Heat Load Tracking

### Summary

Accumulate total heat load `Q = Σ(q · dt)` during the atmospheric pass. Currently only peak heat flux is tracked — this addition enables TPS mass estimation and thermal margin assessment.

### Implementation

**Key discovery:** `state[6]` already accumulates heat load via RK4 integration of `dflux = cq * sqrt(rho) * V^3.05`. The cumulative value is stored in `final_record[28]` as `integrated_flux_mj_m2`. The IMPROVEMENTS.md claim that "it stores instantaneous, not cumulative" is incorrect. The work is **exposing** this existing value, not computing it.

**Output touchpoints (exposure of existing `final_record[28]`):**
- Trajectory data: replace reserved column 15 with cumulative heat load (kJ/m²) from photo array
- Photo CSV: new column `heat_load_kj_m2` (running total at each timestep)
- PyO3 `SimResult`: expose as `.integrated_heat_load` (converts MJ/m² → kJ/m²)
- Charts (`charts.py`): new `chart_heat_load_time()` panel for cumulative heat load vs time

**GA cost function:**
- Add `max_heat_load` (kJ/m²) to `[flight.constraints]` in mission TOMLs
- Add heat load penalty term in `evaluate.py`, same normalized soft-penalty pattern as g-load and heat flux

**Backward compatibility:** All existing configs produce identical results. The new field is additive.

---

## 2. Wind Model

### Summary

Replace the zero-returning stub in `physics/winds.rs` with altitude-dependent zonal and meridional wind profiles, loaded from data files. Includes MC dispersions for wind scaling and direction bias.

### Data Source

Published Mars zonal wind profiles from literature (Forget et al. 1999 or equivalent). Typical profile:
- 0–150 km altitude range
- Zonal winds peaking ~80–100 m/s around 40–60 km altitude
- Meridional winds weaker (~10–20 m/s)

Tabulated as `.dat` files:
- `data/atmosphere/mars_winds.dat` — Mars wind profile
- `data/atmosphere/earth_winds.dat` — Earth wind profile

Format: `altitude_km  zonal_m_s  meridional_m_s` (same column pattern as existing atmosphere tables).

### Implementation

**winds.rs (replace stub):**
1. Load wind table at init (same pattern as atmosphere density table)
2. `get_wind(altitude, latitude) -> (v_zonal, v_meridional)` — linear interpolation on altitude
3. Optional latitude dependence: cosine scaling of zonal component (zonal winds strongest at equator)
4. Return wind velocity in the local horizontal frame (East, North components)

**Integration into physics (runner.rs / aerodynamics.rs):**
- Wind modifies relative velocity: `V_rel = V_inertial - ω×r - V_wind`
- The wind velocity is projected into the trajectory frame before subtraction
- Aero force computation receives the modified `V_rel` — no changes to force equations themselves

**Config (config.rs):**
- Add optional `wind_table` path to `[data]` TOML section
- Absent = no winds (backward compatible with all existing configs)

**Monte Carlo dispersions (dispersions.rs):**
- Wind scaling factor: uniform multiplier on the entire wind profile (e.g., 0.5x to 1.5x)
- Wind direction bias: random angle (uniform in [−δ, +δ]) that rotates the wind vector in the horizontal plane
- Parameters in `[dispersions.wind]` TOML section:
  ```toml
  [dispersions.wind]
  scale_min = 0.5
  scale_max = 1.5
  direction_bias_deg = 30.0  # max rotation, uniform in [-30, +30]
  ```

### Not Included

- No time-varying winds (§1.3 — separate future improvement)
- No MCD interface
- No vertical winds (negligible for aerocapture)
- No horizontal atmosphere variation (§1.5)

### Impact on Existing Results

- Configs without `wind_table` produce identical results (backward compatible)
- New test configs with winds enabled for regression testing

---

## 3. EKF Navigation

### Summary

Replace the current bias-only navigation model with an Extended Kalman Filter that propagates state via IMU measurements and corrects with star tracker observations (when available) and drag-derived altitude updates (during atmospheric pass). Subsumes the existing exponential density filter.

### EKF State Vector (13 states)

| State | Dim | Notes |
|-------|-----|-------|
| Position (r, lon, lat) | 3 | Spherical, same as sim state |
| Velocity (V, γ, ψ) | 3 | Relative, same as sim state |
| Accelerometer bias (x, y, z) | 3 | Body frame, slowly varying |
| Gyro bias (x, y, z) | 3 | Body frame, slowly varying |
| Density correction factor | 1 | Replaces exponential density filter |

The density correction factor replaces the current `coefro` exponential filter — one unified estimation framework.

### IMU Model (new: `gnc/navigation/imu.rs`)

**Accelerometers:**
- `a_meas = (1 + sf_err) * a_true + bias + noise`
- Bias: random walk (Gauss-Markov with long time constant)
- Noise: white Gaussian (typical: 50–100 μg for tactical-grade IMU)

**Gyroscopes:**
- `ω_meas = ω_true + bias + noise`
- Bias: random walk (typical drift: 0.01–0.1 °/hr)
- Noise: angle random walk

All noise/bias parameters are MC-dispersion-eligible.

### Prediction Step

At each navigation cycle (configurable cadence, e.g., 10 Hz):
1. Integrate nav state forward using IMU measurements (strapdown inertial navigation)
2. Propagate covariance: `P = F·P·Fᵀ + Q`
   - F: Jacobian of dynamics w.r.t. state (gravity + aero, linearized)
   - Q: process noise (IMU errors, density model uncertainty)

Replaces the current "true state + bias" model entirely.

### Star Tracker Updates (new: `gnc/navigation/star_tracker.rs`)

- Position + attitude updates (typical accuracy: ~50 m position, ~1 arcmin attitude)
- **Blackout during atmospheric pass:** no updates when `q_dyn > threshold` (configurable, e.g., 100 Pa)
- Update cadence: configurable (e.g., every 10 s when available)
- Standard Kalman update: `K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹`, state and covariance correction

### Drag-Derived Altitude Updates

During atmospheric pass (star tracker blacked out):
- Extract density from measured drag: `ρ_meas = 2·m·|a_drag| / (Cx·S·V²)`
- Compare with onboard atmosphere model: `ρ_model(h) = table_lookup(h) * density_correction_factor`
- Creates altitude-density coupled observation constraining both altitude error and density state
- Replaces the current exponential density filter with a proper estimated state + covariance

### Sensor Availability Timeline

```
Coast-in:    Star tracker ON,  IMU ON,  Drag update OFF
             → full observability, nav converges before entry

Entry:       Star tracker OFF (q_dyn > threshold), IMU ON, Drag update ON
             → coasting on IMU + drag-altitude updates only

Exit:        Star tracker ON (q_dyn < threshold), IMU ON, Drag update OFF
             → nav reconverges after atmospheric pass
```

### File Structure

| File | Purpose |
|------|---------|
| `gnc/navigation/ekf.rs` (new) | EKF core: state, covariance, predict, update |
| `gnc/navigation/imu.rs` (new) | IMU sensor model + strapdown propagation |
| `gnc/navigation/star_tracker.rs` (new) | Star tracker model + availability logic |
| `gnc/navigation/estimator.rs` (modified) | Orchestrator: calls EKF predict/update at correct cadences, replaces bias-only model |
| `data/navigation.rs` (modified) | Add IMU/star tracker params alongside existing nav error profiles |

### TOML Configuration

```toml
[navigation]
mode = "ekf"  # or "bias" for backward compatibility

[navigation.imu]
accel_bias_sigma = 1e-4          # m/s² (1-sigma initial bias)
accel_noise_sigma = 5e-4         # m/s²/√Hz
accel_scale_factor_sigma = 1e-4
gyro_bias_sigma = 5e-6           # rad/s
gyro_noise_sigma = 1e-5          # rad/s/√Hz

[navigation.star_tracker]
position_sigma = 50.0            # m (1-sigma)
attitude_sigma = 3e-4            # rad (~1 arcmin)
update_period = 10.0             # s
blackout_qdyn_threshold = 100.0  # Pa

[navigation.ekf]
process_noise_density = 0.1      # density model uncertainty
```

### Backward Compatibility

- `mode = "bias"` (or absent): preserves current behavior — all existing configs and golden tests unchanged
- `mode = "ekf"`: activates the new filter
- The density filter output (`nav_density_ratio` in trajectory data) maps to the EKF density correction factor

### Not Included

- No GPS (not available at Mars)
- No terrain-relative navigation (LIDAR, camera)
- No attitude estimation (bank angle assumed known from star tracker / IMU integration)
- No UKF — EKF first; UKF is a future upgrade if linearization errors prove problematic

---

## Implementation Sequence

1. **Integrated heat load** — low effort, immediate credibility payoff
2. **Wind model** — moderate effort, removes the most visible physics gap
3. **EKF navigation** — significant effort, largest credibility improvement

Each item is backward compatible. Existing configs and golden tests remain unaffected.

---

## Testing Strategy

### Heat Load
- Unit test: known constant heat flux over known duration → exact integral
- Regression: verify existing golden tests gain the new column with correct values
- Property test: integrated heat load is monotonically non-decreasing

### Wind Model
- Unit test: known altitude → correct interpolated wind
- Integration test: run with winds enabled vs disabled, verify corridor width changes
- Regression: new golden config with winds for reproducibility
- Property test: wind at table boundaries matches table values exactly

### EKF Navigation
- Unit tests: predict step (state propagation matches known trajectory), update step (Kalman gain converges for known measurement)
- IMU model: verify noise statistics over many samples
- Star tracker blackout: verify no updates during atmospheric pass
- Integration test: EKF nav vs bias nav on same MC scenario — EKF should show larger nav errors (more realistic) but guidance should still capture
- Regression: new golden config with EKF mode
- Property tests: covariance remains positive definite, state estimates bounded
