//! Navigation state estimator.
//!
//! Adds navigation errors to the true state to produce measured state,
//! estimates atmospheric density, and manages guidance phase transitions.

use crate::config::{PlanetConfig, SimPhase};
use crate::data::SimData;
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::gnc::navigation::ekf::{EkfConfig, EkfState};
use crate::gnc::navigation::imu::{ImuConfig, ImuState};
use crate::gnc::navigation::star_tracker::{StarTrackerConfig, StarTrackerState};
use crate::orbit::elements;
use crate::physics::atmosphere;
use nalgebra::{SMatrix, SVector};

/// Density multiplicative correction factor bounds.
///
/// The onboard density estimate is `density_gain * rho_model`, so `density_gain`
/// is a multiplicative factor. Shared by the bias-mode filter and the EKF:
///   - Bias mode clamps `density_gain` directly to `[DENSITY_FACTOR_MIN, DENSITY_FACTOR_MAX]`.
///   - EKF `state[12]` is the additive offset from 1, so its bounds are
///     `[DENSITY_FACTOR_MIN - 1.0, DENSITY_FACTOR_MAX - 1.0]` = `[-0.9, 9.0]`.
pub(crate) const DENSITY_FACTOR_MIN: f64 = 0.1;
pub(crate) const DENSITY_FACTOR_MAX: f64 = 10.0;

/// Navigation error biases (constant during a run).
#[derive(Debug, Clone, Copy, Default)]
pub struct NavigationBiases {
    pub pos: [f64; 3], // [altitude, longitude, latitude] bias
    pub vel: [f64; 3], // [velocity, flight_path, azimuth] bias
    pub drag: f64,     // drag acceleration measurement bias
}

/// Navigation filter state (persistent across steps).
#[derive(Debug, Clone, Copy)]
pub struct NavigationState {
    pub density_gain: f64,             // density estimation coefficient
    pub previous_radial_velocity: f64, // previous radial velocity (m/s)
    pub bounce_flag: i32,              // bounce indicator: 0=before, 1=after
    pub guidance_phase: i32,           // guidance phase: 1=capture, 2=exit, 3=emergency
    pub capture_time: f64,             // capture phase duration (s)
    pub exit_phase_locked: bool,       // once true, phase cannot revert from 2 to 1
}

impl Default for NavigationState {
    fn default() -> Self {
        Self::new()
    }
}

impl NavigationState {
    pub fn new() -> Self {
        Self {
            density_gain: 1.0,
            previous_radial_velocity: 0.0,
            bounce_flag: 0,
            guidance_phase: 1,
            capture_time: 0.0,
            exit_phase_locked: false,
        }
    }
}

/// Navigation output for guidance.
#[derive(Debug, Clone, Copy, Default)]
pub struct NavigationOutput {
    // Estimated state (with navigation errors added)
    pub position_estimated: [f64; 3], // [r, lon, lat]
    pub velocity_estimated: [f64; 3], // [V, gamma, psi]
    // Estimated aerodynamic quantities
    pub acceleration_estimated: [f64; 2], // [drag accel, lift accel]
    pub aero_coefficients: [f64; 2],      // [Cx, Cz]
    pub density_guidance: f64,            // estimated density for guidance
    pub density_exit: f64,                // estimated exit density
    pub dynamic_pressure_estimated: f64,  // estimated dynamic pressure
    pub energy_estimated: f64,            // total energy
    // Orbital parameter errors
    pub orbital_errors: [f64; 4], // [Δa, Δe, Δi, ΔΩ]
    // Phase management
    pub bounce_flag: i32,
    pub guidance_phase: i32,
    pub crash_flag: i32,
    pub phase_transition_flag: i32, // phase transition flag
    pub reference_velocity: f64,    // reference radial velocity
    pub capture_time: f64,          // capture duration
    // Thermal state (for guidance limiter and NN inputs)
    pub heat_flux_fraction: f64, // current_heat_flux / max_heat_flux (0.0 if no limit)
    pub heat_load_fraction: f64, // cumulative_heat_load / max_heat_load (0.0 if no limit)
}

/// Run one navigation step.
#[allow(clippy::too_many_arguments)]
pub fn navigate(
    position_true: &[f64; 3], // true position [r, lon, lat]
    velocity_true: &[f64; 3], // true velocity [V, gamma, psi]
    aoa_commanded: f64,       // commanded AoA
    sim_time: f64,            // current time
    biases: &NavigationBiases,
    nav_state: &mut NavigationState,
    data: &SimData,
    planet: &PlanetConfig,
    run_density_bias: f64,
    run_density_perturbation: f64,
    run_cx_bias: f64,
    run_cz_bias: f64,
    run_mass_bias: f64,
    run_incidence_bias: f64,
    run_ref_area_bias: f64,
    run_filter_gain_bias: f64,
) -> NavigationOutput {
    let mut out = NavigationOutput {
        phase_transition_flag: 0,
        crash_flag: 0,
        ..Default::default()
    };

    // Add navigation errors (bias constants)
    out.position_estimated[0] = position_true[0] + biases.pos[0];
    out.position_estimated[1] = position_true[1] + biases.pos[1];
    out.position_estimated[2] = position_true[2] + biases.pos[2];
    out.velocity_estimated[0] = velocity_true[0] + biases.vel[0];
    out.velocity_estimated[1] = velocity_true[1] + biases.vel[1];
    out.velocity_estimated[2] = velocity_true[2] + biases.vel[2];

    let velocity_relative = out.velocity_estimated[0];

    // Compute true drag acceleration (truth model)
    let (alt_true, _) =
        geodetic_from_spherical(position_true[0], position_true[1], position_true[2], planet);
    let rho_true = atmosphere::density(
        &data.atmosphere,
        alt_true,
        run_density_bias,
        run_density_perturbation,
    );
    let aoa_true = aoa_commanded + run_incidence_bias;
    let cx_true = data.aero.interpolate_cx(aoa_true) * (1.0 + run_cx_bias);
    let cz_true = data.aero.interpolate_cz(aoa_true) * (1.0 + run_cz_bias);
    let mass_true = data.capsule.mass * (1.0 + run_mass_bias);
    let ref_area_true = data.capsule.reference_area * (1.0 + run_ref_area_bias);
    let aero_factor_true =
        rho_true * ref_area_true * velocity_true[0] * velocity_true[0] / (2.0 * mass_true);
    let accel_body_x_true =
        aero_factor_true * (cx_true * aoa_true.cos() + cz_true * aoa_true.sin());
    let accel_measured = accel_body_x_true + biases.drag;

    // Compute estimated aero coefficients (onboard model)
    let (alt_est, _) = geodetic_from_spherical(
        out.position_estimated[0],
        out.position_estimated[1],
        out.position_estimated[2],
        planet,
    );
    let cx_est = data.aero.interpolate_cx(aoa_commanded);
    let cz_est = data.aero.interpolate_cz(aoa_commanded);
    out.aero_coefficients[0] = cx_est;
    out.aero_coefficients[1] = cz_est;

    // Density estimation via inverse dynamics (lift-corrected)
    // a_body_x = (rho*S*V^2 / 2m) * (Cx*cos(alpha) + Cz*sin(alpha))
    // => rho = 2*m*|a| / (S*V^2 * (Cx*cos(alpha) + Cz*sin(alpha)))
    // Valid only for a POSITIVE denominator; a lift-dominated (negative) denom
    // yields a non-physical negative density, so reject it (hold at 0).
    let aoa_est = aoa_commanded;
    let denom = cx_est * aoa_est.cos() + cz_est * aoa_est.sin();
    let density_estimated = if denom > 1e-10 && velocity_relative.abs() > 1e-10 {
        2.0 * accel_measured.abs() * data.capsule.mass
            / (denom * data.capsule.reference_area * velocity_relative * velocity_relative)
    } else {
        0.0
    };

    // Model atmosphere density at estimated altitude
    let rho_model = data
        .atmosphere_onboard
        .density_at(alt_est, &data.atmosphere);

    // Exponential filter for density correction
    // density_gain = (1-λ)*density_gain + λ*(density_estimated/rho_model)
    // Skip guard-tripped steps (density_estimated == 0) so a rejected estimate
    // does not drag the gain toward the floor — matches the EKF trigger.
    let lambda = (data.guidance.density_filter_gain + run_filter_gain_bias).clamp(0.01, 0.99);
    if rho_model.abs() > 1e-30 && density_estimated > 0.0 {
        let raw_gain =
            (1.0 - lambda) * nav_state.density_gain + lambda * (density_estimated / rho_model);

        // Rate-of-change limiting
        let max_delta = data.guidance.density_gain_max_delta;
        let delta = (raw_gain - nav_state.density_gain).clamp(-max_delta, max_delta);
        nav_state.density_gain += delta;
    }

    // Gain saturation (hardcoded safety bounds, matches EKF [0.1, 10.0]) — applied
    // unconditionally every tick as a safety net, even when the filter is skipped.
    nav_state.density_gain = nav_state
        .density_gain
        .clamp(DENSITY_FACTOR_MIN, DENSITY_FACTOR_MAX);

    if alt_est > 100e3 {
        nav_state.density_gain = 1.0;
    }

    out.density_guidance = nav_state.density_gain * rho_model;

    // Estimated drag and lift accelerations
    let mass_est = data.capsule.mass;
    let aero_factor = out.density_guidance * data.capsule.reference_area / (2.0 * mass_est);
    out.acceleration_estimated[0] = aero_factor * cx_est * velocity_relative * velocity_relative;
    out.acceleration_estimated[1] = aero_factor * cz_est * velocity_relative * velocity_relative;
    out.dynamic_pressure_estimated =
        0.5 * out.density_guidance * velocity_relative * velocity_relative;

    // Exit density estimation
    let alt_exit = data.guidance.exit_altitude_threshold;
    let rho_exit_model = data
        .atmosphere_onboard
        .density_at(alt_exit, &data.atmosphere);
    out.density_exit = nav_state.density_gain * rho_exit_model;

    compute_energy_and_orbital_errors(&mut out, planet, data);

    let velocity_radial = update_bounce_and_phase(
        &mut out,
        nav_state,
        velocity_relative,
        sim_time,
        data.guidance.exit_velocity_threshold,
    );

    finalize_crash_phase_and_output(
        &mut out,
        nav_state,
        velocity_radial,
        data.periods.navigation,
        data,
    );

    out
}

/// Bounce detection + capture->exit phase management (with the
/// `exit_phase_locked` irreversibility guard). Shared verbatim by the bias
/// and EKF navigation paths. Returns the estimated radial velocity.
fn update_bounce_and_phase(
    out: &mut NavigationOutput,
    ns: &mut NavigationState,
    velocity_relative: f64,
    sim_time: f64,
    exit_velocity_threshold: f64,
) -> f64 {
    if ns.bounce_flag == 0 && out.velocity_estimated[1].sin() > 0.0 {
        ns.bounce_flag = 1;
    }

    let velocity_radial = velocity_relative * out.velocity_estimated[1].sin();

    // Phase management (once exit phase is entered, it cannot revert to capture)
    if ns.bounce_flag == 0 {
        ns.guidance_phase = 1;
    } else if !ns.exit_phase_locked {
        if velocity_relative >= exit_velocity_threshold && velocity_radial < 0.0 {
            ns.guidance_phase = 1;
        }
        if velocity_relative <= exit_velocity_threshold && ns.guidance_phase == 1 {
            ns.guidance_phase = 2;
            ns.exit_phase_locked = true;
            ns.capture_time = sim_time;
            out.phase_transition_flag = 1;
            out.reference_velocity = velocity_radial;
        }
    }
    velocity_radial
}

/// Compute total energy and orbital element errors into `out`.
///
/// Shared by both the bias and EKF navigation paths. Reads `out.position_estimated`
/// and `out.velocity_estimated`; writes `out.energy_estimated` and `out.orbital_errors`.
fn compute_energy_and_orbital_errors(
    out: &mut NavigationOutput,
    planet: &PlanetConfig,
    data: &SimData,
) {
    out.energy_estimated = total_energy(
        out.position_estimated[0],
        out.position_estimated[1],
        out.position_estimated[2],
        out.velocity_estimated[0],
        out.velocity_estimated[1],
        out.velocity_estimated[2],
        planet,
    );

    let orbit = elements::from_spherical(
        out.position_estimated[0],
        out.position_estimated[1],
        out.position_estimated[2],
        out.velocity_estimated[0],
        out.velocity_estimated[1],
        out.velocity_estimated[2],
        planet,
    );
    out.orbital_errors[0] = orbit.semi_major_axis - data.target_orbit.semi_major_axis;
    out.orbital_errors[1] = orbit.eccentricity - data.target_orbit.eccentricity;
    out.orbital_errors[2] = orbit.inclination - data.target_orbit.inclination;
    out.orbital_errors[3] = orbit.raan - data.target_orbit.raan;
}

/// Crash detection, SimPhase gating, capture-time accumulation, and final output population.
///
/// Shared by both the bias and EKF navigation paths. Both callers now share the same
/// guarded phase-management block (with `exit_phase_locked` guard); only the post-phase
/// tail is extracted here.
fn finalize_crash_phase_and_output(
    out: &mut NavigationOutput,
    ns: &mut NavigationState,
    velocity_radial: f64,
    nav_dt: f64,
    data: &SimData,
) {
    // Crash detection after bounce
    if ns.bounce_flag >= 1 {
        let delta_radial_velocity = velocity_radial - ns.previous_radial_velocity;
        ns.previous_radial_velocity = velocity_radial;
        if delta_radial_velocity < 0.0 {
            out.crash_flag = 1;
        }
    }

    if out.crash_flag == 1 {
        ns.guidance_phase = 3;
    }

    // Apply SimPhase gating
    match data.sim_phase {
        SimPhase::CaptureOnly => {
            ns.guidance_phase = 1;
        }
        SimPhase::ExitOnly => {
            ns.guidance_phase = 2;
        }
        SimPhase::Full | SimPhase::Preprogrammed => {
            // Phase logic above already computed the correct phase
        }
    }

    if ns.guidance_phase == 1 {
        ns.capture_time += nav_dt;
    }

    out.bounce_flag = ns.bounce_flag;
    out.guidance_phase = ns.guidance_phase;
    out.capture_time = ns.capture_time;
}

/// Full navigation system state: bias (legacy) or EKF.
pub enum NavigationFilter {
    Bias(NavigationState),
    Ekf {
        ekf: Box<EkfState>,
        imu: Box<ImuState>,
        star_tracker: Box<StarTrackerState>,
        st_config: StarTrackerConfig,
        imu_config: ImuConfig,
        ekf_config: EkfConfig,
        legacy: NavigationState, // still need bounce/phase tracking
    },
}

impl NavigationFilter {
    pub fn new_bias() -> Self {
        NavigationFilter::Bias(NavigationState::new())
    }

    pub fn new_ekf(
        imu_config: ImuConfig,
        st_config: StarTrackerConfig,
        ekf_config: EkfConfig,
        seed: u64,
    ) -> Self {
        NavigationFilter::Ekf {
            ekf: Box::new(EkfState::new(&ekf_config)),
            imu: Box::new(ImuState::new(&imu_config, seed)),
            star_tracker: Box::new(StarTrackerState::new(&st_config, seed.wrapping_add(1000))),
            st_config,
            imu_config,
            ekf_config,
            legacy: NavigationState::new(),
        }
    }

    /// Returns the density gain for photo output.
    pub fn density_gain(&self) -> f64 {
        match self {
            NavigationFilter::Bias(nav_state) => nav_state.density_gain,
            NavigationFilter::Ekf { ekf, .. } => ekf.density_correction(),
        }
    }
}

/// Build sensor configs from TOML navigation config.
pub fn build_ekf_configs(
    toml_nav: &crate::config::TomlNavigation,
) -> (ImuConfig, StarTrackerConfig, EkfConfig) {
    let imu_config = if let Some(ref imu) = toml_nav.imu {
        ImuConfig {
            accel_bias_sigma: imu.accel_bias_sigma,
            accel_noise_sigma: imu.accel_noise_sigma,
            accel_scale_factor_sigma: imu.accel_scale_factor_sigma,
            gyro_bias_sigma: imu.gyro_bias_sigma,
            gyro_noise_sigma: imu.gyro_noise_sigma,
        }
    } else {
        ImuConfig::default()
    };

    let st_config = if let Some(ref st) = toml_nav.star_tracker {
        StarTrackerConfig {
            position_sigma: st.position_sigma,
            attitude_sigma: st.attitude_sigma,
            update_period: st.update_period,
            blackout_qdyn_threshold: st.blackout_qdyn_threshold,
        }
    } else {
        StarTrackerConfig::default()
    };

    let ekf_config = if let Some(ref ekf_toml) = toml_nav.ekf {
        EkfConfig {
            q_density: ekf_toml.process_noise_density,
            ..EkfConfig::default()
        }
    } else {
        EkfConfig::default()
    };

    (imu_config, st_config, ekf_config)
}

/// Run one EKF navigation step.
///
/// Uses the legacy bias model for state corruption, then layers EKF error-state
/// corrections on top. The EKF predict/update cycle runs alongside, providing
/// improved density estimation and (when star tracker is available) position correction.
#[allow(clippy::too_many_arguments)]
pub fn navigate_ekf(
    position_true: &[f64; 3],
    velocity_true: &[f64; 3],
    aoa_commanded: f64,
    sim_time: f64,
    nav_dt: f64,
    biases: &NavigationBiases,
    legacy: &mut NavigationState,
    ekf: &mut EkfState,
    imu: &mut ImuState,
    star_tracker: &mut StarTrackerState,
    st_config: &StarTrackerConfig,
    ekf_config: &EkfConfig,
    data: &SimData,
    planet: &PlanetConfig,
    run_density_bias: f64,
    run_density_perturbation: f64,
    run_cx_bias: f64,
    run_cz_bias: f64,
    run_mass_bias: f64,
    run_incidence_bias: f64,
    run_ref_area_bias: f64,
) -> NavigationOutput {
    let mut out = NavigationOutput {
        phase_transition_flag: 0,
        crash_flag: 0,
        ..Default::default()
    };

    // ── Step 1: Biased estimated state (same as legacy) ──
    let pos_biased = [
        position_true[0] + biases.pos[0],
        position_true[1] + biases.pos[1],
        position_true[2] + biases.pos[2],
    ];
    let vel_biased = [
        velocity_true[0] + biases.vel[0],
        velocity_true[1] + biases.vel[1],
        velocity_true[2] + biases.vel[2],
    ];

    // ── Step 2: Compute true aero acceleration for IMU ──
    let (alt_true, _) =
        geodetic_from_spherical(position_true[0], position_true[1], position_true[2], planet);
    let rho_true = atmosphere::density(
        &data.atmosphere,
        alt_true,
        run_density_bias,
        run_density_perturbation,
    );
    let aoa_true = aoa_commanded + run_incidence_bias;
    let cx_true = data.aero.interpolate_cx(aoa_true) * (1.0 + run_cx_bias);
    let cz_true = data.aero.interpolate_cz(aoa_true) * (1.0 + run_cz_bias);
    let mass_true = data.capsule.mass * (1.0 + run_mass_bias);
    let ref_area_true = data.capsule.reference_area * (1.0 + run_ref_area_bias);
    let aero_factor_true =
        rho_true * ref_area_true * velocity_true[0] * velocity_true[0] / (2.0 * mass_true);
    let accel_body_x_true =
        aero_factor_true * (cx_true * aoa_true.cos() + cz_true * aoa_true.sin());

    // Body-frame x-axis acceleration includes both drag and lift projections
    let true_accel = [accel_body_x_true, 0.0, 0.0];
    let true_gyro = [0.0, 0.0, 0.0]; // simplified: no true rotation rate available here

    // ── Step 3: IMU measurements ──
    let accel_meas = imu.measure_accel(&true_accel);
    let gyro_meas = imu.measure_gyro(&true_gyro);

    // ── Step 4: EKF predict ──
    ekf.predict(nav_dt, &accel_meas, &gyro_meas, ekf_config);

    // ── Step 5: Apply EKF error-state corrections to biased estimate ──
    out.position_estimated[0] = pos_biased[0] + ekf.state[0];
    out.position_estimated[1] = pos_biased[1] + ekf.state[1];
    out.position_estimated[2] = pos_biased[2] + ekf.state[2];
    out.velocity_estimated[0] = vel_biased[0] + ekf.state[3];
    out.velocity_estimated[1] = vel_biased[1] + ekf.state[4];
    out.velocity_estimated[2] = vel_biased[2] + ekf.state[5];

    let velocity_relative = out.velocity_estimated[0];

    // ── Step 6: Onboard aero coefficients + density estimation ──
    let (alt_est, _) = geodetic_from_spherical(
        out.position_estimated[0],
        out.position_estimated[1],
        out.position_estimated[2],
        planet,
    );
    let cx_est = data.aero.interpolate_cx(aoa_commanded);
    let cz_est = data.aero.interpolate_cz(aoa_commanded);
    out.aero_coefficients[0] = cx_est;
    out.aero_coefficients[1] = cz_est;

    // Density estimation via inverse dynamics (lift-corrected).
    // Valid only for a POSITIVE denominator; a lift-dominated (negative) denom
    // yields a non-physical negative density, so reject it (hold at 0).
    let accel_measured_ekf = accel_meas[0];
    let aoa_est = aoa_commanded;
    let denom = cx_est * aoa_est.cos() + cz_est * aoa_est.sin();
    let density_estimated = if denom > 1e-10 && velocity_relative.abs() > 1e-10 {
        2.0 * accel_measured_ekf.abs() * data.capsule.mass
            / (denom * data.capsule.reference_area * velocity_relative * velocity_relative)
    } else {
        0.0
    };

    // Model density at estimated altitude
    let rho_model = data
        .atmosphere_onboard
        .density_at(alt_est, &data.atmosphere);

    // ── Step 7: Star tracker update (if available) ──
    // Compute dynamic pressure for blackout check
    let pdyn_est = 0.5 * rho_model * velocity_relative * velocity_relative;
    if let Some(meas_pos) = star_tracker.measure(position_true, pdyn_est, sim_time, st_config) {
        // Innovation: measured position - estimated position
        let innovation = SVector::<f64, 3>::new(
            meas_pos[0] - out.position_estimated[0],
            meas_pos[1] - out.position_estimated[1],
            meas_pos[2] - out.position_estimated[2],
        );
        // Measurement noise covariance
        let r = out.position_estimated[0]; // radial distance for angular noise
        let r_meas = SMatrix::<f64, 3, 3>::from_diagonal(&SVector::<f64, 3>::new(
            st_config.position_sigma.powi(2),
            (st_config.position_sigma / r).powi(2),
            (st_config.position_sigma / r).powi(2),
        ));
        ekf.update_position(&innovation, &r_meas);

        // Re-apply corrected error states after position update
        out.position_estimated[0] = pos_biased[0] + ekf.state[0];
        out.position_estimated[1] = pos_biased[1] + ekf.state[1];
        out.position_estimated[2] = pos_biased[2] + ekf.state[2];
        out.velocity_estimated[0] = vel_biased[0] + ekf.state[3];
        out.velocity_estimated[1] = vel_biased[1] + ekf.state[4];
        out.velocity_estimated[2] = vel_biased[2] + ekf.state[5];
    }

    // ── Step 8: EKF density update ──
    if rho_model.abs() > 1e-30 && density_estimated > 0.0 {
        let density_ratio = density_estimated / rho_model;
        let innovation = density_ratio - ekf.density_correction();
        let r_density = 0.1_f64; // measurement noise for density ratio
        ekf.update_density(innovation, r_density);
    }

    // Use EKF density correction instead of legacy exponential filter
    out.density_guidance = ekf.density_correction() * rho_model;
    if alt_est > 100e3 {
        // High altitude: reset to model density (same logic as bias mode)
        out.density_guidance = rho_model;
    }

    // ── Step 9: Estimated drag, lift, dynamic pressure ──
    let mass_est = data.capsule.mass;
    let aero_factor = out.density_guidance * data.capsule.reference_area / (2.0 * mass_est);
    out.acceleration_estimated[0] = aero_factor * cx_est * velocity_relative * velocity_relative;
    out.acceleration_estimated[1] = aero_factor * cz_est * velocity_relative * velocity_relative;
    out.dynamic_pressure_estimated =
        0.5 * out.density_guidance * velocity_relative * velocity_relative;

    // Exit density estimation
    let alt_exit = data.guidance.exit_altitude_threshold;
    let rho_exit_model = data
        .atmosphere_onboard
        .density_at(alt_exit, &data.atmosphere);
    out.density_exit = ekf.density_correction() * rho_exit_model;

    // ── Step 10: Energy + orbital elements (same as legacy) ──
    compute_energy_and_orbital_errors(&mut out, planet, data);

    // ── Step 11: Bounce/phase management (delegated to legacy) ──
    let velocity_radial = update_bounce_and_phase(
        &mut out,
        legacy,
        velocity_relative,
        sim_time,
        data.guidance.exit_velocity_threshold,
    );

    finalize_crash_phase_and_output(&mut out, legacy, velocity_radial, nav_dt, data);

    out
}

#[cfg(test)]
#[path = "estimator_tests.rs"]
mod tests;
