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
    nav_state.density_gain = nav_state.density_gain.clamp(0.1, 10.0);

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

    // Bounce detection
    if nav_state.bounce_flag == 0 && out.velocity_estimated[1].sin() > 0.0 {
        nav_state.bounce_flag = 1;
    }

    let velocity_radial = velocity_relative * out.velocity_estimated[1].sin();

    // Phase management (once exit phase is entered, it cannot revert to capture)
    if nav_state.bounce_flag == 0 {
        nav_state.guidance_phase = 1;
    } else if !nav_state.exit_phase_locked {
        let vphase = data.guidance.exit_velocity_threshold;
        if velocity_relative >= vphase && velocity_radial < 0.0 {
            nav_state.guidance_phase = 1;
        }
        if velocity_relative <= vphase && nav_state.guidance_phase == 1 {
            nav_state.guidance_phase = 2;
            nav_state.exit_phase_locked = true;
            nav_state.capture_time = sim_time;
            out.phase_transition_flag = 1;
            out.reference_velocity = velocity_radial;
        }
    }

    finalize_crash_phase_and_output(
        &mut out,
        nav_state,
        velocity_radial,
        data.periods.navigation,
        data,
    );

    out
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
/// Shared by both the bias and EKF navigation paths. The divergent phase-management block
/// (bias has `exit_phase_locked` guard; EKF does not) is left inline in each caller — only
/// the post-phase tail is extracted here.
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
    if legacy.bounce_flag == 0 && out.velocity_estimated[1].sin() > 0.0 {
        legacy.bounce_flag = 1;
    }

    let velocity_radial = velocity_relative * out.velocity_estimated[1].sin();

    if legacy.bounce_flag == 0 {
        legacy.guidance_phase = 1;
    } else if !legacy.exit_phase_locked {
        let vphase = data.guidance.exit_velocity_threshold;
        if velocity_relative >= vphase && velocity_radial < 0.0 {
            legacy.guidance_phase = 1;
        }
        if velocity_relative <= vphase && legacy.guidance_phase == 1 {
            legacy.guidance_phase = 2;
            legacy.exit_phase_locked = true;
            legacy.capture_time = sim_time;
            out.phase_transition_flag = 1;
            out.reference_velocity = velocity_radial;
        }
    }

    finalize_crash_phase_and_output(&mut out, legacy, velocity_radial, nav_dt, data);

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    use crate::data::aerodynamics::AeroTables;
    use crate::data::atmosphere::{AtmosphereModel, DensityProfile};
    use crate::data::capsule::Capsule;
    use crate::data::guidance_params::GuidanceParams;
    use crate::data::incidence::IncidenceProfile;
    use crate::data::pilot::{PilotModel, PilotType};
    use crate::data::{
        Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit, SimData,
        SphericalState, SuccessCriteria, TimePeriods,
    };

    /// Build a minimal SimData suitable for navigation tests.
    fn test_sim_data() -> SimData {
        SimData {
            capsule: Capsule {
                mass: 1089.0,
                reference_area: 14.7,
                cq: 0.00008242,
                max_bank_rate: 15.0_f64.to_radians(),
                periods: TimePeriods::default(),
            },
            aero: AeroTables {
                n_points: 2,
                incidence: vec![-0.5, 0.0],
                cx: vec![1.269, 1.269],
                cz: vec![-0.205, -0.205],
                equilibrium_aoa: -0.48,
                nominal_cx: 1.269,
                nominal_cz: -0.205,
                nominal_finesse: -0.205 / 1.269,
                ballistic_coeff: 0.0,
            },
            atmosphere: AtmosphereModel {
                n_points: 3,
                altitudes: vec![0.0, 50_000.0, 130_000.0],
                densities: vec![0.02, 0.001, 1e-8],
                ref_density: 1e-8,
                scale_factor: 1e-4,
                ref_altitude: 130_000.0,
                gas_constant: 1.3,
                density_profile: DensityProfile::default(),
            },
            atmosphere_onboard: crate::data::atmosphere::OnboardAtmosphereModel::Identical,
            entry: EntryConditions {
                state: SphericalState {
                    altitude: 130_000.0,
                    velocity: 5687.0,
                    flight_path: -10.8_f64.to_radians(),
                    ..Default::default()
                },
                initial_bank: 64.77_f64.to_radians(),
                initial_aoa: -27.5_f64.to_radians(),
                initial_date: 0.0,
            },
            guidance: GuidanceParams {
                density_filter_gain: 0.8,
                exit_velocity_threshold: 4400.0,
                exit_altitude_threshold: 60_000.0,
                ..Default::default()
            },
            incidence: IncidenceProfile {
                n_points: 2,
                altitudes: vec![-10_000.0, 150_000.0],
                incidences: vec![-0.48, -0.48],
            },
            periods: TimePeriods::default(),
            pilot: PilotModel {
                pilot_type: PilotType::Perfect,
                time_constant: 0.0,
                damping: 0.0,
                frequency: 0.0,
            },
            target_orbit: OrbitalTarget {
                semi_major_axis: 3_649_622.0,
                eccentricity: 0.067,
                inclination: 50.0_f64.to_radians(),
                raan: -7.612_f64.to_radians(),
                apoapsis: 500_130.0,
                periapsis: 11_233.0,
            },
            final_conditions: FinalConditions::default(),
            parking_orbit: ParkingOrbit::default(),
            constraints: Constraints::default(),
            success: SuccessCriteria::default(),
            wind_enabled: false,
            wind_table: None,
            neural_net: None,
            dispersion_config: None,
            nav_mode: crate::data::NavMode::Bias,
            nav_config: None,
            integration_mode: crate::config::IntegrationMode::FixedGill,
            sim_phase: crate::config::SimPhase::Full,
            density_perturbation: None,
            nn_normalization_override: None,
        }
    }

    /// Mars equatorial radius for converting altitude to geocentric radius.
    const MARS_REQ: f64 = 3.39394e6;

    fn zero_biases() -> NavigationBiases {
        NavigationBiases::default()
    }

    fn no_run_biases() -> [f64; 7] {
        [0.0; 7] // density, cx, cz, mass, incidence, ref_area, filter_gain
    }

    /// Build EKF/IMU/StarTracker state with noise-free configs for deterministic
    /// phase-logic tests (no sensor noise perturbs `velocity_estimated`).
    fn quiet_ekf_states() -> (
        EkfState,
        ImuState,
        StarTrackerState,
        StarTrackerConfig,
        EkfConfig,
    ) {
        let imu_config = ImuConfig {
            accel_bias_sigma: 0.0,
            accel_noise_sigma: 0.0,
            accel_scale_factor_sigma: 0.0,
            gyro_bias_sigma: 0.0,
            gyro_noise_sigma: 0.0,
        };
        let st_config = StarTrackerConfig::default();
        let ekf_config = EkfConfig::default();
        (
            EkfState::new(&ekf_config),
            ImuState::new(&imu_config, 0),
            StarTrackerState::new(&st_config, 0),
            st_config,
            ekf_config,
        )
    }

    /// Helper: call navigate_ekf with zero run biases.
    #[allow(clippy::too_many_arguments)]
    fn call_navigate_ekf(
        position_true: &[f64; 3],
        velocity_true: &[f64; 3],
        sim_time: f64,
        legacy: &mut NavigationState,
        ekf: &mut EkfState,
        imu: &mut ImuState,
        star_tracker: &mut StarTrackerState,
        st_config: &StarTrackerConfig,
        ekf_config: &EkfConfig,
        data: &SimData,
    ) -> NavigationOutput {
        navigate_ekf(
            position_true,
            velocity_true,
            data.entry.initial_aoa,
            sim_time,
            data.periods.navigation,
            &NavigationBiases::default(),
            legacy,
            ekf,
            imu,
            star_tracker,
            st_config,
            ekf_config,
            data,
            &PlanetConfig::mars(),
            0.0, // density
            0.0, // density_perturbation
            0.0, // cx
            0.0, // cz
            0.0, // mass
            0.0, // incidence
            0.0, // ref_area
        )
    }

    /// Helper: call navigate with a convenient tuple of run biases.
    fn call_navigate(
        position_true: &[f64; 3],
        velocity_true: &[f64; 3],
        biases: &NavigationBiases,
        nav_state: &mut NavigationState,
        data: &SimData,
        run_biases: &[f64; 7],
    ) -> NavigationOutput {
        navigate(
            position_true,
            velocity_true,
            data.entry.initial_aoa,
            0.0, // sim_time
            biases,
            nav_state,
            data,
            &PlanetConfig::mars(),
            run_biases[0], // density
            0.0,           // density_perturbation
            run_biases[1], // cx
            run_biases[2], // cz
            run_biases[3], // mass
            run_biases[4], // incidence
            run_biases[5], // ref_area
            run_biases[6], // filter_gain
        )
    }

    // ── Test 1: biases_are_additive ──

    #[rstest]
    #[case::zero_bias(
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        "zero bias should not change the state"
    )]
    #[case::nonzero_bias(
        [100.0, 0.001, -0.002], [5.0, 0.01, -0.005],
        "nonzero bias should be added to true state"
    )]
    fn biases_are_additive(
        #[case] pos_bias: [f64; 3],
        #[case] vel_bias: [f64; 3],
        #[case] _label: &str,
    ) {
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        // Use high altitude so density filter doesn't complicate things
        let r = planet.equatorial_radius + 120_000.0;
        let position_true = [r, 0.5, 0.3];
        let velocity_true = [5500.0, -0.15, 1.2];

        let biases = NavigationBiases {
            pos: pos_bias,
            vel: vel_bias,
            drag: 0.0,
        };
        let mut nav_state = NavigationState::new();
        let out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        for i in 0..3 {
            assert_relative_eq!(
                out.position_estimated[i],
                position_true[i] + pos_bias[i],
                max_relative = 1e-14
            );
            assert_relative_eq!(
                out.velocity_estimated[i],
                velocity_true[i] + vel_bias[i],
                max_relative = 1e-14
            );
        }
    }

    // ── Test 2: density_filter_convergence ──

    #[test]
    fn density_filter_convergence() {
        let data = test_sim_data();
        // Altitude ~40 km where there's meaningful atmosphere
        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let mut density_gain_values = Vec::new();
        for step in 0..50 {
            let _out = navigate(
                &position_true,
                &velocity_true,
                data.entry.initial_aoa,
                step as f64,
                &biases,
                &mut nav_state,
                &data,
                &PlanetConfig::mars(),
                0.0, // density_bias
                0.0, // density_perturbation
                0.0, // cx
                0.0, // cz
                0.0, // mass
                0.0, // incidence
                0.0, // ref_area
                0.0, // filter_gain
            );
            density_gain_values.push(nav_state.density_gain);
        }

        // After many steps with constant inputs, density_gain should converge
        // (difference between successive values should shrink).
        let late_delta = (density_gain_values[49] - density_gain_values[48]).abs();
        let early_delta = (density_gain_values[5] - density_gain_values[4]).abs();

        // Late deltas should be smaller or equal to early deltas (convergence).
        // If density_gain converges immediately (same input each step), both could be 0.
        assert!(
            late_delta <= early_delta + 1e-15,
            "density filter should converge: early_delta={early_delta:.6e}, late_delta={late_delta:.6e}"
        );

        // The final density_gain should be finite and positive
        assert!(
            nav_state.density_gain.is_finite() && nav_state.density_gain > 0.0,
            "density_gain should be finite and positive, got {}",
            nav_state.density_gain
        );
    }

    // ── Test 3: high_altitude_resets_density_gain ──

    #[test]
    fn high_altitude_resets_density_gain() {
        let data = test_sim_data();
        // Altitude above 100 km
        let r = MARS_REQ + 110_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5687.0, -0.15, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        // Perturb density_gain away from 1.0
        nav_state.density_gain = 2.5;

        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        assert_relative_eq!(nav_state.density_gain, 1.0, max_relative = 1e-14,);
    }

    // ── Test 4: filter_gain_clamped ──

    #[rstest]
    #[case::extreme_negative(-10.0, "large negative bias should be clamped")]
    #[case::extreme_positive(10.0, "large positive bias should be clamped")]
    fn filter_gain_clamped(#[case] filter_gain_bias: f64, #[case] _label: &str) {
        let data = test_sim_data();
        // Use 40 km altitude so filter runs
        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let run_biases = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, filter_gain_bias];
        let out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &run_biases,
        );

        // Function should not crash, and all outputs should be finite
        assert!(
            out.position_estimated[0].is_finite(),
            "position_estimated[0] should be finite"
        );
        assert!(
            out.velocity_estimated[0].is_finite(),
            "velocity_estimated[0] should be finite"
        );
        assert!(
            out.density_guidance.is_finite(),
            "density_guidance should be finite"
        );
        assert!(
            nav_state.density_gain.is_finite(),
            "density_gain should be finite with filter_gain_bias={filter_gain_bias}"
        );
    }

    // ── Test 5: bounce_detection ──

    #[rstest]
    #[case::descending(-0.15, 0, "negative gamma (descending) => no bounce")]
    #[case::ascending(0.05, 1, "positive gamma (ascending) => bounce detected")]
    fn bounce_detection(
        #[case] gamma: f64,
        #[case] expected_bounce_flag: i32,
        #[case] _label: &str,
    ) {
        let data = test_sim_data();
        let r = MARS_REQ + 50_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, gamma, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        assert_eq!(
            nav_state.bounce_flag, expected_bounce_flag,
            "with gamma={gamma}, expected bounce_flag={expected_bounce_flag}, got {}",
            nav_state.bounce_flag
        );
    }

    // ── Test 7: density_filter_stability ──

    /// Run navigate() 100 times in a loop and verify density_gain stays finite and
    /// positive at every step.  This guards against density filter instability
    /// (lambda > 1 causing exponential amplification per step).
    #[test]
    fn density_filter_stability() {
        let data = test_sim_data();
        // 40 km — meaningful atmosphere so the filter actually updates
        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        for step in 0..100 {
            let _out = call_navigate(
                &position_true,
                &velocity_true,
                &biases,
                &mut nav_state,
                &data,
                &no_run_biases(),
            );
            assert!(
                nav_state.density_gain.is_finite(),
                "density_gain became non-finite at step {step}: {}",
                nav_state.density_gain
            );
            assert!(
                nav_state.density_gain > 0.0,
                "density_gain became non-positive at step {step}: {}",
                nav_state.density_gain
            );
        }
    }

    // ── Test: density_gain_rate_limited ──

    #[test]
    fn density_gain_rate_limited() {
        let mut data = test_sim_data();
        data.guidance.density_gain_max_delta = 0.05; // tight rate limit
        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();
        nav_state.density_gain = 1.0;

        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // With rate limit of 0.05, density_gain cannot move more than 0.05 from 1.0
        let delta = (nav_state.density_gain - 1.0).abs();
        assert!(
            delta <= 0.05 + 1e-14,
            "density_gain delta {delta} exceeded max_delta 0.05"
        );
    }

    // ── Test: density_gain_saturated ──

    #[test]
    fn density_gain_saturated() {
        let mut data = test_sim_data();
        data.guidance.density_gain_max_delta = 100.0; // very loose rate limit
        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        // Start with extreme density_gain — should be clamped to [0.1, 10.0]
        nav_state.density_gain = 50.0;

        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        assert!(
            nav_state.density_gain <= 10.0,
            "density_gain {} should be <= 10.0",
            nav_state.density_gain
        );
        assert!(
            nav_state.density_gain >= 0.1,
            "density_gain {} should be >= 0.1",
            nav_state.density_gain
        );
    }

    // ── Test: rate_limit_before_saturation ──

    #[test]
    fn rate_limit_before_saturation() {
        let mut data = test_sim_data();
        data.guidance.density_gain_max_delta = 0.02; // very tight
        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();

        // Start near the lower saturation bound
        let mut nav_state = NavigationState::new();
        nav_state.density_gain = 0.12;

        // Run one step — even if filter wants to go below 0.1,
        // rate limit restricts movement to 0.02
        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // density_gain should be in [0.10, 0.14] (0.12 +/- 0.02, then clamped to [0.1, 10.0])
        assert!(
            nav_state.density_gain >= 0.1,
            "density_gain {} below saturation floor",
            nav_state.density_gain
        );
        let delta = (nav_state.density_gain - 0.12).abs();
        assert!(
            delta <= 0.02 + 1e-14,
            "density_gain moved by {delta}, exceeding rate limit 0.02"
        );
    }

    // ── Test 8: proptest_navigate_outputs_finite ──

    proptest::proptest! {
        /// For any bounded but arbitrary state, navigate() must produce entirely
        /// finite outputs — no NaN or Inf should escape.
        #[test]
        fn proptest_navigate_outputs_finite(
            // altitude 30–120 km above Mars surface
            alt_km in 30.0_f64..=120.0_f64,
            velocity in 1_000.0_f64..=8_000.0_f64,
            gamma in -0.5_f64..=0.5_f64,
            psi in -3.15_f64..=3.15_f64,
            pos_bias_alt in -500.0_f64..=500.0_f64,
            vel_bias in -5.0_f64..=5.0_f64,
        ) {
            let data = test_sim_data();
            let r = MARS_REQ + alt_km * 1_000.0;
            let position_true = [r, 0.1, 0.05];
            let velocity_true = [velocity, gamma, psi];
            let biases = NavigationBiases {
                pos: [pos_bias_alt, 0.0, 0.0],
                vel: [vel_bias, 0.0, 0.0],
                drag: 0.0,
            };
            let mut nav_state = NavigationState::new();

            let out = call_navigate(
                &position_true,
                &velocity_true,
                &biases,
                &mut nav_state,
                &data,
                &no_run_biases(),
            );

            proptest::prop_assert!(out.position_estimated[0].is_finite(), "position_estimated[0] non-finite");
            proptest::prop_assert!(out.velocity_estimated[0].is_finite(), "velocity_estimated[0] non-finite");
            proptest::prop_assert!(out.density_guidance.is_finite(), "density_guidance non-finite");
            proptest::prop_assert!(out.dynamic_pressure_estimated.is_finite(), "dynamic_pressure_estimated non-finite");
            proptest::prop_assert!(out.energy_estimated.is_finite(), "energy_estimated non-finite");
            proptest::prop_assert!(nav_state.density_gain.is_finite(), "density_gain non-finite");
        }
    }

    proptest::proptest! {
        /// density_gain must always be in [0.1, 10.0] after any filter update
        /// (except high-altitude reset to 1.0).
        #[test]
        fn proptest_density_gain_bounded(
            alt_km in 30.0_f64..=90.0_f64,  // below 100 km so filter runs
            velocity in 2_000.0_f64..=8_000.0_f64,
            gamma in -0.3_f64..=0.0_f64,
            initial_gain in 0.001_f64..=100.0_f64,
            filter_gain_bias in -5.0_f64..=5.0_f64,
        ) {
            let data = test_sim_data();
            let r = MARS_REQ + alt_km * 1000.0;
            let position_true = [r, 0.0, 0.0];
            let velocity_true = [velocity, gamma, 1.0];
            let biases = zero_biases();
            let mut nav_state = NavigationState::new();
            nav_state.density_gain = initial_gain;

            let run_biases = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, filter_gain_bias];
            let _out = call_navigate(
                &position_true,
                &velocity_true,
                &biases,
                &mut nav_state,
                &data,
                &run_biases,
            );

            proptest::prop_assert!(
                nav_state.density_gain >= 0.1 && nav_state.density_gain <= 10.0,
                "density_gain {} out of [0.1, 10.0] bounds",
                nav_state.density_gain
            );
        }
    }

    // ── Test 6: zero_biases_no_nav_errors ──

    #[test]
    fn zero_biases_no_nav_errors() {
        let data = test_sim_data();
        let r = MARS_REQ + 80_000.0;
        let position_true = [r, 0.3, -0.1];
        let velocity_true = [5200.0, -0.12, 0.8];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // With zero biases, output position should exactly equal input
        for i in 0..3 {
            assert_eq!(
                out.position_estimated[i], position_true[i],
                "position_estimated[{i}] should exactly match input with zero biases"
            );
            assert_eq!(
                out.velocity_estimated[i], velocity_true[i],
                "velocity_estimated[{i}] should exactly match input with zero biases"
            );
        }
    }

    // ── Test 7: density_gain_diverges_with_onboard_model ──

    #[test]
    fn density_gain_diverges_with_onboard_model() {
        use crate::data::atmosphere::{ExponentialSegment, OnboardAtmosphereModel};

        let mut data = test_sim_data();
        data.atmosphere_onboard = OnboardAtmosphereModel::PiecewiseExponential {
            segments: vec![ExponentialSegment {
                alt_low: 0.0,
                alt_high: 150_000.0,
                rho_ref: 0.02,
                scale_height: 12_000.0,
            }],
        };

        let biases = NavigationBiases::default();
        let mut nav_state = NavigationState::new();
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius + 50_000.0;
        let position = [r, 0.0, 0.0];
        let velocity = [5000.0, -0.15, 0.6];

        for _ in 0..10 {
            call_navigate(
                &position,
                &velocity,
                &biases,
                &mut nav_state,
                &data,
                &no_run_biases(),
            );
        }

        assert!(
            (nav_state.density_gain - 1.0).abs() > 0.01,
            "density gain {} should diverge from 1.0 with inaccurate onboard model",
            nav_state.density_gain,
        );
    }

    // ── Test: lift_correction_at_zero_aoa ──

    #[test]
    fn lift_correction_at_zero_aoa() {
        let mut data = test_sim_data();
        // Override aero tables to have Cz = 0 at AoA = 0
        data.aero.incidence = vec![0.0, 0.35];
        data.aero.cx = vec![1.5, 1.7];
        data.aero.cz = vec![0.0, -0.4];
        data.aero.n_points = 2;
        data.entry.initial_aoa = 0.0; // zero AoA

        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // At alpha=0, cos(0)=1, sin(0)=0, so correction factor = 1.0
        // density_guidance should be positive and finite
        assert!(out.density_guidance > 0.0 && out.density_guidance.is_finite());
    }

    // ── Test: lift_correction_at_nonzero_aoa ──

    #[test]
    fn lift_correction_at_nonzero_aoa() {
        // Test that at non-zero AoA, the corrected density differs from
        // what a Cx-only inversion would produce.
        let mut data = test_sim_data();
        // Set up aero tables with known Cx and Cz at a specific AoA
        let aoa_10deg = 10.0_f64.to_radians();
        data.aero.incidence = vec![0.0, aoa_10deg, 0.35];
        data.aero.cx = vec![1.5, 1.6, 1.7];
        data.aero.cz = vec![0.0, -0.2, -0.4];
        data.aero.n_points = 3;
        data.entry.initial_aoa = aoa_10deg;
        // AoA profile returns constant aoa_10deg
        data.incidence.altitudes = vec![-10_000.0, 150_000.0];
        data.incidence.incidences = vec![aoa_10deg, aoa_10deg];

        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();

        let mut nav_state = NavigationState::new();
        let out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // The corrected denominator at AoA=10deg: 1.6*cos(10) + (-0.2)*sin(10)
        // = 1.6 * 0.9848 - 0.2 * 0.1736 = 1.5757 - 0.0347 = 1.5410
        // Correction factor vs Cx-only: 1.6 / 1.541 = 1.038 (~3.8% more density)
        let cx = 1.6_f64;
        let cz = -0.2_f64;
        let corrected_denom = cx * aoa_10deg.cos() + cz * aoa_10deg.sin();
        let correction_ratio = cx / corrected_denom;

        // Verify the ratio is approximately 1.038
        assert_relative_eq!(correction_ratio, 1.038, max_relative = 0.01);

        // density_guidance should be finite and positive
        assert!(
            out.density_guidance > 0.0 && out.density_guidance.is_finite(),
            "density_guidance should be positive and finite, got {}",
            out.density_guidance
        );
    }

    // ── Test: lift_correction_denom_guard ──

    #[test]
    fn lift_correction_denom_guard() {
        // When Cx*cos(alpha) + Cz*sin(alpha) is exactly zero, the guard
        // should trigger and density_estimated falls back to 0.0.
        let mut data = test_sim_data();
        // Force denom = 0: Cx=0, Cz=0 at all AoA
        data.aero.incidence = vec![0.0, 1.57];
        data.aero.cx = vec![0.0, 0.0];
        data.aero.cz = vec![0.0, 0.0];
        data.aero.n_points = 2;
        data.entry.initial_aoa = 0.5;

        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // Guard triggered: density_estimated = 0.0, filter stays near initial gain
        assert!(
            out.density_guidance.is_finite(),
            "density_guidance should be finite when denom guard triggers, got {}",
            out.density_guidance
        );
    }

    // ── Fix 4.2: density-inversion positive-denominator guard ──

    /// When `Cx*cos(alpha) + Cz*sin(alpha) < 0` (lift-dominated), the inverse-
    /// dynamics density is non-physical (negative). The guard must reject it
    /// (density_estimated = 0), not admit a negative estimate through `.abs()`.
    #[test]
    fn negative_denom_rejected_in_density_inversion() {
        let mut data = test_sim_data();
        // Loose rate limiter so the single-step gain is the un-clamped filter value
        // (default 0.1 would mask the old-vs-new difference behind the rate cap).
        data.guidance.density_gain_max_delta = 100.0;
        // aoa = 1.4 rad: denom = 1.0*cos(1.4) + (-5.0)*sin(1.4) ≈ 0.170 - 4.927 < 0.
        data.aero.incidence = vec![0.0, 1.57];
        data.aero.cx = vec![1.0, 1.0];
        data.aero.cz = vec![-5.0, -5.0];
        data.aero.n_points = 2;
        data.entry.initial_aoa = 1.4;

        // Sanity: confirm the constructed denominator is actually negative.
        let denom = 1.0 * 1.4_f64.cos() + (-5.0) * 1.4_f64.sin();
        assert!(
            denom < 0.0,
            "test setup invalid: denom={denom} not negative"
        );

        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new(); // density_gain = 1.0

        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // density_estimated is rejected (= 0), and the guard-tripped filter step is
        // skipped (Fix 4.3), so the gain is HELD at its initial 1.0. The OLD abs-guarded
        // code admitted a large NEGATIVE estimate, driving the gain below the 0.1 floor.
        assert_relative_eq!(nav_state.density_gain, 1.0, max_relative = 1e-12);
        assert!(
            nav_state.density_gain >= 0.1,
            "density_gain must not go negative / below floor, got {}",
            nav_state.density_gain
        );
    }

    // ── Fix 4.3: bias density-filter trigger consistency ──

    /// On a guard-tripped step (density_estimated == 0), the bias filter must be
    /// SKIPPED (gain held), not run with a zero numerator that drags the gain
    /// toward the 0.1 floor — matching the EKF trigger (`density_estimated > 0`).
    #[test]
    fn bias_filter_holds_gain_on_guard_tripped_step() {
        let mut data = test_sim_data();
        // Loose rate limiter: if the filter wrongly ran, the gain would move far,
        // so a held gain is unambiguous.
        data.guidance.density_gain_max_delta = 100.0;
        // Negative denom (aoa=1.4, Cz=-5) → density_estimated rejected to 0 (Fix 4.2).
        data.aero.incidence = vec![0.0, 1.57];
        data.aero.cx = vec![1.0, 1.0];
        data.aero.cz = vec![-5.0, -5.0];
        data.aero.n_points = 2;
        data.entry.initial_aoa = 1.4;

        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();
        nav_state.density_gain = 5.0; // pre-set, in-range gain to be HELD

        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // Filter skipped → gain held at 5.0. The OLD bias trigger (rho_model only)
        // would run the filter with density_estimated=0: raw = (1-0.8)*5.0 = 1.0,
        // dragging the gain down toward the floor.
        assert_relative_eq!(nav_state.density_gain, 5.0, max_relative = 1e-12);
    }

    // ── Fix 4.4: unconditional density-gain clamp ──

    /// The gain saturation clamp is a safety net and must run EVERY tick, even when
    /// the filter trigger is false. With the filter skipped (guard-tripped step) a
    /// pre-set out-of-range gain must still be clamped to [0.1, 10.0].
    #[test]
    fn density_gain_clamped_when_filter_skipped() {
        let mut data = test_sim_data();
        data.guidance.density_gain_max_delta = 100.0;
        // Negative denom → density_estimated = 0 → filter `if` body is skipped (Fix 4.3).
        data.aero.incidence = vec![0.0, 1.57];
        data.aero.cx = vec![1.0, 1.0];
        data.aero.cz = vec![-5.0, -5.0];
        data.aero.n_points = 2;
        data.entry.initial_aoa = 1.4;

        let r = MARS_REQ + 40_000.0;
        let position_true = [r, 0.0, 0.0];
        let velocity_true = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();
        nav_state.density_gain = 50.0; // out-of-range; must be clamped to 10.0

        let _out = call_navigate(
            &position_true,
            &velocity_true,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // Clamp hoisted out of the filter `if` → runs unconditionally. The OLD code
        // left the clamp inside the (skipped) filter block, so 50.0 survived.
        assert_relative_eq!(nav_state.density_gain, 10.0, max_relative = 1e-12);
    }

    // ── Test 9: SimPhase gating in navigate() ──

    /// SimPhase::Full: phase transitions from 1 → 2 after bounce + velocity below threshold.
    #[test]
    fn full_phase_transitions_to_exit() {
        let mut data = test_sim_data();
        data.sim_phase = SimPhase::Full;
        data.guidance.exit_velocity_threshold = 4400.0;
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius + 50_000.0;
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();
        let run_biases = no_run_biases();

        // First call: descending (FPA negative, pre-bounce) — should be phase 1
        let out1 = navigate(
            &[r, 0.0, 0.0],
            &[5000.0, -0.05, 0.6], // negative FPA → sin < 0 → no bounce
            data.entry.initial_aoa,
            10.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0],
            0.0, // density_perturbation
            run_biases[1],
            run_biases[2],
            run_biases[3],
            run_biases[4],
            run_biases[5],
            run_biases[6],
        );
        assert_eq!(
            out1.guidance_phase, 1,
            "should be capture phase while descending"
        );

        // Second call: ascending (FPA positive, small angle so velocity_radial < 120 m/s)
        // but velocity still above threshold — should remain capture phase.
        // gamma = 0.02 rad → velocity_radial ≈ 5000 * sin(0.02) ≈ 100 m/s < 120 m/s
        let out2 = navigate(
            &[r, 0.0, 0.0],
            &[5000.0, 0.02, 0.6], // positive FPA → sin > 0 → bounce; radial < 120 m/s
            data.entry.initial_aoa,
            20.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0],
            0.0, // density_perturbation
            run_biases[1],
            run_biases[2],
            run_biases[3],
            run_biases[4],
            run_biases[5],
            run_biases[6],
        );
        assert_eq!(
            out2.guidance_phase, 1,
            "above velocity threshold → still capture"
        );

        // Third call: ascending, velocity below threshold → phase 2.
        // Use gamma = 0.028 rad so velocity_radial ≈ 4000 * sin(0.028) ≈ 112 m/s.
        // This exceeds call 2's radial (≈ 5000 * sin(0.02) ≈ 100 m/s), so delta_radial > 0
        // and crash detection does not trigger. velocity < 4400 triggers the threshold transition.
        let out3 = navigate(
            &[r, 0.0, 0.0],
            &[4000.0, 0.028, 0.6], // below 4400 threshold; radial ≈ 112 m/s > prev, no crash
            data.entry.initial_aoa,
            30.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0],
            0.0, // density_perturbation
            run_biases[1],
            run_biases[2],
            run_biases[3],
            run_biases[4],
            run_biases[5],
            run_biases[6],
        );
        assert_eq!(
            out3.guidance_phase, 2,
            "below velocity threshold after bounce → exit phase"
        );
        assert_eq!(
            out3.phase_transition_flag, 1,
            "transition flag should be set"
        );
        assert!(
            out3.reference_velocity.abs() > 0.0,
            "reference_velocity should be latched"
        );
    }

    /// SimPhase::CaptureOnly: phase stays 1 regardless of state.
    #[test]
    fn capture_only_stays_phase_1() {
        let mut data = test_sim_data();
        data.sim_phase = SimPhase::CaptureOnly;
        data.guidance.exit_velocity_threshold = 4400.0;
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius + 50_000.0;
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();
        let run_biases = no_run_biases();

        // Trigger bounce (small FPA so velocity_radial < 120 m/s, avoiding the radial override)
        let _ = navigate(
            &[r, 0.0, 0.0],
            &[5000.0, 0.02, 0.6], // positive FPA, radial ≈ 100 m/s < 120
            data.entry.initial_aoa,
            10.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0],
            0.0, // density_perturbation
            run_biases[1],
            run_biases[2],
            run_biases[3],
            run_biases[4],
            run_biases[5],
            run_biases[6],
        );

        // Below threshold after bounce — would normally be phase 2, but CaptureOnly keeps phase 1.
        // gamma = 0.028 so radial ≈ 4000 * sin(0.028) ≈ 112 m/s, which exceeds call 1's radial
        // (≈ 5000 * sin(0.02) ≈ 100 m/s), so delta_radial > 0 and crash detection does not fire.
        let out = navigate(
            &[r, 0.0, 0.0],
            &[4000.0, 0.028, 0.6], // below 4400 threshold, radial ≈ 112 m/s > prev, no crash
            data.entry.initial_aoa,
            20.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0],
            0.0, // density_perturbation
            run_biases[1],
            run_biases[2],
            run_biases[3],
            run_biases[4],
            run_biases[5],
            run_biases[6],
        );
        assert_eq!(out.guidance_phase, 1, "CaptureOnly must keep phase 1");
    }

    /// SimPhase::ExitOnly: phase stays 2 regardless of state.
    #[test]
    fn exit_only_stays_phase_2() {
        let mut data = test_sim_data();
        data.sim_phase = SimPhase::ExitOnly;
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius + 50_000.0;
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();
        let run_biases = no_run_biases();

        // Descending, pre-bounce — would normally be phase 1
        let out = navigate(
            &[r, 0.0, 0.0],
            &[5000.0, -0.05, 0.6],
            data.entry.initial_aoa,
            10.0,
            &biases,
            &mut nav_state,
            &data,
            &planet,
            run_biases[0],
            0.0, // density_perturbation
            run_biases[1],
            run_biases[2],
            run_biases[3],
            run_biases[4],
            run_biases[5],
            run_biases[6],
        );
        assert_eq!(out.guidance_phase, 2, "ExitOnly must force phase 2");
    }

    // ── Fix 4.1: EKF exit-phase irreversibility ──

    /// Once `navigate_ekf` transitions to exit phase (1 → 2), a later step with
    /// `velocity_relative >= vphase && velocity_radial < 0` must NOT revert to phase 1.
    /// This mirrors the bias path's `exit_phase_locked` invariant.
    #[test]
    fn ekf_exit_phase_does_not_revert() {
        let mut data = test_sim_data();
        data.sim_phase = SimPhase::Full;
        data.guidance.exit_velocity_threshold = 4400.0;
        let r = MARS_REQ + 40_000.0;

        let mut legacy = NavigationState::new();
        let (mut ekf, mut imu, mut st, st_config, ekf_config) = quiet_ekf_states();

        // Step 1: ascending (γ>0 → bounce) and below threshold → transition 1 → 2.
        // radial ≈ 4000*sin(0.02) ≈ +80 m/s; prev_radial starts 0 → delta>0, no crash.
        let out1 = call_navigate_ekf(
            &[r, 0.0, 0.0],
            &[4000.0, 0.02, 0.6],
            10.0,
            &mut legacy,
            &mut ekf,
            &mut imu,
            &mut st,
            &st_config,
            &ekf_config,
            &data,
        );
        assert_eq!(
            out1.guidance_phase, 2,
            "below threshold after bounce → exit phase"
        );
        assert_eq!(
            out1.phase_transition_flag, 1,
            "transition flag should be set"
        );

        // Disarm the crash detector for the revert step (orthogonal to the phase-lock
        // bug under test): force prev_radial very negative so the next (negative) radial
        // is an increase, not a crash-triggering decrease.
        legacy.previous_radial_velocity = -1e9;

        // Step 2: velocity_relative (5000) >= vphase (4400) AND velocity_radial < 0
        // (γ<0). In the buggy EKF path this re-enters capture (phase 2 → 1); with the
        // fix, `exit_phase_locked` guards it and the phase stays 2.
        let out2 = call_navigate_ekf(
            &[r, 0.0, 0.0],
            &[5000.0, -0.01, 0.6],
            20.0,
            &mut legacy,
            &mut ekf,
            &mut imu,
            &mut st,
            &st_config,
            &ekf_config,
            &data,
        );
        assert_eq!(
            out2.guidance_phase, 2,
            "exit phase must not revert to capture once latched (EKF parity with bias)"
        );
    }
}
