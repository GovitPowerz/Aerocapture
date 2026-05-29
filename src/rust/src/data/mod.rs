//! Data file loading and simulation data structures.

pub mod aerodynamics;
pub mod atmosphere;
pub mod capsule;
pub mod dispersions;
pub mod guidance_params;
pub mod incidence;
pub mod neural;
pub mod nn_state;
pub mod pilot;

use crate::config::{
    GuidanceType, IntegrationMode, SimInput, SimPhase, TomlConfig, TomlLayerSpec, TomlMonteCarlo,
    TomlNavigation,
};
use crate::gnc::guidance::lateral::LateralParams;
use crate::gnc::guidance::thermal_limiter::ThermalLimiterParams;
use crate::physics::winds;
use std::fmt;

/// Navigation mode: bias-based (legacy) or Extended Kalman Filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NavMode {
    Bias,
    Ekf,
}

#[derive(Debug)]
pub struct DataError(pub String);

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DataError {}

/// State vector in spherical coordinates
#[derive(Debug, Clone, Copy, Default)]
pub struct SphericalState {
    pub altitude: f64,    // meters
    pub longitude: f64,   // radians
    pub latitude: f64,    // radians
    pub velocity: f64,    // m/s
    pub flight_path: f64, // radians (gamma)
    pub azimuth: f64,     // radians
}

/// Orbital elements
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OrbitalElements {
    pub semi_major_axis: f64, // meters
    pub eccentricity: f64,
    pub inclination: f64,   // radians
    pub raan: f64,          // radians
    pub arg_periapsis: f64, // radians
    pub true_anomaly: f64,  // radians
    pub periapsis_alt: f64, // meters
    pub apoapsis_alt: f64,  // meters
}

/// Target orbital parameters (from mission file)
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OrbitalTarget {
    pub apoapsis: f64,        // meters (altitude)
    pub periapsis: f64,       // meters (altitude)
    pub semi_major_axis: f64, // meters
    pub eccentricity: f64,
    pub inclination: f64, // radians
    pub raan: f64,        // radians
}

/// Mission final conditions (from mission file)
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FinalConditions {
    pub altitude: f64,    // meters
    pub longitude: f64,   // radians
    pub latitude: f64,    // radians
    pub velocity: f64,    // m/s
    pub flight_path: f64, // radians
    pub azimuth: f64,     // radians
    pub energy: f64,      // J/kg (converted from MJ/kg)
    pub radial_vel: f64,  // m/s
}

/// Parking orbit parameters
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ParkingOrbit {
    pub apoapsis: f64,  // meters
    pub periapsis: f64, // meters
}

/// Time periods for different subsystems
#[derive(Debug, Clone, Copy)]
pub struct TimePeriods {
    pub navigation: f64,  // seconds
    pub guidance: f64,    // seconds
    pub pilot: f64,       // seconds
    pub prediction: f64,  // seconds
    pub integration: f64, // seconds
    pub photo: f64,       // seconds
}

impl Default for TimePeriods {
    fn default() -> Self {
        Self {
            navigation: 1.0,
            guidance: 1.0,
            pilot: 0.1,
            prediction: 1.0,
            integration: 1.0,
            photo: 1.0,
        }
    }
}

/// Entry conditions (from rentree file)
#[derive(Debug, Clone, Copy, Default)]
pub struct EntryConditions {
    pub state: SphericalState,
    pub initial_date: f64, // seconds
    pub initial_bank: f64, // radians (gite)
    pub initial_aoa: f64,  // radians (incidence)
}

/// Reentry constraints (converted to SI)
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Constraints {
    pub max_heat_flux: f64,        // W/m^2 (from kW/m^2)
    pub max_load_factor: f64,      // m/s^2 (from g, multiplied by g0=9.81)
    pub max_dynamic_pressure: f64, // Pa (from kPa)
    pub max_heat_load: f64,        // J/m^2 (from kJ/m^2)
}

/// Success criteria
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SuccessCriteria {
    pub inclination_tol: f64, // radians (from deg)
    pub velocity_tol: f64,    // m/s
    pub apoapsis_tol: f64,    // meters (from km)
    pub periapsis_tol: f64,   // meters (from km)
}

/// All loaded simulation data
#[allow(dead_code)]
#[derive(Debug)]
pub struct SimData {
    pub capsule: capsule::Capsule,
    pub aero: aerodynamics::AeroTables,
    pub atmosphere: atmosphere::AtmosphereModel,
    /// Onboard atmosphere model (degraded) for navigation and guidance
    pub atmosphere_onboard: atmosphere::OnboardAtmosphereModel,
    pub entry: EntryConditions,
    pub constraints: Constraints,
    pub final_conditions: FinalConditions,
    pub target_orbit: OrbitalTarget,
    pub parking_orbit: ParkingOrbit,
    pub periods: TimePeriods,
    pub guidance: guidance_params::GuidanceParams,
    pub incidence: incidence::IncidenceProfile,
    pub pilot: pilot::PilotModel,
    pub success: SuccessCriteria,
    pub wind_enabled: bool,
    pub wind_table: Option<winds::WindTable>,
    pub neural_net: Option<neural::NeuralNetModel>,
    /// Domain-based dispersion config (replaces lottery files when present)
    pub dispersion_config: Option<dispersions::DispersionConfig>,
    /// Navigation mode: bias (legacy) or EKF
    pub nav_mode: NavMode,
    /// Raw TOML navigation config for building EKF sensor models
    pub nav_config: Option<TomlNavigation>,
    /// Integration method: fixed Gill RK4 (default) or adaptive DOPRI45
    pub integration_mode: IntegrationMode,
    /// Simulation phase mode (Full, CaptureOnly, ExitOnly, Preprogrammed)
    pub sim_phase: SimPhase,
    /// Gauss-Markov density perturbation config (None = disabled)
    pub density_perturbation: Option<dispersions::DensityPerturbationConfig>,
}

const G0: f64 = 9.81;
const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

impl SimData {
    /// Load simulation data from TOML config (inline data + external files).
    pub fn from_toml(toml: &TomlConfig, config: &SimInput) -> Result<Self, DataError> {
        let v = toml
            .vehicle
            .as_ref()
            .ok_or_else(|| DataError("Missing [vehicle] section".to_string()))?;
        let e = toml
            .entry
            .as_ref()
            .ok_or_else(|| DataError("Missing [entry] section".to_string()))?;
        let f = toml
            .flight
            .as_ref()
            .ok_or_else(|| DataError("Missing [flight] section".to_string()))?;
        let a = toml
            .aerodynamics
            .as_ref()
            .ok_or_else(|| DataError("Missing [aerodynamics] section".to_string()))?;

        // Vehicle / capsule
        let capsule_data = capsule::Capsule {
            mass: v.mass,
            reference_area: v.reference_area,
            cq: v.cq,
            max_bank_rate: v.max_bank_rate * DEG2RAD,
            periods: TimePeriods {
                navigation: v.periods.navigation,
                guidance: v.periods.guidance,
                pilot: v.periods.pilot,
                prediction: v.periods.prediction,
                integration: v.periods.integration,
                photo: v.periods.photo,
            },
        };

        // Pilot
        let pilot_type = match v.pilot.model.as_str() {
            "perfect" => pilot::PilotType::Perfect,
            "first_order" => pilot::PilotType::FirstOrder,
            "second_order" => pilot::PilotType::SecondOrder,
            other => return Err(DataError(format!("Unknown pilot model: {}", other))),
        };
        let pilot_data = pilot::PilotModel {
            pilot_type,
            time_constant: v.pilot.time_constant,
            damping: v.pilot.damping,
            frequency: v.pilot.frequency,
        };

        // Entry conditions
        let entry = EntryConditions {
            state: SphericalState {
                altitude: e.altitude * 1e3,
                longitude: e.longitude * DEG2RAD,
                latitude: e.latitude * DEG2RAD,
                velocity: e.velocity,
                flight_path: e.flight_path_angle * DEG2RAD,
                azimuth: e.azimuth * DEG2RAD,
            },
            initial_date: e.initial_time,
            initial_bank: e.initial_bank_angle * DEG2RAD,
            initial_aoa: e.initial_aoa * DEG2RAD,
        };

        // Aerodynamics (body-axis Ca/Cn → stability-axis Cx/Cz)
        let alfaeq = a.equilibrium_aoa * DEG2RAD;
        let n_aero = a.points.len();
        let mut aero_incidence = Vec::with_capacity(n_aero);
        let mut cx_vec = Vec::with_capacity(n_aero);
        let mut cz_vec = Vec::with_capacity(n_aero);
        for pt in &a.points {
            let alpha = pt.aoa * DEG2RAD;
            let cx_i = pt.ca * alpha.cos() + pt.cn * alpha.sin();
            let cz_i = -pt.ca * alpha.sin() + pt.cn * alpha.cos();
            aero_incidence.push(alpha);
            cx_vec.push(cx_i);
            cz_vec.push(cz_i);
        }
        let nominal_cx = aerodynamics::interpolate(&aero_incidence, &cx_vec, alfaeq);
        let nominal_cz = aerodynamics::interpolate(&aero_incidence, &cz_vec, alfaeq);
        let nominal_finesse = if nominal_cx.abs() > 1e-30 {
            nominal_cz / nominal_cx
        } else {
            0.0
        };
        let aero = aerodynamics::AeroTables {
            equilibrium_aoa: alfaeq,
            n_points: n_aero,
            incidence: aero_incidence,
            cx: cx_vec,
            cz: cz_vec,
            nominal_cx,
            nominal_cz,
            nominal_finesse,
            ballistic_coeff: 0.0,
        };

        // Flight / mission
        let constraints = Constraints {
            max_heat_flux: f.constraints.max_heat_flux * 1e3,
            max_load_factor: f.constraints.max_load_factor * G0,
            max_dynamic_pressure: f.constraints.max_dynamic_pressure * 1e3,
            max_heat_load: f.constraints.max_heat_load * 1e3, // kJ/m^2 -> J/m^2
        };
        let final_conditions = FinalConditions {
            altitude: f.final_conditions.altitude * 1e3,
            longitude: f.final_conditions.longitude * DEG2RAD,
            latitude: f.final_conditions.latitude * DEG2RAD,
            velocity: f.final_conditions.velocity,
            flight_path: f.final_conditions.flight_path_angle * DEG2RAD,
            azimuth: f.final_conditions.azimuth * DEG2RAD,
            energy: f.final_conditions.energy * 1e6,
            radial_vel: f.final_conditions.radial_velocity,
        };
        let target_orbit = OrbitalTarget {
            apoapsis: f.target_orbit.apoapsis * 1e3,
            periapsis: f.target_orbit.periapsis * 1e3,
            semi_major_axis: f.target_orbit.semi_major_axis * 1e3,
            eccentricity: f.target_orbit.eccentricity,
            inclination: f.target_orbit.inclination * DEG2RAD,
            raan: f.target_orbit.raan * DEG2RAD,
        };
        let parking_orbit = ParkingOrbit {
            apoapsis: f.parking_orbit.apoapsis * 1e3,
            periapsis: f.parking_orbit.periapsis * 1e3,
        };

        // Success criteria
        let success = if let Some(ref s) = toml.success {
            SuccessCriteria {
                inclination_tol: s.inclination_tolerance * DEG2RAD,
                velocity_tol: s.velocity_tolerance,
                apoapsis_tol: s.apoapsis_tolerance * 1e3,
                periapsis_tol: s.periapsis_tolerance * 1e3,
            }
        } else {
            SuccessCriteria::default()
        };

        // Incidence profile
        let incidence_data = if let Some(ref inc) = toml.incidence {
            let n = inc.altitudes.len().min(inc.angles.len());
            incidence::IncidenceProfile {
                n_points: n,
                altitudes: inc.altitudes[..n].iter().map(|a| a * 1e3).collect(),
                incidences: inc.angles[..n].iter().map(|a| a * DEG2RAD).collect(),
            }
        } else {
            incidence::IncidenceProfile {
                n_points: 0,
                altitudes: vec![],
                incidences: vec![],
            }
        };

        // Per-scheme guidance params (with defaults if not in TOML)
        let piecewise_constant_params = {
            let pc = &toml.guidance.piecewise_constant;
            let bank_angles_deg = pc.resolve_bank_angles_deg().map_err(DataError)?;
            guidance_params::PiecewiseConstantParams {
                bank_angles: bank_angles_deg.iter().map(|d| d.to_radians()).collect(),
                energy_min: pc.energy_min * 1e6,
                energy_max: pc.energy_max * 1e6,
            }
        };

        let eq_glide_params = if let Some(ref p) = toml.guidance.equilibrium_glide {
            guidance_params::EqGlideParams {
                k_hdot_scale: p.k_hdot_scale,
                v_ratio_threshold: p.v_ratio_threshold,
                velocity_bias_high: p.velocity_bias_high,
                velocity_bias_low: p.velocity_bias_low,
                alt_bias_threshold: p.alt_bias_threshold,
                cos_bank_min: p.cos_bank_min,
                cos_bank_max: p.cos_bank_max,
            }
        } else {
            guidance_params::EqGlideParams::default()
        };

        let energy_ctrl_params = if let Some(ref p) = toml.guidance.energy_controller {
            guidance_params::EnergyCtrlParams {
                gain: p.gain,
                kp: p.kp,
                kd: p.kd,
            }
        } else {
            guidance_params::EnergyCtrlParams::default()
        };

        let pred_guid_params = if let Some(ref p) = toml.guidance.pred_guid {
            guidance_params::PredGuidParams {
                k_drag_high: p.k_drag_high,
                k_drag_low: p.k_drag_low,
                pdyn_threshold: p.pdyn_threshold,
            }
        } else {
            guidance_params::PredGuidParams::default()
        };

        let fnpag_params = if let Some(ref p) = toml.guidance.fnpag {
            guidance_params::FnpagParams {
                energy_tol: p.energy_tol,
                prediction_dt: p.prediction_dt,
                bank_min_deg: p.bank_min_deg,
                bank_max_high_deg: p.bank_max_high_deg,
                bank_max_low_deg: p.bank_max_low_deg,
            }
        } else {
            guidance_params::FnpagParams::default()
        };

        // Navigation density filter params (live in [navigation], used by all schemes)
        let nav_density_filter_gain = toml
            .navigation
            .as_ref()
            .map_or(0.8, |n| n.density_filter_gain);
        let nav_density_gain_max_delta = toml
            .navigation
            .as_ref()
            .map_or(0.1, |n| n.density_gain_max_delta);

        // Neural network guidance mode (full_neural | magnitude_only)
        let neural_mode = match toml
            .guidance
            .neural_network
            .as_ref()
            .and_then(|nn| nn.mode.as_deref())
        {
            None | Some("full_neural") => guidance_params::NeuralNetMode::FullNeural,
            Some("magnitude_only") => guidance_params::NeuralNetMode::MagnitudeOnly,
            Some(other) => {
                return Err(DataError(format!(
                    "guidance.neural_network.mode must be \"full_neural\" or \"magnitude_only\" (got \"{}\")",
                    other
                )));
            }
        };

        // Validate output_parameterization constraints.
        let output_param_toml = toml
            .guidance
            .neural_network
            .as_ref()
            .and_then(|nn| nn.output_parameterization.as_deref());
        validate_output_parameterization(
            output_param_toml,
            neural_mode,
            toml.network
                .as_ref()
                .and_then(|n| n.architecture.as_deref()),
        )?;

        // FTC guidance params
        let guidance = if let Some(ref ftc) = toml.guidance.ftc {
            let energy_scale = 1e6;

            // Load reference trajectory from external file
            let ref_traj = if !config.reference_trajectory {
                if let Some(ref path) = toml.data.reference_trajectory {
                    guidance_params::ReferenceTrajectory::load(path)?
                } else {
                    guidance_params::ReferenceTrajectory::default()
                }
            } else {
                guidance_params::ReferenceTrajectory::default()
            };

            guidance_params::GuidanceParams {
                capture_damping: ftc.capture_damping,
                capture_frequency: ftc.capture_frequency,
                capture_pdyn_margin: ftc.capture_pdyn_margin,
                altitude_damping: ftc.altitude_damping,
                altitude_frequency: ftc.altitude_frequency * DEG2RAD,
                exit_velocity_threshold: ftc.exit_velocity_threshold,
                exit_pdyn_margin: ftc.exit_pdyn_margin,
                exit_altitude_threshold: ftc.exit_altitude_threshold * 1e3,
                exit_radial_vel_gain: ftc.exit_radial_vel_gain,
                exit_apoapsis_threshold: ftc.exit_apoapsis_threshold,
                lateral: if let Some(ref lat) = toml.guidance.lateral {
                    LateralParams {
                        tau: lat.tau,
                        threshold: lat.threshold * DEG2RAD,
                        min_reversal_interval: lat.min_reversal_interval,
                        lateral_activation: lat.lateral_activation * energy_scale,
                        lateral_inhibition: lat.lateral_inhibition * energy_scale,
                        max_reversals: lat.max_reversals,
                    }
                } else {
                    LateralParams::default()
                },
                security_capture: ftc.security_capture,
                security_exit: ftc.security_exit,
                density_filter_gain: nav_density_filter_gain,
                density_gain_max_delta: nav_density_gain_max_delta,
                longi_activation: ftc.longi_activation * energy_scale,
                longi_inhibition: ftc.longi_inhibition * energy_scale,
                pdyn_min: ftc.pdyn_min,
                pressure_coeff_base: ftc.pressure_coeff_base,
                pressure_coeff_scale_height: ftc.pressure_coeff_scale_height,
                gain_fade_start_km: ftc.gain_fade_start_km,
                gain_fade_end_km: ftc.gain_fade_end_km,
                ref_trajectory: ref_traj,
                eq_glide: eq_glide_params.clone(),
                energy_ctrl: energy_ctrl_params.clone(),
                pred_guid: pred_guid_params.clone(),
                fnpag: fnpag_params.clone(),
                piecewise_constant: piecewise_constant_params.clone(),
                thermal_limiter: if let Some(ref tl) = toml.guidance.thermal_limiter {
                    ThermalLimiterParams {
                        heat_flux_activation: tl.heat_flux_activation,
                        heat_load_activation: tl.heat_load_activation,
                        heat_flux_ramp_exponent: tl.heat_flux_ramp_exponent,
                        heat_load_ramp_exponent: tl.heat_load_ramp_exponent,
                    }
                } else {
                    ThermalLimiterParams::default()
                },
                command_shaping: match toml.guidance.command_shaping.as_ref() {
                    Some(cs) if cs.enabled => {
                        if cs.max_bank_acceleration <= 0.0 {
                            return Err(DataError(format!(
                                "command_shaping.max_bank_acceleration must be > 0 when enabled (got {})",
                                cs.max_bank_acceleration
                            )));
                        }
                        Some(guidance_params::CommandShapingConfig {
                            max_bank_acceleration: cs.max_bank_acceleration.to_radians(),
                        })
                    }
                    _ => None,
                },
                neural_mode,
            }
        } else {
            // No FTC params — load from file if guidance suffix available, else defaults
            let ref_traj = if !config.reference_trajectory {
                if let Some(ref path) = toml.data.reference_trajectory {
                    guidance_params::ReferenceTrajectory::load(path)?
                } else {
                    guidance_params::ReferenceTrajectory::default()
                }
            } else {
                guidance_params::ReferenceTrajectory::default()
            };
            guidance_params::GuidanceParams {
                capture_damping: 0.7,
                capture_frequency: 0.072,
                capture_pdyn_margin: 1.75,
                altitude_damping: 0.7,
                altitude_frequency: 0.08 * DEG2RAD,
                exit_velocity_threshold: 4400.0,
                exit_pdyn_margin: 1.75,
                exit_altitude_threshold: 60e3,
                exit_radial_vel_gain: 10.0,
                exit_apoapsis_threshold: 100.0,
                lateral: if let Some(ref lat) = toml.guidance.lateral {
                    LateralParams {
                        tau: lat.tau,
                        threshold: lat.threshold * DEG2RAD,
                        min_reversal_interval: lat.min_reversal_interval,
                        lateral_activation: lat.lateral_activation * 1e6,
                        lateral_inhibition: lat.lateral_inhibition * 1e6,
                        max_reversals: lat.max_reversals,
                    }
                } else {
                    // No lateral config — inactive by default (tau=0 disables)
                    LateralParams::default()
                },
                security_capture: 1,
                security_exit: 3,
                density_filter_gain: nav_density_filter_gain,
                density_gain_max_delta: nav_density_gain_max_delta,
                longi_activation: 1e9,
                longi_inhibition: -1e9,
                pdyn_min: 0.0,
                pressure_coeff_base: -134.4,
                pressure_coeff_scale_height: 6.9,
                gain_fade_start_km: 80.0,
                gain_fade_end_km: 100.0,
                ref_trajectory: ref_traj,
                eq_glide: eq_glide_params,
                energy_ctrl: energy_ctrl_params,
                pred_guid: pred_guid_params,
                fnpag: fnpag_params,
                piecewise_constant: piecewise_constant_params,
                thermal_limiter: if let Some(ref tl) = toml.guidance.thermal_limiter {
                    ThermalLimiterParams {
                        heat_flux_activation: tl.heat_flux_activation,
                        heat_load_activation: tl.heat_load_activation,
                        heat_flux_ramp_exponent: tl.heat_flux_ramp_exponent,
                        heat_load_ramp_exponent: tl.heat_load_ramp_exponent,
                    }
                } else {
                    ThermalLimiterParams::default()
                },
                command_shaping: match toml.guidance.command_shaping.as_ref() {
                    Some(cs) if cs.enabled => {
                        if cs.max_bank_acceleration <= 0.0 {
                            return Err(DataError(format!(
                                "command_shaping.max_bank_acceleration must be > 0 when enabled (got {})",
                                cs.max_bank_acceleration
                            )));
                        }
                        Some(guidance_params::CommandShapingConfig {
                            max_bank_acceleration: cs.max_bank_acceleration.to_radians(),
                        })
                    }
                    _ => None,
                },
                neural_mode,
            }
        };

        // Atmosphere (always external)
        let atm_path = toml
            .data
            .atmosphere
            .as_ref()
            .ok_or_else(|| DataError("Missing data.atmosphere path".to_string()))?;
        let atm = atmosphere::AtmosphereModel::load(atm_path)?;

        // Onboard atmosphere model
        let atm_onboard = match &toml.onboard_atmosphere {
            Some(cfg) if cfg.mode.as_deref() == Some("identical") => {
                atmosphere::OnboardAtmosphereModel::Identical
            }
            Some(cfg) if cfg.segments.is_some() => {
                let segs = cfg.segments.as_ref().unwrap();
                atmosphere::OnboardAtmosphereModel::PiecewiseExponential {
                    segments: segs
                        .iter()
                        .map(|s| atmosphere::ExponentialSegment {
                            alt_low: s.alt_low,
                            alt_high: s.alt_high,
                            rho_ref: s.rho_ref,
                            scale_height: s.scale_height,
                        })
                        .collect(),
                }
            }
            Some(cfg) => {
                let n = cfg.n_segments.unwrap_or(5);
                atmosphere::OnboardAtmosphereModel::fit_from_table(&atm, n)
            }
            None => {
                // Default: auto-fit with 5 segments
                atmosphere::OnboardAtmosphereModel::fit_from_table(&atm, 5)
            }
        };

        // Wind table (optional)
        let wind_table = if let Some(ref wt_path) = toml.data.wind_table {
            Some(winds::WindTable::load(wt_path)?)
        } else {
            None
        };

        // Neural network (external, optional)
        let neural_net = if config.guidance_type == GuidanceType::NeuralNetwork {
            if let Some(ref nn_path) = toml.data.neural_network {
                Some(neural::NeuralNetModel::load(nn_path)?)
            } else {
                None
            }
        } else {
            None
        };

        // Apply TOML [network] overrides to loaded NN model
        let mut neural_net = match (neural_net, &toml.network) {
            (Some(mut nn), Some(net_cfg)) => {
                if let Some(ref mask) = net_cfg.input_mask {
                    nn.input_mask = Some(mask.clone());
                }
                if let Some(ablated) = net_cfg.ablated_input {
                    nn.ablated_input = Some(ablated);
                }
                // Re-validate after override
                neural::NeuralNetModel::validate_mask(&nn.input_mask, nn.layer_sizes[0])?;
                neural::NeuralNetModel::validate_ablated_input(&nn.ablated_input)?;
                Some(nn)
            }
            (nn, _) => nn,
        };

        // Apply TOML [guidance.neural_network] decoder-knob overrides onto the
        // loaded model so the runtime honors the training-time knob even if the
        // deployed JSON predates the field. Done before the immutable cross-check
        // / defense-in-depth borrows below to satisfy the borrow checker.
        if let Some(nn) = neural_net.as_mut()
            && let Some(tnn) = &toml.guidance.neural_network
        {
            if let Some(n) = tnn.scaled_pi_n {
                nn.scaled_pi_n = n;
            }
            if let Some(d) = tnn.delta_max {
                nn.delta_max = d;
            }
        }

        // Cross-check TOML `[guidance.neural_network] output_parameterization`
        // against the loaded model's `output_param`. The TOML is a training-time
        // knob; the JSON model file is the deploy-time source of truth. They MUST
        // agree if both are set — silently ignoring a mismatched TOML would mean
        // the user's intent is unrepresented at runtime.
        if let (Some(nn), Some(tnn)) = (&neural_net, &toml.guidance.neural_network)
            && let Some(toml_param) = &tnn.output_parameterization
        {
            let toml_enum = match toml_param.as_str() {
                "atan2_signed" => neural::OutputParam::Atan2Signed,
                "acos_tanh" => neural::OutputParam::AcosTanh,
                "scaled_pi" => neural::OutputParam::ScaledPi,
                "delta" => neural::OutputParam::Delta,
                other => {
                    return Err(DataError(format!(
                        "[guidance.neural_network] output_parameterization='{}' is not recognized; \
                         expected 'atan2_signed', 'acos_tanh', 'scaled_pi', or 'delta'",
                        other
                    )));
                }
            };
            if nn.output_param != toml_enum {
                return Err(DataError(format!(
                    "TOML output_parameterization='{}' disagrees with the loaded model's \
                     output_param={:?}. Either retrain the model with the TOML knob set to \
                     '{}' before training (the trainer embeds it into best_model.json) or \
                     align the TOML to match the model.",
                    toml_param, nn.output_param, toml_param
                )));
            }
        }

        // Defense-in-depth: regardless of whether the TOML sets
        // `output_parameterization`, an `acos_tanh` model emits a non-negative
        // magnitude in `[0, π]`. Running it under `full_neural` mode would feed
        // an unsigned magnitude through the signed-bank dispatch path, silently
        // suppressing roll reversals. The deployed JSON is the runtime source
        // of truth, so reject this combo at data load even when the TOML key
        // is absent.
        if let Some(nn) = &neural_net {
            match nn.output_param {
                neural::OutputParam::AcosTanh
                    if neural_mode != guidance_params::NeuralNetMode::MagnitudeOnly =>
                {
                    return Err(DataError(
                        "loaded model has output_param='acos_tanh' which requires \
                         [guidance.neural_network] mode='magnitude_only'; the model emits an \
                         unsigned bank magnitude and cannot drive the signed-bank dispatch path."
                            .to_string(),
                    ));
                }
                neural::OutputParam::ScaledPi | neural::OutputParam::Delta
                    if neural_mode != guidance_params::NeuralNetMode::FullNeural =>
                {
                    return Err(DataError(format!(
                        "loaded model output_param={:?} emits a signed bank and requires \
                         [guidance.neural_network] mode='full_neural'.",
                        nn.output_param
                    )));
                }
                _ => {}
            }
        }

        // Domain-based Monte Carlo config (replaces lottery files)
        let dispersion_config = if let Some(ref mc) = toml.monte_carlo {
            Some(build_dispersion_config(mc)?)
        } else {
            None
        };

        let density_perturbation = dispersion_config
            .as_ref()
            .and_then(|dc| dc.density_perturbation);

        // Navigation mode
        let nav_mode = match toml.navigation.as_ref().map(|n| n.mode.as_str()) {
            Some("ekf") => NavMode::Ekf,
            _ => NavMode::Bias,
        };

        Ok(SimData {
            capsule: capsule_data,
            aero,
            atmosphere: atm,
            atmosphere_onboard: atm_onboard,
            entry,
            constraints,
            final_conditions,
            target_orbit,
            parking_orbit,
            periods: capsule_data.periods,
            guidance,
            incidence: incidence_data,
            pilot: pilot_data,
            success,
            wind_enabled: f.wind,
            wind_table,
            neural_net,
            dispersion_config,
            nav_mode,
            nav_config: toml.navigation.clone(),
            integration_mode: IntegrationMode::from_toml(&toml.integration, v.periods.integration),
            sim_phase: config.sim_phase,
            density_perturbation,
        })
    }
}

/// Build a DispersionConfig from TOML [monte_carlo] section.
fn build_dispersion_config(
    mc: &TomlMonteCarlo,
) -> Result<dispersions::DispersionConfig, DataError> {
    use dispersions::*;

    let initial_state = mc.initial_state.as_ref().and_then(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off {
            return None;
        }
        let mut s = InitialStateSigmas::from_level(level);
        if level == DispersionLevel::Custom {
            if let Some(&v) = d.custom.get("altitude") {
                s.altitude = v;
            }
            if let Some(&v) = d.custom.get("longitude") {
                s.longitude = v;
            }
            if let Some(&v) = d.custom.get("latitude") {
                s.latitude = v;
            }
            if let Some(&v) = d.custom.get("velocity") {
                s.velocity = v;
            }
            if let Some(&v) = d.custom.get("flight_path_angle") {
                s.flight_path = v;
            }
            if let Some(&v) = d.custom.get("azimuth") {
                s.azimuth = v;
            }
        }
        Some(s)
    });

    let atmosphere = mc.atmosphere.as_ref().and_then(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off {
            return None;
        }
        let mut s = AtmosphereSigmas::from_level(level);
        if level == DispersionLevel::Custom
            && let Some(&v) = d.custom.get("density")
        {
            s.density = v;
        }
        Some(s)
    });

    let aerodynamics = mc.aerodynamics.as_ref().and_then(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off {
            return None;
        }
        let mut s = AerodynamicsSigmas::from_level(level);
        if level == DispersionLevel::Custom {
            if let Some(&v) = d.custom.get("drag") {
                s.drag = v;
            }
            if let Some(&v) = d.custom.get("lift") {
                s.lift = v;
            }
            if let Some(&v) = d.custom.get("incidence") {
                s.incidence = v;
            }
        }
        Some(s)
    });

    let navigation = mc.navigation.as_ref().and_then(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off {
            return None;
        }
        let mut s = NavigationSigmas::from_level(level);
        if level == DispersionLevel::Custom {
            if let Some(&v) = d.custom.get("altitude") {
                s.altitude = v;
            }
            if let Some(&v) = d.custom.get("longitude") {
                s.longitude = v;
            }
            if let Some(&v) = d.custom.get("latitude") {
                s.latitude = v;
            }
            if let Some(&v) = d.custom.get("velocity") {
                s.velocity = v;
            }
            if let Some(&v) = d.custom.get("flight_path_angle") {
                s.flight_path = v;
            }
            if let Some(&v) = d.custom.get("azimuth") {
                s.azimuth = v;
            }
            if let Some(&v) = d.custom.get("drag_accel") {
                s.drag_accel = v;
            }
        }
        Some(s)
    });

    let mass = mc.mass.as_ref().and_then(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off {
            return None;
        }
        let mut s = MassSigmas::from_level(level);
        if level == DispersionLevel::Custom
            && let Some(&v) = d.custom.get("mass")
        {
            s.mass = v;
        }
        Some(s)
    });

    let vehicle = mc.vehicle.as_ref().and_then(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off {
            return None;
        }
        let mut s = VehicleSigmas::from_level(level);
        if level == DispersionLevel::Custom {
            if let Some(&v) = d.custom.get("ref_area") {
                s.ref_area = v;
            }
            if let Some(&v) = d.custom.get("max_bank_rate") {
                s.max_bank_rate = v;
            }
        }
        Some(s)
    });

    let pilot = mc.pilot.as_ref().and_then(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off {
            return None;
        }
        let mut s = PilotSigmas::from_level(level);
        if level == DispersionLevel::Custom {
            if let Some(&v) = d.custom.get("time_constant") {
                s.time_constant = v;
            }
            if let Some(&v) = d.custom.get("damping") {
                s.damping = v;
            }
            if let Some(&v) = d.custom.get("frequency") {
                s.frequency = v;
            }
        }
        Some(s)
    });

    let nav_filter = mc.nav_filter.as_ref().and_then(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off {
            return None;
        }
        let mut s = NavFilterSigmas::from_level(level);
        if level == DispersionLevel::Custom
            && let Some(&v) = d.custom.get("filter_gain")
        {
            s.filter_gain = v;
        }
        Some(s)
    });

    let wind = mc.wind.as_ref().and_then(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off {
            return None;
        }
        let mut cfg = WindDispersionConfig::from_level(level);
        // Apply custom overrides (for backward compat: existing configs without a level
        // field get level="medium" by default, with their explicit values as overrides)
        if let Some(&v) = d.custom.get("scale_min") {
            cfg.scale_min = v;
        }
        if let Some(&v) = d.custom.get("scale_max") {
            cfg.scale_max = v;
        }
        if let Some(&v) = d.custom.get("direction_bias_deg") {
            cfg.direction_bias_deg = v;
        }
        Some(cfg)
    });

    let density_perturbation = if let Some(d) = mc.density_perturbation.as_ref() {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off {
            None
        } else {
            let mut cfg = DensityPerturbationConfig::from_level(level);
            if level == DispersionLevel::Custom {
                if let Some(&v) = d.custom.get("tau") {
                    cfg.tau = v;
                }
                if let Some(&v) = d.custom.get("sigma") {
                    cfg.sigma = v;
                }
            }
            if cfg.tau < 0.0 || cfg.sigma < 0.0 {
                return Err(DataError(format!(
                    "density_perturbation: tau ({}) and sigma ({}) must be non-negative",
                    cfg.tau, cfg.sigma
                )));
            }
            Some(cfg)
        }
    } else {
        None
    };

    let sampling = if let Some(s) = &mc.sampling {
        dispersions::SamplingMethod::from_str(s)?
    } else {
        dispersions::SamplingMethod::Random
    };

    Ok(DispersionConfig {
        seed: mc.seed,
        sampling,
        initial_state,
        atmosphere,
        aerodynamics,
        navigation,
        mass,
        vehicle,
        pilot,
        nav_filter,
        wind,
        density_perturbation,
    })
}

/// Validate `output_parameterization` against mode and architecture constraints.
/// Extracted so tests can call it directly without a full `TomlConfig`.
fn validate_output_parameterization(
    output_param: Option<&str>,
    neural_mode: guidance_params::NeuralNetMode,
    architecture: Option<&[TomlLayerSpec]>,
) -> Result<(), DataError> {
    use guidance_params::NeuralNetMode::*;
    let param = match output_param {
        Some(p) => p,
        None => return Ok(()),
    };

    // acos_tanh is a magnitude decoder; the new scaled_pi/delta are signed.
    if param == "acos_tanh" && neural_mode != MagnitudeOnly {
        return Err(DataError(
            "output_parameterization='acos_tanh' is only legal with mode='magnitude_only' \
             (it cannot emit signed bank); use 'atan2_signed' for full_neural mode"
                .to_string(),
        ));
    }
    // atan2_signed has no architecture constraint and is the historical default;
    // only the new signed decoders need the full_neural assertion here.
    if matches!(param, "scaled_pi" | "delta") && neural_mode != FullNeural {
        return Err(DataError(format!(
            "output_parameterization='{}' is only legal with mode='full_neural' \
             (it emits a signed bank; magnitude_only expects an unsigned magnitude)",
            param
        )));
    }

    // Single-output tanh-head decoders share the architecture constraints.
    let needs_single_tanh = matches!(param, "acos_tanh" | "scaled_pi" | "delta");
    if !needs_single_tanh {
        return Ok(());
    }

    let arch = architecture.ok_or_else(|| {
        DataError(format!(
            "output_parameterization='{}' requires v2 [[network.architecture]] entries; \
             v1 layer_sizes/activations is not supported. Convert your config to use \
             [[network.architecture]] with last-layer output_size=1, activation='tanh'.",
            param
        ))
    })?;
    {
        let last = arch.last().ok_or_else(|| {
            DataError(format!(
                "output_parameterization='{}' requires [[network.architecture]] entries",
                param
            ))
        })?;
        match last {
            TomlLayerSpec::Dense {
                output_size,
                activation,
                ..
            } => {
                if *output_size != 1 {
                    return Err(DataError(format!(
                        "output_parameterization='{}' requires last layer \
                         output_size=1, got {output_size}",
                        param
                    )));
                }
                if activation.as_str() != "tanh" {
                    return Err(DataError(format!(
                        "output_parameterization='{}' requires last-layer \
                         activation='tanh', got {:?}",
                        param, activation
                    )));
                }
            }
            _ => {
                return Err(DataError(format!(
                    "output_parameterization='{}' requires last layer to be dense",
                    param
                )));
            }
        }
    }
    Ok(())
}

/// Lines whose first whitespace-delimited token parses as f64 are data lines.
/// D-notation (1.23D+04) is handled by replacing D/d with E/e.
pub fn parse_data_file(path: &str) -> Result<Vec<Vec<f64>>, DataError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| DataError(format!("Cannot read {}: {}", path, e)))?;

    let mut rows = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let first = trimmed.split_whitespace().next().unwrap_or("");
        let first_norm = first.replace('D', "E").replace('d', "e");
        if first_norm.parse::<f64>().is_ok() {
            let values: Vec<f64> = trimmed
                .split_whitespace()
                .filter_map(|t| {
                    let norm = t.replace('D', "E").replace('d', "e");
                    norm.parse::<f64>().ok()
                })
                .collect();
            if !values.is_empty() {
                rows.push(values);
            }
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Create a unique temp directory for test files.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aerocapture_test_{}", name));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    /// Write content to a temp file and return its path as a String.
    fn write_temp_file(dir: &Path, filename: &str, content: &str) -> String {
        let path = dir.join(filename);
        fs::write(&path, content).expect("failed to write temp file");
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn parse_data_file_skips_comments() {
        let dir = temp_dir("skips_comments");
        let path = write_temp_file(
            &dir,
            "test.dat",
            "# Header\n  text line\n1.0 2.0 3.0\n4.0D+01 5.0 6.0\n",
        );

        let rows = parse_data_file(&path).expect("parse failed");
        assert_eq!(rows.len(), 2, "expected 2 data rows, got {}", rows.len());
        assert_eq!(rows[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(
            rows[1],
            vec![40.0, 5.0, 6.0],
            "D-notation 4.0D+01 should become 40.0"
        );

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_data_file_handles_d_notation() {
        let dir = temp_dir("d_notation");
        let path = write_temp_file(&dir, "test.dat", "1.23D+04\n-5.67d-03\n");

        let rows = parse_data_file(&path).expect("parse failed");
        assert_eq!(rows.len(), 2);
        assert_abs_diff_eq!(rows[0][0], 12300.0, epsilon = 1e-10);
        assert_abs_diff_eq!(rows[1][0], -0.00567, epsilon = 1e-10);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_data_file_empty_lines_skipped() {
        let dir = temp_dir("empty_lines");
        let path = write_temp_file(&dir, "test.dat", "\n\n1.0\n\n2.0\n\n");

        let rows = parse_data_file(&path).expect("parse failed");
        assert_eq!(
            rows.len(),
            2,
            "expected 2 rows after skipping blanks, got {}",
            rows.len()
        );
        assert_eq!(rows[0], vec![1.0]);
        assert_eq!(rows[1], vec![2.0]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn spherical_state_default_is_zero() {
        let s = SphericalState::default();
        assert_eq!(s.altitude, 0.0);
        assert_eq!(s.longitude, 0.0);
        assert_eq!(s.latitude, 0.0);
        assert_eq!(s.velocity, 0.0);
        assert_eq!(s.flight_path, 0.0);
        assert_eq!(s.azimuth, 0.0);
    }

    // ─── output_parameterization validation tests ───

    fn valid_arch_1out_tanh() -> Vec<TomlLayerSpec> {
        vec![
            TomlLayerSpec::Dense {
                input_size: 16,
                output_size: 32,
                activation: "tanh".to_string(),
            },
            TomlLayerSpec::Dense {
                input_size: 32,
                output_size: 1,
                activation: "tanh".to_string(),
            },
        ]
    }

    #[test]
    fn acos_tanh_with_full_neural_mode_rejects() {
        let arch = valid_arch_1out_tanh();
        let err = validate_output_parameterization(
            Some("acos_tanh"),
            guidance_params::NeuralNetMode::FullNeural,
            Some(&arch),
        )
        .unwrap_err();
        assert!(
            err.0.contains("acos_tanh") && err.0.contains("magnitude_only"),
            "error should mention acos_tanh and magnitude_only, got: {}",
            err.0
        );
    }

    #[test]
    fn acos_tanh_with_output_size_2_rejects() {
        let arch = vec![TomlLayerSpec::Dense {
            input_size: 16,
            output_size: 2,
            activation: "tanh".to_string(),
        }];
        let err = validate_output_parameterization(
            Some("acos_tanh"),
            guidance_params::NeuralNetMode::MagnitudeOnly,
            Some(&arch),
        )
        .unwrap_err();
        assert!(
            err.0.contains("output_size=1"),
            "error should mention output_size=1, got: {}",
            err.0
        );
    }

    #[test]
    fn acos_tanh_with_wrong_activation_rejects() {
        let arch = vec![TomlLayerSpec::Dense {
            input_size: 16,
            output_size: 1,
            activation: "asinh".to_string(),
        }];
        let err = validate_output_parameterization(
            Some("acos_tanh"),
            guidance_params::NeuralNetMode::MagnitudeOnly,
            Some(&arch),
        )
        .unwrap_err();
        assert!(
            err.0.contains("tanh"),
            "error should mention tanh, got: {}",
            err.0
        );
    }

    #[test]
    fn acos_tanh_with_valid_config_accepts() {
        let arch = valid_arch_1out_tanh();
        validate_output_parameterization(
            Some("acos_tanh"),
            guidance_params::NeuralNetMode::MagnitudeOnly,
            Some(&arch),
        )
        .expect("valid acos_tanh config should pass validation");
    }

    #[test]
    fn no_output_parameterization_always_accepts() {
        // None always passes regardless of mode or architecture.
        validate_output_parameterization(None, guidance_params::NeuralNetMode::FullNeural, None)
            .expect("None output_param should always be valid");
    }

    #[test]
    fn acos_tanh_with_v1_architecture_is_rejected() {
        // v1 path (architecture=None) used to silently accept acos_tanh,
        // which combined with the JSON-load NaN bug produced silent runtime
        // NaN trajectories. Now must error with a clear remediation message.
        let err = validate_output_parameterization(
            Some("acos_tanh"),
            guidance_params::NeuralNetMode::MagnitudeOnly,
            None,
        )
        .expect_err("acos_tanh + v1 architecture should be rejected");
        let msg = format!("{:?}", err);
        assert!(msg.contains("v2"), "expected v2 hint, got: {}", msg);
        assert!(
            msg.contains("[[network.architecture]]"),
            "expected architecture hint, got: {}",
            msg
        );
    }

    fn one_output_tanh_arch() -> Vec<TomlLayerSpec> {
        valid_arch_1out_tanh()
    }

    fn one_output_linear_arch() -> Vec<TomlLayerSpec> {
        vec![
            TomlLayerSpec::Dense {
                input_size: 16,
                output_size: 32,
                activation: "tanh".to_string(),
            },
            TomlLayerSpec::Dense {
                input_size: 32,
                output_size: 1,
                activation: "linear".to_string(),
            },
        ]
    }

    #[test]
    fn scaled_pi_with_magnitude_only_rejects() {
        let err = validate_output_parameterization(
            Some("scaled_pi"),
            guidance_params::NeuralNetMode::MagnitudeOnly,
            Some(&one_output_tanh_arch()),
        )
        .unwrap_err();
        assert!(err.0.contains("scaled_pi") && err.0.contains("full_neural"));
    }

    #[test]
    fn delta_with_magnitude_only_rejects() {
        let err = validate_output_parameterization(
            Some("delta"),
            guidance_params::NeuralNetMode::MagnitudeOnly,
            Some(&one_output_tanh_arch()),
        )
        .unwrap_err();
        assert!(err.0.contains("delta") && err.0.contains("full_neural"));
    }

    #[test]
    fn delta_with_full_neural_and_tanh_accepts() {
        assert!(
            validate_output_parameterization(
                Some("delta"),
                guidance_params::NeuralNetMode::FullNeural,
                Some(&one_output_tanh_arch()),
            )
            .is_ok()
        );
    }

    #[test]
    fn scaled_pi_with_full_neural_and_tanh_accepts() {
        assert!(
            validate_output_parameterization(
                Some("scaled_pi"),
                guidance_params::NeuralNetMode::FullNeural,
                Some(&one_output_tanh_arch()),
            )
            .is_ok()
        );
    }

    #[test]
    fn scaled_pi_with_linear_last_activation_rejects() {
        let err = validate_output_parameterization(
            Some("scaled_pi"),
            guidance_params::NeuralNetMode::FullNeural,
            Some(&one_output_linear_arch()),
        )
        .unwrap_err();
        assert!(err.0.contains("tanh") || err.0.contains("activation"));
    }
}
