//! Data file loading and simulation data structures.
//!
//! Matches the Fortran lectci.f reading order exactly.

pub mod aerodynamics;
pub mod atmosphere;
pub mod capsule;
pub mod dispersions;
pub mod guidance_params;
pub mod incidence;
pub mod navigation;
pub mod neural;
pub mod pilot;

use crate::config::{GuidanceType, MissionType, SimInput, TomlConfig, TomlMonteCarlo};
use std::fmt;

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

/// AGA-specific parameters
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AgaParams {
    pub v_infinity: f64,   // m/s
    pub true_anomaly: f64, // radians
}

/// All loaded simulation data
#[allow(dead_code)]
#[derive(Debug)]
pub struct SimData {
    pub capsule: capsule::Capsule,
    pub aero: aerodynamics::AeroTables,
    pub atmosphere: atmosphere::AtmosphereModel,
    pub entry: EntryConditions,
    pub constraints: Constraints,
    pub final_conditions: FinalConditions,
    pub target_orbit: OrbitalTarget,
    pub parking_orbit: ParkingOrbit,
    pub periods: TimePeriods,
    pub guidance: guidance_params::GuidanceParams,
    pub dispersions: dispersions::DispersionParams,
    pub navigation: navigation::NavigationParams,
    pub incidence: incidence::IncidenceProfile,
    pub pilot: pilot::PilotModel,
    pub success: SuccessCriteria,
    pub wind_enabled: bool,
    pub aga: Option<AgaParams>,
    pub neural_net: Option<neural::NeuralNetParams>,
    /// Domain-based dispersion config (replaces lottery files when present)
    pub dispersion_config: Option<dispersions::DispersionConfig>,
}

const G0: f64 = 9.81;
const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

impl SimData {
    /// Load all simulation data from files based on configuration suffixes.
    ///
    /// Matches the Fortran lectci.f reading order.
    pub fn load(config: &SimInput) -> Result<Self, DataError> {
        let capsule =
            capsule::Capsule::load(&config.data_path("capsule", &config.suffixes.capsule))?;

        let aero = aerodynamics::AeroTables::load(
            &config.data_path("aerodynamique", &config.suffixes.aero),
        )?;

        let atm = atmosphere::AtmosphereModel::load(
            &config.data_path("atmosphere", &config.suffixes.atmosphere),
        )?;

        let entry = load_entry_conditions(&config.data_path("rentree", &config.suffixes.reentry))?;

        let (constraints, final_cond, target, parking, wind, aga) = load_mission(
            &config.data_path("mission", &config.suffixes.mission),
            config.mission_type,
        )?;

        let periods = capsule.periods;

        let ref_path = config.data_path("tables_energie_gains", &config.suffixes.guidance);
        let guidance = guidance_params::GuidanceParams::load_with_ref(
            &config.data_path("guidage", &config.suffixes.guidance),
            config.mission_type,
            &ref_path,
            config.reference_trajectory,
        )?;

        let dispersions = dispersions::DispersionParams::load(
            &config.data_path("dispersions", &config.suffixes.dispersions),
            &config.dispersion_multipliers,
        )?;

        let navigation = navigation::NavigationParams::load(
            &config.data_path("navigation", &config.suffixes.navigation),
            &config.dispersion_multipliers,
        )?;

        let incidence = incidence::IncidenceProfile::load(
            &config.data_path("incidence", &config.suffixes.incidence),
        )?;

        // Fortran uses sufmsr (capsule suffix) for pilote file
        let pilot = pilot::PilotModel::load(&config.data_path("pilote", &config.suffixes.capsule))?;

        let success = load_success(&config.data_path("succes", &config.suffixes.success))?;

        let neural_net = if config.guidance_type == GuidanceType::NeuralNetwork {
            let nn_path = config.data_path("nn_param", &config.suffixes.neural);
            Some(neural::NeuralNetParams::load(&nn_path)?)
        } else {
            None
        };

        Ok(SimData {
            capsule,
            aero,
            atmosphere: atm,
            entry,
            constraints,
            final_conditions: final_cond,
            target_orbit: target,
            parking_orbit: parking,
            periods,
            guidance,
            dispersions,
            navigation,
            incidence,
            pilot,
            success,
            wind_enabled: wind,
            aga,
            neural_net,
            dispersion_config: None,
        })
    }

    /// Load simulation data from consolidated TOML config (inline data + external files).
    pub fn from_toml(toml: &TomlConfig, config: &SimInput) -> Result<Self, DataError> {
        let v = toml.vehicle.as_ref()
            .ok_or_else(|| DataError("Missing [vehicle] section".to_string()))?;
        let e = toml.entry.as_ref()
            .ok_or_else(|| DataError("Missing [entry] section".to_string()))?;
        let f = toml.flight.as_ref()
            .ok_or_else(|| DataError("Missing [flight] section".to_string()))?;
        let a = toml.aerodynamics.as_ref()
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
        let nominal_finesse = if nominal_cx.abs() > 1e-30 { nominal_cz / nominal_cx } else { 0.0 };
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
            incidence::IncidenceProfile { n_points: 0, altitudes: vec![], incidences: vec![] }
        };

        // FTC guidance params
        let guidance = if let Some(ref ftc) = toml.guidance.ftc {
            let energy_scale = if config.mission_type == MissionType::Aerocapture { 1e6 } else { 1.0 };
            let pdyn_table = ftc.pdyn_table.iter().map(|e| {
                guidance_params::PdynTableEntry {
                    altitude: e.altitude,
                    coeff_a: e.a,
                    coeff_b: e.b,
                }
            }).collect();

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
                corridor_slope: ftc.corridor_slope,
                corridor_intercept: ftc.corridor_intercept * DEG2RAD,
                max_reversals: ftc.max_reversals,
                security_capture: ftc.security_capture,
                security_exit: ftc.security_exit,
                density_filter_gain: ftc.density_filter_gain,
                longi_activation: ftc.longi_activation * energy_scale,
                longi_inhibition: ftc.longi_inhibition * energy_scale,
                lateral_activation: ftc.lateral_activation * energy_scale,
                lateral_inhibition: ftc.lateral_inhibition * energy_scale,
                pdyn_min: ftc.pdyn_min,
                pdyn_table,
                ref_trajectory: ref_traj,
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
                capture_damping: 0.7, capture_frequency: 0.072,
                capture_pdyn_margin: 1.75, altitude_damping: 0.7,
                altitude_frequency: 0.08 * DEG2RAD,
                exit_velocity_threshold: 4400.0, exit_pdyn_margin: 1.75,
                exit_altitude_threshold: 60e3, exit_radial_vel_gain: 10.0,
                exit_apoapsis_threshold: 100.0, corridor_slope: 13080.458,
                corridor_intercept: 0.0, max_reversals: 5,
                security_capture: 1, security_exit: 3,
                density_filter_gain: 0.8,
                longi_activation: 1e9, longi_inhibition: -1e9,
                lateral_activation: 1.311e6, lateral_inhibition: 1e9,
                pdyn_min: 0.0, pdyn_table: vec![],
                ref_trajectory: ref_traj,
            }
        };

        // Dispersions
        let xm = &config.dispersion_multipliers;
        let disp = if let Some(ref d) = toml.initial_dispersions {
            dispersions::DispersionParams {
                altitude: xm[1] * d.altitude * 1e3,
                longitude: xm[1] * d.longitude * DEG2RAD,
                latitude: xm[1] * d.latitude * DEG2RAD,
                velocity: xm[1] * d.velocity,
                flight_path: xm[1] * d.flight_path_angle * DEG2RAD,
                azimuth: xm[1] * d.azimuth * DEG2RAD,
                drag_coeff: xm[3] * d.drag_coeff / 100.0,
                lift_coeff: xm[3] * d.lift_coeff / 100.0,
                density: d.density / 100.0,
                incidence: xm[3] * d.incidence * DEG2RAD,
                mass: d.mass / 100.0,
            }
        } else {
            dispersions::DispersionParams::default()
        };

        // Navigation errors
        let nav = if let Some(ref n) = toml.navigation_errors {
            navigation::NavigationParams {
                altitude: xm[0] * n.altitude * 1e3,
                latitude: xm[0] * n.latitude * DEG2RAD,
                longitude: xm[0] * n.longitude * DEG2RAD,
                velocity: xm[0] * n.velocity,
                flight_path: xm[0] * n.flight_path_angle * DEG2RAD,
                azimuth: xm[0] * n.azimuth * DEG2RAD,
                drag_accel: xm[2] * n.drag_accel,
            }
        } else {
            navigation::NavigationParams::default()
        };

        // Atmosphere (always external)
        let atm_path = toml.data.atmosphere.as_ref()
            .ok_or_else(|| DataError("Missing data.atmosphere path".to_string()))?;
        let atm = atmosphere::AtmosphereModel::load(atm_path)?;

        // Neural network (external, optional)
        let neural_net = if config.guidance_type == GuidanceType::NeuralNetwork {
            if let Some(ref nn_path) = toml.data.neural_network {
                Some(neural::NeuralNetParams::load(nn_path)?)
            } else {
                None
            }
        } else {
            None
        };

        // Domain-based Monte Carlo config (replaces lottery files)
        let dispersion_config = if let Some(ref mc) = toml.monte_carlo {
            Some(build_dispersion_config(mc)?)
        } else {
            None
        };

        Ok(SimData {
            capsule: capsule_data,
            aero,
            atmosphere: atm,
            entry,
            constraints,
            final_conditions,
            target_orbit,
            parking_orbit,
            periods: capsule_data.periods,
            guidance,
            dispersions: disp,
            navigation: nav,
            incidence: incidence_data,
            pilot: pilot_data,
            success,
            wind_enabled: f.wind,
            aga: None,
            neural_net,
            dispersion_config,
        })
    }
}

/// Build a DispersionConfig from TOML [monte_carlo] section.
fn build_dispersion_config(mc: &TomlMonteCarlo) -> Result<dispersions::DispersionConfig, DataError> {
    use dispersions::*;

    let initial_state = mc.initial_state.as_ref().map(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off { return None; }
        let mut s = InitialStateSigmas::from_level(level);
        if level == DispersionLevel::Custom {
            if let Some(&v) = d.custom.get("altitude") { s.altitude = v; }
            if let Some(&v) = d.custom.get("longitude") { s.longitude = v; }
            if let Some(&v) = d.custom.get("latitude") { s.latitude = v; }
            if let Some(&v) = d.custom.get("velocity") { s.velocity = v; }
            if let Some(&v) = d.custom.get("flight_path_angle") { s.flight_path = v; }
            if let Some(&v) = d.custom.get("azimuth") { s.azimuth = v; }
        }
        Some(s)
    }).flatten();

    let atmosphere = mc.atmosphere.as_ref().map(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off { return None; }
        let mut s = AtmosphereSigmas::from_level(level);
        if level == DispersionLevel::Custom {
            if let Some(&v) = d.custom.get("density") { s.density = v; }
        }
        Some(s)
    }).flatten();

    let aerodynamics = mc.aerodynamics.as_ref().map(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off { return None; }
        let mut s = AerodynamicsSigmas::from_level(level);
        if level == DispersionLevel::Custom {
            if let Some(&v) = d.custom.get("drag") { s.drag = v; }
            if let Some(&v) = d.custom.get("lift") { s.lift = v; }
            if let Some(&v) = d.custom.get("incidence") { s.incidence = v; }
        }
        Some(s)
    }).flatten();

    let navigation = mc.navigation.as_ref().map(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off { return None; }
        let mut s = NavigationSigmas::from_level(level);
        if level == DispersionLevel::Custom {
            if let Some(&v) = d.custom.get("altitude") { s.altitude = v; }
            if let Some(&v) = d.custom.get("longitude") { s.longitude = v; }
            if let Some(&v) = d.custom.get("latitude") { s.latitude = v; }
            if let Some(&v) = d.custom.get("velocity") { s.velocity = v; }
            if let Some(&v) = d.custom.get("flight_path_angle") { s.flight_path = v; }
            if let Some(&v) = d.custom.get("azimuth") { s.azimuth = v; }
            if let Some(&v) = d.custom.get("drag_accel") { s.drag_accel = v; }
        }
        Some(s)
    }).flatten();

    let mass = mc.mass.as_ref().map(|d| {
        let level = DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium);
        if level == DispersionLevel::Off { return None; }
        let mut s = MassSigmas::from_level(level);
        if level == DispersionLevel::Custom {
            if let Some(&v) = d.custom.get("mass") { s.mass = v; }
        }
        Some(s)
    }).flatten();

    Ok(DispersionConfig {
        seed: mc.seed,
        initial_state,
        atmosphere,
        aerodynamics,
        navigation,
        mass,
    })
}

/// Parse a data file, skipping comment/header lines.
///
/// Lines whose first whitespace-delimited token parses as f64 are data lines.
/// Fortran D-notation (1.23D+04) is handled by replacing D/d with E/e.
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

/// Load entry conditions from rentree file.
///
/// Fortran order (unit 102):
///   xaltzd (km), xlonzd (deg), xlatzd (deg), xvitzd (m/s),
///   xpenzd (deg), xazmzd (deg), datini (s), gitpre (deg), alfpre (deg)
fn load_entry_conditions(path: &str) -> Result<EntryConditions, DataError> {
    let rows = parse_data_file(path)?;
    if rows.len() < 9 {
        return Err(DataError(format!(
            "Entry file too short ({} rows, need 9): {}",
            rows.len(),
            path
        )));
    }

    Ok(EntryConditions {
        state: SphericalState {
            altitude: rows[0][0] * 1000.0,
            longitude: rows[1][0] * DEG2RAD,
            latitude: rows[2][0] * DEG2RAD,
            velocity: rows[3][0],
            flight_path: rows[4][0] * DEG2RAD,
            azimuth: rows[5][0] * DEG2RAD,
        },
        initial_date: rows[6][0],
        initial_bank: rows[7][0] * DEG2RAD,
        initial_aoa: rows[8][0] * DEG2RAD,
    })
}

/// Load mission parameters.
///
/// Fortran order (unit 101):
///   ivents, conflu (kW/m2), conacc (g), conpdy (kPa),
///   xaltfn (km), xlonfn (deg), xlatfn (deg), xvitfn (m/s),
///   xpenfn (deg), xazmfn (deg), enrjfn (MJ/kg), vitzfn (m/s),
///   zapoge (km), zperig (km), demiax (km), excorb,
///   xincli (deg), gomega (deg), zapotf (km), zpertf (km),
///   [AGA only: vitinf (m/s), anoinf (deg)]
#[allow(clippy::type_complexity)]
fn load_mission(
    path: &str,
    mission_type: MissionType,
) -> Result<
    (
        Constraints,
        FinalConditions,
        OrbitalTarget,
        ParkingOrbit,
        bool,
        Option<AgaParams>,
    ),
    DataError,
> {
    let rows = parse_data_file(path)?;
    if rows.len() < 20 {
        return Err(DataError(format!(
            "Mission file too short ({} rows, need 20): {}",
            rows.len(),
            path
        )));
    }

    let wind = rows[0][0] as i32 != 0;

    let constraints = Constraints {
        max_heat_flux: rows[1][0] * 1e3,        // kW/m2 → W/m2
        max_load_factor: rows[2][0] * G0,       // g → m/s2
        max_dynamic_pressure: rows[3][0] * 1e3, // kPa → Pa
    };

    let final_cond = FinalConditions {
        altitude: rows[4][0] * 1e3,
        longitude: rows[5][0] * DEG2RAD,
        latitude: rows[6][0] * DEG2RAD,
        velocity: rows[7][0],
        flight_path: rows[8][0] * DEG2RAD,
        azimuth: rows[9][0] * DEG2RAD,
        energy: rows[10][0] * 1e6, // MJ/kg → J/kg
        radial_vel: rows[11][0],
    };

    let target = OrbitalTarget {
        apoapsis: rows[12][0] * 1e3,
        periapsis: rows[13][0] * 1e3,
        semi_major_axis: rows[14][0] * 1e3,
        eccentricity: rows[15][0],
        inclination: rows[16][0] * DEG2RAD,
        raan: rows[17][0] * DEG2RAD,
    };

    let parking = ParkingOrbit {
        apoapsis: rows[18][0] * 1e3,
        periapsis: rows[19][0] * 1e3,
    };

    let aga = if mission_type == MissionType::AeroGravityAssist && rows.len() >= 22 {
        Some(AgaParams {
            v_infinity: rows[20][0],
            true_anomaly: rows[21][0] * DEG2RAD,
        })
    } else {
        None
    };

    Ok((constraints, final_cond, target, parking, wind, aga))
}

/// Load success criteria.
///
/// Fortran order (unit 111):
///   errinc (deg), errvit (m/s), errzap (km), errzpe (km)
fn load_success(path: &str) -> Result<SuccessCriteria, DataError> {
    let rows = parse_data_file(path)?;
    if rows.len() < 4 {
        return Err(DataError(format!(
            "Success file too short ({} rows, need 4): {}",
            rows.len(),
            path
        )));
    }

    Ok(SuccessCriteria {
        inclination_tol: rows[0][0] * DEG2RAD,
        velocity_tol: rows[1][0],
        apoapsis_tol: rows[2][0] * 1e3,
        periapsis_tol: rows[3][0] * 1e3,
    })
}
