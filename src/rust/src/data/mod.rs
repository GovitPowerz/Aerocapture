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

use crate::config::{GuidanceType, MissionType, SimInput};
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
#[derive(Debug, Clone, Copy, Default)]
pub struct OrbitalElements {
    pub semi_major_axis: f64, // meters
    pub eccentricity: f64,
    pub inclination: f64,  // radians
    pub raan: f64,         // radians
    pub arg_periapsis: f64, // radians
    pub true_anomaly: f64, // radians
    pub periapsis_alt: f64, // meters
    pub apoapsis_alt: f64, // meters
}

/// Target orbital parameters (from mission file)
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
    pub initial_date: f64,  // seconds
    pub initial_bank: f64,  // radians (gite)
    pub initial_aoa: f64,   // radians (incidence)
}

/// Reentry constraints (converted to SI)
#[derive(Debug, Clone, Copy, Default)]
pub struct Constraints {
    pub max_heat_flux: f64,        // W/m^2 (from kW/m^2)
    pub max_load_factor: f64,      // m/s^2 (from g, multiplied by g0=9.81)
    pub max_dynamic_pressure: f64, // Pa (from kPa)
}

/// Success criteria
#[derive(Debug, Clone, Copy, Default)]
pub struct SuccessCriteria {
    pub inclination_tol: f64, // radians (from deg)
    pub velocity_tol: f64,    // m/s
    pub apoapsis_tol: f64,    // meters (from km)
    pub periapsis_tol: f64,   // meters (from km)
}

/// AGA-specific parameters
#[derive(Debug, Clone, Copy, Default)]
pub struct AgaParams {
    pub v_infinity: f64,    // m/s
    pub true_anomaly: f64,  // radians
}

/// All loaded simulation data
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
}

const G0: f64 = 9.81;
const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

impl SimData {
    /// Load all simulation data from files based on configuration suffixes.
    ///
    /// Matches the Fortran lectci.f reading order.
    pub fn load(config: &SimInput) -> Result<Self, DataError> {
        let capsule = capsule::Capsule::load(
            &config.data_path("capsule", &config.suffixes.capsule),
        )?;

        let aero = aerodynamics::AeroTables::load(
            &config.data_path("aerodynamique", &config.suffixes.aero),
        )?;

        let atm = atmosphere::AtmosphereModel::load(
            &config.data_path("atmosphere", &config.suffixes.atmosphere),
        )?;

        let entry = load_entry_conditions(
            &config.data_path("rentree", &config.suffixes.reentry),
        )?;

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
        let pilot = pilot::PilotModel::load(
            &config.data_path("pilote", &config.suffixes.capsule),
        )?;

        let success = load_success(
            &config.data_path("succes", &config.suffixes.success),
        )?;

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
        })
    }
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
fn load_mission(
    path: &str,
    mission_type: MissionType,
) -> Result<(Constraints, FinalConditions, OrbitalTarget, ParkingOrbit, bool, Option<AgaParams>), DataError> {
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
        max_load_factor: rows[2][0] * G0,        // g → m/s2
        max_dynamic_pressure: rows[3][0] * 1e3,  // kPa → Pa
    };

    let final_cond = FinalConditions {
        altitude: rows[4][0] * 1e3,
        longitude: rows[5][0] * DEG2RAD,
        latitude: rows[6][0] * DEG2RAD,
        velocity: rows[7][0],
        flight_path: rows[8][0] * DEG2RAD,
        azimuth: rows[9][0] * DEG2RAD,
        energy: rows[10][0] * 1e6,    // MJ/kg → J/kg
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
