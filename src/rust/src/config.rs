//! Parse TOML configuration files + data file suffixes.

use serde::Deserialize;
use std::fmt;

/// Mission type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MissionType {
    Aerocapture,
}

/// Planet identifier
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Planet {
    Moon,
    Earth,
    Mars,
    Jupiter,
}

impl Planet {
    /// Mean equatorial radius in meters
    pub fn equatorial_radius(&self) -> f64 {
        match self {
            Planet::Moon => 6.0518e6,
            Planet::Earth => 6.378137e6,
            Planet::Mars => 3.39394e6,
            Planet::Jupiter => 71.492e6,
        }
    }

    /// Polar radius in meters (must match Fortran lectci.f)
    pub fn polar_radius(&self) -> f64 {
        match self {
            Planet::Moon => 6.0518e6,
            Planet::Earth => 6.356784e6,
            Planet::Mars => 3.376780e6,
            Planet::Jupiter => 66.854e6,
        }
    }

    /// Gravitational parameter mu = GM (m^3/s^2)
    pub fn mu(&self) -> f64 {
        match self {
            Planet::Moon => 3.249e14,
            Planet::Earth => 3.98600418e14,
            Planet::Mars => 4.282829e13,
            Planet::Jupiter => 1.26686e17,
        }
    }

    /// J2 gravitational harmonic coefficient
    pub fn j2(&self) -> f64 {
        match self {
            Planet::Moon => 4.458e-6,
            Planet::Earth => 1.08263e-3,
            Planet::Mars => 1.958616e-3,
            Planet::Jupiter => 14.736e-3,
        }
    }

    /// Rotation rate (rad/s) — must match Fortran lectci.f
    pub fn omega(&self) -> f64 {
        match self {
            Planet::Moon => 2.9924e-7,
            Planet::Earth => 7.292115e-5,
            Planet::Mars => 7.088218e-5,
            Planet::Jupiter => 1.759e-4,
        }
    }
}

/// Simulation phase type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimPhase {
    Full,
    CaptureOnly,
    ExitOnly,
    Preprogrammed,
}

/// Output format for simulation results
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// CSV with named column headers (default)
    #[default]
    Csv,
    /// Legacy Fortran D-notation text format (for regression tests)
    Text,
}

/// Guidance type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GuidanceType {
    Ftc,
    NeuralNetwork,
    EquilibriumGlide,
    EnergyController,
    PredGuid,
    Fnpag,
}

/// Parsed simulation input configuration
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SimInput {
    pub mission_type: MissionType,
    pub planet: Planet,
    pub n_sims: i32,
    pub sim_phase: SimPhase,
    pub guidance_type: GuidanceType,
    pub stats_only: bool,
    pub save_results: bool,
    pub visualize_sim: i32,
    pub screen_output: bool,
    pub random_seed: f64,
    pub reference_trajectory: bool,
    pub reference_bank_angle: f64, // degrees
    pub base_dir: String,
    pub output_dir: String,
    pub output_format: OutputFormat,
    pub results_suffix: String,
}

// ─── TOML deserialization structs ───

#[derive(Debug, Deserialize)]
pub struct TomlConfig {
    pub mission: TomlMission,
    pub guidance: TomlGuidance,
    #[serde(default)]
    pub simulation: TomlSimulation,
    pub data: TomlData,

    // Inline data sections (consolidated mode — replaces 10 external files)
    pub vehicle: Option<TomlVehicle>,
    pub entry: Option<TomlEntry>,
    pub aerodynamics: Option<TomlAero>,
    pub flight: Option<TomlFlight>,
    pub success: Option<TomlSuccess>,
    pub incidence: Option<TomlIncidence>,
    // Domain-based Monte Carlo config (consolidated mode)
    pub monte_carlo: Option<TomlMonteCarlo>,
}

#[derive(Debug, Deserialize)]
pub struct TomlMission {
    #[serde(rename = "type")]
    pub mission_type: String,
    pub planet: String,
    #[serde(default = "default_phase")]
    pub phase: String,
}

fn default_phase() -> String {
    "full".to_string()
}

#[derive(Debug, Deserialize)]
pub struct TomlGuidance {
    #[serde(rename = "type")]
    pub guidance_type: String,
    #[serde(default)]
    pub reference_trajectory: bool,
    #[serde(default = "default_ref_bank")]
    pub reference_bank_angle: f64,
    /// FTC-specific parameters (consolidated mode, from guidage.* files)
    pub ftc: Option<TomlFtcParams>,
    /// Equilibrium glide parameters
    pub equilibrium_glide: Option<TomlEqGlideParams>,
    /// Energy controller parameters
    pub energy_controller: Option<TomlEnergyCtrlParams>,
    /// PredGuid (drag tracking) parameters
    pub pred_guid: Option<TomlPredGuidParams>,
    /// FNPAG (numerical predictor-corrector) parameters
    pub fnpag: Option<TomlFnpagParams>,
}

fn default_ref_bank() -> f64 {
    0.0
}

#[derive(Debug, Deserialize, Default)]
pub struct TomlSimulation {
    #[serde(default = "default_one_i32")]
    pub n_sims: i32,
    #[serde(default)]
    pub random_seed: f64,
    #[serde(default)]
    pub screen_output: bool,
    #[serde(default)]
    pub stats_only: bool,
    #[serde(default = "default_true")]
    pub save_results: bool,
    #[serde(default)]
    pub visualize_sim: i32,
}

fn default_one_i32() -> i32 {
    1
}
fn default_one() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct TomlData {
    #[serde(default = "default_base_dir")]
    pub base_dir: String,
    #[serde(default = "default_output_dir")]
    pub output_dir: String,
    // Direct file paths for external data
    pub atmosphere: Option<String>,
    pub reference_trajectory: Option<String>,
    pub neural_network: Option<String>,
    pub results_suffix: Option<String>,
    #[serde(default)]
    pub output_format: OutputFormat,
}

fn default_base_dir() -> String {
    "data".to_string()
}
fn default_output_dir() -> String {
    "output".to_string()
}

// ─── Inline data TOML structs (consolidated mode) ───

#[derive(Debug, Deserialize, Clone)]
pub struct TomlVehicle {
    pub mass: f64,           // kg
    pub reference_area: f64, // m²
    pub cq: f64,             // heat flux coefficient
    pub max_bank_rate: f64,  // deg/s
    #[serde(default)]
    pub periods: TomlPeriods,
    #[serde(default)]
    pub pilot: TomlPilot,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlPeriods {
    #[serde(default = "default_one")]
    pub navigation: f64,
    #[serde(default = "default_one")]
    pub guidance: f64,
    #[serde(default = "default_one")]
    pub pilot: f64,
    #[serde(default = "default_one")]
    pub prediction: f64,
    #[serde(default = "default_one")]
    pub integration: f64,
    #[serde(default = "default_one")]
    pub photo: f64,
}

impl Default for TomlPeriods {
    fn default() -> Self {
        Self {
            navigation: 1.0,
            guidance: 1.0,
            pilot: 1.0,
            prediction: 1.0,
            integration: 1.0,
            photo: 1.0,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlPilot {
    #[serde(default = "default_pilot_model")]
    pub model: String,
    #[serde(default = "default_one")]
    pub time_constant: f64,
    #[serde(default = "default_pilot_damping")]
    pub damping: f64,
    #[serde(default = "default_pilot_freq")]
    pub frequency: f64,
}

fn default_pilot_model() -> String {
    "perfect".to_string()
}
fn default_pilot_damping() -> f64 {
    0.7
}
fn default_pilot_freq() -> f64 {
    0.072
}

impl Default for TomlPilot {
    fn default() -> Self {
        Self {
            model: "perfect".to_string(),
            time_constant: 1.0,
            damping: 0.7,
            frequency: 0.072,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlEntry {
    pub altitude: f64, // km
    #[serde(default)]
    pub longitude: f64, // deg
    #[serde(default)]
    pub latitude: f64, // deg
    pub velocity: f64, // m/s
    pub flight_path_angle: f64, // deg
    pub azimuth: f64,  // deg
    #[serde(default)]
    pub initial_time: f64, // s
    pub initial_bank_angle: f64, // deg
    pub initial_aoa: f64, // deg
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlAeroPoint {
    pub aoa: f64, // deg
    pub ca: f64,  // axial force coeff (body axis)
    pub cn: f64,  // normal force coeff (body axis)
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlAero {
    pub equilibrium_aoa: f64, // deg
    pub points: Vec<TomlAeroPoint>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlFlight {
    #[serde(default)]
    pub wind: bool,
    pub constraints: TomlConstraints,
    pub final_conditions: TomlFinalConditions,
    pub target_orbit: TomlTargetOrbit,
    pub parking_orbit: TomlParkingOrbit,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlConstraints {
    pub max_heat_flux: f64,        // kW/m²
    pub max_load_factor: f64,      // g
    pub max_dynamic_pressure: f64, // kPa
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlFinalConditions {
    pub altitude: f64,          // km
    pub longitude: f64,         // deg
    pub latitude: f64,          // deg
    pub velocity: f64,          // m/s
    pub flight_path_angle: f64, // deg
    pub azimuth: f64,           // deg
    pub energy: f64,            // MJ/kg
    pub radial_velocity: f64,   // m/s
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlTargetOrbit {
    pub apoapsis: f64,        // km
    pub periapsis: f64,       // km
    pub semi_major_axis: f64, // km
    pub eccentricity: f64,
    pub inclination: f64, // deg
    pub raan: f64,        // deg
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlParkingOrbit {
    pub apoapsis: f64,  // km
    pub periapsis: f64, // km
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlSuccess {
    pub inclination_tolerance: f64, // deg
    pub velocity_tolerance: f64,    // m/s
    pub apoapsis_tolerance: f64,    // km
    pub periapsis_tolerance: f64,   // km
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlIncidence {
    pub altitudes: Vec<f64>, // km
    pub angles: Vec<f64>,    // deg
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlFtcParams {
    pub capture_damping: f64,
    pub capture_frequency: f64, // rad/s
    pub capture_pdyn_margin: f64,
    pub altitude_damping: f64,
    pub altitude_frequency: f64,      // deg/s (converted to rad/s)
    pub exit_velocity_threshold: f64, // m/s
    pub exit_pdyn_margin: f64,
    pub exit_altitude_threshold: f64, // km
    pub exit_radial_vel_gain: f64,    // Pa/(m/s)
    pub exit_apoapsis_threshold: f64, // m
    pub corridor_slope: f64,          // m/s
    #[serde(default)]
    pub corridor_intercept: f64, // deg
    #[serde(default = "default_five_i32")]
    pub max_reversals: i32,
    #[serde(default = "default_one_i32")]
    pub security_capture: i32,
    #[serde(default = "default_three_i32")]
    pub security_exit: i32,
    pub density_filter_gain: f64,
    #[serde(default = "default_longi_act")]
    pub longi_activation: f64, // MJ/kg
    #[serde(default = "default_longi_inh")]
    pub longi_inhibition: f64, // MJ/kg
    pub lateral_activation: f64, // MJ/kg
    #[serde(default = "default_longi_act")]
    pub lateral_inhibition: f64, // MJ/kg
    #[serde(default)]
    pub pdyn_min: f64, // Pa
    #[serde(default)]
    pub pdyn_table: Vec<TomlPdynEntry>,
}

fn default_five_i32() -> i32 {
    5
}
fn default_three_i32() -> i32 {
    3
}
fn default_longi_act() -> f64 {
    1000.0
}
fn default_longi_inh() -> f64 {
    -1000.0
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlPdynEntry {
    pub altitude: f64,
    pub a: f64,
    pub b: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlEqGlideParams {
    #[serde(default = "default_0_3")]
    pub k_hdot_scale: f64,
    #[serde(default = "default_1_1")]
    pub v_ratio_threshold: f64,
    #[serde(default = "default_0_15")]
    pub velocity_bias_high: f64,
    #[serde(default = "default_0_3")]
    pub velocity_bias_low: f64,
    #[serde(default = "default_40")]
    pub alt_bias_threshold: f64,
    #[serde(default = "default_neg_0_5")]
    pub cos_bank_min: f64,
    #[serde(default = "default_0_95")]
    pub cos_bank_max: f64,
}

fn default_0_3() -> f64 {
    0.3
}
fn default_1_1() -> f64 {
    1.1
}
fn default_0_15() -> f64 {
    0.15
}
fn default_40() -> f64 {
    40.0
}
fn default_neg_0_5() -> f64 {
    -0.5
}
fn default_0_95() -> f64 {
    0.95
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlEnergyCtrlParams {
    #[serde(default = "default_5e_7")]
    pub gain: f64,
    #[serde(default = "default_one")]
    pub kp: f64,
    #[serde(default = "default_0_5")]
    pub kd: f64,
}

fn default_5e_7() -> f64 {
    5e-7
}
fn default_0_5() -> f64 {
    0.5
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlPredGuidParams {
    #[serde(default = "default_0_8")]
    pub k_drag_high: f64,
    #[serde(default = "default_0_3")]
    pub k_drag_low: f64,
    #[serde(default = "default_100")]
    pub pdyn_threshold: f64,
}

fn default_0_8() -> f64 {
    0.8
}
fn default_100() -> f64 {
    100.0
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlFnpagParams {
    #[serde(default = "default_1e4")]
    pub energy_tol: f64,
    #[serde(default = "default_2")]
    pub prediction_dt: f64,
    #[serde(default = "default_20")]
    pub bank_min_deg: f64,
    #[serde(default = "default_140")]
    pub bank_max_high_deg: f64,
    #[serde(default = "default_100")]
    pub bank_max_low_deg: f64,
}

fn default_1e4() -> f64 {
    1e4
}
fn default_2() -> f64 {
    2.0
}
fn default_20() -> f64 {
    20.0
}
fn default_140() -> f64 {
    140.0
}

// ─── Domain-based Monte Carlo TOML structs ───

#[derive(Debug, Deserialize, Clone)]
pub struct TomlMonteCarlo {
    pub seed: u64,
    pub initial_state: Option<TomlMcDomain>,
    pub atmosphere: Option<TomlMcDomain>,
    pub aerodynamics: Option<TomlMcDomain>,
    pub navigation: Option<TomlMcDomain>,
    pub mass: Option<TomlMcDomain>,
    pub vehicle: Option<TomlMcDomain>,
    pub pilot: Option<TomlMcDomain>,
    pub nav_filter: Option<TomlMcDomain>,
}

/// A single dispersion domain config.
/// `level` selects a preset ("off", "low", "medium", "high", "custom").
/// Custom sigma overrides are only used when level = "custom".
#[derive(Debug, Deserialize, Clone)]
pub struct TomlMcDomain {
    #[serde(default = "default_mc_level")]
    pub level: String,
    // Custom overrides — field names match the domain, optional
    #[serde(flatten)]
    pub custom: std::collections::HashMap<String, f64>,
}

fn default_mc_level() -> String {
    "medium".to_string()
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

impl SimInput {
    /// Parse a TOML configuration string. Returns (SimInput, TomlConfig).
    /// The TomlConfig is needed for inline data loading in consolidated mode.
    pub fn from_toml(content: &str) -> Result<(Self, TomlConfig), ParseError> {
        let config: TomlConfig =
            toml::from_str(content).map_err(|e| ParseError(format!("TOML parse error: {}", e)))?;

        let mission_type = match config.mission.mission_type.as_str() {
            "aerocapture" => MissionType::Aerocapture,
            other => return Err(ParseError(format!("Unknown mission type: {}", other))),
        };

        let planet = match config.mission.planet.as_str() {
            "moon" => Planet::Moon,
            "earth" => Planet::Earth,
            "mars" => Planet::Mars,
            "jupiter" => Planet::Jupiter,
            other => return Err(ParseError(format!("Unknown planet: {}", other))),
        };

        let sim_phase = match config.mission.phase.as_str() {
            "full" => SimPhase::Full,
            "capture_only" => SimPhase::CaptureOnly,
            "exit_only" => SimPhase::ExitOnly,
            "preprogrammed" => SimPhase::Preprogrammed,
            other => return Err(ParseError(format!("Unknown phase: {}", other))),
        };

        let guidance_type = match config.guidance.guidance_type.as_str() {
            "ftc" => GuidanceType::Ftc,
            "neural_network" => GuidanceType::NeuralNetwork,
            "equilibrium_glide" => GuidanceType::EquilibriumGlide,
            "energy_controller" => GuidanceType::EnergyController,
            "pred_guid" => GuidanceType::PredGuid,
            "fnpag" => GuidanceType::Fnpag,
            other => return Err(ParseError(format!("Unknown guidance type: {}", other))),
        };

        let sim_input = SimInput {
            mission_type,
            planet,
            n_sims: config.simulation.n_sims,
            sim_phase,
            guidance_type,
            stats_only: config.simulation.stats_only,
            save_results: config.simulation.save_results,
            visualize_sim: config.simulation.visualize_sim,
            screen_output: config.simulation.screen_output,
            random_seed: config.simulation.random_seed,
            reference_trajectory: config.guidance.reference_trajectory,
            reference_bank_angle: config.guidance.reference_bank_angle,
            base_dir: config.data.base_dir.clone(),
            output_dir: config.data.output_dir.clone(),
            output_format: config.data.output_format,
            results_suffix: config
                .data
                .results_suffix
                .clone()
                .unwrap_or_else(|| ".out".to_string()),
        };

        Ok((sim_input, config))
    }

    /// Build an output file path
    pub fn output_path(&self, filename: &str) -> String {
        format!("{}/{}", self.output_dir, filename)
    }
}
