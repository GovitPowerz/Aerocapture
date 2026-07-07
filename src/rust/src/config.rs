//! Parse TOML configuration files + data file suffixes.

use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::path::Path;

/// Mission type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MissionType {
    Aerocapture,
}

/// Planet physical constants, parsed from TOML [planet] section.
#[derive(Debug, Clone, Deserialize)]
pub struct PlanetConfig {
    pub name: String,
    pub mu: f64,
    pub equatorial_radius: f64,
    pub polar_radius: f64,
    pub omega: f64,
    pub j2: f64,
    #[serde(default)]
    pub j3: f64,
    #[serde(default)]
    pub j4: f64,
}

#[cfg(test)]
impl PlanetConfig {
    pub fn mars() -> Self {
        Self {
            name: "mars".into(),
            mu: 4.282829e13,
            equatorial_radius: 3393940.0,
            polar_radius: 3376780.0,
            omega: 7.088218e-5,
            j2: 1.958616e-3,
            j3: 3.145e-5,
            j4: -1.538e-5,
        }
    }

    pub fn earth() -> Self {
        Self {
            name: "earth".into(),
            mu: 3.98600418e14,
            equatorial_radius: 6378137.0,
            polar_radius: 6356784.0,
            omega: 7.292115e-5,
            j2: 1.08263e-3,
            j3: -2.5327e-6,
            j4: -1.6196e-6,
        }
    }

    pub fn moon() -> Self {
        Self {
            name: "moon".into(),
            mu: 3.249e14,
            equatorial_radius: 6051800.0,
            polar_radius: 6051800.0,
            omega: 2.9924e-7,
            j2: 4.458e-6,
            j3: 0.0,
            j4: 0.0,
        }
    }

    /// Mars-like planet with J3=J4=0 for backward-compat tests.
    pub fn mars_j2_only() -> Self {
        Self {
            j3: 0.0,
            j4: 0.0,
            ..Self::mars()
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

/// Adaptive integration configuration.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveConfig {
    pub rtol: f64,       // relative tolerance
    pub initial_dt: f64, // initial sub-step guess (seconds)
    pub min_dt: f64,     // floor (seconds)
    pub max_dt: f64,     // ceiling (seconds)
}

/// Integration method selection.
#[derive(Debug, Clone, Copy, Default)]
pub enum IntegrationMode {
    /// Fixed-step Gill-variant RK4 (legacy, default).
    #[default]
    FixedGill,
    /// Adaptive Dormand-Prince 4(5) with error control.
    AdaptiveDopri45(AdaptiveConfig),
}

impl IntegrationMode {
    /// Build from TOML config. `integration_period` is the outer tick dt from [vehicle.periods].
    pub fn from_toml(
        toml: &Option<TomlIntegration>,
        integration_period: f64,
    ) -> Result<Self, String> {
        let Some(cfg) = toml else {
            return Ok(Self::FixedGill);
        };
        match cfg.mode.as_str() {
            "fixed" => Ok(Self::FixedGill),
            "adaptive" => Ok(Self::AdaptiveDopri45(AdaptiveConfig {
                rtol: cfg.rtol.unwrap_or(1e-6),
                initial_dt: cfg.initial_dt.unwrap_or(0.1),
                min_dt: cfg.min_dt.unwrap_or(1e-6),
                max_dt: cfg.max_dt.unwrap_or(integration_period),
            })),
            other => Err(format!(
                "unknown integration mode: {other:?} (expected \"fixed\" or \"adaptive\")"
            )),
        }
    }
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
    PiecewiseConstant,
}

/// Parsed simulation input configuration
#[derive(Debug, Clone)]
pub struct SimInput {
    pub mission_type: MissionType,
    pub planet: PlanetConfig,
    pub n_sims: i32,
    pub sim_phase: SimPhase,
    pub guidance_type: GuidanceType,
    pub visualize_sim: i32,
    pub screen_output: bool,
    pub random_seed: f64,
    pub reference_trajectory: bool,
    pub collect_supervised: bool,
    pub reference_bank_angle: f64, // degrees
    pub base_dir: String,
    pub output_dir: String,
    pub results_suffix: String,
    pub max_time: f64,
}

// ─── TOML deserialization structs ───

#[derive(Debug, Deserialize)]
pub struct TomlConfig {
    pub mission: TomlMission,
    pub planet: PlanetConfig,
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
    // Navigation mode config (bias vs EKF)
    pub navigation: Option<TomlNavigation>,
    /// Onboard atmosphere model config
    #[serde(default)]
    pub onboard_atmosphere: Option<TomlAtmosphereOnboard>,
    /// Integration method config (adaptive DOPRI45 vs fixed Gill RK4)
    pub integration: Option<TomlIntegration>,
    /// Neural network architecture/mask overrides
    #[serde(default)]
    pub network: Option<TomlNetwork>,
}

// ─── Network TOML struct ───

/// v2 layer spec mirrored into TOML via [[network.architecture]] array-of-tables.
/// Mirrors the Rust `LayerSpec` in `data/neural.rs`; kept separate to keep the
/// TOML layer Activation-as-string (TOML parsing) vs the data layer Activation-enum
/// (runtime typing). `to_layer_spec()` bridges them.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TomlLayerSpec {
    Dense {
        input_size: usize,
        output_size: usize,
        activation: String,
    },
    Gru {
        input_size: usize,
        hidden_size: usize,
    },
    Lstm {
        input_size: usize,
        hidden_size: usize,
    },
    Window {
        input_size: usize,
        n_steps: usize,
    },
    Transformer {
        d_model: usize,
        n_heads: usize,
        d_ffn: usize,
        n_seq: usize,
    },
    Mamba {
        input_size: usize,
        d_state: usize,
        #[serde(default)]
        dt_rank: Option<usize>,
    },
    Mamba3 {
        input_size: usize,
        d_state: usize,
        #[serde(default)]
        dt_rank: Option<usize>,
        #[serde(default = "default_discretization")]
        discretization: String,
        #[serde(default = "default_state_mode")]
        state_mode: String,
    },
}

fn default_discretization() -> String {
    "euler".to_string()
}

fn default_state_mode() -> String {
    "real".to_string()
}

impl TomlLayerSpec {
    pub fn to_layer_spec(&self) -> Result<crate::data::neural::LayerSpec, ParseError> {
        use crate::data::neural::LayerSpec;
        match self {
            TomlLayerSpec::Dense {
                input_size,
                output_size,
                activation,
            } => {
                let act = crate::data::neural::parse_activation(activation).map_err(|e| {
                    ParseError(format!("unknown activation {:?}: {}", activation, e))
                })?;
                Ok(LayerSpec::Dense {
                    input_size: *input_size,
                    output_size: *output_size,
                    activation: act,
                })
            }
            TomlLayerSpec::Gru {
                input_size,
                hidden_size,
            } => Ok(LayerSpec::Gru {
                input_size: *input_size,
                hidden_size: *hidden_size,
            }),
            TomlLayerSpec::Lstm {
                input_size,
                hidden_size,
            } => Ok(LayerSpec::Lstm {
                input_size: *input_size,
                hidden_size: *hidden_size,
            }),
            TomlLayerSpec::Window {
                input_size,
                n_steps,
            } => {
                if *input_size == 0 || *n_steps == 0 {
                    return Err(ParseError(format!(
                        "Window layer input_size and n_steps must be positive (got input_size={}, n_steps={})",
                        input_size, n_steps
                    )));
                }
                Ok(LayerSpec::Window {
                    input_size: *input_size,
                    n_steps: *n_steps,
                })
            }
            TomlLayerSpec::Transformer {
                d_model,
                n_heads,
                d_ffn,
                n_seq,
            } => {
                if *n_heads == 0 || *d_model % n_heads != 0 {
                    return Err(ParseError(format!(
                        "(transformer) d_model={d_model} not divisible by n_heads={n_heads}"
                    )));
                }
                if *d_model == 0 || *d_ffn == 0 || *n_seq == 0 {
                    return Err(ParseError(
                        "(transformer) all shape fields must be positive".into(),
                    ));
                }
                Ok(LayerSpec::Transformer {
                    d_model: *d_model,
                    n_heads: *n_heads,
                    d_ffn: *d_ffn,
                    n_seq: *n_seq,
                })
            }
            TomlLayerSpec::Mamba {
                input_size,
                d_state,
                dt_rank,
            } => {
                if *input_size == 0 {
                    return Err(ParseError("Mamba: input_size must be > 0".into()));
                }
                if *d_state == 0 {
                    return Err(ParseError("Mamba: d_state must be > 0".into()));
                }
                let resolved = dt_rank.unwrap_or_else(|| (*input_size / 16).max(1));
                if resolved == 0 {
                    return Err(ParseError("Mamba: dt_rank must be > 0".into()));
                }
                if resolved > *input_size {
                    return Err(ParseError(format!(
                        "Mamba: dt_rank ({resolved}) must be <= input_size ({input_size})"
                    )));
                }
                Ok(LayerSpec::Mamba {
                    input_size: *input_size,
                    d_state: *d_state,
                    dt_rank: resolved,
                })
            }
            TomlLayerSpec::Mamba3 {
                input_size,
                d_state,
                dt_rank,
                discretization,
                state_mode,
            } => {
                if *input_size == 0 {
                    return Err(ParseError("Mamba3: input_size must be > 0".into()));
                }
                if *d_state == 0 {
                    return Err(ParseError("Mamba3: d_state must be > 0".into()));
                }
                let resolved = dt_rank.unwrap_or_else(|| (*input_size / 16).max(1));
                if resolved == 0 || resolved > *input_size {
                    return Err(ParseError(format!(
                        "Mamba3: dt_rank ({resolved}) must be in 1..={input_size}"
                    )));
                }
                let trapezoidal = match discretization.as_str() {
                    "euler" => false,
                    "trapezoidal" => true,
                    other => {
                        return Err(ParseError(format!(
                            "Mamba3: discretization must be euler|trapezoidal, got {other:?}"
                        )));
                    }
                };
                let complex = match state_mode.as_str() {
                    "real" => false,
                    "complex" => true,
                    other => {
                        return Err(ParseError(format!(
                            "Mamba3: state_mode must be real|complex, got {other:?}"
                        )));
                    }
                };
                Ok(LayerSpec::Mamba3 {
                    input_size: *input_size,
                    d_state: *d_state,
                    dt_rank: resolved,
                    trapezoidal,
                    complex,
                })
            }
        }
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TomlNetwork {
    #[serde(default)]
    pub input_mask: Option<Vec<usize>>,
    #[serde(default)]
    pub ablated_input: Option<usize>,
    /// v2 path: heterogeneous architecture spec as [[network.architecture]] TOML array-of-tables.
    /// When present, downstream consumers use it to describe the network shape;
    /// when absent, the existing v1 JSON-file-driven path applies (backward compatible).
    #[serde(default)]
    pub architecture: Option<Vec<TomlLayerSpec>>,
    /// Optional per-input normalization override. When present, REPLACES the loaded
    /// model's normalization table (must be exactly NN_FULL_INPUT_SIZE entries).
    #[serde(default)]
    pub normalization: Option<Vec<crate::data::neural::NormSpec>>,
}

// ─── Onboard Atmosphere TOML structs ───

/// TOML config for explicit exponential segment override.
#[derive(Debug, Clone, Deserialize)]
pub struct TomlExponentialSegment {
    pub alt_low: f64,
    pub alt_high: f64,
    pub rho_ref: f64,
    pub scale_height: f64,
}

/// TOML config for the onboard atmosphere model.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct TomlAtmosphereOnboard {
    pub mode: Option<String>,
    pub n_segments: Option<usize>,
    pub segments: Option<Vec<TomlExponentialSegment>>,
}

// ─── Integration TOML structs ───

/// TOML config for the integration method.
#[derive(Debug, Clone, Deserialize)]
pub struct TomlIntegration {
    pub mode: String,            // "fixed" or "adaptive"
    pub rtol: Option<f64>,       // relative tolerance (default 1e-6)
    pub initial_dt: Option<f64>, // initial sub-step guess in seconds (default 0.1)
    pub min_dt: Option<f64>,     // floor to prevent sub-step collapse (default 1e-6)
    pub max_dt: Option<f64>,     // ceiling in seconds (default = periods.integration)
}

// ─── Navigation TOML structs ───

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TomlNavigation {
    #[serde(default = "default_nav_mode")]
    pub mode: String, // "bias" or "ekf"
    #[serde(default = "default_density_filter_gain")]
    pub density_filter_gain: f64,
    #[serde(default = "default_density_gain_max_delta")]
    pub density_gain_max_delta: f64,
    pub imu: Option<TomlImu>,
    pub star_tracker: Option<TomlStarTracker>,
    pub ekf: Option<TomlEkf>,
}

fn default_nav_mode() -> String {
    "bias".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TomlImu {
    #[serde(default = "default_accel_bias_sigma")]
    pub accel_bias_sigma: f64,
    #[serde(default = "default_accel_noise_sigma")]
    pub accel_noise_sigma: f64,
    #[serde(default = "default_accel_sf_sigma")]
    pub accel_scale_factor_sigma: f64,
    #[serde(default = "default_gyro_bias_sigma")]
    pub gyro_bias_sigma: f64,
    #[serde(default = "default_gyro_noise_sigma")]
    pub gyro_noise_sigma: f64,
}

fn default_accel_bias_sigma() -> f64 {
    1e-4
}
fn default_accel_noise_sigma() -> f64 {
    5e-4
}
fn default_accel_sf_sigma() -> f64 {
    1e-4
}
fn default_gyro_bias_sigma() -> f64 {
    5e-6
}
fn default_gyro_noise_sigma() -> f64 {
    1e-5
}

#[derive(Debug, Clone, Deserialize)]
pub struct TomlStarTracker {
    #[serde(default = "default_st_pos_sigma")]
    pub position_sigma: f64, // 50.0 m
    #[serde(default = "default_st_att_sigma")]
    pub attitude_sigma: f64, // 3e-4 rad
    #[serde(default = "default_st_period")]
    pub update_period: f64, // 10.0 s
    #[serde(default = "default_st_blackout")]
    pub blackout_qdyn_threshold: f64, // 100.0 Pa
}

fn default_st_pos_sigma() -> f64 {
    50.0
}
fn default_st_att_sigma() -> f64 {
    3e-4
}
fn default_st_period() -> f64 {
    10.0
}
fn default_st_blackout() -> f64 {
    100.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct TomlEkf {
    #[serde(default = "default_q_density")]
    pub process_noise_density: f64, // 0.1
}

fn default_q_density() -> f64 {
    0.1
}

#[derive(Debug, Deserialize)]
pub struct TomlMission {
    #[serde(rename = "type")]
    pub mission_type: String,
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
    pub reference_bank_angle: Option<f64>,
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
    /// Piecewise-constant bank angle parameters
    #[serde(default)]
    pub piecewise_constant: TomlPiecewiseConstantParams,
    /// Lateral guidance parameters (shared by unsigned-magnitude schemes)
    #[serde(default)]
    pub lateral: Option<TomlLateralParams>,
    /// Thermal safety limiter parameters (shared by unsigned-magnitude schemes)
    #[serde(default)]
    pub thermal_limiter: Option<TomlThermalLimiterParams>,
    /// Command shaping parameters (acceleration-limited rate shaping)
    #[serde(default)]
    pub command_shaping: Option<TomlCommandShapingParams>,
    /// Neural network guidance section (mode, etc.). Architecture stays under [network].
    #[serde(default)]
    pub neural_network: Option<TomlNeuralNetworkParams>,
}

#[derive(Debug, Deserialize, Default)]
pub struct TomlSimulation {
    #[serde(default = "default_n_sims")]
    pub n_sims: i32,
    #[serde(default)]
    pub random_seed: f64,
    #[serde(default)]
    pub screen_output: bool,
    #[serde(default)]
    pub visualize_sim: i32,
    #[serde(default = "default_max_time")]
    pub max_time: f64,
}

fn default_max_time() -> f64 {
    3000.0
}

fn default_n_sims() -> i32 {
    1
}
fn default_one() -> f64 {
    1.0
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
    #[serde(default)]
    pub wind_table: Option<String>,
    pub results_suffix: Option<String>,
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
    #[serde(default = "default_pilot_time_constant")]
    pub time_constant: f64,
    #[serde(default = "default_pilot_damping")]
    pub damping: f64,
    #[serde(default = "default_pilot_freq")]
    pub frequency: f64,
}

fn default_pilot_model() -> String {
    "perfect".to_string()
}
fn default_pilot_time_constant() -> f64 {
    1.0
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
    #[serde(default)]
    pub max_heat_load: f64, // kJ/m^2
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

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TomlFtcParams {
    #[serde(default)]
    pub capture_damping: f64,
    #[serde(default)]
    pub capture_frequency: f64, // rad/s
    #[serde(default)]
    pub capture_pdyn_margin: f64,
    #[serde(default)]
    pub altitude_damping: f64,
    #[serde(default)]
    pub altitude_frequency: f64, // deg/s (converted to rad/s)
    #[serde(default)]
    pub exit_velocity_threshold: f64, // m/s
    #[serde(default)]
    pub exit_pdyn_margin: f64,
    #[serde(default = "default_exit_altitude_km")]
    pub exit_altitude_threshold: f64, // km
    #[serde(default)]
    pub exit_radial_vel_gain: f64, // Pa/(m/s)
    #[serde(default = "default_security_capture")]
    pub security_capture: i32,
    #[serde(default = "default_three_i32")]
    pub security_exit: i32,
    #[serde(default = "default_longi_act")]
    pub longi_activation: f64, // MJ/kg
    #[serde(default = "default_longi_inh")]
    pub longi_inhibition: f64, // MJ/kg
    #[serde(default)]
    pub pdyn_min: f64, // Pa
    #[serde(default = "default_pressure_coeff_base")]
    pub pressure_coeff_base: f64,
    #[serde(default = "default_pressure_coeff_scale_height")]
    pub pressure_coeff_scale_height: f64, // km
    #[serde(default = "default_gain_fade_start_km")]
    pub gain_fade_start_km: f64,
    #[serde(default = "default_gain_fade_end_km")]
    pub gain_fade_end_km: f64,
}

fn default_five_i32() -> i32 {
    5
}
fn default_three_i32() -> i32 {
    3
}
fn default_security_capture() -> i32 {
    1
}
fn default_exit_altitude_km() -> f64 {
    60.0
}
fn default_density_filter_gain() -> f64 {
    0.8
}
fn default_density_gain_max_delta() -> f64 {
    0.1
}
fn default_longi_act() -> f64 {
    1000.0
}
fn default_longi_inh() -> f64 {
    -1000.0
}
fn default_pressure_coeff_base() -> f64 {
    -134.4
}
fn default_pressure_coeff_scale_height() -> f64 {
    6.9
}
fn default_gain_fade_start_km() -> f64 {
    80.0
}
fn default_gain_fade_end_km() -> f64 {
    100.0
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlEqGlideParams {
    #[serde(default = "default_k_hdot_scale")]
    pub k_hdot_scale: f64,
    #[serde(default = "default_1_1")]
    pub v_ratio_threshold: f64,
    #[serde(default = "default_0_15")]
    pub velocity_bias_high: f64,
    #[serde(default = "default_velocity_bias_low")]
    pub velocity_bias_low: f64,
    #[serde(default = "default_40")]
    pub alt_bias_threshold: f64,
    #[serde(default = "default_neg_0_5")]
    pub cos_bank_min: f64,
    #[serde(default = "default_0_95")]
    pub cos_bank_max: f64,
}

fn default_k_hdot_scale() -> f64 {
    0.3
}
fn default_velocity_bias_low() -> f64 {
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
    #[serde(default = "default_kp")]
    pub kp: f64,
    #[serde(default = "default_0_5")]
    pub kd: f64,
}

fn default_5e_7() -> f64 {
    5e-7
}
fn default_kp() -> f64 {
    1.0
}
fn default_0_5() -> f64 {
    0.5
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlPredGuidParams {
    #[serde(default = "default_0_8")]
    pub k_drag_high: f64,
    #[serde(default = "default_k_drag_low")]
    pub k_drag_low: f64,
    #[serde(default = "default_pdyn_threshold")]
    pub pdyn_threshold: f64,
}

fn default_0_8() -> f64 {
    0.8
}
fn default_k_drag_low() -> f64 {
    0.3
}
fn default_pdyn_threshold() -> f64 {
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
    #[serde(default = "default_bank_max_low_deg")]
    pub bank_max_low_deg: f64,
    #[serde(default = "default_2")]
    pub replan_period: f64,
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
fn default_bank_max_low_deg() -> f64 {
    100.0
}

/// Piecewise-constant guidance config. Three accepted shapes:
///   1. `bank_angles = [...]` (canonical; `n_segments` derived from length).
///   2. `n_segments = N` + individual `bank_angle_0..N-1` keys
///      (override-friendly; the GA writes per-element overrides this way).
///   3. Neither: defaults to 10 segments at 65 deg.
///
/// `n_segments` is validated against `bank_angles.len()` when both are
/// present, and against the highest `bank_angle_N` index found.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TomlPiecewiseConstantParams {
    #[serde(default)]
    pub n_segments: Option<usize>,
    #[serde(default)]
    pub bank_angles: Option<Vec<f64>>,
    #[serde(default = "default_energy_min")]
    pub energy_min: f64, // MJ/kg in TOML, converted to J/kg at load time
    #[serde(default = "default_energy_max")]
    pub energy_max: f64, // MJ/kg in TOML, converted to J/kg at load time
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

const DEFAULT_PIECEWISE_N_SEGMENTS: usize = 10;
const DEFAULT_PIECEWISE_BANK_DEG: f64 = 65.0;

impl TomlPiecewiseConstantParams {
    /// Resolve the per-segment bank angles in degrees. Errors on
    /// inconsistent `n_segments` / `bank_angles` / `bank_angle_N` inputs.
    pub fn resolve_bank_angles_deg(&self) -> Result<Vec<f64>, String> {
        // Highest index N referenced by any `bank_angle_N` extra key.
        let max_idx_from_extras = self
            .extra
            .keys()
            .filter_map(|k| {
                k.strip_prefix("bank_angle_")
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max();

        let n_from_array = self.bank_angles.as_ref().map(|v| v.len());
        let n_from_extras = max_idx_from_extras.map(|i| i + 1);

        // Resolve n_segments: explicit > array length > extras count > default.
        let n = self
            .n_segments
            .or(n_from_array)
            .or(n_from_extras)
            .unwrap_or(DEFAULT_PIECEWISE_N_SEGMENTS);

        if n == 0 {
            return Err("piecewise_constant: n_segments must be >= 1".to_string());
        }
        if let (Some(declared), Some(arr_len)) = (self.n_segments, n_from_array)
            && declared != arr_len
        {
            return Err(format!(
                "piecewise_constant: n_segments={} disagrees with bank_angles array length {}",
                declared, arr_len
            ));
        }
        if let (Some(declared), Some(extras_n)) = (self.n_segments, n_from_extras)
            && extras_n > declared
        {
            return Err(format!(
                "piecewise_constant: bank_angle_{} present but n_segments={}",
                extras_n - 1,
                declared
            ));
        }

        // Base vector: explicit `bank_angles` array if present, else
        // default-filled. Then overlay any `bank_angle_N` extras so per-element
        // overrides ALWAYS win. The GA writes its chromosome as flat
        // `bank_angle_N` keys, so an early return of the array would silently
        // nullify GA optimization on array-seeded configs.
        // Precedence: flat `bank_angle_N` keys > `bank_angles` array > default.
        let mut angles = match &self.bank_angles {
            Some(arr) => arr.clone(),
            None => vec![DEFAULT_PIECEWISE_BANK_DEG; n],
        };
        for (key, val) in &self.extra {
            let Some(idx_str) = key.strip_prefix("bank_angle_") else {
                continue;
            };
            let Ok(idx) = idx_str.parse::<usize>() else {
                continue;
            };
            let f = val
                .as_float()
                .or_else(|| val.as_integer().map(|i| i as f64))
                .ok_or_else(|| format!("piecewise_constant: bank_angle_{} is not numeric", idx))?;
            if idx >= n {
                return Err(format!(
                    "piecewise_constant: bank_angle_{} out of range (n_segments={})",
                    idx, n
                ));
            }
            angles[idx] = f;
        }
        Ok(angles)
    }
}

fn default_energy_min() -> f64 {
    -6.0
}
fn default_energy_max() -> f64 {
    5.0
}

fn default_heat_flux_activation() -> f64 {
    1.0
}
fn default_heat_load_activation() -> f64 {
    1.0
}
fn default_heat_flux_ramp_exponent() -> f64 {
    1.0
}
fn default_heat_load_ramp_exponent() -> f64 {
    1.0
}
fn default_command_shaping_enabled() -> bool {
    true
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlLateralParams {
    #[serde(default)]
    pub tau: f64, // s (0.0 = inactive)
    #[serde(default)]
    pub threshold: f64, // deg (converted to rad)
    #[serde(default)]
    pub min_reversal_interval: f64, // s
    #[serde(default = "default_five_i32")]
    pub max_reversals: i32,
    #[serde(default)]
    pub lateral_activation: f64, // MJ/kg
    #[serde(default)]
    pub lateral_inhibition: f64, // MJ/kg
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlThermalLimiterParams {
    #[serde(default = "default_heat_flux_activation")]
    pub heat_flux_activation: f64,
    #[serde(default = "default_heat_load_activation")]
    pub heat_load_activation: f64,
    #[serde(default = "default_heat_flux_ramp_exponent")]
    pub heat_flux_ramp_exponent: f64,
    #[serde(default = "default_heat_load_ramp_exponent")]
    pub heat_load_ramp_exponent: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TomlCommandShapingParams {
    #[serde(default = "default_command_shaping_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub max_bank_acceleration: f64, // deg/s^2 (converted to rad/s^2 at load time)
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct TomlNeuralNetworkParams {
    /// "full_neural" (default) | "magnitude_only".
    #[serde(default)]
    pub mode: Option<String>,
    /// Output parameterization: None (default, atan2 signed) | "acos_tanh" (magnitude via acos of tanh output).
    /// "acos_tanh" requires mode = "magnitude_only" and a last Dense layer with output_size=1 and activation="tanh".
    #[serde(default)]
    pub output_parameterization: Option<String>,
    /// Half-range multiplier for output_parameterization="scaled_pi".
    #[serde(default)]
    pub scaled_pi_n: Option<f64>,
    /// Per-step increment bound (rad) for output_parameterization="delta".
    #[serde(default)]
    pub delta_max: Option<f64>,
}

// ─── Domain-based Monte Carlo TOML structs ───

#[derive(Debug, Deserialize, Clone)]
pub struct TomlMonteCarlo {
    pub seed: u64,
    #[serde(default)]
    pub sampling: Option<String>,
    pub initial_state: Option<TomlMcDomain>,
    pub atmosphere: Option<TomlMcDomain>,
    pub aerodynamics: Option<TomlMcDomain>,
    pub navigation: Option<TomlMcDomain>,
    pub mass: Option<TomlMcDomain>,
    pub vehicle: Option<TomlMcDomain>,
    pub pilot: Option<TomlMcDomain>,
    pub nav_filter: Option<TomlMcDomain>,
    pub wind: Option<TomlMcDomain>,
    pub density_perturbation: Option<TomlMcDomain>,
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

/// Deep-merge `overlay` into `base`. Tables merge recursively;
/// all other types (scalars, arrays) replace the base value.
pub fn deep_merge(base: &mut toml::Value, overlay: toml::Value) {
    match (base.is_table(), overlay.is_table()) {
        (true, true) => {
            let base_table = base.as_table_mut().unwrap();
            if let toml::Value::Table(overlay_table) = overlay {
                for (key, overlay_val) in overlay_table {
                    if let Some(base_val) = base_table.get_mut(&key) {
                        deep_merge(base_val, overlay_val);
                    } else {
                        base_table.insert(key, overlay_val);
                    }
                }
            }
        }
        _ => {
            *base = overlay;
        }
    }
}

/// Resolve `base` key(s) in a TOML value tree, loading and merging parent configs.
/// Supports single string (`base = "file.toml"`) or array (`base = ["a.toml", "b.toml"]`).
/// Detects cycles via the `visited` set.
pub fn resolve_toml_bases(
    mut root: toml::Value,
    file_path: &Path,
    visited: &mut HashSet<std::path::PathBuf>,
) -> Result<toml::Value, ParseError> {
    let base_dir = file_path.parent().unwrap_or_else(|| Path::new("."));

    let base_paths: Vec<String> = match root.as_table_mut().and_then(|t| t.remove("base")) {
        None => return Ok(root),
        Some(toml::Value::String(s)) => vec![s],
        Some(toml::Value::Array(arr)) => arr
            .into_iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| ParseError("base array elements must be strings".into()))
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => {
            return Err(ParseError(
                "base must be a string or array of strings".into(),
            ));
        }
    };

    let canonical = file_path.canonicalize().map_err(|e| {
        ParseError(format!(
            "Cannot canonicalize '{}': {}",
            file_path.display(),
            e
        ))
    })?;
    if !visited.insert(canonical.clone()) {
        return Err(ParseError(format!(
            "Cycle detected: '{}' was already visited",
            file_path.display()
        )));
    }

    let result = (|| {
        let mut merged = toml::Value::Table(toml::map::Map::new());
        for base_rel in &base_paths {
            let base_abs = base_dir.join(base_rel);
            let base_content = std::fs::read_to_string(&base_abs).map_err(|e| {
                ParseError(format!(
                    "Cannot read base '{}' (referenced from '{}'): {}",
                    base_abs.display(),
                    file_path.display(),
                    e
                ))
            })?;
            let base_value: toml::Value = toml::from_str(&base_content).map_err(|e| {
                ParseError(format!(
                    "TOML parse error in '{}': {}",
                    base_abs.display(),
                    e
                ))
            })?;
            let resolved_base = resolve_toml_bases(base_value, &base_abs, visited)?;
            deep_merge(&mut merged, resolved_base);
        }

        deep_merge(&mut merged, root);
        Ok(merged)
    })();

    // Always clean up visited, even on error paths, to prevent
    // false cycle errors for sibling references (diamond inheritance).
    visited.remove(&canonical);
    result
}

impl SimInput {
    /// Load a TOML config file with base inheritance resolution.
    /// Returns (SimInput, TomlConfig).
    pub fn from_toml_file(path: &Path) -> Result<(Self, TomlConfig), ParseError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ParseError(format!("Cannot read '{}': {}", path.display(), e)))?;
        let root: toml::Value = toml::from_str(&content)
            .map_err(|e| ParseError(format!("TOML parse error in '{}': {}", path.display(), e)))?;
        let mut visited = HashSet::new();
        let resolved = resolve_toml_bases(root, path, &mut visited)?;
        let resolved_str = toml::to_string(&resolved)
            .map_err(|e| ParseError(format!("TOML serialize error: {}", e)))?;
        Self::from_toml(&resolved_str)
    }

    /// Parse a TOML configuration string. Returns (SimInput, TomlConfig).
    /// The TomlConfig is needed for inline data loading in consolidated mode.
    pub fn from_toml(content: &str) -> Result<(Self, TomlConfig), ParseError> {
        let config: TomlConfig =
            toml::from_str(content).map_err(|e| ParseError(format!("TOML parse error: {}", e)))?;

        let mission_type = match config.mission.mission_type.as_str() {
            "aerocapture" => MissionType::Aerocapture,
            other => return Err(ParseError(format!("Unknown mission type: {}", other))),
        };

        let planet = config.planet.clone();

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
            "piecewise_constant" => GuidanceType::PiecewiseConstant,
            other => return Err(ParseError(format!("Unknown guidance type: {}", other))),
        };

        let sim_input = SimInput {
            mission_type,
            planet,
            n_sims: config.simulation.n_sims,
            sim_phase,
            guidance_type,
            visualize_sim: config.simulation.visualize_sim,
            screen_output: config.simulation.screen_output,
            random_seed: config.simulation.random_seed,
            reference_trajectory: config.guidance.reference_trajectory,
            collect_supervised: false,
            reference_bank_angle: config.guidance.reference_bank_angle.unwrap_or_else(|| {
                config
                    .entry
                    .as_ref()
                    .map(|e| e.initial_bank_angle)
                    .unwrap_or(0.0)
            }),
            base_dir: config.data.base_dir.clone(),
            output_dir: config.data.output_dir.clone(),
            results_suffix: config
                .data
                .results_suffix
                .clone()
                .unwrap_or_else(|| ".out".to_string()),
            max_time: config.simulation.max_time,
        };

        Ok((sim_input, config))
    }

    /// Build an output file path
    pub fn output_path(&self, filename: &str) -> String {
        format!("{}/{}", self.output_dir, filename)
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
