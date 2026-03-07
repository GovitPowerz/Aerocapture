//! Parse .in configuration files + data file suffixes.
//!
//! Handles both the original (30-field) and neural (32-field) input formats.

use serde::Deserialize;
use std::fmt;

/// Mission type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MissionType {
    Aerocapture,
    AeroGravityAssist,
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
    pub fn from_id(id: i32) -> Result<Self, String> {
        match id {
            2 => Ok(Planet::Moon),
            3 => Ok(Planet::Earth),
            4 => Ok(Planet::Mars),
            5 => Ok(Planet::Jupiter),
            _ => Err(format!("Unknown planet id: {}", id)),
        }
    }

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

/// Data file suffix configuration
#[derive(Debug, Clone)]
pub struct DataSuffixes {
    pub capsule: String,     // sufmsr
    pub reentry: String,     // sufren
    pub mission: String,     // sufmis
    pub guidance: String,    // sufgui
    pub neural: String,      // sufgnn (nn variant only)
    pub incidence: String,   // sufinc
    pub aero: String,        // sufaer
    pub atmosphere: String,  // sufatm
    pub dispersions: String, // sufdis
    pub navigation: String,  // sufnav
    pub lottery: String,     // suflot
    pub success: String,     // sufsuc
    pub results: String,     // sufres
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
    pub create_dispersions: bool,
    pub replay_sim: i32,
    pub save_results: bool,
    pub visualize_sim: i32,
    pub screen_output: bool,
    pub random_seed: f64,
    pub reference_trajectory: bool,
    pub reference_bank_angle: f64, // degrees
    pub dispersion_multipliers: [f64; 4],
    pub suffixes: DataSuffixes,
    pub base_dir: String,
    pub output_dir: String,
}

// ─── TOML deserialization structs ───

#[derive(Debug, Deserialize)]
pub struct TomlConfig {
    pub mission: TomlMission,
    pub guidance: TomlGuidance,
    #[serde(default)]
    pub simulation: TomlSimulation,
    #[serde(default)]
    pub dispersions: TomlDispersions,
    pub data: TomlData,
}

#[derive(Debug, Deserialize)]
pub struct TomlMission {
    #[serde(rename = "type")]
    pub mission_type: String,
    pub planet: String,
    #[serde(default = "default_phase")]
    pub phase: String,
}

fn default_phase() -> String { "full".to_string() }

#[derive(Debug, Deserialize)]
pub struct TomlGuidance {
    #[serde(rename = "type")]
    pub guidance_type: String,
    #[serde(default)]
    pub reference_trajectory: bool,
    #[serde(default = "default_ref_bank")]
    pub reference_bank_angle: f64,
}

fn default_ref_bank() -> f64 { 0.0 }

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
    #[serde(default)]
    pub create_dispersions: bool,
    #[serde(default)]
    pub replay_sim: i32,
    #[serde(default = "default_true")]
    pub save_results: bool,
    #[serde(default)]
    pub visualize_sim: i32,
}

fn default_one_i32() -> i32 { 1 }
fn default_true() -> bool { true }

#[derive(Debug, Deserialize)]
pub struct TomlDispersions {
    #[serde(default = "default_one")]
    pub nav_aerocapture: f64,
    #[serde(default = "default_one")]
    pub nav_interplanetary: f64,
    #[serde(default = "default_one")]
    pub accelerometer: f64,
    #[serde(default = "default_one")]
    pub aero_model: f64,
}

fn default_one() -> f64 { 1.0 }

impl Default for TomlDispersions {
    fn default() -> Self {
        Self {
            nav_aerocapture: 1.0,
            nav_interplanetary: 1.0,
            accelerometer: 1.0,
            aero_model: 1.0,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct TomlData {
    #[serde(default = "default_base_dir")]
    pub base_dir: String,
    #[serde(default = "default_output_dir")]
    pub output_dir: String,
    pub files: TomlDataFiles,
}

fn default_base_dir() -> String { "old_codebase/donnees".to_string() }
fn default_output_dir() -> String { "old_codebase/sorties".to_string() }

#[derive(Debug, Deserialize)]
pub struct TomlDataFiles {
    pub capsule: String,
    pub entry: String,
    pub mission: String,
    pub guidance: String,
    #[serde(default)]
    pub neural_network: String,
    pub incidence: String,
    pub aerodynamics: String,
    pub atmosphere: String,
    #[serde(default = "default_nul")]
    pub dispersions: String,
    #[serde(default = "default_nul")]
    pub navigation: String,
    pub lottery: String,
    pub success: String,
    pub results: String,
}

fn default_nul() -> String { ".nul".to_string() }

#[derive(Debug)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

fn parse_line<T: std::str::FromStr>(
    lines: &[String],
    idx: usize,
    name: &str,
) -> Result<T, ParseError> {
    let line = lines
        .get(idx)
        .ok_or_else(|| ParseError(format!("Missing line {} for {}", idx, name)))?;
    let token = line.split_whitespace().next().unwrap_or("");
    token.parse::<T>().map_err(|_| {
        ParseError(format!(
            "Cannot parse '{}' as {} for {}",
            token,
            std::any::type_name::<T>(),
            name
        ))
    })
}

fn parse_string(lines: &[String], idx: usize, name: &str) -> Result<String, ParseError> {
    let line = lines
        .get(idx)
        .ok_or_else(|| ParseError(format!("Missing line {} for {}", idx, name)))?;
    let token = line.split_whitespace().next().unwrap_or("");
    Ok(token.to_string())
}

impl SimInput {
    /// Parse a TOML configuration string into SimInput.
    pub fn from_toml(content: &str) -> Result<Self, ParseError> {
        let config: TomlConfig = toml::from_str(content)
            .map_err(|e| ParseError(format!("TOML parse error: {}", e)))?;

        let mission_type = match config.mission.mission_type.as_str() {
            "aerocapture" => MissionType::Aerocapture,
            "aero_gravity_assist" => MissionType::AeroGravityAssist,
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

        Ok(SimInput {
            mission_type,
            planet,
            n_sims: config.simulation.n_sims,
            sim_phase,
            guidance_type,
            stats_only: config.simulation.stats_only,
            create_dispersions: config.simulation.create_dispersions,
            replay_sim: config.simulation.replay_sim,
            save_results: config.simulation.save_results,
            visualize_sim: config.simulation.visualize_sim,
            screen_output: config.simulation.screen_output,
            random_seed: config.simulation.random_seed,
            reference_trajectory: config.guidance.reference_trajectory,
            reference_bank_angle: config.guidance.reference_bank_angle,
            dispersion_multipliers: [
                config.dispersions.nav_aerocapture,
                config.dispersions.nav_interplanetary,
                config.dispersions.accelerometer,
                config.dispersions.aero_model,
            ],
            suffixes: DataSuffixes {
                capsule: config.data.files.capsule,
                reentry: config.data.files.entry,
                mission: config.data.files.mission,
                guidance: config.data.files.guidance,
                neural: config.data.files.neural_network,
                incidence: config.data.files.incidence,
                aero: config.data.files.aerodynamics,
                atmosphere: config.data.files.atmosphere,
                dispersions: config.data.files.dispersions,
                navigation: config.data.files.navigation,
                lottery: config.data.files.lottery,
                success: config.data.files.success,
                results: config.data.files.results,
            },
            base_dir: config.data.base_dir,
            output_dir: config.data.output_dir,
        })
    }

    /// Parse input configuration from lines (stdin).
    ///
    /// Auto-detects whether this is the neural variant (32 fields with natgnn + sufgnn)
    /// or original variant (30 fields without those).
    pub fn parse(lines: &[String]) -> Result<Self, ParseError> {
        // Count non-empty lines to determine format
        let data_lines: Vec<&String> = lines.iter().filter(|l| !l.trim().is_empty()).collect();
        let is_neural = data_lines.len() >= 32;

        let mut i = 0;

        let natman: i32 = parse_line(lines, i, "natman")?;
        i += 1;
        let mission_type = match natman {
            1 => MissionType::Aerocapture,
            2 => MissionType::AeroGravityAssist,
            _ => return Err(ParseError(format!("Invalid natman: {}", natman))),
        };

        let natpla: i32 = parse_line(lines, i, "natpla")?;
        i += 1;
        let planet = Planet::from_id(natpla).map_err(ParseError)?;

        let n_sims: i32 = parse_line(lines, i, "nbsimu")?;
        i += 1;

        let natsim: i32 = parse_line(lines, i, "natsim")?;
        i += 1;
        let sim_phase = match natsim {
            1 => SimPhase::Full,
            2 => SimPhase::CaptureOnly,
            3 => SimPhase::ExitOnly,
            4 => SimPhase::Preprogrammed,
            _ => return Err(ParseError(format!("Invalid natsim: {}", natsim))),
        };

        let guidance_type = if is_neural {
            let natgnn: i32 = parse_line(lines, i, "natgnn")?;
            i += 1;
            match natgnn {
                1 => GuidanceType::Ftc,
                2 => GuidanceType::NeuralNetwork,
                3 => GuidanceType::EquilibriumGlide,
                4 => GuidanceType::EnergyController,
                5 => GuidanceType::PredGuid,
                6 => GuidanceType::Fnpag,
                _ => return Err(ParseError(format!("Invalid natgnn: {}", natgnn))),
            }
        } else {
            GuidanceType::Ftc
        };

        let istats: i32 = parse_line(lines, i, "istats")?;
        i += 1;
        let itirag: i32 = parse_line(lines, i, "itirag")?;
        i += 1;
        let numsim: i32 = parse_line(lines, i, "numsim")?;
        i += 1;
        let isauve: i32 = parse_line(lines, i, "isauve")?;
        i += 1;
        let numvis: i32 = parse_line(lines, i, "numvis")?;
        i += 1;
        let iecran: i32 = parse_line(lines, i, "iecran")?;
        i += 1;
        let xgalea: f64 = parse_line(lines, i, "xgalea")?;
        i += 1;
        let irefer: i32 = parse_line(lines, i, "irefer")?;
        i += 1;
        let gitref: f64 = parse_line(lines, i, "gitref")?;
        i += 1;

        let mut xmulti = [1.0f64; 4];
        for (j, item) in xmulti.iter_mut().enumerate() {
            *item = parse_line(lines, i, &format!("xmulti({})", j + 1))?;
            i += 1;
        }

        let sufmsr = parse_string(lines, i, "sufmsr")?;
        i += 1;
        let sufren = parse_string(lines, i, "sufren")?;
        i += 1;
        let sufmis = parse_string(lines, i, "sufmis")?;
        i += 1;
        let sufgui = parse_string(lines, i, "sufgui")?;
        i += 1;

        let sufgnn = if is_neural {
            let s = parse_string(lines, i, "sufgnn")?;
            i += 1;
            s
        } else {
            String::new()
        };

        let sufinc = parse_string(lines, i, "sufinc")?;
        i += 1;
        let sufaer = parse_string(lines, i, "sufaer")?;
        i += 1;
        let sufatm = parse_string(lines, i, "sufatm")?;
        i += 1;
        let sufdis = parse_string(lines, i, "sufdis")?;
        i += 1;
        let sufnav = parse_string(lines, i, "sufnav")?;
        i += 1;
        let suflot = parse_string(lines, i, "suflot")?;
        i += 1;
        let sufsuc = parse_string(lines, i, "sufsuc")?;
        i += 1;
        let sufres = parse_string(lines, i, "sufres")?;
        // confirmation line is read but not used

        Ok(SimInput {
            mission_type,
            planet,
            n_sims,
            sim_phase,
            guidance_type,
            stats_only: istats == 1,
            create_dispersions: itirag == 1,
            replay_sim: numsim,
            save_results: isauve == 1,
            visualize_sim: numvis,
            screen_output: iecran == 1,
            random_seed: xgalea,
            reference_trajectory: irefer == 1,
            reference_bank_angle: gitref,
            dispersion_multipliers: xmulti,
            suffixes: DataSuffixes {
                capsule: sufmsr,
                reentry: sufren,
                mission: sufmis,
                guidance: sufgui,
                neural: sufgnn,
                incidence: sufinc,
                aero: sufaer,
                atmosphere: sufatm,
                dispersions: sufdis,
                navigation: sufnav,
                lottery: suflot,
                success: sufsuc,
                results: sufres,
            },
            base_dir: "../donnees".to_string(),
            output_dir: "../sorties".to_string(),
        })
    }

    /// Build the data file path for a given category
    pub fn data_path(&self, category: &str, suffix: &str) -> String {
        format!("{}/{}{}", self.base_dir, category, suffix)
    }

    /// Build an output file path
    pub fn output_path(&self, filename: &str) -> String {
        format!("{}/{}", self.output_dir, filename)
    }
}
