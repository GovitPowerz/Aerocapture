//! Parse .in configuration files + data file suffixes.
//!
//! Handles both the original (30-field) and neural (32-field) input formats.

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
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

fn parse_line<T: std::str::FromStr>(lines: &[String], idx: usize, name: &str) -> Result<T, ParseError> {
    let line = lines.get(idx).ok_or_else(|| ParseError(format!("Missing line {} for {}", idx, name)))?;
    let token = line.split_whitespace().next().unwrap_or("");
    token
        .parse::<T>()
        .map_err(|_| ParseError(format!("Cannot parse '{}' as {} for {}", token, std::any::type_name::<T>(), name)))
}

fn parse_string(lines: &[String], idx: usize, name: &str) -> Result<String, ParseError> {
    let line = lines.get(idx).ok_or_else(|| ParseError(format!("Missing line {} for {}", idx, name)))?;
    let token = line.split_whitespace().next().unwrap_or("");
    Ok(token.to_string())
}

impl SimInput {
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
        for j in 0..4 {
            xmulti[j] = parse_line(lines, i, &format!("xmulti({})", j + 1))?;
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
        })
    }

    /// Build the data file path for a given category
    pub fn data_path(&self, category: &str, suffix: &str) -> String {
        format!("../donnees/{}{}", category, suffix)
    }
}
