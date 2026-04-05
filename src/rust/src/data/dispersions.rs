//! Monte Carlo dispersion system.
//!
//! Domain-based: sigma values from TOML config with preset levels + runtime RNG
//! draw generation.
//!
//! Distribution types:
//! - Initial state + navigation + nav filter: Gaussian (1-sigma)
//! - Atmosphere, aerodynamics, incidence, mass, vehicle, pilot: Uniform[-sigma, +sigma]

use super::DataError;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand_distr::{Normal, Uniform};

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// Preset dispersion severity level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DispersionLevel {
    Off,
    Low,
    Medium,
    High,
    Custom,
}

impl DispersionLevel {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, DataError> {
        match s {
            "off" => Ok(DispersionLevel::Off),
            "low" => Ok(DispersionLevel::Low),
            "medium" => Ok(DispersionLevel::Medium),
            "high" => Ok(DispersionLevel::High),
            "custom" => Ok(DispersionLevel::Custom),
            _ => Err(DataError(format!("Unknown dispersion level: '{}'", s))),
        }
    }
}

/// Sampling method for Monte Carlo draws.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SamplingMethod {
    #[default]
    Random,
    Lhs,
    Sobol,
}

impl SamplingMethod {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, DataError> {
        match s.to_lowercase().as_str() {
            "random" => Ok(SamplingMethod::Random),
            "lhs" => Ok(SamplingMethod::Lhs),
            "sobol" => Ok(SamplingMethod::Sobol),
            _ => Err(DataError(format!("Unknown sampling method: '{}'", s))),
        }
    }
}

/// Initial state dispersion sigmas (Gaussian, 1-sigma).
/// Calibrated from MSR/ESR/ATPE mission files.
#[derive(Debug, Clone, Copy)]
pub struct InitialStateSigmas {
    pub altitude: f64,    // km
    pub longitude: f64,   // deg
    pub latitude: f64,    // deg
    pub velocity: f64,    // m/s
    pub flight_path: f64, // deg
    pub azimuth: f64,     // deg
}

impl InitialStateSigmas {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self {
                altitude: 0.0,
                longitude: 0.0,
                latitude: 0.0,
                velocity: 0.0,
                flight_path: 0.0,
                azimuth: 0.0,
            },
            DispersionLevel::Low => Self {
                altitude: 0.0,
                longitude: 0.01,
                latitude: 0.01,
                velocity: 0.13,
                flight_path: 0.043,
                azimuth: 0.043,
            },
            DispersionLevel::Medium => Self {
                altitude: 0.1,
                longitude: 0.1,
                latitude: 0.05,
                velocity: 1.0,
                flight_path: 0.1,
                azimuth: 0.05,
            },
            DispersionLevel::High => Self {
                altitude: 0.5,
                longitude: 0.5,
                latitude: 0.1,
                velocity: 2.0,
                flight_path: 0.2,
                azimuth: 0.1,
            },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }
}

/// Atmosphere dispersion sigmas (Uniform, half-width).
#[derive(Debug, Clone, Copy)]
pub struct AtmosphereSigmas {
    pub density: f64, // %
}

impl AtmosphereSigmas {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self { density: 0.0 },
            DispersionLevel::Low => Self { density: 20.0 },
            DispersionLevel::Medium => Self { density: 50.0 },
            DispersionLevel::High => Self { density: 100.0 },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }
}

/// Aerodynamics dispersion sigmas (Uniform, half-width).
#[derive(Debug, Clone, Copy)]
pub struct AerodynamicsSigmas {
    pub drag: f64,      // %
    pub lift: f64,      // %
    pub incidence: f64, // deg
}

impl AerodynamicsSigmas {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self {
                drag: 0.0,
                lift: 0.0,
                incidence: 0.0,
            },
            DispersionLevel::Low => Self {
                drag: 3.0,
                lift: 5.0,
                incidence: 0.5,
            },
            DispersionLevel::Medium => Self {
                drag: 5.0,
                lift: 10.0,
                incidence: 1.0,
            },
            DispersionLevel::High => Self {
                drag: 10.0,
                lift: 15.0,
                incidence: 2.0,
            },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }
}

/// Navigation error sigmas (Gaussian, 1-sigma).
#[derive(Debug, Clone, Copy)]
pub struct NavigationSigmas {
    pub altitude: f64,    // km
    pub longitude: f64,   // deg
    pub latitude: f64,    // deg
    pub velocity: f64,    // m/s
    pub flight_path: f64, // deg
    pub azimuth: f64,     // deg
    pub drag_accel: f64,  // m/s²
}

impl NavigationSigmas {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self {
                altitude: 0.0,
                longitude: 0.0,
                latitude: 0.0,
                velocity: 0.0,
                flight_path: 0.0,
                azimuth: 0.0,
                drag_accel: 0.0,
            },
            DispersionLevel::Low => Self {
                altitude: 0.3,
                longitude: 0.01,
                latitude: 0.01,
                velocity: 0.2,
                flight_path: 0.02,
                azimuth: 0.02,
                drag_accel: 0.05,
            },
            DispersionLevel::Medium => Self {
                altitude: 0.667,
                longitude: 0.05,
                latitude: 0.05,
                velocity: 0.4,
                flight_path: 0.03,
                azimuth: 0.03,
                drag_accel: 0.1,
            },
            DispersionLevel::High => Self {
                altitude: 1.0,
                longitude: 0.1,
                latitude: 0.1,
                velocity: 1.0,
                flight_path: 0.05,
                azimuth: 0.05,
                drag_accel: 0.2,
            },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }
}

/// Mass dispersion sigmas (Uniform, half-width).
#[derive(Debug, Clone, Copy)]
pub struct MassSigmas {
    pub mass: f64, // %
}

impl MassSigmas {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self { mass: 0.0 },
            DispersionLevel::Low => Self { mass: 0.5 },
            DispersionLevel::Medium => Self { mass: 1.0 },
            DispersionLevel::High => Self { mass: 2.0 },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }
}

/// Vehicle dispersion sigmas (Uniform, half-width).
/// Covers manufacturing tolerance and ablation uncertainty.
#[derive(Debug, Clone, Copy)]
pub struct VehicleSigmas {
    pub ref_area: f64,      // % — reference area
    pub max_bank_rate: f64, // % — max bank rate
}

impl VehicleSigmas {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self {
                ref_area: 0.0,
                max_bank_rate: 0.0,
            },
            DispersionLevel::Low => Self {
                ref_area: 1.0,
                max_bank_rate: 5.0,
            },
            DispersionLevel::Medium => Self {
                ref_area: 2.0,
                max_bank_rate: 10.0,
            },
            DispersionLevel::High => Self {
                ref_area: 5.0,
                max_bank_rate: 20.0,
            },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }
}

/// Pilot dynamics dispersion sigmas (Uniform, half-width).
/// Actuator performance uncertainty.
#[derive(Debug, Clone, Copy)]
pub struct PilotSigmas {
    pub time_constant: f64, // % — tau
    pub damping: f64,       // % — zeta
    pub frequency: f64,     // % — omega
}

impl PilotSigmas {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self {
                time_constant: 0.0,
                damping: 0.0,
                frequency: 0.0,
            },
            DispersionLevel::Low => Self {
                time_constant: 5.0,
                damping: 5.0,
                frequency: 5.0,
            },
            DispersionLevel::Medium => Self {
                time_constant: 10.0,
                damping: 10.0,
                frequency: 10.0,
            },
            DispersionLevel::High => Self {
                time_constant: 20.0,
                damping: 20.0,
                frequency: 20.0,
            },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }
}

/// Navigation filter dispersion sigmas (Gaussian, 1-sigma).
/// Density filter gain (lambda) tuning uncertainty.
#[derive(Debug, Clone, Copy)]
pub struct NavFilterSigmas {
    pub filter_gain: f64, // absolute delta on lambda (e.g. 0.1)
}

impl NavFilterSigmas {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self { filter_gain: 0.0 },
            DispersionLevel::Low => Self { filter_gain: 0.05 },
            DispersionLevel::Medium => Self { filter_gain: 0.10 },
            DispersionLevel::High => Self { filter_gain: 0.15 },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }
}

/// Wind dispersion parameters.
#[derive(Debug, Clone, Copy)]
pub struct WindDispersionConfig {
    pub scale_min: f64,          // lower bound for uniform draw (e.g. 0.5)
    pub scale_max: f64,          // upper bound for uniform draw (e.g. 1.5)
    pub direction_bias_deg: f64, // max rotation ±deg
}

impl WindDispersionConfig {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self {
                scale_min: 1.0,
                scale_max: 1.0,
                direction_bias_deg: 0.0,
            },
            DispersionLevel::Low => Self {
                scale_min: 0.7,
                scale_max: 1.3,
                direction_bias_deg: 5.0,
            },
            DispersionLevel::Medium => Self {
                scale_min: 0.5,
                scale_max: 1.5,
                direction_bias_deg: 10.0,
            },
            DispersionLevel::High => Self {
                scale_min: 0.2,
                scale_max: 2.0,
                direction_bias_deg: 20.0,
            },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }
}

/// Gauss-Markov (Ornstein-Uhlenbeck) density perturbation config.
/// Produces time-varying density multiplier that evolves during each run.
#[derive(Debug, Clone, Copy)]
pub struct DensityPerturbationConfig {
    pub tau: f64,   // correlation time (seconds)
    pub sigma: f64, // steady-state RMS amplitude (fractional)
}

impl DensityPerturbationConfig {
    pub fn from_level(level: DispersionLevel) -> Self {
        match level {
            DispersionLevel::Off => Self {
                tau: 0.0,
                sigma: 0.0,
            },
            DispersionLevel::Low => Self {
                tau: 120.0,
                sigma: 0.05,
            },
            DispersionLevel::Medium => Self {
                tau: 60.0,
                sigma: 0.10,
            },
            DispersionLevel::High => Self {
                tau: 30.0,
                sigma: 0.20,
            },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }

    /// Returns true if the perturbation is effectively disabled.
    pub fn is_disabled(&self) -> bool {
        self.sigma <= 0.0 || self.tau <= 0.0
    }
}

/// Inverse standard normal CDF via Peter Acklam's rational approximation.
/// Accurate to ~1.15e-9. Input p in (0,1); output z such that P(Z<=z)=p.
pub fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -norm_ppf(1.0 - p)
    }
}

/// Per-dimension transform: maps a unit uniform sample u in [0,1) to the draw value.
#[derive(Debug, Clone, PartialEq)]
pub enum DimTransform {
    Gaussian { sigma: f64 },
    Uniform { half_width: f64 },
    UniformRange { min: f64, max: f64 },
    Fixed(f64),
}

impl DimTransform {
    pub fn apply(&self, u: f64) -> f64 {
        match self {
            DimTransform::Gaussian { sigma } => norm_ppf(u) * sigma,
            DimTransform::Uniform { half_width } => (2.0 * u - 1.0) * half_width,
            DimTransform::UniformRange { min, max } => min + u * (max - min),
            DimTransform::Fixed(v) => *v,
        }
    }
}

/// Advance the Ornstein-Uhlenbeck density perturbation by one timestep.
///
/// Exact transition: x(t+dt) = x(t)*exp(-dt/tau) + sigma*sqrt(1 - exp(-2*dt/tau))*N(0,1)
///
/// Returns 0.0 when disabled (sigma <= 0 or tau <= 0).
pub fn step_density_perturbation(x: f64, dt: f64, tau: f64, sigma: f64, normal_sample: f64) -> f64 {
    if sigma <= 0.0 || tau <= 0.0 {
        return 0.0;
    }
    let decay = (-dt / tau).exp();
    x * decay + sigma * (1.0 - decay * decay).sqrt() * normal_sample
}

/// Full domain-based dispersion configuration.
#[derive(Debug, Clone)]
pub struct DispersionConfig {
    pub seed: u64,
    pub sampling: SamplingMethod,
    pub initial_state: Option<InitialStateSigmas>,
    pub atmosphere: Option<AtmosphereSigmas>,
    pub aerodynamics: Option<AerodynamicsSigmas>,
    pub navigation: Option<NavigationSigmas>,
    pub mass: Option<MassSigmas>,
    pub vehicle: Option<VehicleSigmas>,
    pub pilot: Option<PilotSigmas>,
    pub nav_filter: Option<NavFilterSigmas>,
    pub wind: Option<WindDispersionConfig>,
    pub density_perturbation: Option<DensityPerturbationConfig>,
}

// ────────────────────────────────────────────────────────────────────
// Draw generation
// ────────────────────────────────────────────────────────────────────

/// One simulation's dispersion draws.
/// Values are in SI units, ready to apply directly.
#[derive(Debug, Clone)]
pub struct DispersionDraw {
    // Initial state (Gaussian draws × sigma, SI units)
    pub altitude: f64,    // meters
    pub longitude: f64,   // radians
    pub latitude: f64,    // radians
    pub velocity: f64,    // m/s
    pub flight_path: f64, // radians
    pub azimuth: f64,     // radians

    // Atmosphere (Uniform draw × sigma, fractional)
    pub density: f64, // fractional (e.g. 0.15 = +15%)

    // Aerodynamics (Uniform draws × sigma, fractional/radians)
    pub drag_coeff: f64, // fractional
    pub lift_coeff: f64, // fractional
    pub incidence: f64,  // radians

    // Navigation (Gaussian draws × sigma, SI units)
    pub nav_altitude: f64,    // meters
    pub nav_longitude: f64,   // radians
    pub nav_latitude: f64,    // radians
    pub nav_velocity: f64,    // m/s
    pub nav_flight_path: f64, // radians
    pub nav_azimuth: f64,     // radians
    pub nav_drag_accel: f64,  // m/s²

    // Mass (Uniform draw × sigma, fractional)
    pub mass: f64, // fractional

    // Vehicle (Uniform draws × sigma, fractional)
    pub ref_area: f64,      // fractional
    pub max_bank_rate: f64, // fractional

    // Pilot dynamics (Uniform draws × sigma, fractional)
    pub pilot_tau: f64,       // fractional
    pub pilot_damping: f64,   // fractional
    pub pilot_frequency: f64, // fractional

    // Navigation filter (Gaussian draw × sigma, absolute)
    pub filter_gain: f64, // absolute delta on lambda

    // Wind dispersions
    pub wind_scale: f64,          // multiplicative (1.0 = no change)
    pub wind_direction_bias: f64, // rotation in radians
}

impl Default for DispersionDraw {
    fn default() -> Self {
        Self {
            altitude: 0.0,
            longitude: 0.0,
            latitude: 0.0,
            velocity: 0.0,
            flight_path: 0.0,
            azimuth: 0.0,
            density: 0.0,
            drag_coeff: 0.0,
            lift_coeff: 0.0,
            incidence: 0.0,
            nav_altitude: 0.0,
            nav_longitude: 0.0,
            nav_latitude: 0.0,
            nav_velocity: 0.0,
            nav_flight_path: 0.0,
            nav_azimuth: 0.0,
            nav_drag_accel: 0.0,
            mass: 0.0,
            ref_area: 0.0,
            max_bank_rate: 0.0,
            pilot_tau: 0.0,
            pilot_damping: 0.0,
            pilot_frequency: 0.0,
            filter_gain: 0.0,
            wind_scale: 1.0,          // 1.0 = no scaling (identity)
            wind_direction_bias: 0.0, // no rotation
        }
    }
}

/// Number of fields in [`DispersionDraw`] — keep in sync with [`DispersionDraw::to_array`].
pub const DISPERSION_DRAW_LEN: usize = 26;

impl DispersionDraw {
    /// Serialize all fields to a flat array in struct field order.
    pub fn to_array(&self) -> [f64; DISPERSION_DRAW_LEN] {
        [
            self.altitude,
            self.longitude,
            self.latitude,
            self.velocity,
            self.flight_path,
            self.azimuth,
            self.density,
            self.drag_coeff,
            self.lift_coeff,
            self.incidence,
            self.nav_altitude,
            self.nav_longitude,
            self.nav_latitude,
            self.nav_velocity,
            self.nav_flight_path,
            self.nav_azimuth,
            self.nav_drag_accel,
            self.mass,
            self.ref_area,
            self.max_bank_rate,
            self.pilot_tau,
            self.pilot_damping,
            self.pilot_frequency,
            self.filter_gain,
            self.wind_scale,
            self.wind_direction_bias,
        ]
    }

    /// Deserialize all fields from a flat array in struct field order (inverse of `to_array()`).
    pub fn from_array(a: [f64; DISPERSION_DRAW_LEN]) -> Self {
        Self {
            altitude: a[0], longitude: a[1], latitude: a[2], velocity: a[3],
            flight_path: a[4], azimuth: a[5], density: a[6], drag_coeff: a[7],
            lift_coeff: a[8], incidence: a[9], nav_altitude: a[10], nav_longitude: a[11],
            nav_latitude: a[12], nav_velocity: a[13], nav_flight_path: a[14],
            nav_azimuth: a[15], nav_drag_accel: a[16], mass: a[17], ref_area: a[18],
            max_bank_rate: a[19], pilot_tau: a[20], pilot_damping: a[21],
            pilot_frequency: a[22], filter_gain: a[23], wind_scale: a[24],
            wind_direction_bias: a[25],
        }
    }
}

impl DispersionConfig {
    /// Build per-dimension transforms from the sigma config.
    /// Index order matches `DispersionDraw::to_array()` / `DISPERSION_DRAW_LEN`.
    pub fn build_dim_transforms(&self) -> [DimTransform; DISPERSION_DRAW_LEN] {
        // Helper closures
        let gauss = |sigma: f64| DimTransform::Gaussian { sigma };
        let unif = |hw: f64| DimTransform::Uniform { half_width: hw };

        // dims 0-5: initial state
        let (alt_sigma, lon_sigma, lat_sigma, vel_sigma, fpa_sigma, az_sigma) =
            if let Some(ref s) = self.initial_state {
                (s.altitude * 1e3, s.longitude * DEG2RAD, s.latitude * DEG2RAD, s.velocity, s.flight_path * DEG2RAD, s.azimuth * DEG2RAD)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            };

        // dim 6: atmosphere density
        let atm_hw = self.atmosphere.as_ref().map(|s| s.density / 100.0).unwrap_or(0.0);

        // dims 7-9: aero drag, lift, incidence
        let (drag_hw, lift_hw, inc_hw) = self.aerodynamics.as_ref()
            .map(|s| (s.drag / 100.0, s.lift / 100.0, s.incidence * DEG2RAD))
            .unwrap_or((0.0, 0.0, 0.0));

        // dims 10-16: navigation
        let (nav_alt, nav_lon, nav_lat, nav_vel, nav_fpa, nav_az, nav_drag) =
            if let Some(ref s) = self.navigation {
                (s.altitude * 1e3, s.longitude * DEG2RAD, s.latitude * DEG2RAD, s.velocity, s.flight_path * DEG2RAD, s.azimuth * DEG2RAD, s.drag_accel)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            };

        // dim 17: mass
        let mass_hw = self.mass.as_ref().map(|s| s.mass / 100.0).unwrap_or(0.0);

        // dims 18-19: vehicle ref_area, max_bank_rate
        let (area_hw, bank_rate_hw) = self.vehicle.as_ref()
            .map(|s| (s.ref_area / 100.0, s.max_bank_rate / 100.0))
            .unwrap_or((0.0, 0.0));

        // dims 20-22: pilot tau, damping, freq
        let (tau_hw, damp_hw, freq_hw) = self.pilot.as_ref()
            .map(|s| (s.time_constant / 100.0, s.damping / 100.0, s.frequency / 100.0))
            .unwrap_or((0.0, 0.0, 0.0));

        // dim 23: nav_filter (Gaussian)
        let nav_filter_sigma = self.nav_filter.as_ref().map(|s| s.filter_gain).unwrap_or(0.0);

        // dims 24-25: wind scale (UniformRange or Fixed), direction bias (Uniform or Fixed)
        let (wind_scale_tx, wind_dir_tx) = if let Some(ref w) = self.wind {
            (
                DimTransform::UniformRange { min: w.scale_min, max: w.scale_max },
                unif(w.direction_bias_deg * DEG2RAD),
            )
        } else {
            (DimTransform::Fixed(1.0), DimTransform::Fixed(0.0))
        };

        [
            gauss(alt_sigma),           // 0: altitude
            gauss(lon_sigma),           // 1: longitude
            gauss(lat_sigma),           // 2: latitude
            gauss(vel_sigma),           // 3: velocity
            gauss(fpa_sigma),           // 4: flight_path
            gauss(az_sigma),            // 5: azimuth
            unif(atm_hw),               // 6: density
            unif(drag_hw),              // 7: drag_coeff
            unif(lift_hw),              // 8: lift_coeff
            unif(inc_hw),               // 9: incidence
            gauss(nav_alt),             // 10: nav_altitude
            gauss(nav_lon),             // 11: nav_longitude
            gauss(nav_lat),             // 12: nav_latitude
            gauss(nav_vel),             // 13: nav_velocity
            gauss(nav_fpa),             // 14: nav_flight_path
            gauss(nav_az),              // 15: nav_azimuth
            gauss(nav_drag),            // 16: nav_drag_accel
            unif(mass_hw),              // 17: mass
            unif(area_hw),              // 18: ref_area
            unif(bank_rate_hw),         // 19: max_bank_rate
            unif(tau_hw),               // 20: pilot_tau
            unif(damp_hw),              // 21: pilot_damping
            unif(freq_hw),              // 22: pilot_frequency
            gauss(nav_filter_sigma),    // 23: filter_gain
            wind_scale_tx,              // 24: wind_scale
            wind_dir_tx,                // 25: wind_direction_bias
        ]
    }

    /// Generate LHS unit samples: N samples x 26 dimensions, values in [0,1).
    /// Each stratum [k/N, (k+1)/N) contains exactly one sample per dimension.
    pub fn generate_lhs_unit_samples(&self, n: usize) -> Vec<[f64; DISPERSION_DRAW_LEN]> {
        use rand::RngExt;
        if n == 0 {
            return Vec::new();
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        // Build one permutation per dimension via Fisher-Yates
        let dim_perms: Vec<Vec<usize>> = (0..DISPERSION_DRAW_LEN)
            .map(|_| {
                let mut perm: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    let j = rng.random_range(0..=i);
                    perm.swap(i, j);
                }
                perm
            })
            .collect();

        let mut samples: Vec<[f64; DISPERSION_DRAW_LEN]> = vec![[0.0; DISPERSION_DRAW_LEN]; n];
        for (i, row) in samples.iter_mut().enumerate() {
            for (d, perm) in dim_perms.iter().enumerate() {
                row[d] = (perm[i] as f64 + rng.random::<f64>()) / n as f64;
            }
        }
        samples
    }

    /// Generate Sobol quasi-random unit samples: N samples x 26 dimensions, values in [0,1].
    pub fn generate_sobol_unit_samples(&self, n: usize) -> Vec<[f64; DISPERSION_DRAW_LEN]> {
        let seed = self.seed as u32;
        (0..n)
            .map(|i| {
                let mut sample = [0.0f64; DISPERSION_DRAW_LEN];
                for d in 0..DISPERSION_DRAW_LEN {
                    sample[d] = sobol_burley::sample(i as u32, d as u32, seed) as f64;
                }
                sample
            })
            .collect()
    }

    /// Map a slice of unit samples (each row is one sim, 26 dims in [0,1]) through the
    /// per-dimension transforms to produce `DispersionDraw` values.
    fn draws_from_unit_samples(&self, unit_samples: &[[f64; DISPERSION_DRAW_LEN]]) -> Vec<DispersionDraw> {
        let transforms = self.build_dim_transforms();
        unit_samples
            .iter()
            .map(|row| {
                let arr: [f64; DISPERSION_DRAW_LEN] = std::array::from_fn(|d| transforms[d].apply(row[d]));
                DispersionDraw::from_array(arr)
            })
            .collect()
    }

    /// Generate all dispersion draws for a batch of simulations.
    ///
    /// Dispatches to the configured sampling method:
    /// - `Random`: seeded PRNG (backward-compatible, same seed = same draws)
    /// - `Lhs`: Latin Hypercube Sampling via `generate_lhs_unit_samples`
    /// - `Sobol`: Sobol quasi-random sequence (limited to 65536 samples)
    pub fn generate_draws(&self, n_sims: usize) -> Vec<DispersionDraw> {
        match self.sampling {
            SamplingMethod::Random => self.generate_draws_random(n_sims),
            SamplingMethod::Lhs => {
                let unit_samples = self.generate_lhs_unit_samples(n_sims);
                self.draws_from_unit_samples(&unit_samples)
            }
            SamplingMethod::Sobol => {
                assert!(
                    n_sims <= 65_536,
                    "Sobol sampling limited to 65536 samples, got {}",
                    n_sims,
                );
                let unit_samples = self.generate_sobol_unit_samples(n_sims);
                self.draws_from_unit_samples(&unit_samples)
            }
        }
    }

    /// Random draw generation (legacy PRNG path). Used by `generate_draws` for `SamplingMethod::Random`.
    ///
    /// Uses a seeded RNG for reproducibility. Draw order:
    /// initial state (Gaussian), atmosphere (Uniform), aero (Uniform),
    /// nav (Gaussian), mass (Uniform), vehicle (Uniform), pilot (Uniform),
    /// nav_filter (Gaussian), wind (Uniform scale + Uniform direction bias).
    fn generate_draws_random(&self, n_sims: usize) -> Vec<DispersionDraw> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let uniform = Uniform::new(-1.0_f64, 1.0).unwrap();

        (0..n_sims)
            .map(|_| {
                let mut draw = DispersionDraw::default();

                // Initial state (Gaussian)
                if let Some(ref s) = self.initial_state {
                    draw.altitude = normal.sample(&mut rng) * s.altitude * 1e3;
                    draw.longitude = normal.sample(&mut rng) * s.longitude * DEG2RAD;
                    draw.latitude = normal.sample(&mut rng) * s.latitude * DEG2RAD;
                    draw.velocity = normal.sample(&mut rng) * s.velocity;
                    draw.flight_path = normal.sample(&mut rng) * s.flight_path * DEG2RAD;
                    draw.azimuth = normal.sample(&mut rng) * s.azimuth * DEG2RAD;
                }

                // Atmosphere (Uniform)
                if let Some(ref s) = self.atmosphere {
                    draw.density = uniform.sample(&mut rng) * s.density / 100.0;
                }

                // Aerodynamics (Uniform)
                if let Some(ref s) = self.aerodynamics {
                    draw.drag_coeff = uniform.sample(&mut rng) * s.drag / 100.0;
                    draw.lift_coeff = uniform.sample(&mut rng) * s.lift / 100.0;
                    draw.incidence = uniform.sample(&mut rng) * s.incidence * DEG2RAD;
                }

                // Navigation (Gaussian)
                if let Some(ref s) = self.navigation {
                    draw.nav_altitude = normal.sample(&mut rng) * s.altitude * 1e3;
                    draw.nav_longitude = normal.sample(&mut rng) * s.longitude * DEG2RAD;
                    draw.nav_latitude = normal.sample(&mut rng) * s.latitude * DEG2RAD;
                    draw.nav_velocity = normal.sample(&mut rng) * s.velocity;
                    draw.nav_flight_path = normal.sample(&mut rng) * s.flight_path * DEG2RAD;
                    draw.nav_azimuth = normal.sample(&mut rng) * s.azimuth * DEG2RAD;
                    draw.nav_drag_accel = normal.sample(&mut rng) * s.drag_accel;
                }

                // Mass (Uniform)
                if let Some(ref s) = self.mass {
                    draw.mass = uniform.sample(&mut rng) * s.mass / 100.0;
                }

                // Vehicle (Uniform)
                if let Some(ref s) = self.vehicle {
                    draw.ref_area = uniform.sample(&mut rng) * s.ref_area / 100.0;
                    draw.max_bank_rate = uniform.sample(&mut rng) * s.max_bank_rate / 100.0;
                }

                // Pilot dynamics (Uniform)
                if let Some(ref s) = self.pilot {
                    draw.pilot_tau = uniform.sample(&mut rng) * s.time_constant / 100.0;
                    draw.pilot_damping = uniform.sample(&mut rng) * s.damping / 100.0;
                    draw.pilot_frequency = uniform.sample(&mut rng) * s.frequency / 100.0;
                }

                // Navigation filter (Gaussian)
                if let Some(ref s) = self.nav_filter {
                    draw.filter_gain = normal.sample(&mut rng) * s.filter_gain;
                }

                // Wind (Uniform scale in [min, max], Uniform direction bias in [-deg, +deg])
                if let Some(ref w) = self.wind {
                    let scale_uniform = Uniform::new(w.scale_min, w.scale_max).unwrap();
                    draw.wind_scale = scale_uniform.sample(&mut rng);
                    draw.wind_direction_bias =
                        uniform.sample(&mut rng) * w.direction_bias_deg * DEG2RAD;
                } else {
                    draw.wind_scale = 1.0; // no-op scale
                    draw.wind_direction_bias = 0.0;
                }

                draw
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn medium_config(seed: u64) -> DispersionConfig {
        DispersionConfig {
            seed,
            sampling: SamplingMethod::Random,
            initial_state: Some(InitialStateSigmas::from_level(DispersionLevel::Medium)),
            atmosphere: Some(AtmosphereSigmas::from_level(DispersionLevel::Medium)),
            aerodynamics: Some(AerodynamicsSigmas::from_level(DispersionLevel::Medium)),
            navigation: Some(NavigationSigmas::from_level(DispersionLevel::Medium)),
            mass: Some(MassSigmas::from_level(DispersionLevel::Medium)),
            vehicle: Some(VehicleSigmas::from_level(DispersionLevel::Medium)),
            pilot: Some(PilotSigmas::from_level(DispersionLevel::Medium)),
            nav_filter: Some(NavFilterSigmas::from_level(DispersionLevel::Medium)),
            wind: None,
            density_perturbation: None,
        }
    }

    #[test]
    fn test_generate_draws_reproducible() {
        let draws_a = medium_config(42).generate_draws(10);
        let draws_b = medium_config(42).generate_draws(10);
        for (a, b) in draws_a.iter().zip(draws_b.iter()) {
            assert_eq!(a.altitude, b.altitude);
            assert_eq!(a.velocity, b.velocity);
            assert_eq!(a.density, b.density);
            assert_eq!(a.drag_coeff, b.drag_coeff);
            assert_eq!(a.nav_altitude, b.nav_altitude);
            assert_eq!(a.mass, b.mass);
            assert_eq!(a.ref_area, b.ref_area);
            assert_eq!(a.pilot_tau, b.pilot_tau);
            assert_eq!(a.filter_gain, b.filter_gain);
        }
    }

    #[test]
    fn test_generate_draws_different_seeds() {
        let draws_a = medium_config(42).generate_draws(5);
        let draws_b = medium_config(99).generate_draws(5);
        // With different seeds, at least one draw should differ
        let any_differ = draws_a
            .iter()
            .zip(draws_b.iter())
            .any(|(a, b)| a.velocity != b.velocity);
        assert!(any_differ, "Different seeds should produce different draws");
    }

    #[test]
    fn test_generate_draws_count() {
        for n in [0, 1, 5, 100] {
            let draws = medium_config(42).generate_draws(n);
            assert_eq!(draws.len(), n);
        }
    }

    #[test]
    fn test_all_none_gives_zeros() {
        let config = DispersionConfig {
            seed: 42,
            sampling: SamplingMethod::Random,
            initial_state: None,
            atmosphere: None,
            aerodynamics: None,
            navigation: None,
            mass: None,
            vehicle: None,
            pilot: None,
            nav_filter: None,
            wind: None,
            density_perturbation: None,
        };
        let draws = config.generate_draws(10);
        for d in &draws {
            assert_eq!(d.altitude, 0.0);
            assert_eq!(d.longitude, 0.0);
            assert_eq!(d.velocity, 0.0);
            assert_eq!(d.density, 0.0);
            assert_eq!(d.drag_coeff, 0.0);
            assert_eq!(d.nav_altitude, 0.0);
            assert_eq!(d.mass, 0.0);
            assert_eq!(d.ref_area, 0.0);
            assert_eq!(d.max_bank_rate, 0.0);
            assert_eq!(d.pilot_tau, 0.0);
            assert_eq!(d.pilot_damping, 0.0);
            assert_eq!(d.pilot_frequency, 0.0);
            assert_eq!(d.filter_gain, 0.0);
            assert_eq!(
                d.wind_scale, 1.0,
                "wind_scale default should be 1.0 (identity)"
            );
            assert_eq!(d.wind_direction_bias, 0.0);
        }
    }

    #[test]
    fn test_sigma_presets_nonzero() {
        for level in [
            DispersionLevel::Low,
            DispersionLevel::Medium,
            DispersionLevel::High,
        ] {
            let s = InitialStateSigmas::from_level(level);
            assert!(
                s.velocity > 0.0,
                "velocity sigma should be > 0 for {:?}",
                level
            );

            let a = AtmosphereSigmas::from_level(level);
            assert!(a.density > 0.0);

            let n = NavigationSigmas::from_level(level);
            assert!(n.altitude > 0.0);

            let v = VehicleSigmas::from_level(level);
            assert!(v.ref_area > 0.0);
            assert!(v.max_bank_rate > 0.0);

            let p = PilotSigmas::from_level(level);
            assert!(p.time_constant > 0.0);

            let nf = NavFilterSigmas::from_level(level);
            assert!(nf.filter_gain > 0.0);
        }

        let s_off = InitialStateSigmas::from_level(DispersionLevel::Off);
        assert_eq!(s_off.altitude, 0.0);
        assert_eq!(s_off.velocity, 0.0);

        let v_off = VehicleSigmas::from_level(DispersionLevel::Off);
        assert_eq!(v_off.ref_area, 0.0);
        assert_eq!(v_off.max_bank_rate, 0.0);
    }

    #[test]
    fn test_wind_config_off() {
        let cfg = WindDispersionConfig::from_level(DispersionLevel::Off);
        assert_eq!(cfg.scale_min, 1.0);
        assert_eq!(cfg.scale_max, 1.0);
        assert_eq!(cfg.direction_bias_deg, 0.0);
    }

    #[test]
    fn test_wind_config_medium() {
        let cfg = WindDispersionConfig::from_level(DispersionLevel::Medium);
        assert_eq!(cfg.scale_min, 0.5);
        assert_eq!(cfg.scale_max, 1.5);
        assert_eq!(cfg.direction_bias_deg, 10.0);
    }

    #[test]
    fn test_wind_config_high() {
        let cfg = WindDispersionConfig::from_level(DispersionLevel::High);
        assert_eq!(cfg.scale_min, 0.2);
        assert_eq!(cfg.scale_max, 2.0);
        assert_eq!(cfg.direction_bias_deg, 20.0);
    }

    #[test]
    fn test_wind_config_custom_defaults_to_medium() {
        let cfg = WindDispersionConfig::from_level(DispersionLevel::Custom);
        let med = WindDispersionConfig::from_level(DispersionLevel::Medium);
        assert_eq!(cfg.scale_min, med.scale_min);
        assert_eq!(cfg.scale_max, med.scale_max);
        assert_eq!(cfg.direction_bias_deg, med.direction_bias_deg);
    }

    #[test]
    fn test_uniform_fields_bounded() {
        let config = DispersionConfig {
            seed: 12345,
            sampling: SamplingMethod::Random,
            initial_state: None,
            atmosphere: Some(AtmosphereSigmas { density: 50.0 }),
            aerodynamics: Some(AerodynamicsSigmas {
                drag: 5.0,
                lift: 10.0,
                incidence: 1.0,
            }),
            navigation: None,
            mass: Some(MassSigmas { mass: 1.0 }),
            vehicle: Some(VehicleSigmas {
                ref_area: 2.0,
                max_bank_rate: 10.0,
            }),
            pilot: Some(PilotSigmas {
                time_constant: 10.0,
                damping: 10.0,
                frequency: 10.0,
            }),
            nav_filter: None,
            wind: None,
            density_perturbation: None,
        };
        let draws = config.generate_draws(1000);
        for d in &draws {
            // Uniform[-1,1] * sigma/100, so |value| <= sigma/100
            assert!(
                d.density.abs() <= 0.50 + 1e-10,
                "density out of bounds: {}",
                d.density
            );
            assert!(
                d.drag_coeff.abs() <= 0.05 + 1e-10,
                "drag out of bounds: {}",
                d.drag_coeff
            );
            assert!(
                d.lift_coeff.abs() <= 0.10 + 1e-10,
                "lift out of bounds: {}",
                d.lift_coeff
            );
            assert!(
                d.incidence.abs() <= 1.0 * DEG2RAD + 1e-10,
                "incidence out of bounds: {}",
                d.incidence
            );
            assert!(
                d.mass.abs() <= 0.01 + 1e-10,
                "mass out of bounds: {}",
                d.mass
            );
            assert!(
                d.ref_area.abs() <= 0.02 + 1e-10,
                "ref_area out of bounds: {}",
                d.ref_area
            );
            assert!(
                d.max_bank_rate.abs() <= 0.10 + 1e-10,
                "max_bank_rate out of bounds: {}",
                d.max_bank_rate
            );
            assert!(
                d.pilot_tau.abs() <= 0.10 + 1e-10,
                "pilot_tau out of bounds: {}",
                d.pilot_tau
            );
            assert!(
                d.pilot_damping.abs() <= 0.10 + 1e-10,
                "pilot_damping out of bounds: {}",
                d.pilot_damping
            );
            assert!(
                d.pilot_frequency.abs() <= 0.10 + 1e-10,
                "pilot_frequency out of bounds: {}",
                d.pilot_frequency
            );
        }
    }

    #[test]
    fn test_filter_gain_gaussian_range() {
        let config = DispersionConfig {
            seed: 54321,
            sampling: SamplingMethod::Random,
            initial_state: None,
            atmosphere: None,
            aerodynamics: None,
            navigation: None,
            mass: None,
            vehicle: None,
            pilot: None,
            nav_filter: Some(NavFilterSigmas { filter_gain: 0.10 }),
            wind: None,
            density_perturbation: None,
        };
        let draws = config.generate_draws(1000);
        // Gaussian: most draws within ±3sigma = ±0.30
        let within_3sigma = draws.iter().filter(|d| d.filter_gain.abs() <= 0.30).count();
        assert!(
            within_3sigma > 990,
            "Expected >99% within 3-sigma, got {}/1000",
            within_3sigma
        );
        // At least some should be nonzero
        let any_nonzero = draws.iter().any(|d| d.filter_gain.abs() > 0.001);
        assert!(any_nonzero, "Filter gain draws should not all be zero");
    }

    #[test]
    fn dispersion_draw_to_array_roundtrip() {
        let draw = DispersionDraw {
            altitude: 1.0,
            longitude: 2.0,
            latitude: 3.0,
            velocity: 4.0,
            flight_path: 5.0,
            azimuth: 6.0,
            density: 7.0,
            drag_coeff: 8.0,
            lift_coeff: 9.0,
            incidence: 10.0,
            nav_altitude: 11.0,
            nav_longitude: 12.0,
            nav_latitude: 13.0,
            nav_velocity: 14.0,
            nav_flight_path: 15.0,
            nav_azimuth: 16.0,
            nav_drag_accel: 17.0,
            mass: 18.0,
            ref_area: 19.0,
            max_bank_rate: 20.0,
            pilot_tau: 21.0,
            pilot_damping: 22.0,
            pilot_frequency: 23.0,
            filter_gain: 24.0,
            wind_scale: 25.0,
            wind_direction_bias: 26.0,
        };
        let arr = draw.to_array();
        assert_eq!(arr.len(), 26);
        for (i, &val) in arr.iter().enumerate() {
            assert_eq!(val, (i + 1) as f64);
        }
    }

    #[test]
    fn dispersion_draw_default_to_array_len() {
        let arr = DispersionDraw::default().to_array();
        assert_eq!(arr.len(), 26);
        // All zeros except wind_scale which defaults to 1.0
        assert!(arr[..24].iter().all(|&v| v == 0.0));
        assert_eq!(arr[24], 1.0, "wind_scale default is 1.0");
        assert_eq!(arr[25], 0.0, "wind_direction_bias default is 0.0");
    }

    #[test]
    fn test_dispersion_level_parsing() {
        assert_eq!(
            DispersionLevel::from_str("off").unwrap(),
            DispersionLevel::Off
        );
        assert_eq!(
            DispersionLevel::from_str("low").unwrap(),
            DispersionLevel::Low
        );
        assert_eq!(
            DispersionLevel::from_str("medium").unwrap(),
            DispersionLevel::Medium
        );
        assert_eq!(
            DispersionLevel::from_str("high").unwrap(),
            DispersionLevel::High
        );
        assert_eq!(
            DispersionLevel::from_str("custom").unwrap(),
            DispersionLevel::Custom
        );
        assert!(DispersionLevel::from_str("invalid").is_err());
    }

    #[test]
    fn test_density_perturbation_config_off() {
        let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Off);
        assert_eq!(cfg.sigma, 0.0);
        assert_eq!(cfg.tau, 0.0);
    }

    #[test]
    fn test_density_perturbation_config_low() {
        let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Low);
        assert_eq!(cfg.tau, 120.0);
        assert_eq!(cfg.sigma, 0.05);
    }

    #[test]
    fn test_density_perturbation_config_medium() {
        let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Medium);
        assert_eq!(cfg.tau, 60.0);
        assert_eq!(cfg.sigma, 0.10);
    }

    #[test]
    fn test_density_perturbation_config_high() {
        let cfg = DensityPerturbationConfig::from_level(DispersionLevel::High);
        assert_eq!(cfg.tau, 30.0);
        assert_eq!(cfg.sigma, 0.20);
    }

    #[test]
    fn test_density_perturbation_config_custom_defaults_to_medium() {
        let cfg = DensityPerturbationConfig::from_level(DispersionLevel::Custom);
        assert_eq!(cfg.tau, 60.0);
        assert_eq!(cfg.sigma, 0.10);
    }

    #[test]
    fn test_density_perturbation_is_disabled() {
        assert!(
            DensityPerturbationConfig::from_level(DispersionLevel::Off).is_disabled(),
            "Off preset should be disabled"
        );
        assert!(
            !DensityPerturbationConfig::from_level(DispersionLevel::Medium).is_disabled(),
            "Medium preset should not be disabled"
        );
    }

    #[test]
    fn test_step_density_perturbation_disabled_sigma_zero() {
        assert_eq!(step_density_perturbation(0.5, 0.1, 60.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn test_step_density_perturbation_disabled_tau_zero() {
        assert_eq!(step_density_perturbation(0.5, 0.1, 0.0, 0.10, 1.0), 0.0);
    }

    #[test]
    fn test_step_density_perturbation_deterministic() {
        let a = step_density_perturbation(0.0, 0.1, 60.0, 0.10, 0.5);
        let b = step_density_perturbation(0.0, 0.1, 60.0, 0.10, 0.5);
        assert_eq!(a, b);
    }

    #[test]
    fn test_step_density_perturbation_decay() {
        // With zero noise (normal_sample=0), the state should decay toward 0
        let x = step_density_perturbation(1.0, 0.1, 60.0, 0.10, 0.0);
        assert!(x < 1.0, "state should decay: got {}", x);
        assert!(
            x > 0.0,
            "state should remain positive with no noise: got {}",
            x
        );
    }

    #[test]
    fn test_step_density_perturbation_statistical_properties() {
        // Run many steps from x=0 and check steady-state variance ~ sigma^2
        let tau = 60.0;
        let sigma = 0.10;
        let dt = 0.1;
        let n_steps = 100_000;

        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut x = 0.0;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let burn_in = 10_000; // let it reach steady state

        for i in 0..n_steps {
            let z = normal.sample(&mut rng);
            x = step_density_perturbation(x, dt, tau, sigma, z);
            if i >= burn_in {
                sum += x;
                sum_sq += x * x;
            }
        }

        let n = (n_steps - burn_in) as f64;
        let mean = sum / n;
        let variance = sum_sq / n - mean * mean;

        // Mean should be ~0
        assert!(mean.abs() < 0.02, "mean should be ~0, got {}", mean);
        // Variance should be ~sigma^2 = 0.01
        assert!(
            (variance - sigma * sigma).abs() < 0.002,
            "variance should be ~{}, got {}",
            sigma * sigma,
            variance
        );
    }

    #[test]
    fn test_sampling_method_parsing() {
        assert_eq!(SamplingMethod::from_str("random").unwrap(), SamplingMethod::Random);
        assert_eq!(SamplingMethod::from_str("lhs").unwrap(), SamplingMethod::Lhs);
        assert_eq!(SamplingMethod::from_str("sobol").unwrap(), SamplingMethod::Sobol);
        // case-insensitive
        assert_eq!(SamplingMethod::from_str("LHS").unwrap(), SamplingMethod::Lhs);
        assert_eq!(SamplingMethod::from_str("Random").unwrap(), SamplingMethod::Random);
        assert_eq!(SamplingMethod::from_str("SOBOL").unwrap(), SamplingMethod::Sobol);
        // unknown string errors
        assert!(SamplingMethod::from_str("invalid").is_err());
        assert!(SamplingMethod::from_str("").is_err());
    }

    #[test]
    fn test_sampling_method_default_is_random() {
        assert_eq!(SamplingMethod::default(), SamplingMethod::Random);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn step_always_finite(
                x in -10.0f64..10.0,
                dt in 0.001f64..10.0,
                tau in 0.01f64..1000.0,
                sigma in 0.0f64..1.0,
                z in -5.0f64..5.0,
            ) {
                let result = step_density_perturbation(x, dt, tau, sigma, z);
                prop_assert!(result.is_finite(), "got {}", result);
            }

            #[test]
            fn all_sampling_methods_produce_finite_draws(
                seed in 0u64..10_000,
                n_sims in 1usize..200,
                method_idx in 0u32..3,
            ) {
                let method = match method_idx {
                    0 => SamplingMethod::Random,
                    1 => SamplingMethod::Lhs,
                    _ => SamplingMethod::Sobol,
                };
                let config = DispersionConfig {
                    seed,
                    sampling: method,
                    initial_state: Some(InitialStateSigmas::from_level(DispersionLevel::Medium)),
                    atmosphere: Some(AtmosphereSigmas::from_level(DispersionLevel::Medium)),
                    aerodynamics: Some(AerodynamicsSigmas::from_level(DispersionLevel::Medium)),
                    navigation: Some(NavigationSigmas::from_level(DispersionLevel::Medium)),
                    mass: Some(MassSigmas::from_level(DispersionLevel::Medium)),
                    vehicle: Some(VehicleSigmas::from_level(DispersionLevel::Medium)),
                    pilot: Some(PilotSigmas::from_level(DispersionLevel::Medium)),
                    nav_filter: Some(NavFilterSigmas::from_level(DispersionLevel::Medium)),
                    wind: None,
                    density_perturbation: None,
                };
                let draws = config.generate_draws(n_sims);
                prop_assert_eq!(draws.len(), n_sims);
                for draw in &draws {
                    let arr = draw.to_array();
                    for &val in &arr {
                        prop_assert!(val.is_finite(), "non-finite draw value: {}", val);
                    }
                }
            }
        }
    }

    // ── Task 2: norm_ppf + DimTransform tests ──────────────────────────────

    #[test]
    fn test_norm_ppf_known_values() {
        let tol = 1e-6;
        assert!((norm_ppf(0.5) - 0.0).abs() < tol, "p=0.5 -> 0");
        assert!((norm_ppf(0.841344746) - 1.0).abs() < tol, "p=0.841 -> 1");
        assert!((norm_ppf(0.158655254) - (-1.0)).abs() < tol, "p=0.159 -> -1");
        assert!((norm_ppf(0.977249868) - 2.0).abs() < tol, "p=0.977 -> 2");
        assert!((norm_ppf(0.022750132) - (-2.0)).abs() < tol, "p=0.023 -> -2");
        assert!((norm_ppf(0.998650102) - 3.0).abs() < tol, "p~0.99865 -> 3");
    }

    #[test]
    fn test_norm_ppf_symmetry() {
        let tol = 1e-12;
        for p in [0.01, 0.1, 0.25, 0.4] {
            let sum = norm_ppf(p) + norm_ppf(1.0 - p);
            assert!(sum.abs() < tol, "symmetry failed at p={}: sum={}", p, sum);
        }
    }

    #[test]
    fn test_dim_transform_gaussian() {
        let tx = DimTransform::Gaussian { sigma: 2.0 };
        // u=0.5 -> norm_ppf(0.5)=0.0 -> 0.0*2.0=0.0
        assert!((tx.apply(0.5) - 0.0).abs() < 1e-12);
        // u=0.841344746 -> ~1.0 * 2.0 = ~2.0
        assert!((tx.apply(0.841344746) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_dim_transform_uniform() {
        let tx = DimTransform::Uniform { half_width: 5.0 };
        // u=0.5 -> (2*0.5-1)*5 = 0.0
        assert_eq!(tx.apply(0.5), 0.0);
        // u=1.0 -> (2*1.0-1)*5 = 5.0
        assert_eq!(tx.apply(1.0), 5.0);
        // u=0.0 -> (2*0.0-1)*5 = -5.0
        assert_eq!(tx.apply(0.0), -5.0);
    }

    #[test]
    fn test_dim_transform_uniform_range() {
        let tx = DimTransform::UniformRange { min: 0.5, max: 1.5 };
        // u=0.0 -> 0.5 + 0.0*1.0 = 0.5
        assert_eq!(tx.apply(0.0), 0.5);
        // u=1.0 -> 0.5 + 1.0*1.0 = 1.5
        assert_eq!(tx.apply(1.0), 1.5);
        // u=0.5 -> 1.0
        assert_eq!(tx.apply(0.5), 1.0);
    }

    #[test]
    fn test_dim_transform_fixed() {
        let tx = DimTransform::Fixed(42.0);
        assert_eq!(tx.apply(0.0), 42.0);
        assert_eq!(tx.apply(0.5), 42.0);
        assert_eq!(tx.apply(1.0), 42.0);
    }

    #[test]
    fn test_build_dim_transforms_medium_config() {
        let cfg = medium_config(42);
        let txs = cfg.build_dim_transforms();
        // dim 0 (altitude) should be Gaussian
        assert!(matches!(txs[0], DimTransform::Gaussian { .. }), "dim 0 should be Gaussian");
        // dim 6 (density) should be Uniform
        assert!(matches!(txs[6], DimTransform::Uniform { .. }), "dim 6 should be Uniform");
        // wind=None -> dim 24 = Fixed(1.0), dim 25 = Fixed(0.0)
        assert_eq!(txs[24], DimTransform::Fixed(1.0), "dim 24 wind=None should be Fixed(1.0)");
        assert_eq!(txs[25], DimTransform::Fixed(0.0), "dim 25 wind=None should be Fixed(0.0)");
    }

    // ── Task 3: LHS tests ──────────────────────────────────────────────────

    #[test]
    fn test_lhs_stratification() {
        let cfg = medium_config(42);
        let n = 100usize;
        let samples = cfg.generate_lhs_unit_samples(n);
        assert_eq!(samples.len(), n);
        // Each stratum [k/n, (k+1)/n) must contain exactly one sample per dimension
        for d in 0..DISPERSION_DRAW_LEN {
            let mut stratum_counts = vec![0u32; n];
            for row in &samples {
                let v = row[d];
                assert!(v >= 0.0 && v < 1.0, "dim {} value {} out of [0,1)", d, v);
                let k = (v * n as f64) as usize;
                stratum_counts[k] += 1;
            }
            for (k, &count) in stratum_counts.iter().enumerate() {
                assert_eq!(
                    count, 1,
                    "dim {} stratum {} has {} samples (expected 1)",
                    d, k, count
                );
            }
        }
    }

    #[test]
    fn test_lhs_deterministic() {
        let a = medium_config(7).generate_lhs_unit_samples(50);
        let b = medium_config(7).generate_lhs_unit_samples(50);
        for (row_a, row_b) in a.iter().zip(b.iter()) {
            for (va, vb) in row_a.iter().zip(row_b.iter()) {
                assert_eq!(va, vb);
            }
        }
    }

    // ── Task 4: Sobol tests ────────────────────────────────────────────────

    #[test]
    fn test_sobol_bounds() {
        let cfg = medium_config(0);
        let samples = cfg.generate_sobol_unit_samples(1000);
        assert_eq!(samples.len(), 1000);
        for row in &samples {
            for (d, &v) in row.iter().enumerate() {
                assert!(v >= 0.0 && v <= 1.0, "dim {} value {} out of [0,1]", d, v);
            }
        }
    }

    #[test]
    fn test_sobol_deterministic() {
        let a = medium_config(123).generate_sobol_unit_samples(100);
        let b = medium_config(123).generate_sobol_unit_samples(100);
        for (row_a, row_b) in a.iter().zip(b.iter()) {
            for (va, vb) in row_a.iter().zip(row_b.iter()) {
                assert_eq!(va, vb);
            }
        }
    }

    #[test]
    fn test_sobol_different_seeds() {
        let a = medium_config(1).generate_sobol_unit_samples(50);
        let b = medium_config(2).generate_sobol_unit_samples(50);
        let any_differ = a.iter().zip(b.iter()).any(|(ra, rb)| {
            ra.iter().zip(rb.iter()).any(|(va, vb)| va != vb)
        });
        assert!(any_differ, "different seeds should produce different Sobol samples");
    }

    // ── Task 5: generate_draws() dispatch + from_array() tests ────────────

    #[test]
    fn test_from_array_roundtrip() {
        let draw = DispersionDraw {
            altitude: 1.0, longitude: 2.0, latitude: 3.0, velocity: 4.0,
            flight_path: 5.0, azimuth: 6.0, density: 7.0, drag_coeff: 8.0,
            lift_coeff: 9.0, incidence: 10.0, nav_altitude: 11.0, nav_longitude: 12.0,
            nav_latitude: 13.0, nav_velocity: 14.0, nav_flight_path: 15.0,
            nav_azimuth: 16.0, nav_drag_accel: 17.0, mass: 18.0, ref_area: 19.0,
            max_bank_rate: 20.0, pilot_tau: 21.0, pilot_damping: 22.0,
            pilot_frequency: 23.0, filter_gain: 24.0, wind_scale: 25.0,
            wind_direction_bias: 26.0,
        };
        let arr = draw.to_array();
        let roundtrip = DispersionDraw::from_array(arr);
        let arr2 = roundtrip.to_array();
        assert_eq!(arr, arr2);
    }

    #[test]
    fn test_generate_draws_lhs_produces_valid_draws() {
        let mut config = medium_config(42);
        config.sampling = SamplingMethod::Lhs;
        let draws = config.generate_draws(100);
        assert_eq!(draws.len(), 100);
        for draw in &draws {
            assert!(draw.altitude.is_finite());
            assert!(draw.velocity.is_finite());
            assert!(draw.density.is_finite());
            assert!(draw.wind_scale.is_finite());
        }
    }

    #[test]
    fn test_generate_draws_sobol_produces_valid_draws() {
        let mut config = medium_config(42);
        config.sampling = SamplingMethod::Sobol;
        let draws = config.generate_draws(100);
        assert_eq!(draws.len(), 100);
        for draw in &draws {
            assert!(draw.altitude.is_finite());
            assert!(draw.velocity.is_finite());
            assert!(draw.density.is_finite());
            assert!(draw.wind_scale.is_finite());
        }
    }

    #[test]
    #[should_panic(expected = "Sobol sampling limited to 65536")]
    fn test_sobol_rejects_too_many_sims() {
        let mut config = medium_config(42);
        config.sampling = SamplingMethod::Sobol;
        config.generate_draws(70_000);
    }
}
