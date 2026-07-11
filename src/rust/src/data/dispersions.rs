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
    // Guard: clamp to avoid -inf/+inf at the boundaries.
    // LHS can produce p=0.0 (stratum 0 + jitter exactly 0.0); Sobol can too.
    let p = p.clamp(1e-300, 1.0 - 1e-15);

    const A: [f64; 6] = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383_577_518_672_69e2,
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
            altitude: a[0],
            longitude: a[1],
            latitude: a[2],
            velocity: a[3],
            flight_path: a[4],
            azimuth: a[5],
            density: a[6],
            drag_coeff: a[7],
            lift_coeff: a[8],
            incidence: a[9],
            nav_altitude: a[10],
            nav_longitude: a[11],
            nav_latitude: a[12],
            nav_velocity: a[13],
            nav_flight_path: a[14],
            nav_azimuth: a[15],
            nav_drag_accel: a[16],
            mass: a[17],
            ref_area: a[18],
            max_bank_rate: a[19],
            pilot_tau: a[20],
            pilot_damping: a[21],
            pilot_frequency: a[22],
            filter_gain: a[23],
            wind_scale: a[24],
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
                (
                    s.altitude * 1e3,
                    s.longitude * DEG2RAD,
                    s.latitude * DEG2RAD,
                    s.velocity,
                    s.flight_path * DEG2RAD,
                    s.azimuth * DEG2RAD,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            };

        // dim 6: atmosphere density
        let atm_hw = self
            .atmosphere
            .as_ref()
            .map(|s| s.density / 100.0)
            .unwrap_or(0.0);

        // dims 7-9: aero drag, lift, incidence
        let (drag_hw, lift_hw, inc_hw) = self
            .aerodynamics
            .as_ref()
            .map(|s| (s.drag / 100.0, s.lift / 100.0, s.incidence * DEG2RAD))
            .unwrap_or((0.0, 0.0, 0.0));

        // dims 10-16: navigation
        let (nav_alt, nav_lon, nav_lat, nav_vel, nav_fpa, nav_az, nav_drag) =
            if let Some(ref s) = self.navigation {
                (
                    s.altitude * 1e3,
                    s.longitude * DEG2RAD,
                    s.latitude * DEG2RAD,
                    s.velocity,
                    s.flight_path * DEG2RAD,
                    s.azimuth * DEG2RAD,
                    s.drag_accel,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            };

        // dim 17: mass
        let mass_hw = self.mass.as_ref().map(|s| s.mass / 100.0).unwrap_or(0.0);

        // dims 18-19: vehicle ref_area, max_bank_rate
        let (area_hw, bank_rate_hw) = self
            .vehicle
            .as_ref()
            .map(|s| (s.ref_area / 100.0, s.max_bank_rate / 100.0))
            .unwrap_or((0.0, 0.0));

        // dims 20-22: pilot tau, damping, freq
        let (tau_hw, damp_hw, freq_hw) = self
            .pilot
            .as_ref()
            .map(|s| {
                (
                    s.time_constant / 100.0,
                    s.damping / 100.0,
                    s.frequency / 100.0,
                )
            })
            .unwrap_or((0.0, 0.0, 0.0));

        // dim 23: nav_filter (Gaussian)
        let nav_filter_sigma = self
            .nav_filter
            .as_ref()
            .map(|s| s.filter_gain)
            .unwrap_or(0.0);

        // dims 24-25: wind scale (UniformRange or Fixed), direction bias (Uniform or Fixed)
        let (wind_scale_tx, wind_dir_tx) = if let Some(ref w) = self.wind {
            (
                DimTransform::UniformRange {
                    min: w.scale_min,
                    max: w.scale_max,
                },
                unif(w.direction_bias_deg * DEG2RAD),
            )
        } else {
            (DimTransform::Fixed(1.0), DimTransform::Fixed(0.0))
        };

        [
            gauss(alt_sigma),        // 0: altitude
            gauss(lon_sigma),        // 1: longitude
            gauss(lat_sigma),        // 2: latitude
            gauss(vel_sigma),        // 3: velocity
            gauss(fpa_sigma),        // 4: flight_path
            gauss(az_sigma),         // 5: azimuth
            unif(atm_hw),            // 6: density
            unif(drag_hw),           // 7: drag_coeff
            unif(lift_hw),           // 8: lift_coeff
            unif(inc_hw),            // 9: incidence
            gauss(nav_alt),          // 10: nav_altitude
            gauss(nav_lon),          // 11: nav_longitude
            gauss(nav_lat),          // 12: nav_latitude
            gauss(nav_vel),          // 13: nav_velocity
            gauss(nav_fpa),          // 14: nav_flight_path
            gauss(nav_az),           // 15: nav_azimuth
            gauss(nav_drag),         // 16: nav_drag_accel
            unif(mass_hw),           // 17: mass
            unif(area_hw),           // 18: ref_area
            unif(bank_rate_hw),      // 19: max_bank_rate
            unif(tau_hw),            // 20: pilot_tau
            unif(damp_hw),           // 21: pilot_damping
            unif(freq_hw),           // 22: pilot_frequency
            gauss(nav_filter_sigma), // 23: filter_gain
            wind_scale_tx,           // 24: wind_scale
            wind_dir_tx,             // 25: wind_direction_bias
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
            .map(|i| std::array::from_fn(|d| sobol_burley::sample(i as u32, d as u32, seed) as f64))
            .collect()
    }

    /// Map a slice of unit samples (each row is one sim, 26 dims in [0,1]) through the
    /// per-dimension transforms to produce `DispersionDraw` values.
    fn draws_from_unit_samples(
        &self,
        unit_samples: &[[f64; DISPERSION_DRAW_LEN]],
    ) -> Vec<DispersionDraw> {
        let transforms = self.build_dim_transforms();
        unit_samples
            .iter()
            .map(|row| {
                let arr: [f64; DISPERSION_DRAW_LEN] =
                    std::array::from_fn(|d| transforms[d].apply(row[d]));
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

                // Wind (Uniform scale in [min, max], Uniform direction bias in [-deg, +deg]).
                // A degenerate range (scale_min == scale_max, e.g. custom configs pinning
                // the scale while keeping direction bias) is a fixed value: rand's
                // `Uniform::new` errors on an empty range. scale_min > scale_max is
                // rejected at config load (`build_dispersion_config`).
                if let Some(ref w) = self.wind {
                    draw.wind_scale = if w.scale_max > w.scale_min {
                        Uniform::new(w.scale_min, w.scale_max)
                            .unwrap()
                            .sample(&mut rng)
                    } else {
                        w.scale_min
                    };
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
#[path = "dispersions_tests.rs"]
mod tests;
