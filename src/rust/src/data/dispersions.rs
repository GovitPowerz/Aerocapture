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
            DispersionLevel::Off => Self { tau: 0.0, sigma: 0.0 },
            DispersionLevel::Low => Self { tau: 120.0, sigma: 0.05 },
            DispersionLevel::Medium => Self { tau: 60.0, sigma: 0.10 },
            DispersionLevel::High => Self { tau: 30.0, sigma: 0.20 },
            DispersionLevel::Custom => Self::from_level(DispersionLevel::Medium),
        }
    }

    /// Returns true if the perturbation is effectively disabled.
    pub fn is_disabled(&self) -> bool {
        self.sigma <= 0.0 || self.tau <= 0.0
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
    pub initial_state: Option<InitialStateSigmas>,
    pub atmosphere: Option<AtmosphereSigmas>,
    pub aerodynamics: Option<AerodynamicsSigmas>,
    pub navigation: Option<NavigationSigmas>,
    pub mass: Option<MassSigmas>,
    pub vehicle: Option<VehicleSigmas>,
    pub pilot: Option<PilotSigmas>,
    pub nav_filter: Option<NavFilterSigmas>,
    pub wind: Option<WindDispersionConfig>,
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
}

impl DispersionConfig {
    /// Generate all dispersion draws for a batch of simulations.
    ///
    /// Uses a seeded RNG for reproducibility. Draw order:
    /// initial state (Gaussian), atmosphere (Uniform), aero (Uniform),
    /// nav (Gaussian), mass (Uniform), vehicle (Uniform), pilot (Uniform),
    /// nav_filter (Gaussian), wind (Uniform scale + Uniform direction bias).
    pub fn generate_draws(&self, n_sims: usize) -> Vec<DispersionDraw> {
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
            initial_state: Some(InitialStateSigmas::from_level(DispersionLevel::Medium)),
            atmosphere: Some(AtmosphereSigmas::from_level(DispersionLevel::Medium)),
            aerodynamics: Some(AerodynamicsSigmas::from_level(DispersionLevel::Medium)),
            navigation: Some(NavigationSigmas::from_level(DispersionLevel::Medium)),
            mass: Some(MassSigmas::from_level(DispersionLevel::Medium)),
            vehicle: Some(VehicleSigmas::from_level(DispersionLevel::Medium)),
            pilot: Some(PilotSigmas::from_level(DispersionLevel::Medium)),
            nav_filter: Some(NavFilterSigmas::from_level(DispersionLevel::Medium)),
            wind: None,
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
            initial_state: None,
            atmosphere: None,
            aerodynamics: None,
            navigation: None,
            mass: None,
            vehicle: None,
            pilot: None,
            nav_filter: None,
            wind: None,
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
    fn test_uniform_fields_bounded() {
        let config = DispersionConfig {
            seed: 12345,
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
            initial_state: None,
            atmosphere: None,
            aerodynamics: None,
            navigation: None,
            mass: None,
            vehicle: None,
            pilot: None,
            nav_filter: Some(NavFilterSigmas { filter_gain: 0.10 }),
            wind: None,
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
        assert!(x > 0.0, "state should remain positive with no noise: got {}", x);
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
        }
    }
}
