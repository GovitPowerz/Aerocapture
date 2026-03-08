//! Monte Carlo dispersion system.
//!
//! Domain-based: sigma values from TOML config with preset levels + runtime RNG
//! draw generation.
//!
//! Distribution types match the original Fortran (loteri.f):
//! - Initial state + navigation: Gaussian (bgauss.f — sum of 12 uniforms, CLT)
//! - Atmosphere, aerodynamics, incidence, mass: Uniform[-sigma, +sigma] (bunifo.f)

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
                altitude: 0.0,
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

/// Full domain-based dispersion configuration.
#[derive(Debug, Clone)]
pub struct DispersionConfig {
    pub seed: u64,
    pub initial_state: Option<InitialStateSigmas>,
    pub atmosphere: Option<AtmosphereSigmas>,
    pub aerodynamics: Option<AerodynamicsSigmas>,
    pub navigation: Option<NavigationSigmas>,
    pub mass: Option<MassSigmas>,
}

// ────────────────────────────────────────────────────────────────────
// Draw generation
// ────────────────────────────────────────────────────────────────────

/// One simulation's dispersion draws (replaces LotteryDraw).
/// Values are in SI units, ready to apply directly.
#[derive(Debug, Clone, Default)]
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
}

impl DispersionConfig {
    /// Generate all dispersion draws for a batch of simulations.
    ///
    /// Uses a seeded RNG for reproducibility. The draw order matches
    /// the Fortran loteri.f sequence: initial state (Gaussian),
    /// atmosphere (Uniform), aero (Uniform), nav (Gaussian),
    /// incidence (Uniform), mass (Uniform).
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
            // At least velocity should be non-zero for all non-off levels
            assert!(
                s.velocity > 0.0,
                "velocity sigma should be > 0 for {:?}",
                level
            );

            let a = AtmosphereSigmas::from_level(level);
            assert!(a.density > 0.0);

            let n = NavigationSigmas::from_level(level);
            assert!(n.altitude > 0.0);
        }

        let s_off = InitialStateSigmas::from_level(DispersionLevel::Off);
        assert_eq!(s_off.altitude, 0.0);
        assert_eq!(s_off.velocity, 0.0);
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
        }
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
}
