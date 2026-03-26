//! Star tracker sensor model.
//! Provides position updates with dynamic pressure blackout.

use rand::SeedableRng;
use rand::distr::Distribution;
use rand_distr::Normal;

/// Configuration parameters for the star tracker sensor model.
#[derive(Debug, Clone)]
pub struct StarTrackerConfig {
    /// 1-sigma position noise (m)
    pub position_sigma: f64,
    /// 1-sigma attitude noise (rad)
    pub attitude_sigma: f64,
    /// Measurement update period (s)
    pub update_period: f64,
    /// Dynamic pressure above which the star tracker is blacked out (Pa)
    pub blackout_qdyn_threshold: f64,
}

impl Default for StarTrackerConfig {
    fn default() -> Self {
        Self {
            position_sigma: 50.0,
            attitude_sigma: 3e-4,
            update_period: 10.0,
            blackout_qdyn_threshold: 100.0,
        }
    }
}

impl StarTrackerConfig {
    /// Returns `true` when the star tracker is not blacked out by aeroheating.
    pub fn is_available(&self, dynamic_pressure_pa: f64) -> bool {
        dynamic_pressure_pa < self.blackout_qdyn_threshold
    }

    /// Returns `true` when enough time has elapsed since the last update.
    pub fn is_update_due(&self, last_update_time: f64, current_time: f64) -> bool {
        (current_time - last_update_time) >= self.update_period
    }
}

/// Runtime state of the star tracker sensor model.
pub struct StarTrackerState {
    /// Simulation time of the last successful measurement (s). Initialised to -1e10
    /// so the first update is always "due" at t=0.
    pub last_update_time: f64,
    rng: rand::rngs::StdRng,
    pos_noise: Normal<f64>,
}

impl StarTrackerState {
    /// Create a new star tracker state seeded with `seed`.
    pub fn new(config: &StarTrackerConfig, seed: u64) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        let pos_noise = Normal::new(0.0, config.position_sigma).unwrap();
        Self {
            last_update_time: -1e10,
            rng,
            pos_noise,
        }
    }

    /// Attempt a measurement.
    ///
    /// Returns `None` when:
    /// - the star tracker is blacked out (`qdyn >= threshold`), or
    /// - an update is not yet due.
    ///
    /// Otherwise returns `true_position + noise` and updates `last_update_time`.
    /// Position noise in metres is divided by the radial distance (`true_position[0]`)
    /// to convert to radians for longitude/latitude components.
    pub fn measure(
        &mut self,
        true_position: &[f64; 3],
        dynamic_pressure_pa: f64,
        sim_time: f64,
        config: &StarTrackerConfig,
    ) -> Option<[f64; 3]> {
        if !config.is_available(dynamic_pressure_pa) {
            return None;
        }
        if !config.is_update_due(self.last_update_time, sim_time) {
            return None;
        }

        self.last_update_time = sim_time;

        let r = true_position[0];
        // Component 0 is radial (metres); components 1 and 2 are angular (rad).
        // Convert metre-level noise to radians for angular components.
        let noise_r = self.pos_noise.sample(&mut self.rng);
        let noise_lon = self.pos_noise.sample(&mut self.rng) / r;
        let noise_lat = self.pos_noise.sample(&mut self.rng) / r;

        Some([
            true_position[0] + noise_r,
            true_position[1] + noise_lon,
            true_position[2] + noise_lat,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_when_qdyn_below_threshold() {
        let config = StarTrackerConfig::default();
        assert!(config.is_available(50.0));
        assert!(config.is_available(0.0));
        assert!(config.is_available(99.9));
    }

    #[test]
    fn blacked_out_when_qdyn_above_threshold() {
        let config = StarTrackerConfig::default();
        assert!(!config.is_available(100.0));
        assert!(!config.is_available(500.0));
        assert!(!config.is_available(1e6));
    }

    #[test]
    fn update_due_at_correct_cadence() {
        let config = StarTrackerConfig::default(); // period = 10.0 s
        assert!(config.is_update_due(-1e10, 0.0));
        assert!(config.is_update_due(0.0, 10.0));
        assert!(config.is_update_due(0.0, 15.0));
        assert!(!config.is_update_due(0.0, 9.9));
        assert!(!config.is_update_due(10.0, 19.9));
    }

    #[test]
    fn measure_returns_none_when_blacked_out() {
        let config = StarTrackerConfig::default();
        let mut st = StarTrackerState::new(&config, 1);
        let true_pos = [3.6e6_f64, 0.1, 0.05];
        // High dynamic pressure — should black out.
        let result = st.measure(&true_pos, 1e4, 0.0, &config);
        assert!(result.is_none(), "Expected None when blacked out");
    }

    #[test]
    fn measure_returns_some_when_available_and_due() {
        let config = StarTrackerConfig::default();
        let mut st = StarTrackerState::new(&config, 2);
        let true_pos = [3.6e6_f64, 0.1, 0.05];
        // Low dynamic pressure + update due (last_update_time initialised to -1e10).
        let result = st.measure(&true_pos, 1.0, 0.0, &config);
        assert!(result.is_some(), "Expected Some measurement");
        let meas = result.unwrap();
        // Radial component noise: 50 m on a ~3.6 Mm radius is tiny in relative terms.
        assert!((meas[0] - true_pos[0]).abs() < 1000.0);
        // Second call right away should return None (not due yet).
        let result2 = st.measure(&true_pos, 1.0, 5.0, &config);
        assert!(result2.is_none(), "Expected None before next period");
    }
}
