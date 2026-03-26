//! IMU sensor model (accelerometers + gyroscopes).
//! Models bias, scale factor error, and white noise.

use rand::SeedableRng;
use rand::distr::Distribution;
use rand_distr::Normal;

/// Configuration parameters for the IMU sensor model.
#[derive(Debug, Clone)]
pub struct ImuConfig {
    /// 1-sigma initial accelerometer bias (m/s²)
    pub accel_bias_sigma: f64,
    /// Accelerometer white noise per sample (m/s²)
    pub accel_noise_sigma: f64,
    /// Accelerometer scale factor error (1-sigma, dimensionless)
    pub accel_scale_factor_sigma: f64,
    /// 1-sigma initial gyroscope bias (rad/s)
    pub gyro_bias_sigma: f64,
    /// Gyroscope white noise per sample (rad/s)
    pub gyro_noise_sigma: f64,
}

impl Default for ImuConfig {
    fn default() -> Self {
        Self {
            accel_bias_sigma: 1e-4,
            accel_noise_sigma: 5e-4,
            accel_scale_factor_sigma: 1e-4,
            gyro_bias_sigma: 5e-6,
            gyro_noise_sigma: 1e-5,
        }
    }
}

/// Runtime state of the IMU sensor model.
pub struct ImuState {
    /// Per-axis accelerometer biases drawn at init (m/s²)
    pub accel_bias: [f64; 3],
    /// Per-axis accelerometer scale factor errors drawn at init (dimensionless)
    pub accel_scale_factor: [f64; 3],
    /// Per-axis gyroscope biases drawn at init (rad/s)
    pub gyro_bias: [f64; 3],
    rng: rand::rngs::StdRng,
    accel_noise: Normal<f64>,
    gyro_noise: Normal<f64>,
}

impl ImuState {
    /// Create a new IMU state. Biases and scale factors are drawn from their
    /// respective 1-sigma Gaussian distributions using `seed`.
    pub fn new(config: &ImuConfig, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let bias_dist = Normal::new(0.0, config.accel_bias_sigma).unwrap();
        let sf_dist = Normal::new(0.0, config.accel_scale_factor_sigma).unwrap();
        let gyro_bias_dist = Normal::new(0.0, config.gyro_bias_sigma).unwrap();

        let accel_bias = [
            bias_dist.sample(&mut rng),
            bias_dist.sample(&mut rng),
            bias_dist.sample(&mut rng),
        ];
        let accel_scale_factor = [
            sf_dist.sample(&mut rng),
            sf_dist.sample(&mut rng),
            sf_dist.sample(&mut rng),
        ];
        let gyro_bias = [
            gyro_bias_dist.sample(&mut rng),
            gyro_bias_dist.sample(&mut rng),
            gyro_bias_dist.sample(&mut rng),
        ];

        let accel_noise = Normal::new(0.0, config.accel_noise_sigma).unwrap();
        let gyro_noise = Normal::new(0.0, config.gyro_noise_sigma).unwrap();

        Self { accel_bias, accel_scale_factor, gyro_bias, rng, accel_noise, gyro_noise }
    }

    /// Apply accelerometer model: `(1 + scale_factor) * true_accel + bias + noise`.
    pub fn measure_accel(&mut self, true_accel: &[f64; 3]) -> [f64; 3] {
        [
            (1.0 + self.accel_scale_factor[0]) * true_accel[0]
                + self.accel_bias[0]
                + self.accel_noise.sample(&mut self.rng),
            (1.0 + self.accel_scale_factor[1]) * true_accel[1]
                + self.accel_bias[1]
                + self.accel_noise.sample(&mut self.rng),
            (1.0 + self.accel_scale_factor[2]) * true_accel[2]
                + self.accel_bias[2]
                + self.accel_noise.sample(&mut self.rng),
        ]
    }

    /// Apply gyroscope model: `true_rate + bias + noise`.
    pub fn measure_gyro(&mut self, true_rate: &[f64; 3]) -> [f64; 3] {
        [
            true_rate[0] + self.gyro_bias[0] + self.gyro_noise.sample(&mut self.rng),
            true_rate[1] + self.gyro_bias[1] + self.gyro_noise.sample(&mut self.rng),
            true_rate[2] + self.gyro_bias[2] + self.gyro_noise.sample(&mut self.rng),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_true_accel_returns_small_values() {
        let config = ImuConfig::default();
        let mut imu = ImuState::new(&config, 42);
        let meas = imu.measure_accel(&[0.0, 0.0, 0.0]);
        // With zero true input, output is bias + noise.
        // Default bias sigma is 1e-4 and noise sigma is 5e-4; values should be small.
        for v in meas {
            assert!(v.abs() < 0.1, "accel measurement too large: {v}");
        }
    }

    #[test]
    fn noise_statistics_reasonable() {
        let config = ImuConfig { accel_bias_sigma: 0.0, accel_scale_factor_sigma: 0.0, gyro_bias_sigma: 0.0, ..ImuConfig::default() };
        let mut imu = ImuState::new(&config, 99);
        let n = 10_000usize;
        let mut sum = [0.0f64; 3];
        let zero = [0.0f64; 3];
        for _ in 0..n {
            let m = imu.measure_accel(&zero);
            for i in 0..3 {
                sum[i] += m[i];
            }
        }
        // With zero bias and scale factor, mean of noise samples should be near zero.
        for s in sum {
            let mean = s / n as f64;
            assert!(mean.abs() < 0.01, "accel noise mean too large: {mean}");
        }
    }

    #[test]
    fn gyro_noise_statistics_reasonable() {
        let config = ImuConfig { gyro_bias_sigma: 0.0, ..ImuConfig::default() };
        let mut imu = ImuState::new(&config, 7);
        let n = 10_000usize;
        let mut sum = [0.0f64; 3];
        let zero = [0.0f64; 3];
        for _ in 0..n {
            let m = imu.measure_gyro(&zero);
            for i in 0..3 {
                sum[i] += m[i];
            }
        }
        for s in sum {
            let mean = s / n as f64;
            assert!(mean.abs() < 0.001, "gyro noise mean too large: {mean}");
        }
    }
}
