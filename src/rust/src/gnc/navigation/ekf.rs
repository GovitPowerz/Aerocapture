//! Extended Kalman Filter (13-state error-state formulation).
//!
//! State vector layout:
//! - [0..3] position errors: dr (m), dlon (rad), dlat (rad)
//! - [3..6] velocity errors: dV (m/s), dgamma (rad), dpsi (rad)
//! - [6..9] accelerometer biases x,y,z (m/s²)
//! - [9..12] gyro biases x,y,z (rad/s)
//! - [12] density correction factor (centered at 0; actual = 1 + state[12])

use nalgebra::{SMatrix, SVector};

/// Number of states in the EKF.
pub const N_STATES: usize = 13;

// ─── Configuration ──────────────────────────────────────────────────────────

/// EKF tuning parameters: initial covariances and process noise.
pub struct EkfConfig {
    /// Initial position covariance diagonal [dr², dlon², dlat²].
    pub p0_pos: [f64; 3],
    /// Initial velocity covariance diagonal [dV², dgamma², dpsi²].
    pub p0_vel: [f64; 3],
    /// Initial accelerometer bias covariance (per axis).
    pub p0_accel_bias: f64,
    /// Initial gyro bias covariance (per axis).
    pub p0_gyro_bias: f64,
    /// Initial density correction covariance.
    pub p0_density: f64,
    /// Process noise: accelerometer bias random walk (per axis).
    pub q_accel_bias: f64,
    /// Process noise: gyro bias random walk (per axis).
    pub q_gyro_bias: f64,
    /// Process noise: density correction random walk.
    pub q_density: f64,
}

impl Default for EkfConfig {
    fn default() -> Self {
        Self {
            // 50 m position, ~0.001° lon/lat
            p0_pos: [50.0_f64.powi(2), 1e-5_f64.powi(2), 1e-5_f64.powi(2)],
            // 1 m/s velocity, ~0.01° FPA/heading
            p0_vel: [1.0_f64.powi(2), 1e-4_f64.powi(2), 1e-4_f64.powi(2)],
            p0_accel_bias: 1e-4_f64.powi(2),
            p0_gyro_bias: 1e-6_f64.powi(2),
            p0_density: 0.5_f64.powi(2),
            q_accel_bias: 1e-8,
            q_gyro_bias: 1e-12,
            q_density: 1e-4,
        }
    }
}

// ─── State ──────────────────────────────────────────────────────────────────

/// EKF state: error-state vector and covariance matrix.
pub struct EkfState {
    /// 13-element error-state vector.
    pub state: SVector<f64, N_STATES>,
    /// 13×13 covariance matrix.
    pub covariance: SMatrix<f64, N_STATES, N_STATES>,
}

impl EkfState {
    /// Create a new EKF with zero state and diagonal covariance from config.
    pub fn new(config: &EkfConfig) -> Self {
        let state = SVector::<f64, N_STATES>::zeros();
        let mut p = SMatrix::<f64, N_STATES, N_STATES>::zeros();

        // Position errors
        for (i, &var) in config.p0_pos.iter().enumerate() {
            p[(i, i)] = var;
        }
        // Velocity errors
        for (i, &var) in config.p0_vel.iter().enumerate() {
            p[(3 + i, 3 + i)] = var;
        }
        // Accelerometer biases
        for i in 0..3 {
            p[(6 + i, 6 + i)] = config.p0_accel_bias;
        }
        // Gyro biases
        for i in 0..3 {
            p[(9 + i, 9 + i)] = config.p0_gyro_bias;
        }
        // Density correction
        p[(12, 12)] = config.p0_density;

        Self { state, covariance: p }
    }

    /// Prediction step: propagate error-state and covariance forward by `dt`.
    ///
    /// Uses a simplified error-state transition where position errors grow
    /// with velocity errors and velocity errors grow with IMU biases.
    /// Biases and density correction are modeled as random walks.
    ///
    /// TODO: Incorporate IMU measurements into the prediction step for proper
    /// strapdown inertial navigation. Currently the state transition matrix F
    /// is time-invariant and does not depend on the measured accelerations or
    /// angular rates. The filter still provides value through its density
    /// estimation and covariance tracking, but the error-state propagation
    /// is open-loop with respect to flight dynamics.
    pub fn predict(&mut self, dt: f64, _accel_meas: &[f64; 3], _gyro_meas: &[f64; 3], config: &EkfConfig) {
        // ── State transition matrix F ────────────────────────────────────
        let mut f = SMatrix::<f64, N_STATES, N_STATES>::identity();

        // Position errors grow with velocity errors: d(pos)/dt ~ vel
        // dr    += dV     * dt
        // dlon  += dgamma * dt  (simplified coupling)
        // dlat  += dpsi   * dt  (simplified coupling)
        f[(0, 3)] = dt;
        f[(1, 4)] = dt;
        f[(2, 5)] = dt;

        // Velocity errors grow with accelerometer biases
        // dV     += -accel_bias_x * dt
        // dgamma += -accel_bias_y * dt
        // dpsi   += -accel_bias_z * dt
        f[(3, 6)] = -dt;
        f[(4, 7)] = -dt;
        f[(5, 8)] = -dt;

        // FPA/heading errors grow with gyro biases
        // dgamma += -gyro_bias_x * dt  (simplified)
        // dpsi   += -gyro_bias_y * dt  (simplified)
        f[(4, 9)] = -dt;
        f[(5, 10)] = -dt;

        // Biases (6..12) and density (12) are identity (random walk).

        // ── Process noise Q ──────────────────────────────────────────────
        let mut q = SMatrix::<f64, N_STATES, N_STATES>::zeros();

        // Position process noise: drift from velocity uncertainty
        let dt2 = dt * dt;
        let dt3_3 = dt2 * dt / 3.0;
        // Simplified: q_pos ~ q_vel * dt² (cross terms from F*Q_vel*F^T)
        q[(0, 0)] = dt3_3 * config.q_accel_bias; // radial position from accel noise
        q[(1, 1)] = dt3_3 * config.q_gyro_bias;  // longitude from gyro noise
        q[(2, 2)] = dt3_3 * config.q_gyro_bias;  // latitude from gyro noise
        // Velocity process noise: dV from accel bias, dgamma/dpsi from gyro bias
        q[(3, 3)] = dt2 * config.q_accel_bias;
        q[(4, 4)] = dt2 * config.q_gyro_bias;  // FPA error driven by gyro
        q[(5, 5)] = dt2 * config.q_gyro_bias;  // heading error driven by gyro
        // Accelerometer bias random walk
        for i in 6..9 {
            q[(i, i)] = dt * config.q_accel_bias;
        }
        // Gyro bias random walk
        for i in 9..12 {
            q[(i, i)] = dt * config.q_gyro_bias;
        }
        // Density correction random walk
        q[(12, 12)] = dt * config.q_density;

        // ── Propagate ────────────────────────────────────────────────────
        // x = F * x  (error-state propagation — biases persist, errors grow)
        self.state = f * self.state;

        // P = F * P * F^T + Q
        self.covariance = f * self.covariance * f.transpose() + q;

        // Enforce symmetry
        self.covariance = (self.covariance + self.covariance.transpose()) * 0.5;
    }

    /// Star tracker measurement update (3-element position innovation).
    pub fn update_position(&mut self, innovation: &SVector<f64, 3>, r_meas: &SMatrix<f64, 3, 3>) {
        // H = [I_3x3 | 0_3x10]
        let mut h = SMatrix::<f64, 3, N_STATES>::zeros();
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;

        kalman_update(&mut self.state, &mut self.covariance, &h, innovation, r_meas);
    }

    /// Drag-derived density measurement update (scalar innovation).
    ///
    /// After update, clamps the density correction state to [-0.9, 9.0]
    /// so the multiplicative factor stays in [0.1, 10.0].
    pub fn update_density(&mut self, innovation: f64, r_meas: f64) {
        // H = [0 … 0, 1]  (1×13, only state 12)
        let mut h = SMatrix::<f64, 1, N_STATES>::zeros();
        h[(0, 12)] = 1.0;

        let innov_vec = SVector::<f64, 1>::new(innovation);
        let r_mat = SMatrix::<f64, 1, 1>::new(r_meas);

        kalman_update(&mut self.state, &mut self.covariance, &h, &innov_vec, &r_mat);

        // Clamp density correction to keep factor in [0.1, 10.0]
        self.state[12] = self.state[12].clamp(-0.9, 9.0);
    }

    /// Returns the density multiplicative correction factor: 1 + state[12].
    pub fn density_correction(&self) -> f64 {
        1.0 + self.state[12]
    }
}

// ─── Generic Kalman update (Joseph form) ────────────────────────────────────

/// Perform a Kalman measurement update with Joseph form for numerical stability.
///
/// Generic over measurement dimension `M`.
fn kalman_update<const M: usize>(
    state: &mut SVector<f64, N_STATES>,
    covariance: &mut SMatrix<f64, N_STATES, N_STATES>,
    h: &SMatrix<f64, M, N_STATES>,
    innovation: &SVector<f64, M>,
    r: &SMatrix<f64, M, M>,
) {
    // S = H * P * H^T + R
    let s = h * *covariance * h.transpose() + r;

    // Invert S; skip update if singular
    let Some(s_inv) = s.try_inverse() else {
        return;
    };

    // K = P * H^T * S^{-1}
    let k = *covariance * h.transpose() * s_inv;

    // State update: x += K * innovation
    *state += k * innovation;

    // Joseph form: P = (I - K*H) * P * (I - K*H)^T + K*R*K^T
    let i_kh = SMatrix::<f64, N_STATES, N_STATES>::identity() - k * h;
    *covariance = i_kh * *covariance * i_kh.transpose() + k * r * k.transpose();

    // Enforce symmetry
    *covariance = (*covariance + covariance.transpose()) * 0.5;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_zeros_with_positive_covariance() {
        let ekf = EkfState::new(&EkfConfig::default());
        assert!(ekf.state.iter().all(|&x| x == 0.0));
        for i in 0..13 {
            assert!(ekf.covariance[(i, i)] > 0.0, "P[{i},{i}] should be positive");
        }
    }

    #[test]
    fn predict_preserves_covariance_symmetry() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        ekf.predict(1.0, &[0.0; 3], &[0.0; 3], &config);
        for i in 0..13 {
            for j in 0..13 {
                let diff = (ekf.covariance[(i, j)] - ekf.covariance[(j, i)]).abs();
                assert!(diff < 1e-12, "P[{i},{j}] vs P[{j},{i}]: diff = {diff}");
            }
        }
    }

    #[test]
    fn predict_grows_covariance() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        let p0 = ekf.covariance[(0, 0)];
        ekf.predict(1.0, &[0.0; 3], &[0.0; 3], &config);
        assert!(ekf.covariance[(0, 0)] > p0, "position covariance should grow after predict");
    }

    #[test]
    fn position_update_reduces_covariance() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        ekf.predict(10.0, &[0.0; 3], &[0.0; 3], &config);
        let p_before = ekf.covariance[(0, 0)];
        let innovation = SVector::<f64, 3>::new(100.0, 0.001, 0.001);
        let r = SMatrix::<f64, 3, 3>::identity() * 100.0;
        ekf.update_position(&innovation, &r);
        assert!(ekf.covariance[(0, 0)] < p_before, "position covariance should decrease after update");
    }

    #[test]
    fn density_update_corrects_state() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        assert_eq!(ekf.density_correction(), 1.0);
        ekf.update_density(0.1, 0.01);
        assert!(ekf.density_correction() > 1.0, "density correction should increase with positive innovation");
    }

    #[test]
    fn density_correction_is_clamped() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        // Force extreme state
        ekf.state[12] = 20.0;
        ekf.update_density(0.0, 0.01);
        assert!(ekf.state[12] <= 9.0, "density state should be clamped to 9.0 max");
    }

    #[test]
    fn position_update_corrects_state() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        let innovation = SVector::<f64, 3>::new(100.0, 0.0, 0.0);
        let r = SMatrix::<f64, 3, 3>::identity() * 100.0;
        ekf.update_position(&innovation, &r);
        // State should move toward the innovation
        assert!(ekf.state[0] > 0.0, "radial error should shift toward positive innovation");
    }

    #[test]
    fn multiple_predict_update_cycles_stay_stable() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        let r_pos = SMatrix::<f64, 3, 3>::identity() * 100.0;
        for _ in 0..100 {
            ekf.predict(0.1, &[0.0; 3], &[0.0; 3], &config);
            let innov = SVector::<f64, 3>::new(1.0, 0.0001, 0.0001);
            ekf.update_position(&innov, &r_pos);
        }
        // Covariance should remain finite and positive definite
        for i in 0..13 {
            assert!(ekf.covariance[(i, i)].is_finite(), "P[{i},{i}] is not finite");
            assert!(ekf.covariance[(i, i)] >= 0.0, "P[{i},{i}] is negative");
        }
    }

    #[test]
    fn density_clamp_lower_bound() {
        let config = EkfConfig::default();
        let mut ekf = EkfState::new(&config);
        ekf.state[12] = -5.0;
        ekf.update_density(0.0, 0.01);
        assert!(ekf.state[12] >= -0.9, "density state should be clamped to -0.9 min");
        assert!(ekf.density_correction() >= 0.1 - 1e-12, "density factor should be >= 0.1");
    }
}
