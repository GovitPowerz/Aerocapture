//! Lateral guidance — inclination corridor roll reversal logic.
//!
//! Shared by all unsigned-magnitude guidance schemes (EqGlide, EnergyController,
//! PredGuid, FNPAG). Schemes that produce signed bank angles (NeuralNetwork,
//! PiecewiseConstant) bypass this entirely.

use crate::config::Planet;
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;

/// Lateral guidance configuration (TOML-configurable, per-scheme tunable).
#[derive(Debug, Clone)]
pub struct LateralParams {
    /// Velocity scaling for corridor width (m/s).
    pub corridor_slope: f64,
    /// Baseline corridor width at low velocity (rad).
    pub corridor_intercept: f64,
    /// Energy at which lateral guidance arms (J/kg). Upper bound of the active window.
    pub lateral_activation: f64,
    /// Energy below which lateral guidance disarms (J/kg). Lower bound of the active window.
    pub lateral_inhibition: f64,
    /// Maximum number of roll reversals per trajectory.
    pub max_reversals: i32,
}

impl Default for LateralParams {
    fn default() -> Self {
        Self {
            corridor_slope: 0.0,
            corridor_intercept: 0.0,
            lateral_activation: 0.0,
            lateral_inhibition: 0.0,
            max_reversals: 0,
        }
    }
}

/// Lateral guidance mutable state (per-run).
#[derive(Debug, Clone)]
pub struct LateralState {
    /// Current roll direction sign (±1.0).
    pub roll_sign: f64,
    /// Number of roll reversals executed so far.
    pub n_reversals: i32,
}

impl LateralState {
    pub fn new(initial_bank: f64) -> Self {
        Self {
            roll_sign: if initial_bank >= 0.0 { 1.0 } else { -1.0 },
            n_reversals: 0,
        }
    }
}
