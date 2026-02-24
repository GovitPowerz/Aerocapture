//! Pilot dynamics model.
//!
//! Matches Fortran pilote.f.
//! Models the delay/dynamics in realizing commanded bank angle.

use crate::data::pilot::{PilotModel, PilotType};

/// Pilot state for dynamic models
#[derive(Debug, Clone, Copy, Default)]
pub struct PilotState {
    pub bank_angle: f64,      // current realized bank angle (rad)
    pub bank_rate: f64,       // current bank angle rate (rad/s) — for 2nd order
}

/// Apply pilot dynamics to compute realized bank angle.
///
/// Returns (new_bank_angle, new_bank_rate).
pub fn apply_pilot(
    model: &PilotModel,
    commanded: f64,
    state: &PilotState,
    dt: f64,
    max_rate: f64,
) -> PilotState {
    match model.pilot_type {
        PilotType::Perfect => PilotState {
            bank_angle: commanded,
            bank_rate: 0.0,
        },
        PilotType::FirstOrder => {
            // First order: tau * d(phi)/dt + phi = phi_cmd
            let tau = model.time_constant;
            let error = commanded - state.bank_angle;
            let rate = (error / tau).clamp(-max_rate, max_rate);
            PilotState {
                bank_angle: state.bank_angle + rate * dt,
                bank_rate: rate,
            }
        }
        PilotType::SecondOrder => {
            // Second order: d2(phi)/dt2 + 2*zeta*omega*d(phi)/dt + omega^2*(phi - phi_cmd) = 0
            let omega = model.frequency;
            let zeta = model.damping;
            let error = state.bank_angle - commanded;
            let accel = -2.0 * zeta * omega * state.bank_rate - omega * omega * error;
            let new_rate = (state.bank_rate + accel * dt).clamp(-max_rate, max_rate);
            PilotState {
                bank_angle: state.bank_angle + new_rate * dt,
                bank_rate: new_rate,
            }
        }
    }
}
