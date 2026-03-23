//! Pilot dynamics model.
//!
//! Models the delay/dynamics in realizing commanded bank angle.

use crate::data::pilot::{PilotModel, PilotType};
use crate::gnc::control::angle_utils::shortest_angle_diff;

/// Pilot state for dynamic models
#[derive(Debug, Clone, Copy, Default)]
pub struct PilotState {
    pub bank_angle: f64, // current realized bank angle (rad)
    pub bank_rate: f64,  // current bank angle rate (rad/s) — for 2nd order
}

/// Fractional biases on pilot dynamics parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct PilotBiases {
    pub tau: f64,       // fractional bias on time constant
    pub damping: f64,   // fractional bias on damping ratio
    pub frequency: f64, // fractional bias on natural frequency
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
    biases: &PilotBiases,
) -> PilotState {
    match model.pilot_type {
        PilotType::Perfect => PilotState {
            bank_angle: commanded,
            bank_rate: 0.0,
        },
        PilotType::FirstOrder => {
            // First order: tau * d(phi)/dt + phi = phi_cmd
            let tau = model.time_constant * (1.0 + biases.tau);
            let error = shortest_angle_diff(state.bank_angle, commanded);
            let rate = (error / tau).clamp(-max_rate, max_rate);
            PilotState {
                bank_angle: state.bank_angle + rate * dt,
                bank_rate: rate,
            }
        }
        PilotType::SecondOrder => {
            // Second order: d2(phi)/dt2 + 2*zeta*omega*d(phi)/dt + omega^2*(phi - phi_cmd) = 0
            let omega = model.frequency * (1.0 + biases.frequency);
            let zeta = model.damping * (1.0 + biases.damping);
            let error = shortest_angle_diff(commanded, state.bank_angle);
            let accel = -2.0 * zeta * omega * state.bank_rate - omega * omega * error;
            let new_rate = (state.bank_rate + accel * dt).clamp(-max_rate, max_rate);
            PilotState {
                bank_angle: state.bank_angle + new_rate * dt,
                bank_rate: new_rate,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn make_model(pilot_type: PilotType, tau: f64, zeta: f64, omega: f64) -> PilotModel {
        PilotModel {
            pilot_type,
            time_constant: tau,
            damping: zeta,
            frequency: omega,
        }
    }

    #[test]
    fn perfect_pilot_tracks_immediately() {
        let model = make_model(PilotType::Perfect, 0.0, 0.0, 0.0);
        let state = PilotState {
            bank_angle: 0.5,
            bank_rate: 0.0,
        };
        let result = apply_pilot(&model, 1.0, &state, 0.1, 10.0, &PilotBiases::default());
        assert!((result.bank_angle - 1.0).abs() < EPS);
        assert!((result.bank_rate).abs() < EPS);
    }

    #[test]
    fn first_order_moves_toward_command() {
        // tau=1.0, dt=0.1, cmd=1.0 from 0.0: rate = (1-0)/1 = 1.0, angle = 0 + 1.0*0.1 = 0.1
        let model = make_model(PilotType::FirstOrder, 1.0, 0.0, 0.0);
        let state = PilotState::default();
        let result = apply_pilot(&model, 1.0, &state, 0.1, 10.0, &PilotBiases::default());
        assert!((result.bank_angle - 0.1).abs() < EPS);
        assert!((result.bank_rate - 1.0).abs() < EPS);
    }

    #[test]
    fn first_order_rate_clamped() {
        // tau=0.1, cmd=2.0 rad (~115°) from 0: shortest_diff = 2.0, unclamped rate = 2.0/0.1 = 20, clamped to 0.5
        // angle = 0 + 0.5 * 0.1 = 0.05
        let model = make_model(PilotType::FirstOrder, 0.1, 0.0, 0.0);
        let state = PilotState::default();
        let result = apply_pilot(&model, 2.0, &state, 0.1, 0.5, &PilotBiases::default());
        assert!((result.bank_angle - 0.05).abs() < EPS);
        assert!((result.bank_rate - 0.5).abs() < EPS);
    }

    #[test]
    fn second_order_at_rest_accelerates() {
        // omega=2, zeta=0.7, cmd=1.0, state=(0,0), dt=0.1
        // error = 0 - 1 = -1, accel = -2*0.7*2*0 - 4*(-1) = 4
        // rate = 0 + 4*0.1 = 0.4, angle = 0 + 0.4*0.1 = 0.04
        let model = make_model(PilotType::SecondOrder, 0.0, 0.7, 2.0);
        let state = PilotState::default();
        let result = apply_pilot(&model, 1.0, &state, 0.1, 10.0, &PilotBiases::default());
        assert!((result.bank_rate - 0.4).abs() < EPS);
        assert!((result.bank_angle - 0.04).abs() < EPS);
    }

    #[test]
    fn first_order_wraps_through_pi_shortest_path() {
        use std::f64::consts::PI;
        let deg = PI / 180.0;
        let model = make_model(PilotType::FirstOrder, 1.0, 0.0, 0.0);
        let state = PilotState {
            bank_angle: 170.0 * deg,
            bank_rate: 0.0,
        };
        let result = apply_pilot(
            &model,
            -170.0 * deg,
            &state,
            0.1,
            10.0,
            &PilotBiases::default(),
        );
        // Shortest diff is +20° (0.349 rad), rate = 0.349/1.0 = 0.349, new = 170° + 0.349*0.1 = 170.035°
        assert!(
            result.bank_rate > 0.0,
            "rate should be positive (through +180°), got {}",
            result.bank_rate
        );
        assert!(
            result.bank_angle > 170.0 * deg,
            "should move toward +180°, got {} deg",
            result.bank_angle / deg
        );
    }

    #[test]
    fn first_order_bias_slows_response() {
        // bias.tau = 1.0 → effective tau = 1.0 * (1+1) = 2.0
        // rate = 1.0/2.0 = 0.5, angle = 0.5*0.1 = 0.05
        let model = make_model(PilotType::FirstOrder, 1.0, 0.0, 0.0);
        let state = PilotState::default();
        let biases = PilotBiases {
            tau: 1.0,
            ..Default::default()
        };
        let result = apply_pilot(&model, 1.0, &state, 0.1, 10.0, &biases);
        assert!((result.bank_angle - 0.05).abs() < EPS);
        assert!((result.bank_rate - 0.5).abs() < EPS);
    }
}
