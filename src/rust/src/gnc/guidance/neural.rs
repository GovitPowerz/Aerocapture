//! Neural network guidance.
//!
//! Placeholder for the NN guidance from Fortran guidnn.f.

use super::{Guidance, GuidanceCommand};
use crate::data::SphericalState;

pub struct NeuralGuidance {
    pub bank_angle: f64,
    pub aoa: f64,
    // TODO: NN weights, architecture
}

impl Guidance for NeuralGuidance {
    fn compute(&mut self, _state: &SphericalState, _time: f64) -> GuidanceCommand {
        // TODO: Implement NN forward pass
        GuidanceCommand {
            bank_angle: self.bank_angle,
            aoa: self.aoa,
        }
    }
}
