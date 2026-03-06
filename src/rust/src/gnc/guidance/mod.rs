//! Guidance algorithms.

pub mod ftc;
pub mod neural;
pub mod reference;

use crate::data::SphericalState;

/// Guidance command output
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GuidanceCommand {
    pub bank_angle: f64, // radians (commanded roll/bank angle)
    pub aoa: f64,        // radians (commanded angle of attack)
}

/// Guidance algorithm trait
#[allow(dead_code)]
pub trait Guidance {
    /// Compute guidance command given current measured state and simulation context.
    fn compute(&mut self, state: &SphericalState, time: f64) -> GuidanceCommand;
}
