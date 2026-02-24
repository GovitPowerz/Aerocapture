//! Reference trajectory guidance (constant bank angle).
//!
//! Used when irefer=1. Simply holds a fixed bank angle throughout.

use super::{Guidance, GuidanceCommand};
use crate::data::SphericalState;

pub struct ReferenceGuidance {
    pub bank_angle: f64,
    pub aoa: f64,
}

impl Guidance for ReferenceGuidance {
    fn compute(&mut self, _state: &SphericalState, _time: f64) -> GuidanceCommand {
        GuidanceCommand {
            bank_angle: self.bank_angle,
            aoa: self.aoa,
        }
    }
}
