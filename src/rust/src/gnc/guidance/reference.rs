//! Reference trajectory guidance (constant bank angle).
//!
//! Used when irefer=1. Simply holds a fixed bank angle throughout.

use super::{Guidance, GuidanceCommand};
use crate::data::SphericalState;

#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_configured_bank_angle() {
        let bank = 1.13; // ~64.77°
        let aoa = 0.175; // ~10°
        let mut guidance = ReferenceGuidance { bank_angle: bank, aoa };
        let cmd = guidance.compute(&Default::default(), 0.0);
        assert_eq!(cmd.bank_angle, bank);
        assert_eq!(cmd.aoa, aoa);
    }

    #[test]
    fn ignores_state_and_time() {
        let bank = 0.5;
        let aoa = -0.48;
        let mut guidance = ReferenceGuidance { bank_angle: bank, aoa };

        // Different states and times should all return the same command
        let cmd1 = guidance.compute(&Default::default(), 0.0);
        let cmd2 = guidance.compute(&Default::default(), 999.0);
        assert_eq!(cmd1.bank_angle, cmd2.bank_angle);
        assert_eq!(cmd1.aoa, cmd2.aoa);
    }
}
