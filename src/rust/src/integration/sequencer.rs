//! Module cadence scheduling.
//!
//! Matches Fortran sequen.f.
//! Determines when each subsystem (navigation, guidance, pilot) should execute
//! based on configured time periods.

use crate::data::TimePeriods;

/// Check if a subsystem should execute at the current time.
///
/// A subsystem executes when the elapsed time since its last execution
/// exceeds its configured period.
pub fn should_execute(time: f64, last_time: f64, period: f64) -> bool {
    if period <= 0.0 {
        return true;
    }
    (time - last_time) >= period - 1e-10
}

/// Sequencer state tracking last execution times
#[derive(Debug, Clone, Copy)]
pub struct SequencerState {
    pub last_navigation: f64,
    pub last_guidance: f64,
    pub last_pilot: f64,
    pub last_prediction: f64,
    pub last_photo: f64,
}

impl Default for SequencerState {
    fn default() -> Self {
        Self::new()
    }
}

impl SequencerState {
    pub fn new() -> Self {
        Self {
            last_navigation: f64::NEG_INFINITY,
            last_guidance: f64::NEG_INFINITY,
            last_pilot: f64::NEG_INFINITY,
            last_prediction: f64::NEG_INFINITY,
            last_photo: f64::NEG_INFINITY,
        }
    }

    /// Check which subsystems should execute and update their timestamps.
    pub fn update(&mut self, time: f64, periods: &TimePeriods) -> SequencerFlags {
        let nav = should_execute(time, self.last_navigation, periods.navigation);
        let guid = should_execute(time, self.last_guidance, periods.guidance);
        let pilot = should_execute(time, self.last_pilot, periods.pilot);
        let pred = should_execute(time, self.last_prediction, periods.prediction);
        let photo = should_execute(time, self.last_photo, periods.photo);

        if nav {
            self.last_navigation = time;
        }
        if guid {
            self.last_guidance = time;
        }
        if pilot {
            self.last_pilot = time;
        }
        if pred {
            self.last_prediction = time;
        }
        if photo {
            self.last_photo = time;
        }

        SequencerFlags {
            nav,
            guid,
            pilot,
            pred,
            photo,
        }
    }
}

/// Flags indicating which subsystems should execute this step
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct SequencerFlags {
    pub nav: bool,
    pub guid: bool,
    pub pilot: bool,
    pub pred: bool,
    pub photo: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build TimePeriods with uniform photo/prediction/integration periods.
    fn make_periods(nav: f64, guid: f64, pilot: f64) -> TimePeriods {
        TimePeriods {
            navigation: nav,
            guidance: guid,
            pilot,
            prediction: 1.0,
            integration: 1.0,
            photo: 1.0,
        }
    }

    #[test]
    fn first_call_always_fires() {
        let mut seq = SequencerState::new();
        let periods = make_periods(1.0, 2.0, 0.5);
        let flags = seq.update(0.0, &periods);
        assert!(flags.nav, "nav should fire on first call");
        assert!(flags.guid, "guid should fire on first call");
        assert!(flags.pilot, "pilot should fire on first call");
        assert!(flags.pred, "pred should fire on first call");
        assert!(flags.photo, "photo should fire on first call");
    }

    #[test]
    fn respects_cadence() {
        let mut seq = SequencerState::new();
        let periods = make_periods(1.0, 2.0, 0.5);

        // First call at t=0 — fires everything and records timestamps.
        let _ = seq.update(0.0, &periods);

        // At t=0.5: only pilot (period=0.5) should fire.
        let flags = seq.update(0.5, &periods);
        assert!(!flags.nav, "nav (period=1.0) should NOT fire at t=0.5");
        assert!(!flags.guid, "guid (period=2.0) should NOT fire at t=0.5");
        assert!(flags.pilot, "pilot (period=0.5) should fire at t=0.5");
    }

    #[test]
    fn fires_at_period() {
        let mut seq = SequencerState::new();
        let periods = make_periods(1.0, 2.0, 0.5);

        // t=0: fire everything
        let _ = seq.update(0.0, &periods);
        // t=0.5: fire pilot
        let _ = seq.update(0.5, &periods);

        // t=1.0: nav (period=1.0) and pilot (period=0.5) should fire; guid (period=2.0) should not.
        let flags = seq.update(1.0, &periods);
        assert!(flags.nav, "nav should fire at t=1.0 (period=1.0)");
        assert!(!flags.guid, "guid should NOT fire at t=1.0 (period=2.0)");
        assert!(flags.pilot, "pilot should fire at t=1.0 (period=0.5)");
    }

    #[test]
    fn zero_period_always_fires() {
        assert!(should_execute(0.0, 0.0, 0.0));
        assert!(should_execute(100.0, 99.0, 0.0));
        assert!(should_execute(0.0, 0.0, -1.0));
    }

    #[test]
    fn tolerance_handling() {
        // Elapsed = 1.0, period = 1.0 + 1e-11. The tolerance is 1e-10,
        // so 1.0 >= (1.0 + 1e-11) - 1e-10 = 1.0 - 8.9e-11 ≈ true.
        assert!(
            should_execute(1.0, 0.0, 1.0 + 1e-11),
            "1e-11 overshoot should still fire thanks to 1e-10 tolerance"
        );

        // Elapsed = 1.0, period = 1.0 + 1e-8 — well outside tolerance, should NOT fire.
        assert!(
            !should_execute(1.0, 0.0, 1.0 + 1e-8),
            "1e-8 overshoot exceeds tolerance and should NOT fire"
        );
    }
}
