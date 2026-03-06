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
