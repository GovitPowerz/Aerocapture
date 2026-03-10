//! Wind model.
//!
//! Placeholder for Fortran fvents.f wind model.
//! Currently returns zero wind velocity.

/// Wind velocity components (m/s) in local frame.
#[allow(dead_code)]
pub struct WindVelocity {
    pub north: f64,
    pub east: f64,
    pub vertical: f64,
}

/// Compute wind velocity at a given position.
///
/// Currently returns zero (no wind model).
#[allow(dead_code)]
pub fn wind_velocity(
    _altitude: f64,
    _latitude: f64,
    _longitude: f64,
    _enabled: bool,
) -> WindVelocity {
    WindVelocity {
        north: 0.0,
        east: 0.0,
        vertical: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With wind disabled the stub always returns a zero velocity vector.
    #[test]
    fn disabled_returns_zero() {
        let w = wind_velocity(40_000.0, 0.3, 1.2, false);
        assert_eq!(w.north, 0.0, "north should be zero when disabled");
        assert_eq!(w.east, 0.0, "east should be zero when disabled");
        assert_eq!(w.vertical, 0.0, "vertical should be zero when disabled");
    }

    /// With wind enabled the stub still returns zero — documents stub-only behaviour.
    #[test]
    fn enabled_returns_zero_for_stub() {
        let w = wind_velocity(10_000.0, -0.5, 2.0, true);
        assert_eq!(w.north, 0.0, "stub: north should be zero even when enabled");
        assert_eq!(w.east, 0.0, "stub: east should be zero even when enabled");
        assert_eq!(
            w.vertical, 0.0,
            "stub: vertical should be zero even when enabled"
        );
    }
}
