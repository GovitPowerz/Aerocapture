//! Wind model.
//!
//! Placeholder for Fortran fvents.f wind model.
//! Currently returns zero wind velocity.

/// Wind velocity components (m/s) in local frame.
pub struct WindVelocity {
    pub north: f64,
    pub east: f64,
    pub vertical: f64,
}

/// Compute wind velocity at a given position.
///
/// Currently returns zero (no wind model).
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
