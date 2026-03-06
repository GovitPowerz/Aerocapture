//! Gravity model with J2 oblate planet correction.
//!
//! Matches Fortran fgravi.f exactly.

use crate::config::Planet;

/// Compute gravitational acceleration components in spherical coordinates.
///
/// Returns (gravtl, gravtr):
///   - gravtl: lateral (latitudinal) component from J2 (m/s^2)
///   - gravtr: radial component, positive outward (m/s^2)
///
/// Matches Fortran fgravi.f equations.
pub fn gravity(radius: f64, latitude: f64, planet: &Planet) -> (f64, f64) {
    let mu = planet.mu();
    let req = planet.equatorial_radius();
    let j2 = planet.j2();

    let r2 = radius * radius;
    let r4 = r2 * r2;
    let sin_lat = latitude.sin();
    let cos_lat = latitude.cos();
    let sin2 = sin_lat * sin_lat;
    let req2 = req * req;

    // Lateral component (from J2)
    let gravtl = 3.0 * mu * j2 * req2 * sin_lat * cos_lat / r4;

    // Radial component (positive outward)
    let gravtr = mu / r2 + 3.0 * mu * j2 * req2 * (1.0 - 3.0 * sin2) / (2.0 * r4);

    (gravtl, gravtr)
}
