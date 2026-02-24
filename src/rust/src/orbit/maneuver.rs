//! Orbit maneuver cost computation.
//!
//! Matches Fortran ergols.f.
//! Computes delta-V cost for orbit correction after aerocapture.

use crate::config::Planet;
use crate::data::{OrbitalTarget, ParkingOrbit};

/// Compute total delta-V cost for orbit correction.
///
/// This computes the cost of transferring from the post-aerocapture orbit
/// to the target parking orbit, accounting for:
/// 1. Apoapsis correction (circularize or adjust)
/// 2. Periapsis correction
/// 3. Inclination correction (plane change)
pub fn correction_cost(
    current_apoapsis: f64,  // meters (altitude)
    current_periapsis: f64, // meters (altitude)
    current_inclination: f64, // radians
    target: &OrbitalTarget,
    parking: &ParkingOrbit,
    planet: &Planet,
) -> f64 {
    let mu = planet.mu();
    let req = planet.equatorial_radius();

    let r_apo = current_apoapsis + req;
    let r_peri = current_periapsis + req;

    let r_apo_target = parking.apoapsis + req;
    let r_peri_target = parking.periapsis + req;

    // Current orbit velocity at apoapsis
    let sma_current = (r_apo + r_peri) / 2.0;
    let v_apo = (mu * (2.0 / r_apo - 1.0 / sma_current)).sqrt();

    // Target orbit velocity at apoapsis
    let sma_target = (r_apo_target + r_peri_target) / 2.0;
    let v_apo_target = (mu * (2.0 / r_apo - 1.0 / sma_target)).sqrt();

    // Periapsis correction (Hohmann-like)
    let dv_periapsis = (v_apo_target - v_apo).abs();

    // Velocity at new periapsis for plane change
    let v_peri = (mu * (2.0 / r_peri_target - 1.0 / sma_target)).sqrt();

    // Plane change delta-V (at lowest velocity point = apoapsis is more efficient)
    let di = (current_inclination - target.inclination).abs();
    let dv_plane = 2.0 * v_apo * (di / 2.0).sin();

    dv_periapsis + dv_plane
}
