//! Atmospheric density computation.
//!
//! Wraps the atmosphere data model for use in the simulation loop.
//! Matches Fortran fatmos.f.

use crate::data::atmosphere::AtmosphereModel;

/// Compute atmospheric density at a given geodetic altitude.
///
/// Applies optional density bias for Monte Carlo dispersions.
#[allow(dead_code)]
pub fn density(atm: &AtmosphereModel, altitude: f64, density_bias: f64) -> f64 {
    let rho = atm.density_at(altitude);
    rho * (1.0 + density_bias)
}
