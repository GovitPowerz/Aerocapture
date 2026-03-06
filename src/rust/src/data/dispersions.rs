//! Monte Carlo dispersion parameters loader.
//!
//! Reads from `dispersions.*` data files (unit 106 in Fortran).
//!
//! Format: 3 header lines, then 11 standard deviations.
//! Multipliers xmulti(2) and xmulti(4) are applied per Fortran lectci.f.

use super::{DataError, parse_data_file};

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DispersionParams {
    // Initial state dispersions (1-sigma, after multiplier xmulti(2))
    pub altitude: f64,    // meters (from km)
    pub longitude: f64,   // radians (from deg)
    pub latitude: f64,    // radians (from deg)
    pub velocity: f64,    // m/s
    pub flight_path: f64, // radians (from deg)
    pub azimuth: f64,     // radians (from deg)

    // Aerodynamic/model dispersions (1-sigma, after multiplier xmulti(4))
    pub drag_coeff: f64, // fractional (from %)
    pub lift_coeff: f64, // fractional (from %)
    pub density: f64,    // fractional (from %)
    pub incidence: f64,  // radians (from deg)
    pub mass: f64,       // fractional (from %)
}

impl DispersionParams {
    pub fn load(path: &str, xmulti: &[f64; 4]) -> Result<Self, DataError> {
        let rows = parse_data_file(path)?;
        if rows.len() < 11 {
            return Err(DataError(format!(
                "Dispersions file too short ({} rows, need 11): {}",
                rows.len(),
                path
            )));
        }

        // Matches Fortran lectci.f conversion exactly
        Ok(DispersionParams {
            altitude: xmulti[1] * rows[0][0] * 1e3, // km -> m
            longitude: xmulti[1] * rows[1][0] * DEG2RAD,
            latitude: xmulti[1] * rows[2][0] * DEG2RAD,
            velocity: xmulti[1] * rows[3][0], // m/s (no multiplier in Fortran? actually dvitzd = xmulti(2)*dvitzd without conversion)
            flight_path: xmulti[1] * rows[4][0] * DEG2RAD,
            azimuth: xmulti[1] * rows[5][0] * DEG2RAD,
            drag_coeff: xmulti[3] * rows[6][0] / 100.0,
            lift_coeff: xmulti[3] * rows[7][0] / 100.0,
            density: rows[8][0] / 100.0, // droatm — no multiplier in Fortran
            incidence: xmulti[3] * rows[9][0] * DEG2RAD,
            mass: rows[10][0] / 100.0,
        })
    }
}
