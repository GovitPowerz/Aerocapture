//! Navigation error parameters loader.
//!
//! Reads from `navigation.*` data files (unit 107 in Fortran).
//!
//! Format: 3 header lines, then 7 standard deviations.
//! Multipliers xmulti(1) and xmulti(3) are applied per Fortran lectci.f.

use super::{parse_data_file, DataError};

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

#[derive(Debug, Clone, Copy, Default)]
pub struct NavigationParams {
    pub altitude: f64,    // meters (from km)
    pub latitude: f64,    // radians (from deg)
    pub longitude: f64,   // radians (from deg)
    pub velocity: f64,    // m/s
    pub flight_path: f64, // radians (from deg)
    pub azimuth: f64,     // radians (from deg)
    pub drag_accel: f64,  // m/s^2
}

impl NavigationParams {
    pub fn load(path: &str, xmulti: &[f64; 4]) -> Result<Self, DataError> {
        let rows = parse_data_file(path)?;
        if rows.len() < 7 {
            return Err(DataError(format!(
                "Navigation file too short ({} rows, need 7): {}",
                rows.len(),
                path
            )));
        }

        // Fortran order: dnaval, dnavla, dnavlo, dnavvi, dnavpe, dnavaz, dnavad
        // With multiplier xmulti(1) for position/velocity, xmulti(3) for drag accel
        Ok(NavigationParams {
            altitude: xmulti[0] * rows[0][0] * 1e3,
            latitude: xmulti[0] * rows[1][0] * DEG2RAD,
            longitude: xmulti[0] * rows[2][0] * DEG2RAD,
            velocity: xmulti[0] * rows[3][0],
            flight_path: xmulti[0] * rows[4][0] * DEG2RAD,
            azimuth: xmulti[0] * rows[5][0] * DEG2RAD,
            drag_accel: xmulti[2] * rows[6][0],
        })
    }
}
