//! Capsule properties loader.
//!
//! Reads from `capsule.*` data files (unit 100 in Fortran).
//!
//! Format: 3 header lines, then:
//!   xmasse (kg), srefer (m2), cq, vgitmx (deg/s),
//!   tnavig (s), tguida (s), tpilot (s), tpredi (s), tinteg (s), tphoto (s)

use super::{DataError, TimePeriods, parse_data_file};

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

#[derive(Debug, Clone, Copy)]
pub struct Capsule {
    pub mass: f64,           // kg
    pub reference_area: f64, // m^2
    pub cq: f64,             // heat flux coefficient
    pub max_bank_rate: f64,  // rad/s
    pub periods: TimePeriods,
}

impl Capsule {
    pub fn load(path: &str) -> Result<Self, DataError> {
        let rows = parse_data_file(path)?;
        if rows.len() < 10 {
            return Err(DataError(format!(
                "Capsule file too short ({} rows, need 10): {}",
                rows.len(),
                path
            )));
        }

        Ok(Capsule {
            mass: rows[0][0],
            reference_area: rows[1][0],
            cq: rows[2][0],
            max_bank_rate: rows[3][0] * DEG2RAD,
            periods: TimePeriods {
                navigation: rows[4][0],
                guidance: rows[5][0],
                pilot: rows[6][0],
                prediction: rows[7][0],
                integration: rows[8][0],
                photo: rows[9][0],
            },
        })
    }
}
