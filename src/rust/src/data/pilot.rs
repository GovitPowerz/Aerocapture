//! Pilot dynamics model loader.
//!
//! Reads from `pilote.*` data files (unit 115 in Fortran).
//!
//! Format: 3 header lines, then:
//!   natpil — pilot model type (0: perfect, 1: first order, 2: second order)
//!   cstpil — first-order time constant (s)
//!   amrpil — second-order damping ratio
//!   omgpil — second-order natural frequency (rad/s)

use super::{parse_data_file, DataError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PilotType {
    Perfect,     // natpil = 0
    FirstOrder,  // natpil = 1
    SecondOrder, // natpil = 2
}

#[derive(Debug, Clone, Copy)]
pub struct PilotModel {
    pub pilot_type: PilotType,
    pub time_constant: f64,   // first-order tau (s)
    pub damping: f64,         // second-order zeta
    pub frequency: f64,       // second-order omega (rad/s)
}

impl PilotModel {
    pub fn load(path: &str) -> Result<Self, DataError> {
        let rows = parse_data_file(path)?;
        if rows.len() < 4 {
            return Err(DataError(format!(
                "Pilot file too short ({} rows, need 4): {}",
                rows.len(),
                path
            )));
        }

        let natpil = rows[0][0] as i32;
        let pilot_type = match natpil {
            0 => PilotType::Perfect,
            1 => PilotType::FirstOrder,
            2 => PilotType::SecondOrder,
            _ => {
                return Err(DataError(format!(
                    "Invalid pilot type {}: {}",
                    natpil, path
                )))
            }
        };

        Ok(PilotModel {
            pilot_type,
            time_constant: rows[1][0],
            damping: rows[2][0],
            frequency: rows[3][0],
        })
    }
}
