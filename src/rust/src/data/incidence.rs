//! Incidence (angle of attack) profile loader.
//!
//! Reads from `incidence.*` data files (unit 110 in Fortran).
//!
//! Format: 3 header lines, then:
//!   nbalfa — number of profile points
//!   nbalfa altitude values (km)
//!   nbalfa incidence values (deg)

use super::{parse_data_file, DataError};

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

#[derive(Debug, Clone)]
pub struct IncidenceProfile {
    pub n_points: usize,
    pub altitudes: Vec<f64>,   // meters (from km)
    pub incidences: Vec<f64>,  // radians (from deg)
}

impl IncidenceProfile {
    pub fn load(path: &str) -> Result<Self, DataError> {
        let rows = parse_data_file(path)?;
        if rows.is_empty() {
            return Err(DataError(format!("Incidence file empty: {}", path)));
        }

        let n_points = rows[0][0] as usize;
        if rows.len() < 1 + 2 * n_points {
            return Err(DataError(format!(
                "Incidence file too short ({} rows, need {}): {}",
                rows.len(),
                1 + 2 * n_points,
                path
            )));
        }

        let mut altitudes = Vec::with_capacity(n_points);
        let mut incidences = Vec::with_capacity(n_points);

        for i in 0..n_points {
            altitudes.push(rows[1 + i][0] * 1e3); // km -> m
        }
        for i in 0..n_points {
            incidences.push(rows[1 + n_points + i][0] * DEG2RAD);
        }

        Ok(IncidenceProfile {
            n_points,
            altitudes,
            incidences,
        })
    }

    /// Interpolate commanded incidence at a given altitude
    pub fn incidence_at(&self, altitude: f64) -> f64 {
        if self.n_points == 0 {
            return 0.0;
        }
        if self.n_points == 1 || altitude <= self.altitudes[0] {
            return self.incidences[0];
        }
        if altitude >= self.altitudes[self.n_points - 1] {
            return self.incidences[self.n_points - 1];
        }
        for i in 1..self.n_points {
            if altitude <= self.altitudes[i] {
                let frac = (altitude - self.altitudes[i - 1])
                    / (self.altitudes[i] - self.altitudes[i - 1]);
                return self.incidences[i - 1]
                    + frac * (self.incidences[i] - self.incidences[i - 1]);
            }
        }
        self.incidences[self.n_points - 1]
    }
}
