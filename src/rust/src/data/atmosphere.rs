//! Atmosphere model loader.
//!
//! Reads from `atmosphere.*` data files (unit 105 in Fortran).
//!
//! Format: 3 header lines, then:
//!   naltit — number of altitude points
//!   naltit lines of: altitude(m)  density(kg/m3)
//!   density dispersion profile boundary marker (-1)
//!   4 altitude breakpoints (km) — or fewer if marker is not -1
//!   4 density error breakpoints (%)
//!   rozmod — exponential model reference density (kg/m3)
//!   facech — exponential model scale factor (1/m)
//!   zromod — exponential model reference altitude (m)
//!   cstgam — gas constant ratio (gamma)

use super::{parse_data_file, DataError};

/// Density dispersion profile (altitude-dependent bias envelope)
#[derive(Debug, Clone)]
pub struct DensityProfile {
    pub altitudes: Vec<f64>,       // meters (up to 5 breakpoints)
    pub max_dispersion: Vec<f64>,  // fractional (converted from %)
    pub slopes: Vec<f64>,          // linear interpolation slopes
    pub intercepts: Vec<f64>,      // linear interpolation intercepts
}

#[derive(Debug, Clone)]
pub struct AtmosphereModel {
    pub n_points: usize,
    pub altitudes: Vec<f64>,    // meters
    pub densities: Vec<f64>,    // kg/m^3
    pub ref_density: f64,       // exponential model rho0 (kg/m3)
    pub scale_factor: f64,      // exponential model H^-1 (1/m)
    pub ref_altitude: f64,      // exponential model z0 (m)
    pub gas_constant: f64,      // gamma (ratio of specific heats)
    pub density_profile: DensityProfile,
}

impl AtmosphereModel {
    pub fn load(path: &str) -> Result<Self, DataError> {
        let rows = parse_data_file(path)?;
        if rows.is_empty() {
            return Err(DataError(format!("Atmosphere file empty: {}", path)));
        }

        let n_points = rows[0][0] as usize;
        if rows.len() < 1 + n_points + 10 {
            return Err(DataError(format!(
                "Atmosphere file too short ({} rows): {}",
                rows.len(),
                path
            )));
        }

        let mut altitudes = Vec::with_capacity(n_points);
        let mut densities = Vec::with_capacity(n_points);
        for i in 0..n_points {
            let row = &rows[1 + i];
            altitudes.push(row[0]);
            densities.push(row[1]);
        }

        // After the density table: marker line (-1), then profile breakpoints
        let mut idx = 1 + n_points;

        // Skip the marker line (contains -1)
        idx += 1;

        // Read 4 altitude breakpoints (km) and 4 dispersion breakpoints (%)
        // Fortran reads 5 of each but first altitude is 0
        let mut prof_alts = vec![0.0f64]; // first breakpoint is 0
        for _ in 0..4 {
            if idx >= rows.len() {
                break;
            }
            prof_alts.push(rows[idx][0] * 1e3); // km -> m
            idx += 1;
        }

        let mut prof_disp = vec![0.0f64]; // first dispersion is 0%
        for _ in 0..4 {
            if idx >= rows.len() {
                break;
            }
            prof_disp.push(rows[idx][0] / 100.0); // % -> fraction
            idx += 1;
        }

        // Compute slopes and intercepts for piecewise-linear profile
        // (matches Fortran xgabro computation)
        let n_prof = prof_alts.len();
        let mut slopes = vec![0.0f64; n_prof];
        let mut intercepts = vec![0.0f64; n_prof];
        for i in 1..n_prof {
            let dalt = prof_alts[i] - prof_alts[i - 1];
            if dalt.abs() > 1e-30 {
                slopes[i] = (prof_disp[i] - prof_disp[i - 1]) / dalt;
                intercepts[i] = prof_disp[i] - slopes[i] * prof_alts[i];
            }
        }

        // Exponential model parameters
        let ref_density = if idx < rows.len() { rows[idx][0] } else { 0.0 };
        idx += 1;
        let scale_factor = if idx < rows.len() { rows[idx][0] } else { 0.0 };
        idx += 1;
        let ref_altitude = if idx < rows.len() { rows[idx][0] } else { 0.0 };
        idx += 1;
        let gas_constant = if idx < rows.len() { rows[idx][0] } else { 1.3 };

        Ok(AtmosphereModel {
            n_points,
            altitudes,
            densities,
            ref_density,
            scale_factor,
            ref_altitude,
            gas_constant,
            density_profile: DensityProfile {
                altitudes: prof_alts,
                max_dispersion: prof_disp,
                slopes,
                intercepts,
            },
        })
    }

    /// Interpolate density at a given altitude (linear interpolation in table)
    pub fn density_at(&self, altitude: f64) -> f64 {
        let n = self.n_points;
        if n == 0 {
            return self.exponential_density(altitude);
        }
        if altitude <= self.altitudes[0] {
            return self.densities[0];
        }
        if altitude >= self.altitudes[n - 1] {
            return self.exponential_density(altitude);
        }
        for i in 1..n {
            if altitude <= self.altitudes[i] {
                let frac = (altitude - self.altitudes[i - 1])
                    / (self.altitudes[i] - self.altitudes[i - 1]);
                return self.densities[i - 1]
                    + frac * (self.densities[i] - self.densities[i - 1]);
            }
        }
        self.exponential_density(altitude)
    }

    /// Exponential atmosphere model: rho = rho0 * exp(-H * (z - z0))
    pub fn exponential_density(&self, altitude: f64) -> f64 {
        self.ref_density * (-self.scale_factor * (altitude - self.ref_altitude)).exp()
    }
}
