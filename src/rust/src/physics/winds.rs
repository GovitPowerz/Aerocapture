//! Wind model.
//!
//! Loads an altitude-tabulated wind profile and interpolates zonal/meridional
//! components. Zonal (eastward) wind is cosine-scaled with latitude; meridional
//! (northward) wind is returned without latitude scaling.

use crate::data::DataError;

/// Wind velocity components (m/s) in local frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct WindVelocity {
    pub north: f64,    // meridional, positive = northward (m/s)
    pub east: f64,     // zonal, positive = eastward (m/s)
    pub vertical: f64, // always 0.0
}

/// Altitude-tabulated wind profile.
#[derive(Debug, Clone, Default)]
pub struct WindTable {
    pub n_points: usize,
    pub altitudes_m: Vec<f64>,    // meters
    pub zonal_m_s: Vec<f64>,      // eastward wind (m/s)
    pub meridional_m_s: Vec<f64>, // northward wind (m/s)
}

impl WindTable {
    /// Load a wind table from a data file.
    ///
    /// Format:
    /// - Lines starting with `#` are comments (skipped)
    /// - First non-comment line: integer count N
    /// - Next N lines: `altitude_km  zonal_m_s  meridional_m_s`
    pub fn load(path: &str) -> Result<Self, DataError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| DataError(format!("Cannot read wind file {}: {}", path, e)))?;

        let mut data_lines = content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'));

        // First non-comment line is the count
        let count_str = data_lines
            .next()
            .ok_or_else(|| DataError(format!("Wind file empty: {}", path)))?;
        let n_points: usize = count_str
            .parse()
            .map_err(|_| DataError(format!("Wind file: bad count line '{}': {}", count_str, path)))?;

        let mut altitudes_m = Vec::with_capacity(n_points);
        let mut zonal_m_s = Vec::with_capacity(n_points);
        let mut meridional_m_s = Vec::with_capacity(n_points);

        for (i, line) in data_lines.take(n_points).enumerate() {
            let cols: Vec<f64> = line
                .split_whitespace()
                .filter_map(|t| t.parse::<f64>().ok())
                .collect();
            if cols.len() < 3 {
                return Err(DataError(format!(
                    "Wind file {}: row {} has only {} columns (need 3)",
                    path,
                    i + 1,
                    cols.len()
                )));
            }
            altitudes_m.push(cols[0] * 1e3); // km → m
            zonal_m_s.push(cols[1]);
            meridional_m_s.push(cols[2]);
        }

        if altitudes_m.len() < n_points {
            return Err(DataError(format!(
                "Wind file {}: expected {} rows, got {}",
                path,
                n_points,
                altitudes_m.len()
            )));
        }

        Ok(WindTable {
            n_points,
            altitudes_m,
            zonal_m_s,
            meridional_m_s,
        })
    }

    /// Interpolate wind at a given altitude and latitude.
    ///
    /// - Below table: clamp to first entry.
    /// - Above table: clamp to last entry.
    /// - Zonal component is cosine-scaled with latitude.
    pub fn wind_at(&self, altitude_m: f64, latitude_rad: f64) -> WindVelocity {
        let n = self.n_points;
        if n == 0 {
            return WindVelocity::default();
        }

        let (zonal_interp, meridional_interp) = if altitude_m <= self.altitudes_m[0] {
            (self.zonal_m_s[0], self.meridional_m_s[0])
        } else if altitude_m >= self.altitudes_m[n - 1] {
            (self.zonal_m_s[n - 1], self.meridional_m_s[n - 1])
        } else {
            // Binary search for the interval
            let idx = self
                .altitudes_m
                .partition_point(|&a| a < altitude_m)
                .min(n - 1);
            let i = idx - 1; // lower bound index
            let alt_lo = self.altitudes_m[i];
            let alt_hi = self.altitudes_m[idx];
            let frac = (altitude_m - alt_lo) / (alt_hi - alt_lo);
            let zonal = self.zonal_m_s[i] + frac * (self.zonal_m_s[idx] - self.zonal_m_s[i]);
            let merid =
                self.meridional_m_s[i] + frac * (self.meridional_m_s[idx] - self.meridional_m_s[i]);
            (zonal, merid)
        };

        WindVelocity {
            north: meridional_interp,
            east: zonal_interp * latitude_rad.cos(),
            vertical: 0.0,
        }
    }
}

/// Compute wind velocity at a given position.
///
/// Returns zero if `table` is `None`; otherwise delegates to `WindTable::wind_at`.
pub fn wind_velocity(
    altitude_m: f64,
    latitude_rad: f64,
    _longitude_rad: f64,
    table: Option<&WindTable>,
) -> WindVelocity {
    match table {
        None => WindVelocity::default(),
        Some(t) => t.wind_at(altitude_m, latitude_rad),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::fs;
    use std::path::PathBuf;

    /// Build a simple in-memory WindTable for unit tests.
    fn make_table() -> WindTable {
        WindTable {
            n_points: 3,
            altitudes_m: vec![0.0, 10_000.0, 20_000.0],
            zonal_m_s: vec![10.0, 20.0, 30.0],
            meridional_m_s: vec![1.0, 2.0, 3.0],
        }
    }

    #[test]
    fn interpolation_at_table_points() {
        let t = make_table();
        let w0 = t.wind_at(0.0, 0.0);
        assert_abs_diff_eq!(w0.east, 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w0.north, 1.0, epsilon = 1e-10);

        let w1 = t.wind_at(10_000.0, 0.0);
        assert_abs_diff_eq!(w1.east, 20.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w1.north, 2.0, epsilon = 1e-10);

        let w2 = t.wind_at(20_000.0, 0.0);
        assert_abs_diff_eq!(w2.east, 30.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w2.north, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn interpolation_between_points() {
        let t = make_table();
        // Midpoint between 0 and 10 km → zonal=15, meridional=1.5
        let w = t.wind_at(5_000.0, 0.0);
        assert_abs_diff_eq!(w.east, 15.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w.north, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn above_table_returns_last() {
        let t = make_table();
        let w = t.wind_at(50_000.0, 0.0);
        assert_abs_diff_eq!(w.east, 30.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w.north, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn below_table_returns_first() {
        let t = make_table();
        let w = t.wind_at(-500.0, 0.0);
        assert_abs_diff_eq!(w.east, 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w.north, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn latitude_cosine_scaling() {
        let t = make_table();
        // At equator (lat=0): cos(0) = 1.0, east = zonal unchanged
        let w_eq = t.wind_at(10_000.0, 0.0);
        assert_abs_diff_eq!(w_eq.east, 20.0, epsilon = 1e-10);

        // At pole (lat=π/2): cos(π/2) ≈ 0, east ≈ 0
        let w_pole = t.wind_at(10_000.0, std::f64::consts::FRAC_PI_2);
        assert_abs_diff_eq!(w_pole.east, 0.0, epsilon = 1e-10);

        // Meridional not scaled — same at both latitudes
        assert_abs_diff_eq!(w_eq.north, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(w_pole.north, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn vertical_always_zero() {
        let t = make_table();
        for alt in [0.0, 5_000.0, 10_000.0, 25_000.0] {
            let w = t.wind_at(alt, 0.3);
            assert_eq!(w.vertical, 0.0);
        }
    }

    #[test]
    fn disabled_returns_zero() {
        let w = wind_velocity(40_000.0, 0.3, 1.2, None);
        assert_eq!(w.north, 0.0);
        assert_eq!(w.east, 0.0);
        assert_eq!(w.vertical, 0.0);
    }

    #[test]
    fn load_mars_wind_file() {
        // Resolve path relative to the workspace root
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        let path = PathBuf::from(&manifest)
            .join("../../data/atmosphere/mars_winds.dat");
        let path_str = path.to_str().expect("path to str");

        let table = WindTable::load(path_str).expect("load mars_winds.dat");
        assert_eq!(table.n_points, 18, "expected 18 altitude points");

        // Peak zonal should be ~100 m/s around 50 km (50_000 m)
        let w = table.wind_at(50_000.0, 0.0);
        assert!(
            (w.east - 100.0).abs() < 5.0,
            "expected zonal ~100 m/s at 50 km, got {}",
            w.east
        );

        // Sanity: above table should clamp to last entry (0.0 zonal at 150 km)
        let w_high = table.wind_at(200_000.0, 0.0);
        assert_abs_diff_eq!(w_high.east, 0.0, epsilon = 1e-10);

        // Cleanup temp dir if any (none used here)
        let _ = fs::metadata(path_str); // no-op, just suppress unused import warning
    }
}
