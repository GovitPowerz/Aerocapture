//! Aerodynamic tables loader.
//!
//! Reads from `aerodynamique.*` data files (unit 104 in Fortran).
//!
//! Format: 3 header lines, then:
//!   alfaeq (deg) — equilibrium angle of attack
//!   nbmach — number of incidence points
//!   nbmach lines of: incidence(deg)  Ca  Cn  (axial and normal aero coefficients)
//!
//! The Fortran code converts from body-axis (Ca, Cn) to stability-axis (Cx, Cz):
//!   Cx = Ca*cos(alpha) + Cn*sin(alpha)   (drag coefficient)
//!   Cz = -Ca*sin(alpha) + Cn*cos(alpha)  (lift coefficient)

use super::{DataError, parse_data_file};

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct AeroTables {
    pub equilibrium_aoa: f64, // radians
    pub n_points: usize,
    pub incidence: Vec<f64>,  // radians
    pub cx: Vec<f64>,         // drag coefficient (stability axis)
    pub cz: Vec<f64>,         // lift coefficient (stability axis)
    pub nominal_cx: f64,      // Cx at equilibrium AoA
    pub nominal_cz: f64,      // Cz at equilibrium AoA
    pub nominal_finesse: f64, // Cz/Cx (L/D ratio)
    pub ballistic_coeff: f64, // 1/(m/(S*Cx_nom)), set after capsule loading
}

impl AeroTables {
    pub fn load(path: &str) -> Result<Self, DataError> {
        let rows = parse_data_file(path)?;
        if rows.len() < 3 {
            return Err(DataError(format!(
                "Aero file too short ({} rows, need >= 3): {}",
                rows.len(),
                path
            )));
        }

        let alfaeq = rows[0][0] * DEG2RAD;
        let n_points = rows[1][0] as usize;

        if rows.len() < 2 + n_points {
            return Err(DataError(format!(
                "Aero file: expected {} data rows, got {}: {}",
                n_points,
                rows.len() - 2,
                path
            )));
        }

        let mut incidence = Vec::with_capacity(n_points);
        let mut cx = Vec::with_capacity(n_points);
        let mut cz = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let row = &rows[2 + i];
            if row.len() < 3 {
                return Err(DataError(format!(
                    "Aero row {} too short ({} cols, need 3): {}",
                    i,
                    row.len(),
                    path
                )));
            }
            let alpha = row[0] * DEG2RAD;
            let ca = row[1];
            let cn = row[2];

            // Body-axis to stability-axis conversion (matches Fortran lectci.f)
            let cx_i = ca * alpha.cos() + cn * alpha.sin();
            let cz_i = -ca * alpha.sin() + cn * alpha.cos();

            incidence.push(alpha);
            cx.push(cx_i);
            cz.push(cz_i);
        }

        // Interpolate at equilibrium AoA to get nominal coefficients
        let nominal_cx = interpolate(&incidence, &cx, alfaeq);
        let nominal_cz = interpolate(&incidence, &cz, alfaeq);
        let nominal_finesse = if nominal_cx.abs() > 1e-30 {
            nominal_cz / nominal_cx
        } else {
            0.0
        };

        Ok(AeroTables {
            equilibrium_aoa: alfaeq,
            n_points,
            incidence,
            cx,
            cz,
            nominal_cx,
            nominal_cz,
            nominal_finesse,
            ballistic_coeff: 0.0, // Set later when capsule mass/area known
        })
    }

    /// Interpolate Cx at a given angle of attack
    pub fn interpolate_cx(&self, alpha: f64) -> f64 {
        interpolate(&self.incidence, &self.cx, alpha)
    }

    /// Interpolate Cz at a given angle of attack
    pub fn interpolate_cz(&self, alpha: f64) -> f64 {
        interpolate(&self.incidence, &self.cz, alpha)
    }
}

/// Linear interpolation in a table (matches Fortran intrmo.f)
pub fn interpolate(x_table: &[f64], y_table: &[f64], x: f64) -> f64 {
    let n = x_table.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 || x <= x_table[0] {
        return y_table[0];
    }
    if x >= x_table[n - 1] {
        return y_table[n - 1];
    }
    for i in 1..n {
        if x <= x_table[i] {
            let frac = (x - x_table[i - 1]) / (x_table[i] - x_table[i - 1]);
            return y_table[i - 1] + frac * (y_table[i] - y_table[i - 1]);
        }
    }
    y_table[n - 1]
}
