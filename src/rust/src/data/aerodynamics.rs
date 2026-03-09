//! Aerodynamic tables and interpolation.
//!
//! Body-axis (Ca, Cn) to stability-axis (Cx, Cz) conversion:
//!   Cx = Ca*cos(alpha) + Cn*sin(alpha)   (drag coefficient)
//!   Cz = -Ca*sin(alpha) + Cn*cos(alpha)  (lift coefficient)

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
