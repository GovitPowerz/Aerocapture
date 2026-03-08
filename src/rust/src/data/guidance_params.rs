//! Guidance parameters loader.
//!
//! Reads from `guidage.*` data files (unit 109 in Fortran).
//!
//! Contains FTC predictor-corrector parameters for longitudinal
//! and lateral guidance during capture and exit phases.

use super::{DataError, parse_data_file};
use crate::config::MissionType;

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// Pdyn reference table entry: altitude, slope_a, slope_b
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct PdynTableEntry {
    pub altitude: f64, // km (stored as-is from file)
    pub coeff_a: f64,
    pub coeff_b: f64,
}

/// Equilibrium glide tunable parameters.
#[derive(Debug, Clone)]
pub struct EqGlideParams {
    pub k_hdot_scale: f64,       // radial velocity damping numerator
    pub v_ratio_threshold: f64,  // velocity ratio breakpoint
    pub velocity_bias_high: f64, // cos(bank) bias magnitude above v_ratio
    pub velocity_bias_low: f64,  // cos(bank) bias magnitude below v_ratio
    pub alt_bias_threshold: f64, // altitude (km) for lift-up bias
    pub cos_bank_min: f64,       // lower clamp on cos(bank)
    pub cos_bank_max: f64,       // upper clamp on cos(bank)
}

impl Default for EqGlideParams {
    fn default() -> Self {
        Self {
            k_hdot_scale: 0.3,
            v_ratio_threshold: 1.1,
            velocity_bias_high: 0.15,
            velocity_bias_low: 0.3,
            alt_bias_threshold: 40.0,
            cos_bank_min: -0.5,
            cos_bank_max: 0.95,
        }
    }
}

/// Energy controller tunable parameters.
#[derive(Debug, Clone)]
pub struct EnergyCtrlParams {
    pub gain: f64, // energy error gain (1/Pa)
    pub kp: f64,   // pressure proportional gain
    pub kd: f64,   // radial velocity damping gain
}

impl Default for EnergyCtrlParams {
    fn default() -> Self {
        Self {
            gain: 5e-7,
            kp: 1.0,
            kd: 0.5,
        }
    }
}

/// PredGuid (drag tracking) tunable parameters.
#[derive(Debug, Clone)]
pub struct PredGuidParams {
    pub k_drag_high: f64,    // gain when pdyn > threshold
    pub k_drag_low: f64,     // gain when pdyn <= threshold
    pub pdyn_threshold: f64, // switchover dynamic pressure (Pa)
}

impl Default for PredGuidParams {
    fn default() -> Self {
        Self {
            k_drag_high: 0.8,
            k_drag_low: 0.3,
            pdyn_threshold: 100.0,
        }
    }
}

/// FNPAG (fully numerical predictor-corrector) tunable parameters.
#[derive(Debug, Clone)]
pub struct FnpagParams {
    pub energy_tol: f64,       // convergence tolerance (J/kg)
    pub prediction_dt: f64,    // forward prediction timestep (s)
    pub bank_min_deg: f64,     // minimum bank angle (deg)
    pub bank_max_high_deg: f64, // max bank above 50 km (deg)
    pub bank_max_low_deg: f64, // max bank below 50 km (deg)
}

impl Default for FnpagParams {
    fn default() -> Self {
        Self {
            energy_tol: 1e4,
            prediction_dt: 2.0,
            bank_min_deg: 20.0,
            bank_max_high_deg: 140.0,
            bank_max_low_deg: 100.0,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GuidanceParams {
    // Capture phase — trajectory tracking
    pub capture_damping: f64,     // amorft — damping ratio
    pub capture_frequency: f64,   // pulsft — natural frequency (rad/s)
    pub capture_pdyn_margin: f64, // margmu(1) — Pdyn reference margin

    // Capture phase — altitude oscillation
    pub altitude_damping: f64,   // amorth
    pub altitude_frequency: f64, // pulsah (rad/s, converted from deg/s)

    // Exit phase
    pub exit_velocity_threshold: f64, // vsorti — radial velocity threshold (m/s)
    pub exit_pdyn_margin: f64,        // margmu(2)
    pub exit_altitude_threshold: f64, // altcst — constant accel altitude (m, from km)
    pub exit_radial_vel_gain: f64,    // gaindh — gain on radial velocity error (Pa/(m/s))
    pub exit_apoapsis_threshold: f64, // dzalim — apoapsis comparison threshold (m)

    // Lateral guidance
    pub corridor_slope: f64,     // coridx — inclination corridor slope (m/s)
    pub corridor_intercept: f64, // coridy — inclination corridor intercept (rad, from deg)
    pub max_reversals: i32,      // irevrs — max number of bank reversals

    // Security modes
    pub security_capture: i32, // iseccp — capture phase security mode
    pub security_exit: i32,    // isecex — exit phase security mode

    // Density estimation
    pub density_filter_gain: f64, // lambda — low-pass filter gain

    // Activation/inhibition thresholds
    pub longi_activation: f64, // pdacti — longitudinal guidance activation (J/kg or Pa)
    pub longi_inhibition: f64, // pdinib — longitudinal guidance inhibition
    pub lateral_activation: f64, // enrlat(1) — lateral guidance activation
    pub lateral_inhibition: f64, // enrlat(2) — lateral guidance inhibition
    pub pdyn_min: f64,         // pdymax — min Pdyn for tracking (Pa)

    // Pdyn = f(altitude) reference table
    pub pdyn_table: Vec<PdynTableEntry>,

    // Reference trajectory tables (from tables_energie_gains file)
    pub ref_trajectory: ReferenceTrajectory,

    // Per-scheme tunable parameters
    pub eq_glide: EqGlideParams,
    pub energy_ctrl: EnergyCtrlParams,
    pub pred_guid: PredGuidParams,
    pub fnpag: FnpagParams,
}

/// Reference trajectory tables loaded from tables_energie_gains file.
///
/// Matches Fortran common blocks tabnrj, reftab (unit 113).
/// When irefer=1, these are empty (reference trajectory is being generated).
/// When irefer=0, these are loaded from file and used by guicap.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct ReferenceTrajectory {
    pub n_points: usize,
    pub energy: Vec<f64>,        // nrjval — energy (J/kg)
    pub pressure: Vec<f64>,      // refpre — dynamic pressure (Pa)
    pub radial_vel: Vec<f64>,    // refhdt — radial velocity (m/s)
    pub altitude_rate: Vec<f64>, // refhtt — hpp (altitude derivative)
    pub inclination: Vec<f64>,   // refincli — inclination (rad)
    pub time: Vec<f64>,          // refdates — time (s)
    pub cos_bank: Vec<f64>,      // refcmu — cos(bank angle)
}

impl ReferenceTrajectory {
    /// Load reference trajectory from tables_energie_gains file.
    ///
    /// Matches Fortran lectci.f lines 419-444.
    /// File format: 7 columns per line (E-notation floats).
    /// Column order: energy/1e6, pdyneq, vitrad, hpp, xinccr, temsim, cos(gitref)
    pub fn load(path: &str) -> Result<Self, DataError> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return Ok(ReferenceTrajectory::default()),
        };

        let mut energy = Vec::new();
        let mut pressure = Vec::new();
        let mut radial_vel = Vec::new();
        let mut altitude_rate = Vec::new();
        let mut inclination = Vec::new();
        let mut time = Vec::new();
        let mut cos_bank = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let tokens: Vec<f64> = trimmed
                .split_whitespace()
                .filter_map(|t| {
                    let norm = t.replace('D', "E").replace('d', "e");
                    norm.parse::<f64>().ok()
                })
                .collect();
            if tokens.len() >= 7 {
                energy.push(tokens[0] * 1e6); // MJ/kg → J/kg
                pressure.push(tokens[1]);
                radial_vel.push(tokens[2]);
                altitude_rate.push(tokens[3]);
                inclination.push(tokens[4]);
                time.push(tokens[5]);
                cos_bank.push(tokens[6]);
            }
        }

        let n_points = energy.len();
        Ok(ReferenceTrajectory {
            n_points,
            energy,
            pressure,
            radial_vel,
            altitude_rate,
            inclination,
            time,
            cos_bank,
        })
    }

    /// 1D interpolation on the reference trajectory tables.
    ///
    /// Exact reimplementation of Fortran intrde.f.
    /// Uses iterative search starting from k=2 (1-based) / k=1 (0-based).
    /// Only finds descending brackets: tablxx(k) <= val < tablxx(k-1).
    /// Non-descending portions are skipped (k increments past them).
    pub fn interpolate(&self, energy_val: f64, table: &[f64]) -> f64 {
        if self.n_points == 0 {
            return 0.0;
        }
        if self.n_points == 1 {
            return table[0];
        }

        // Fortran intrde.f: k starts at kinter (=2, 1-based).
        // In 0-based indexing, this is k=1.
        // Each call resets kinter=2 in guicap, so we always start at k=1.
        let mut k: usize = 1;

        for _ in 0..self.n_points {
            let x_prev = self.energy[k - 1]; // tablxx(k-1) in Fortran
            let x_curr = self.energy[k]; // tablxx(k) in Fortran

            if energy_val >= x_curr && energy_val < x_prev {
                // Found descending bracket — linear interpolation
                let frac = (energy_val - x_prev) / (x_curr - x_prev);
                return table[k - 1] + frac * (table[k] - table[k - 1]);
            } else if energy_val < x_curr {
                // Query below current entry — move forward (toward smaller values)
                if k == self.n_points - 1 {
                    // Reached end of table — clamp to last entry
                    return table[self.n_points - 1];
                }
                k += 1;
            } else {
                // Query above previous entry — move backward (toward larger values)
                if k <= 1 {
                    // Reached start of table — clamp to first entry
                    return table[0];
                }
                k -= 1;
            }
        }

        // Fallback after max iterations (shouldn't normally reach here)
        table[0]
    }
}

impl GuidanceParams {
    #[allow(dead_code)]
    pub fn load(path: &str, mission_type: MissionType) -> Result<Self, DataError> {
        Self::load_with_ref(path, mission_type, "", false)
    }

    pub fn load_with_ref(
        path: &str,
        mission_type: MissionType,
        ref_path: &str,
        is_reference: bool,
    ) -> Result<Self, DataError> {
        let rows = parse_data_file(path)?;
        if rows.len() < 22 {
            return Err(DataError(format!(
                "Guidance file too short ({} rows, need >= 22): {}",
                rows.len(),
                path
            )));
        }

        let mut i = 0;
        let capture_damping = rows[i][0];
        i += 1; // amorft
        let capture_frequency = rows[i][0];
        i += 1; // pulsft
        let capture_pdyn_margin = rows[i][0];
        i += 1; // margmu(1)
        let altitude_damping = rows[i][0];
        i += 1; // amorth
        let altitude_frequency = rows[i][0] * DEG2RAD;
        i += 1; // pulsah (deg/s -> rad/s)
        let exit_velocity_threshold = rows[i][0];
        i += 1; // vsorti
        let exit_pdyn_margin = rows[i][0];
        i += 1; // margmu(2)
        let exit_altitude_threshold = rows[i][0] * 1e3;
        i += 1; // altcst (km -> m)
        let exit_radial_vel_gain = rows[i][0];
        i += 1; // gaindh
        let exit_apoapsis_threshold = rows[i][0];
        i += 1; // dzalim
        let corridor_slope = rows[i][0];
        i += 1; // coridx
        let corridor_intercept = rows[i][0] * DEG2RAD;
        i += 1; // coridy (deg -> rad)
        let max_reversals = rows[i][0] as i32;
        i += 1; // irevrs
        let security_capture = rows[i][0] as i32;
        i += 1; // iseccp
        let security_exit = rows[i][0] as i32;
        i += 1; // isecex
        let density_filter_gain = rows[i][0];
        i += 1; // lambda

        // Activation thresholds — for aerocapture, multiply by 1e6 (MJ/kg -> J/kg)
        let pdacti_raw = rows[i][0];
        i += 1;
        let pdinib_raw = rows[i][0];
        i += 1;
        let enrlat1_raw = rows[i][0];
        i += 1;
        let enrlat2_raw = rows[i][0];
        i += 1;

        let energy_scale = if mission_type == MissionType::Aerocapture {
            1e6
        } else {
            1.0
        };

        let longi_activation = pdacti_raw * energy_scale;
        let longi_inhibition = pdinib_raw * energy_scale;
        let lateral_activation = enrlat1_raw * energy_scale;
        let lateral_inhibition = enrlat2_raw * energy_scale;

        let pdyn_min = rows[i][0];
        i += 1; // pdymax (Pa)
        let n_pdyn = rows[i][0] as usize;
        i += 1; // number of Pdyn table points

        let mut pdyn_table = Vec::with_capacity(n_pdyn);
        for j in 0..n_pdyn {
            if i + j >= rows.len() {
                break;
            }
            let row = &rows[i + j];
            pdyn_table.push(PdynTableEntry {
                altitude: row[0],
                coeff_a: if row.len() > 1 { row[1] } else { 0.0 },
                coeff_b: if row.len() > 2 { row[2] } else { 0.0 },
            });
        }

        // Load reference trajectory if not in reference mode
        let ref_trajectory = if !is_reference && !ref_path.is_empty() {
            ReferenceTrajectory::load(ref_path)?
        } else {
            ReferenceTrajectory::default()
        };

        Ok(GuidanceParams {
            capture_damping,
            capture_frequency,
            capture_pdyn_margin,
            altitude_damping,
            altitude_frequency,
            exit_velocity_threshold,
            exit_pdyn_margin,
            exit_altitude_threshold,
            exit_radial_vel_gain,
            exit_apoapsis_threshold,
            corridor_slope,
            corridor_intercept,
            max_reversals,
            security_capture,
            security_exit,
            density_filter_gain,
            longi_activation,
            longi_inhibition,
            lateral_activation,
            lateral_inhibition,
            pdyn_min,
            pdyn_table,
            ref_trajectory,
            eq_glide: EqGlideParams::default(),
            energy_ctrl: EnergyCtrlParams::default(),
            pred_guid: PredGuidParams::default(),
            fnpag: FnpagParams::default(),
        })
    }
}
