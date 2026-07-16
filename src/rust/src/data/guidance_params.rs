//! Guidance parameters.
//!
//! Contains FTC predictor-corrector parameters for longitudinal
//! and lateral guidance during capture and exit phases.

use super::DataError;
use crate::gnc::guidance::lateral::LateralParams;
use crate::gnc::guidance::thermal_limiter::ThermalLimiterParams;
use std::sync::Arc;

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
    // Apoapsis-radius convergence tolerance (m) since the corrector retargeted
    // from energy to osculating apoapsis. TOML key stays `energy_tol` for
    // config / GA-chromosome compatibility.
    pub energy_tol: f64,
    pub prediction_dt: f64,     // forward prediction timestep (s)
    pub bank_min_deg: f64,      // minimum bank angle (deg)
    pub bank_max_high_deg: f64, // max bank above 50 km (deg)
    pub bank_max_low_deg: f64,  // max bank below 50 km (deg)
    pub replan_period: f64,     // seconds between replans; bank held in between
}

impl Default for FnpagParams {
    fn default() -> Self {
        Self {
            energy_tol: 1e4,
            prediction_dt: 2.0,
            bank_min_deg: 20.0,
            bank_max_high_deg: 140.0,
            bank_max_low_deg: 100.0,
            replan_period: 2.0,
        }
    }
}

/// CPAG (convex predictor-corrector) guidance parameters.
///
/// Defaults are the Stage C0 validated set (docs/plans/2026-07-16-cpag-c0-
/// findings.md). `seg_dt`/`n_sub`/`max_iters*` are deliberately NOT GA genes
/// (the wall-time-blind GA would floor them — the FNPAG `replan_period`
/// lesson); the weights and trust knobs are the C2 chromosome.
#[derive(Debug, Clone)]
pub struct CpagParams {
    pub replan_period: f64, // seconds between replans; profile played back in between
    pub seg_dt: f64,        // ZOH bank-rate segment duration (s)
    pub n_sub: usize,       // RK4 substeps per segment
    pub horizon_max: f64,   // planning horizon cap (s)
    pub max_iters: usize,   // SCP iteration cap, cold first call
    pub max_iters_warm: usize, // SCP iteration cap, warm-started replans
    pub tol_apo: f64,       // apoapsis feasibility tolerance (m)
    pub alpha1: f64,        // control effort weight (per (rad/s)^2 s)
    pub alpha2: f64,        // terminal cos-inclination exact penalty
    pub alpha3: f64,        // terminal eps exact penalty (per MJ/kg)
    pub alpha5: f64,        // path-constraint slack penalty (per unit fraction)
    pub lambda_di: f64,     // intermediate inclination slack penalty
    pub di_node_fraction: f64, // inclination corridor on nodes >= this fraction
    pub di_deadband_deg: f64, // free mid-arc inclination error (deg)
    pub trust_init: f64,    // |du| trust box, in U_SCALE units
    pub trust_min: f64,
    pub trust_max: f64,
    pub sigma_max_deg: f64, // |sigma| node box (anti-winding)
    pub enforce_heat_flux: bool,
    pub enforce_g_load: bool,
    pub enforce_pdyn: bool, // default OFF: the mission nominal exceeds the config value
    pub enforce_inclination: bool,
}

impl Default for CpagParams {
    fn default() -> Self {
        Self {
            replan_period: 2.0,
            seg_dt: 8.0,
            n_sub: 4,
            horizon_max: 1200.0,
            max_iters: 20,
            max_iters_warm: 4,
            tol_apo: 1e4,
            alpha1: 1e-2,
            alpha2: 5.0,
            alpha3: 1000.0,
            alpha5: 1000.0,
            lambda_di: 1000.0,
            di_node_fraction: 0.8,
            di_deadband_deg: 0.5,
            trust_init: 4.0,
            trust_min: 0.02,
            trust_max: 8.0,
            sigma_max_deg: 180.0,
            enforce_heat_flux: true,
            enforce_g_load: true,
            enforce_pdyn: false,
            enforce_inclination: true,
        }
    }
}

/// Piecewise-constant bank angle guidance parameters.
/// `bank_angles.len()` segments uniformly distributed over the energy range.
/// Bank angles are signed (negative = implicit roll reversal).
#[derive(Debug, Clone)]
pub struct PiecewiseConstantParams {
    pub bank_angles: Vec<f64>, // radians, signed; length = n_segments
    pub energy_min: f64,       // J/kg (NOT MJ/kg)
    pub energy_max: f64,       // J/kg (NOT MJ/kg)
}

impl Default for PiecewiseConstantParams {
    fn default() -> Self {
        Self {
            bank_angles: vec![65.0_f64.to_radians(); 10],
            energy_min: -6.0e6,
            energy_max: 5.0e6,
        }
    }
}

/// Command shaping: acceleration-limited rate shaping in the dispatch layer.
/// When `None`, dispatch falls back to hard-clamp rate saturation.
#[derive(Debug, Clone, Copy)]
pub struct CommandShapingConfig {
    pub max_bank_acceleration: f64, // rad/s^2
}

/// NN guidance routing mode.
///
/// `FullNeural` (default, backward compatible): the NN emits a signed bank
/// angle via `atan2(out[0], out[1])` and bypasses the exit, lateral, and
/// thermal-limiter modules entirely.
///
/// `MagnitudeOnly`: the NN's signed bank is reduced to its absolute value and
/// fed into the unsigned-magnitude pipeline (exit guidance in phase 2,
/// thermal limiter, lateral guidance for sign selection). Lets the NN replace
/// only the capture-phase predictor-corrector while reusing the rest of the
/// FTC stack.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NeuralNetMode {
    #[default]
    FullNeural,
    MagnitudeOnly,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GuidanceParams {
    // Capture phase — trajectory tracking
    pub capture_damping: f64,     // damping ratio
    pub capture_frequency: f64,   // natural frequency (rad/s)
    pub capture_pdyn_margin: f64, // dynamic pressure reference margin

    // Capture phase — altitude oscillation
    pub altitude_damping: f64,   // altitude damping ratio
    pub altitude_frequency: f64, // altitude natural frequency (rad/s, converted from deg/s)

    // Exit phase
    pub exit_velocity_threshold: f64, // radial velocity threshold (m/s)
    pub exit_pdyn_margin: f64,        // exit dynamic pressure reference margin
    pub exit_altitude_threshold: f64, // constant-acceleration altitude (m, converted from km)
    pub exit_radial_vel_gain: f64,    // gain on radial velocity error (Pa/(m/s))

    // Lateral guidance
    pub lateral: LateralParams,

    // Security modes
    pub security_capture: i32, // capture phase security mode flag
    pub security_exit: i32,    // exit phase security mode flag

    // Density estimation
    pub density_filter_gain: f64, // low-pass filter gain for density estimation
    pub density_gain_max_delta: f64, // max per-step change in density_gain (rate limiter)

    // Activation/inhibition thresholds
    pub longi_activation: f64, // longitudinal guidance activation threshold (J/kg)
    pub longi_inhibition: f64, // longitudinal guidance inhibition threshold (J/kg)
    pub pdyn_min: f64,         // minimum dynamic pressure for tracking (Pa)

    // Analytical gain model (replaces pdyn altitude table)
    pub pressure_coeff_base: f64, // base pressure coefficient for exponential decay
    pub pressure_coeff_scale_height: f64, // exponential decay scale height (km)
    pub gain_fade_start_km: f64,  // altitude where gain fade begins (km)
    pub gain_fade_end_km: f64,    // altitude where gains reach zero (km)

    // Reference trajectory tables (from tables_energie_gains file)
    pub ref_trajectory: Arc<ReferenceTrajectory>,

    // Per-scheme tunable parameters
    pub eq_glide: EqGlideParams,
    pub energy_ctrl: EnergyCtrlParams,
    pub pred_guid: PredGuidParams,
    pub fnpag: FnpagParams,
    pub cpag: CpagParams,
    pub piecewise_constant: PiecewiseConstantParams,
    pub thermal_limiter: ThermalLimiterParams,
    pub command_shaping: Option<CommandShapingConfig>,

    // Neural network guidance routing mode (FullNeural | MagnitudeOnly)
    pub neural_mode: NeuralNetMode,

    // Eval-only state-ablation control: zero the NnState before every guidance
    // tick, making a stateful NN memoryless (paper R4/R5). Default false.
    pub nn_reset_state_every_tick: bool,
}

/// Reference trajectory tables loaded from the reference trajectory data file.
///
/// When `reference_trajectory = true` in config, these tables are empty
/// (the simulation is generating the reference trajectory).
/// When `reference_trajectory = false`, these are loaded from file and used by FTC guidance.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct ReferenceTrajectory {
    pub n_points: usize,
    pub energy: Vec<f64>,        // energy (J/kg)
    pub pressure: Vec<f64>,      // dynamic pressure (Pa)
    pub radial_vel: Vec<f64>,    // radial velocity (m/s)
    pub altitude_rate: Vec<f64>, // altitude rate (m/s)
    pub inclination: Vec<f64>,   // inclination (rad)
    pub time: Vec<f64>,          // time (s)
    pub cos_bank: Vec<f64>,      // cos(bank angle)
    /// True when `energy` is strictly descending (computed by the loader).
    /// Enables the O(log n) bracket lookup in `interpolate`; false (including
    /// the `Default`) falls back to the legacy walk, which also handles
    /// non-monotonic tables.
    pub monotonic_descending: bool,
}

impl ReferenceTrajectory {
    /// Load reference trajectory from the reference trajectory data file.
    ///
    /// File format: 7 whitespace-separated columns per line (E-notation floats).
    /// Column order: energy (MJ/kg), dynamic_pressure_equilibrium (Pa), velocity_radial (m/s),
    ///               altitude_rate (m/s), inclination_error (rad), time (s), cos(reference_bank_angle)
    pub fn load(path: &str) -> Result<Self, DataError> {
        // A missing/unreadable file is a hard error, NOT an empty table: the
        // ref-tracking schemes silently interpolate 0.0 everywhere on an empty
        // table (the silent-wiring failure class). The legitimate no-file cases
        // (reference-generating run, no path configured) construct `default()`
        // explicitly upstream and never call `load`.
        let content = std::fs::read_to_string(path).map_err(|e| {
            DataError(format!(
                "Cannot read reference trajectory '{}': {}",
                path, e
            ))
        })?;

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

        // Units sanity: the file contract is MJ/kg in column 0 (converted to
        // J/kg above). Orbital specific energies are tens of MJ/kg, so a
        // post-conversion magnitude beyond 1e9 J/kg means the file is almost
        // certainly in J/kg already (the historical train.py writer bug) and
        // every interpolation query would collapse into the table's tail.
        const MAX_SANE_ENERGY_J_PER_KG: f64 = 1e9;
        let worst = energy.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
        if worst > MAX_SANE_ENERGY_J_PER_KG {
            return Err(DataError(format!(
                "reference trajectory '{path}': energy column magnitude up to {worst:.3e} J/kg after MJ/kg->J/kg conversion; \
                 the file's first column must be in MJ/kg (this one looks like J/kg)"
            )));
        }

        let n_points = energy.len();
        let monotonic_descending = energy.windows(2).all(|w| w[0] > w[1]);
        Ok(ReferenceTrajectory {
            n_points,
            energy,
            pressure,
            radial_vel,
            altitude_rate,
            inclination,
            time,
            cos_bank,
            monotonic_descending,
        })
    }

    /// 1D interpolation on the reference trajectory tables.
    ///
    /// Uses iterative bracket search starting from k=1 (0-based).
    /// Only finds descending brackets: energy[k] <= val < energy[k-1].
    /// Non-descending portions are skipped (k increments past them).
    pub fn interpolate(&self, energy_val: f64, table: &[f64]) -> f64 {
        if self.n_points == 0 {
            return 0.0;
        }
        if self.n_points == 1 {
            return table[0];
        }

        // Fast path for strictly descending tables (every table the loader
        // accepts in practice): binary search for the first k with
        // energy[k] <= energy_val, which is exactly the bracket the legacy
        // walk below converges to -- same bracket, same interpolation
        // expression, bit-identical output.
        if self.monotonic_descending {
            let k = self.energy.partition_point(|&e| e > energy_val);
            if k == 0 {
                return table[0];
            }
            if k == self.n_points {
                return table[self.n_points - 1];
            }
            let x_prev = self.energy[k - 1];
            let x_curr = self.energy[k];
            let frac = (energy_val - x_prev) / (x_curr - x_prev);
            return table[k - 1] + frac * (table[k] - table[k - 1]);
        }

        // k starts at 1 (0-based), equivalent to the original 1-based k=2 starting index.
        // Each call always resets to k=1 (no persistent state across calls).
        let mut k: usize = 1;

        for _ in 0..self.n_points {
            let x_prev = self.energy[k - 1]; // upper bound of bracket
            let x_curr = self.energy[k]; // lower bound of bracket (energy is descending)

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

impl Default for GuidanceParams {
    fn default() -> Self {
        Self {
            capture_damping: 0.0,
            capture_frequency: 0.0,
            capture_pdyn_margin: 0.0,
            altitude_damping: 0.0,
            altitude_frequency: 0.0,
            exit_velocity_threshold: 0.0,
            exit_pdyn_margin: 0.0,
            exit_altitude_threshold: 0.0,
            exit_radial_vel_gain: 0.0,
            lateral: LateralParams::default(),
            security_capture: 0,
            security_exit: 0,
            density_filter_gain: 0.0,
            density_gain_max_delta: 0.1,
            longi_activation: 0.0,
            longi_inhibition: 0.0,
            pdyn_min: 0.0,
            pressure_coeff_base: -134.4,
            pressure_coeff_scale_height: 6.9,
            gain_fade_start_km: 80.0,
            gain_fade_end_km: 100.0,
            ref_trajectory: Arc::new(ReferenceTrajectory::default()),
            eq_glide: EqGlideParams::default(),
            energy_ctrl: EnergyCtrlParams::default(),
            pred_guid: PredGuidParams::default(),
            fnpag: FnpagParams::default(),
            cpag: CpagParams::default(),
            piecewise_constant: PiecewiseConstantParams::default(),
            thermal_limiter: ThermalLimiterParams::default(),
            command_shaping: None,
            neural_mode: NeuralNetMode::default(),
            nn_reset_state_every_tick: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_ref_file(dir: &tempfile::TempDir, rows: &[[f64; 7]]) -> String {
        let path = dir.path().join("ref.dat");
        let mut f = std::fs::File::create(&path).unwrap();
        for r in rows {
            writeln!(
                f,
                "  {:.16E}  {:.16E}  {:.16E}  {:.16E}  {:.16E}  {:.16E}  {:.16E}",
                r[0], r[1], r[2], r[3], r[4], r[5], r[6]
            )
            .unwrap();
        }
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn load_converts_mj_per_kg_to_j_per_kg() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_ref_file(
            &dir,
            &[
                [4.9, 0.02, -1066.0, -1066.0, 0.87, 0.0, 0.43],
                [0.0, 800.0, -50.0, -50.0, 0.87, 200.0, 0.30],
                [-5.3, 10.0, 100.0, 100.0, 0.87, 400.0, 0.50],
            ],
        );

        let rt = ReferenceTrajectory::load(&path).unwrap();

        assert_eq!(rt.n_points, 3);
        assert!((rt.energy[0] - 4.9e6).abs() < 1.0);
        assert!((rt.energy[2] + 5.3e6).abs() < 1.0);
    }

    #[test]
    fn load_rejects_missing_file() {
        // A typo'd path must be a hard error, not a silent empty table
        // (ref-tracking schemes interpolate 0.0 everywhere on an empty table).
        let err = ReferenceTrajectory::load("/nonexistent/definitely_not_here.dat");
        assert!(
            err.is_err(),
            "loader must reject a missing reference trajectory file"
        );
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("definitely_not_here.dat"),
            "error must name the path, got: {msg}"
        );
    }

    #[test]
    fn interpolate_fast_path_bit_identical_to_legacy_walk() {
        // Strictly descending table -> loader sets monotonic_descending = true.
        // The binary-search fast path must return bit-identical values to the
        // legacy walk for every query class: above table, at knots, interior,
        // below table.
        let energy: Vec<f64> = (0..100).map(|i| 5.0e6 - 1.1e5 * i as f64).collect();
        let table: Vec<f64> = (0..100).map(|i| (i as f64 * 0.37).sin() * 800.0).collect();
        let mk = |fast: bool| ReferenceTrajectory {
            n_points: energy.len(),
            energy: energy.clone(),
            pressure: table.clone(),
            radial_vel: table.clone(),
            altitude_rate: table.clone(),
            inclination: table.clone(),
            time: table.clone(),
            cos_bank: table.clone(),
            monotonic_descending: fast,
        };
        let fast = mk(true);
        let slow = mk(false);
        let mut queries: Vec<f64> = (-60..60).map(|i| i as f64 * 1.07e5).collect();
        queries.extend(energy.iter().copied()); // exact knot hits
        queries.push(6.0e6); // above table
        queries.push(-7.0e6); // below table
        for q in queries {
            let a = fast.interpolate(q, &fast.pressure);
            let b = slow.interpolate(q, &slow.pressure);
            assert_eq!(a, b, "fast/legacy mismatch at energy_val={q}");
        }
    }

    #[test]
    fn load_rejects_j_per_kg_energy_column() {
        // A file whose first column is in J/kg (the train.py writer bug shipped
        // exactly this) would, after the loader's MJ/kg -> J/kg conversion, give
        // an energy axis 1e6x too large -- every runtime interpolation query
        // collapses into the table's near-zero tail. Hard-error instead.
        let dir = tempfile::tempdir().unwrap();
        let path = write_ref_file(
            &dir,
            &[
                [4.9e6, 0.02, -1066.0, -1066.0, 0.87, 0.0, 0.43],
                [-5.3e6, 10.0, 100.0, 100.0, 0.87, 400.0, 0.50],
            ],
        );

        let err = ReferenceTrajectory::load(&path);

        assert!(
            err.is_err(),
            "loader must reject a J/kg-scaled energy column, got {:?}",
            err.map(|rt| rt.energy[0])
        );
    }
}
