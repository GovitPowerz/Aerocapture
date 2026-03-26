//! Python-facing result types wrapping `RunOutput`.
//!
//! `SimResult` wraps a single run; `BatchResults` wraps a batch of runs
//! and provides efficient NumPy views over the final-record matrix.

use aerocapture::RunOutput;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

/// Result of a single simulation run.
#[pyclass]
pub struct SimResult {
    output: RunOutput,
}

#[pymethods]
impl SimResult {
    /// Per-timestep trajectory as an (N, 16) NumPy array.
    ///
    /// Columns: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg, heat_flux_kw_m2,
    ///           time_s, energy_mj_kg, pdyn_kpa, bank_angle_deg, inclination_deg,
    ///           g_load_g, nav_density_ratio, truth_density_kg_m3, heat_load_kj_m2].
    /// Empty if trajectories were not requested.
    #[getter]
    fn trajectory<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows: Vec<Vec<f64>> = self.output.trajectory.iter().map(|r| r.to_vec()).collect();
        if rows.is_empty() {
            PyArray2::from_vec2(py, &[]).unwrap()
        } else {
            PyArray2::from_vec2(py, &rows).unwrap()
        }
    }

    /// Full 52-element final record as a NumPy array.
    #[getter]
    fn final_record<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.output.final_record)
    }

    /// Whether the spacecraft was captured (bound orbit: ecc < 1 and energy < 0).
    #[getter]
    fn captured(&self) -> bool {
        self.output.captured
    }

    // ── Convenience getters for commonly used final-record indices ──

    /// Specific orbital energy (MJ/kg) — final_record[7].
    #[getter]
    fn energy(&self) -> f64 {
        self.output.final_record[7]
    }

    /// Eccentricity — final_record[9].
    #[getter]
    fn ecc(&self) -> f64 {
        self.output.final_record[9]
    }

    /// Periapsis altitude (km) — final_record[14].
    #[getter]
    fn periapsis_alt(&self) -> f64 {
        self.output.final_record[14]
    }

    /// Apoapsis altitude (km) — final_record[15].
    #[getter]
    fn apoapsis_alt(&self) -> f64 {
        self.output.final_record[15]
    }

    /// Total delta-V cost (m/s) — final_record[41].
    #[getter]
    fn delta_v(&self) -> f64 {
        self.output.final_record[41]
    }

    /// Periapsis error (km) — final_record[29].
    #[getter]
    fn peri_err(&self) -> f64 {
        self.output.final_record[29]
    }

    /// Apoapsis error (km) — final_record[30].
    #[getter]
    fn apo_err(&self) -> f64 {
        self.output.final_record[30]
    }

    /// Integrated heat load (kJ/m²) — from final_record[28]
    #[getter]
    fn integrated_heat_load(&self) -> f64 {
        self.output.final_record[28] * 1e3  // MJ/m² → kJ/m²
    }

    /// Dispersion draws as a 1D NumPy array (24 elements).
    #[getter]
    fn dispersions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.output.dispersions)
    }
}

impl SimResult {
    /// Construct from a `RunOutput`.
    pub fn from_output(output: RunOutput) -> Self {
        Self { output }
    }
}

/// Results from a batch of simulation runs.
#[pyclass]
pub struct BatchResults {
    outputs: Vec<RunOutput>,
}

#[pymethods]
impl BatchResults {
    /// All final records stacked as an (N, 52) NumPy array.
    #[getter]
    fn final_records<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows: Vec<Vec<f64>> = self
            .outputs
            .iter()
            .map(|o| o.final_record.to_vec())
            .collect();
        PyArray2::from_vec2(py, &rows).unwrap()
    }

    /// Per-run capture flag as a NumPy bool array of length N.
    #[getter]
    fn captured<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let flags: Vec<bool> = self.outputs.iter().map(|o| o.captured).collect();
        PyArray1::from_vec(py, flags)
    }

    /// Per-run trajectories as a list of (T_i, 16) NumPy arrays.
    ///
    /// Only populated if `include_trajectories=True` was passed; otherwise
    /// returns a list of empty (0, 16) arrays.
    #[getter]
    fn trajectories<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.outputs
            .iter()
            .map(|o| {
                let rows: Vec<Vec<f64>> = o.trajectory.iter().map(|r| r.to_vec()).collect();
                if rows.is_empty() {
                    PyArray2::from_vec2(py, &[]).unwrap()
                } else {
                    PyArray2::from_vec2(py, &rows).unwrap()
                }
            })
            .collect()
    }

    /// Dispersion draws as an (N, 24) NumPy array — always populated.
    #[getter]
    fn dispersions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows: Vec<Vec<f64>> = self
            .outputs
            .iter()
            .map(|o| o.dispersions.to_vec())
            .collect();
        PyArray2::from_vec2(py, &rows).unwrap()
    }

    /// Number of runs in the batch.
    fn __len__(&self) -> usize {
        self.outputs.len()
    }
}

impl BatchResults {
    /// Construct from a vector of `RunOutput`.
    pub fn from_outputs(outputs: Vec<RunOutput>, include_trajectories: bool) -> Self {
        let outputs = if include_trajectories {
            outputs
        } else {
            // Strip trajectories to save memory.
            outputs
                .into_iter()
                .map(|mut o| {
                    o.trajectory = Vec::new();
                    o
                })
                .collect()
        };
        Self { outputs }
    }
}
