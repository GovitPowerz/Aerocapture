//! Python-facing result type wrapping a batch of `RunOutput`s.
//!
//! `BatchResults` provides NumPy views over the final-record matrix, the capture
//! flags, the dispersions and (when requested) the per-run trajectories. A single
//! run is `run_batch(toml, [{}])` row 0; there is no single-run type.

use aerocapture::RunOutput;
use aerocapture::data::dispersions::DISPERSION_DRAW_LEN;
use aerocapture::simulation::final_record::FINAL_RECORD_LEN;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Results from a batch of simulation runs.
#[pyclass]
pub struct BatchResults {
    outputs: Vec<RunOutput>,
}

#[pymethods]
impl BatchResults {
    /// All final records stacked as an (N, FINAL_RECORD_LEN) NumPy array.
    #[getter]
    fn final_records<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.outputs.len();
        let arr = PyArray2::<f64>::zeros(py, [n, FINAL_RECORD_LEN], false);
        // SAFETY: `arr` was just allocated here and is not aliased.
        let mut view = unsafe { arr.as_array_mut() };
        for (i, o) in self.outputs.iter().enumerate() {
            for (j, &v) in o.final_record.iter().enumerate() {
                view[[i, j]] = v;
            }
        }
        arr
    }

    /// Per-run capture flag as a NumPy bool array of length N.
    #[getter]
    fn captured<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let flags: Vec<bool> = self.outputs.iter().map(|o| o.captured).collect();
        PyArray1::from_vec(py, flags)
    }

    /// Per-run trajectories as a list of (T_i, 17) NumPy arrays.
    ///
    /// Only populated if `include_trajectories=True` was passed; otherwise
    /// returns a list of empty (0, 17) arrays.
    #[getter]
    fn trajectories<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        self.outputs
            .iter()
            .map(|o| {
                let rows: Vec<Vec<f64>> = o.trajectory.iter().map(|r| r.to_vec()).collect();
                if rows.is_empty() {
                    Ok(PyArray2::<f64>::zeros(py, [0, 17], false))
                } else {
                    PyArray2::from_vec2(py, &rows).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "trajectory array error: {e}"
                        ))
                    })
                }
            })
            .collect()
    }

    /// Dispersion draws as an (N, DISPERSION_DRAW_LEN) NumPy array — always populated.
    #[getter]
    fn dispersions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.outputs.len();
        let arr = PyArray2::<f64>::zeros(py, [n, DISPERSION_DRAW_LEN], false);
        // SAFETY: `arr` was just allocated here and is not aliased.
        let mut view = unsafe { arr.as_array_mut() };
        for (i, o) in self.outputs.iter().enumerate() {
            for (j, &v) in o.dispersions.iter().enumerate() {
                view[[i, j]] = v;
            }
        }
        arr
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
