//! PyO3 bindings for the aerocapture trajectory simulator.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

mod batch;
mod config;
mod results;

use config::OverrideValue;
use results::{BatchResults, SimResult};

/// Extract a Python dict of overrides into a Vec of (key, OverrideValue).
///
/// Type detection order matters: check `bool` before `int` because in Python
/// `isinstance(True, int)` is `True`.
fn extract_overrides(dict: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, OverrideValue)>> {
    let dict = match dict {
        Some(d) => d,
        None => return Ok(Vec::new()),
    };

    let mut result = Vec::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;

        // Check bool before int (Python bool is a subclass of int).
        let override_val = if value.is_instance_of::<PyBool>() {
            OverrideValue::Bool(value.extract()?)
        } else if value.is_instance_of::<PyFloat>() {
            OverrideValue::Float(value.extract()?)
        } else if value.is_instance_of::<PyInt>() {
            OverrideValue::Int(value.extract()?)
        } else if value.is_instance_of::<PyString>() {
            OverrideValue::Str(value.extract()?)
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported override type for key '{}': {}",
                key_str,
                value.get_type().name()?
            )));
        };

        result.push((key_str, override_val));
    }
    Ok(result)
}

/// Run a single simulation from a TOML config file.
///
/// Args:
///     toml_path: Path to the TOML config file.
///     overrides: Optional dict of "dotted.key" -> value overrides.
///
/// Returns:
///     SimResult with trajectory, final_record, captured flag, and
///     convenience getters (energy, ecc, periapsis_alt, etc.).
#[pyfunction]
#[pyo3(signature = (toml_path, overrides=None))]
fn run(toml_path: &str, overrides: Option<&Bound<'_, PyDict>>) -> PyResult<SimResult> {
    let toml_content = std::fs::read_to_string(toml_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot read '{}': {}", toml_path, e))
    })?;

    let overrides = extract_overrides(overrides)?;

    let (sim_input, sim_data) = config::load_and_override(&toml_content, &overrides)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let outputs =
        aerocapture::simulation::runner::run_for_api(&sim_input, &sim_data).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Simulation error: {}", e))
        })?;

    let output = outputs.into_iter().next().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("Simulation produced no results")
    })?;

    Ok(SimResult::from_output(output))
}

/// Run a Monte Carlo simulation returning all results.
///
/// Unlike `run()` which returns only the first result, this function
/// returns all n_sims results as a `BatchResults` object. Use this
/// for MC evaluations where you need the full distribution.
///
/// Args:
///     toml_path: Path to the TOML config file.
///     overrides: Optional dict of "dotted.key" -> value overrides.
///     include_trajectories: If True, keep per-timestep trajectory data
///         (default: False to save memory).
///
/// Returns:
///     BatchResults with final_records (N,52), captured (N,), and
///     optionally trajectories.
#[pyfunction]
#[pyo3(signature = (toml_path, overrides=None, include_trajectories=false))]
fn run_mc(
    toml_path: &str,
    overrides: Option<&Bound<'_, PyDict>>,
    include_trajectories: bool,
) -> PyResult<BatchResults> {
    let toml_content = std::fs::read_to_string(toml_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot read '{}': {}", toml_path, e))
    })?;

    let overrides = extract_overrides(overrides)?;

    let (sim_input, sim_data) = config::load_and_override(&toml_content, &overrides)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let outputs =
        aerocapture::simulation::runner::run_for_api(&sim_input, &sim_data).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Simulation error: {}", e))
        })?;

    Ok(BatchResults::from_outputs(outputs, include_trajectories))
}

/// Run a batch of simulations with per-run overrides, in parallel.
///
/// Args:
///     toml_path: Path to the base TOML config file.
///     overrides_list: List of dicts, one per run. Each dict maps
///         "dotted.key" -> value.
///     n_threads: Number of Rayon threads (default: number of CPUs).
///     include_trajectories: If True, keep per-timestep trajectory data
///         (default: False to save memory).
///
/// Returns:
///     BatchResults with final_records (N,52), captured (N,), and
///     optionally trajectories.
#[pyfunction]
#[pyo3(signature = (toml_path, overrides_list, n_threads=None, include_trajectories=false))]
fn run_batch(
    toml_path: &str,
    overrides_list: &Bound<'_, PyList>,
    n_threads: Option<usize>,
    include_trajectories: bool,
) -> PyResult<BatchResults> {
    let toml_content = std::fs::read_to_string(toml_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot read '{}': {}", toml_path, e))
    })?;

    let n_threads = n_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });

    // Extract each dict from the Python list.
    let mut overrides_vec = Vec::new();
    for item in overrides_list.iter() {
        let dict: &Bound<'_, PyDict> = item.cast()?;
        overrides_vec.push(extract_overrides(Some(dict))?);
    }

    let outputs = batch::run_batch(&toml_content, overrides_vec, n_threads)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok(BatchResults::from_outputs(outputs, include_trajectories))
}

/// Load and return a TOML config file as a plain Python dict.
///
/// Useful for inspecting or modifying config before passing overrides.
#[pyfunction]
fn load_config(py: Python<'_>, toml_path: &str) -> PyResult<Py<PyAny>> {
    let content = std::fs::read_to_string(toml_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot read '{}': {}", toml_path, e))
    })?;

    let value: toml::Value = toml::from_str::<toml::Table>(&content)
        .map(toml::Value::Table)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("TOML parse error: {}", e)))?;

    toml_to_py(py, &value)
}

/// Convert a TOML value tree into Python objects (dict, list, str, int, float, bool).
fn toml_to_py(py: Python<'_>, value: &toml::Value) -> PyResult<Py<PyAny>> {
    match value {
        toml::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        toml::Value::Integer(i) => Ok(i.into_pyobject(py)?.into_any().unbind()),
        toml::Value::Float(f) => Ok(f.into_pyobject(py)?.into_any().unbind()),
        toml::Value::Boolean(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        toml::Value::Datetime(dt) => {
            // Represent as string in Python.
            Ok(dt.to_string().into_pyobject(py)?.into_any().unbind())
        }
        toml::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(toml_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        toml::Value::Table(table) => {
            let dict = PyDict::new(py);
            for (k, v) in table {
                dict.set_item(k, toml_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Aerocapture trajectory simulator Python bindings.
#[pymodule]
fn aerocapture_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<SimResult>()?;
    m.add_class::<BatchResults>()?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_mc, m)?)?;
    m.add_function(wrap_pyfunction!(run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(load_config, m)?)?;
    Ok(())
}
