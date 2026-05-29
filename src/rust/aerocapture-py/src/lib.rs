//! PyO3 bindings for the aerocapture trajectory simulator.

use std::collections::HashSet;
use std::time::Duration;

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

mod batch;
mod config;
mod env;
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
///     sim_timeout_secs: Optional wall-clock timeout per simulation in seconds.
///         If the simulation exceeds this duration it is terminated and returns
///         a timeout result. Default None (no timeout).
///
/// Returns:
///     SimResult with trajectory, final_record, captured flag, and
///     convenience getters (energy, ecc, periapsis_alt, etc.).
#[pyfunction]
#[pyo3(signature = (toml_path, overrides=None, sim_timeout_secs=None))]
fn run(
    toml_path: &str,
    overrides: Option<&Bound<'_, PyDict>>,
    sim_timeout_secs: Option<f64>,
) -> PyResult<SimResult> {
    let overrides = extract_overrides(overrides)?;
    let wall_timeout = sim_timeout_secs.map(Duration::from_secs_f64);

    let (sim_input, sim_data) =
        config::load_and_override(std::path::Path::new(toml_path), &overrides)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let outputs =
        aerocapture::simulation::runner::run_for_api(&sim_input, &sim_data, false, wall_timeout)
            .map_err(|e| {
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
///     sim_timeout_secs: Optional wall-clock timeout per simulation in seconds.
///         If a simulation exceeds this duration it is terminated and returns
///         a timeout result. Default None (no timeout).
///
/// Returns:
///     BatchResults with final_records (N,52), captured (N,), and
///     optionally trajectories.
#[pyfunction]
#[pyo3(signature = (toml_path, overrides=None, include_trajectories=false, sim_timeout_secs=None))]
fn run_mc(
    toml_path: &str,
    overrides: Option<&Bound<'_, PyDict>>,
    include_trajectories: bool,
    sim_timeout_secs: Option<f64>,
) -> PyResult<BatchResults> {
    let overrides = extract_overrides(overrides)?;
    let wall_timeout = sim_timeout_secs.map(Duration::from_secs_f64);

    let (sim_input, sim_data) =
        config::load_and_override(std::path::Path::new(toml_path), &overrides)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let outputs = aerocapture::simulation::runner::run_for_api(
        &sim_input,
        &sim_data,
        include_trajectories,
        wall_timeout,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Simulation error: {}", e)))?;

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
///     sim_timeout_secs: Optional wall-clock timeout per simulation in seconds.
///         If a simulation exceeds this duration it is terminated and returns
///         a timeout result. Default None (no timeout).
///
/// Returns:
///     BatchResults with final_records (N,52), captured (N,), and
///     optionally trajectories.
#[pyfunction]
#[pyo3(signature = (toml_path, overrides_list, n_threads=None, include_trajectories=false, sim_timeout_secs=None))]
fn run_batch(
    toml_path: &str,
    overrides_list: &Bound<'_, PyList>,
    n_threads: Option<usize>,
    include_trajectories: bool,
    sim_timeout_secs: Option<f64>,
) -> PyResult<BatchResults> {
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

    let wall_timeout = sim_timeout_secs.map(Duration::from_secs_f64);

    let outputs = batch::run_batch(
        std::path::Path::new(toml_path),
        overrides_vec,
        n_threads,
        include_trajectories,
        wall_timeout,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok(BatchResults::from_outputs(outputs, include_trajectories))
}

/// Run simulations with pre-computed dispersion draws from Python.
///
/// Accepts a numpy array of shape (N, 26) where each row is a dispersion draw.
/// Bypasses internal draw generation entirely -- use this when you need
/// deterministic or specially-structured draws (e.g. SALib sensitivity matrices).
///
/// Args:
///     toml_path: Path to the TOML config file.
///     draws: numpy array of shape (N, 26), dtype float64. Each row is one draw.
///     overrides: Optional dict of "dotted.key" -> value overrides applied to all runs.
///     include_trajectories: If True, keep per-timestep trajectory data.
///     sim_timeout_secs: Optional wall-clock timeout per simulation in seconds.
///
/// Returns:
///     BatchResults with final_records (N, 52), dispersions (N, 26), captured (N,).
#[pyfunction]
#[pyo3(signature = (toml_path, draws, overrides=None, include_trajectories=false, sim_timeout_secs=None))]
fn run_with_draws(
    toml_path: &str,
    draws: PyReadonlyArray2<'_, f64>,
    overrides: Option<&Bound<'_, PyDict>>,
    include_trajectories: bool,
    sim_timeout_secs: Option<f64>,
) -> PyResult<BatchResults> {
    let shape = draws.shape();
    if shape.len() != 2 || shape[1] != 26 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "draws must have 26 columns (got shape {:?})",
            shape
        )));
    }
    let n_rows = shape[0];

    let draw_vec: Vec<[f64; 26]> = (0..n_rows)
        .map(|i| {
            let mut row = [0.0f64; 26];
            for j in 0..26 {
                row[j] = *draws.get([i, j]).unwrap();
            }
            row
        })
        .collect();

    let overrides = extract_overrides(overrides)?;
    let wall_timeout = sim_timeout_secs.map(Duration::from_secs_f64);

    let outputs = batch::run_with_external_draws(
        std::path::Path::new(toml_path),
        overrides,
        draw_vec,
        include_trajectories,
        wall_timeout,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    Ok(BatchResults::from_outputs(outputs, include_trajectories))
}

/// Load a v2 NN JSON in Rust and run a stateful forward pass on a single input.
///
/// Used exclusively by the Rust<>Python cross-language equivalence test
/// (Phase 0 integration gate). Applies `input_mask` when present; otherwise
/// passes the input through unchanged. Per-call `NnState` is fresh, so this
/// helper is stateless across calls (Phase 0 dense-only; Phase 1+ stateful
/// layer equivalence tests will need a state-carrying variant).
#[pyfunction]
fn nn_forward(json_path: String, input: Vec<f64>) -> PyResult<Vec<f64>> {
    use aerocapture::data::neural::NeuralNetModel;
    use aerocapture::data::nn_state::NnState;

    let model = NeuralNetModel::load(&json_path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let expected_len = match &model.input_mask {
        Some(mask) => mask.iter().copied().max().map(|m| m + 1).unwrap_or(0),
        None => model.layer_sizes[0],
    };
    if input.len() < expected_len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "nn_forward: input length {} < expected {}",
            input.len(),
            expected_len
        )));
    }

    let masked: Vec<f64> = match &model.input_mask {
        Some(mask) => mask.iter().map(|&i| input[i]).collect(),
        None => input,
    };
    let mut state = NnState::for_model(&model);
    Ok(model.forward(&mut state, &masked))
}

/// Load a v2 NN JSON and run a **stateful** forward pass over a sequence of inputs.
///
/// A single `NnState` is constructed once and threaded across all inputs, so
/// recurrent layers (GRU, LSTM, future SSM/attention) carry hidden state from
/// step `t-1` into step `t`. This is the cross-language equivalence driver for
/// stateful layers: Python runs its own multi-step forward with persistent
/// state, this helper runs Rust's multi-step forward with persistent state,
/// and the two output sequences must match to machine epsilon.
///
/// Args:
///     json_path: path to v2 JSON model.
///     inputs: list of input vectors, one per time step. Each vector must have
///             the same length (the model's layer-0 input size or mask span).
///
/// Returns: list of output vectors, one per time step (same length as inputs).
#[pyfunction]
fn nn_forward_sequence(json_path: String, inputs: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    use aerocapture::data::neural::NeuralNetModel;
    use aerocapture::data::nn_state::NnState;

    let model = NeuralNetModel::load(&json_path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let expected_len = match &model.input_mask {
        Some(mask) => mask.iter().copied().max().map(|m| m + 1).unwrap_or(0),
        None => model.layer_sizes[0],
    };
    for (t, input) in inputs.iter().enumerate() {
        if input.len() < expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "nn_forward_sequence: input[{}] length {} < expected {}",
                t,
                input.len(),
                expected_len
            )));
        }
    }

    let mut state = NnState::for_model(&model);
    let mut outputs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let masked: Vec<f64> = match &model.input_mask {
            Some(mask) => mask.iter().map(|&i| input[i]).collect(),
            None => input,
        };
        outputs.push(model.forward(&mut state, &masked));
    }
    Ok(outputs)
}

/// Construct a NeuralNetModel from flat PSO weights + v2 architecture (JSON string)
/// and write it as v2 JSON. All PSO NN output flows through this helper so the
/// Rust LayerWeights trait is the single source of truth for weight serialization.
///
/// The network output_size is validated to equal 2 (bank = atan2(out[0], out[1])).
///
/// Args:
///     flat: flat weight vector (length must equal sum of per-layer n_params).
///     architecture_json: JSON-serialized list of LayerSpec dicts, e.g.
///         '[{"type":"dense","input_size":16,"output_size":32,"activation":"tanh"},...]'.
///     path: output JSON file path.
///     input_mask: optional list of input indices (length == layer[0] input_size).
///     output_param: optional output parameterization string. One of:
///         "atan2_signed" (default), "acos_tanh", "scaled_pi", or "delta".
///         None defaults to "atan2_signed".
///     scaled_pi_n: optional half-range multiplier for the "scaled_pi" decoder
///         (bank = scaled_pi_n * pi * tanh(out[0])). None defaults to 1.0.
///     delta_max: optional per-step increment bound for the "delta" decoder
///         (bank = prev_realized + delta_max * tanh(out[0])). None defaults to 0.35.
#[pyfunction]
#[pyo3(signature = (flat, architecture_json, path, input_mask=None, output_param=None, scaled_pi_n=None, delta_max=None))]
fn flat_weights_to_json(
    flat: Vec<f64>,
    architecture_json: String,
    path: String,
    input_mask: Option<Vec<usize>>,
    output_param: Option<String>,
    scaled_pi_n: Option<f64>,
    delta_max: Option<f64>,
) -> PyResult<()> {
    use aerocapture::data::neural::{LayerSpec, NeuralNetModel, OutputParam};

    let specs: Vec<LayerSpec> = serde_json::from_str(&architecture_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "flat_weights_to_json: architecture_json parse error: {}",
            e
        ))
    })?;
    let output_param: OutputParam = match output_param.as_deref() {
        None | Some("atan2_signed") => OutputParam::default(),
        Some("acos_tanh") => OutputParam::AcosTanh,
        Some("scaled_pi") => OutputParam::ScaledPi,
        Some("delta") => OutputParam::Delta,
        Some(other) => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "output_param must be 'atan2_signed', 'acos_tanh', 'scaled_pi', or 'delta' (got {other:?})"
            )));
        }
    };
    let model = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &specs,
        input_mask,
        output_param,
        scaled_pi_n.unwrap_or(1.0),
        delta_max.unwrap_or(0.35),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    model
        .save_json(&path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(())
}

/// Collect supervised training data from a non-NN guidance scheme.
///
/// Runs the simulator with `collect_supervised = true` over each seed and
/// returns per-seed dicts preserving trajectory boundaries for downstream BPTT.
///
/// Args:
///     toml_path: Path to the TOML config file.
///     seeds: List of MC seeds to run (one simulation per seed, n_sims=1 each).
///     overrides: Optional dict of "dotted.key" -> value overrides applied to all runs.
///     scheme: Non-NN unsigned-magnitude guidance scheme to use as teacher.
///         One of: "ftc", "equilibrium_glide", "energy_controller", "pred_guid",
///         "fnpag", "piecewise_constant".
///     sim_timeout_secs: Optional wall-clock timeout per simulation in seconds.
///
/// Returns:
///     List of dicts (one per seed) with keys:
///       - "seed": int, the MC seed.
///       - "X": numpy.ndarray of shape (T, 31), per-tick NN input vectors.
///       - "y_signed": numpy.ndarray of shape (T,), final signed bank command
///         (radians) after thermal limiter, lateral, and command shaper.
///       - "prev_realized": numpy.ndarray of shape (T,), the previous-tick
///         pilot-realized bank (radians), consistent with the X row at each
///         step (captured before the per-tick telemetry update). Supervised
///         target for the `delta` bank decoder warm-start.
///       - "dv": float, total orbital-correction DV from the final record (m/s).
///       - "captured": bool, whether the trajectory captured.
#[pyfunction]
#[pyo3(signature = (toml_path, seeds, overrides=None, scheme="ftc".to_string(), sim_timeout_secs=None))]
fn collect_supervised(
    py: Python<'_>,
    toml_path: String,
    seeds: Vec<u64>,
    overrides: Option<&Bound<'_, PyDict>>,
    scheme: String,
    sim_timeout_secs: Option<f64>,
) -> PyResult<Py<PyList>> {
    use aerocapture::config::GuidanceType;

    let scheme_enum = match scheme.as_str() {
        "ftc" => GuidanceType::Ftc,
        "equilibrium_glide" => GuidanceType::EquilibriumGlide,
        "energy_controller" => GuidanceType::EnergyController,
        "pred_guid" => GuidanceType::PredGuid,
        "fnpag" => GuidanceType::Fnpag,
        "piecewise_constant" => GuidanceType::PiecewiseConstant,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "scheme must be a non-NN unsigned-magnitude scheme; got '{other}'"
            )));
        }
    };

    let base_overrides = extract_overrides(overrides)?;
    let wall_timeout = sim_timeout_secs.map(std::time::Duration::from_secs_f64);

    // Collected outside py.detach: (seed, supervised_trace, dv_total_m_s, captured).
    let mut per_seed: Vec<(u64, Vec<(Vec<f64>, f64, f64)>, f64, bool)> = Vec::with_capacity(seeds.len());

    py.detach(|| {
        for seed in &seeds {
            // Build per-seed overrides: force n_sims=1, set seed, force guidance type
            // BEFORE config load so the TOML-driven NN-file load (gated on
            // guidance.type == "neural_network") is skipped. Without this override,
            // running collect_supervised on a TOML that points `[data] neural_network`
            // at a not-yet-trained best_model.json would error at SimData construction.
            let mut seed_overrides = base_overrides.clone();
            seed_overrides.push((
                "simulation.n_sims".to_string(),
                config::OverrideValue::Int(1),
            ));
            seed_overrides.push((
                "monte_carlo.seed".to_string(),
                config::OverrideValue::Int(*seed as i64),
            ));
            seed_overrides.push((
                "guidance.type".to_string(),
                config::OverrideValue::Str(scheme.clone()),
            ));

            let (mut sim_input, sim_data) =
                config::load_and_override(std::path::Path::new(&toml_path), &seed_overrides)
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

            sim_input.collect_supervised = true;
            // guidance_type was already set by the TOML override above; this is belt-and-braces
            // in case load_and_override's override resolution diverges from from_toml's gating.
            sim_input.guidance_type = scheme_enum;

            let outputs = aerocapture::simulation::runner::run_for_api(
                &sim_input,
                &sim_data,
                false,
                wall_timeout,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Simulation error: {e}"))
            })?;

            // n_sims=1 contract: expect exactly one output. Erroring out on an
            // empty Vec keeps the (seed, trace, dv, captured) tuple downstream
            // from quietly carrying NaN -- which `_select_best_teacher_per_seed`
            // would convert into "captured=false, drop the seed" without any
            // signal that something went wrong.
            if outputs.is_empty() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "collect_supervised: run_for_api returned 0 outputs for seed {} (expected 1)",
                    seed
                )));
            }
            let mut combined_trace: Vec<(Vec<f64>, f64, f64)> = Vec::new();
            let mut dv = f64::NAN;
            let mut captured = false;
            for output in outputs {
                combined_trace.extend(output.supervised_trace);
                dv = output
                    .final_record
                    .get(41) // dv_total_m_s column
                    .copied()
                    .unwrap_or(f64::NAN);
                captured = output.captured;
            }
            per_seed.push((*seed, combined_trace, dv, captured));
        }
        Ok::<_, PyErr>(())
    })?;

    // PyDict / PyArray construction requires the GIL, so it happens after py.detach() returns.
    // NN input width is always 31 (the full FULL_MASK applied in tick.rs).
    const NN_INPUT_WIDTH: usize = 31;
    let result_list = PyList::empty(py);
    for (seed, supervised_trace, dv, captured) in per_seed {
        let n_steps = supervised_trace.len();
        let mut x_rows: Vec<Vec<f64>> = Vec::with_capacity(n_steps);
        let mut y_signed: Vec<f64> = Vec::with_capacity(n_steps);
        let mut prev_realized: Vec<f64> = Vec::with_capacity(n_steps);
        for (nn_input, bank, realized) in supervised_trace {
            x_rows.push(nn_input);
            y_signed.push(bank);
            prev_realized.push(realized);
        }
        // Preserve shape (0, 31) on empty traces so downstream code can rely on width.
        let x_array = if x_rows.is_empty() {
            numpy::PyArray2::<f64>::zeros(py, [0, NN_INPUT_WIDTH], false)
        } else {
            numpy::PyArray2::from_vec2(py, &x_rows).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build X array: {e}"))
            })?
        };
        let y_array = numpy::PyArray1::from_vec(py, y_signed);
        let pr_array = numpy::PyArray1::from_vec(py, prev_realized);

        let dict = PyDict::new(py);
        dict.set_item("seed", seed)?;
        dict.set_item("X", x_array)?;
        dict.set_item("y_signed", y_array)?;
        dict.set_item("prev_realized", pr_array)?;
        dict.set_item("dv", dv)?;
        dict.set_item("captured", captured)?;
        result_list.append(dict)?;
    }
    Ok(result_list.unbind())
}

/// Load and return a TOML config file as a plain Python dict.
///
/// Useful for inspecting or modifying config before passing overrides.
#[pyfunction]
fn load_config(py: Python<'_>, toml_path: &str) -> PyResult<Py<PyAny>> {
    let path = std::path::Path::new(toml_path);
    let content = std::fs::read_to_string(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot read '{}': {}", toml_path, e))
    })?;

    let value: toml::Value = toml::from_str::<toml::Table>(&content)
        .map(toml::Value::Table)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("TOML parse error: {}", e)))?;

    // Resolve base inheritance.
    let mut visited = HashSet::new();
    let resolved =
        aerocapture::config::resolve_toml_bases(value, path, &mut visited).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Base resolution error: {}", e))
        })?;

    toml_to_py(py, &resolved)
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
    m.add_class::<env::BatchedSimulation>()?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_mc, m)?)?;
    m.add_function(wrap_pyfunction!(run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(run_with_draws, m)?)?;
    m.add_function(wrap_pyfunction!(load_config, m)?)?;
    m.add_function(wrap_pyfunction!(nn_forward, m)?)?;
    m.add_function(wrap_pyfunction!(nn_forward_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(flat_weights_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(collect_supervised, m)?)?;
    Ok(())
}
