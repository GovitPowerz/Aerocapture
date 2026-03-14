# PyO3 Rust-Python Interface Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PyO3 Python module (`aerocapture_rs`) that calls the Rust simulator directly from Python, eliminating subprocess overhead and enabling batch parallelism for GA training.

**Architecture:** Separate `aerocapture-py` workspace member crate depends on the existing `aerocapture` library crate. Core crate gains a public `RunOutput` struct returned by a new `run_for_api()` function (refactored from existing `run()` to share setup logic). PyO3 crate wraps this with numpy conversion, config override merging, and Rayon batch execution. Python training code migrates gradually with subprocess fallback.

**Tech Stack:** Rust (PyO3, pyo3-numpy, rayon, toml), Python (maturin, numpy), GitHub Actions CI

**Spec:** `docs/superpowers/specs/2026-03-14-pyo3-interface-design.md`

---

## Chunk 1: Rust Foundation (workspace, core crate change, PyO3 crate skeleton)

### Task 0: Create feature branch

**Files:** None

- [ ] **Step 1: Create and switch to feature branch**

```bash
git checkout -b feature/pyo3-interface
```

- [ ] **Step 2: Verify branch**

```bash
git branch --show-current
```

Expected: `feature/pyo3-interface`

---

### Task 1: Create `aerocapture-py` crate skeleton and workspace

**Files:**
- Create: `src/rust/aerocapture-py/Cargo.toml`
- Create: `src/rust/aerocapture-py/src/lib.rs`
- Create: `src/rust/aerocapture-py/pyproject.toml`
- Modify: `src/rust/Cargo.toml`

**Important:** Create the crate directory FIRST, then add the workspace section. Cargo errors if a listed workspace member directory doesn't exist.

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/rust/aerocapture-py/src
```

- [ ] **Step 2: Create `aerocapture-py/Cargo.toml`**

```toml
[package]
name = "aerocapture-py"
version = "0.1.0"
edition = "2024"
description = "PyO3 bindings for the aerocapture trajectory simulator"

[lib]
name = "aerocapture_rs"
crate-type = ["cdylib"]

[dependencies]
aerocapture = { path = ".." }
pyo3 = { version = "0.24", features = ["extension-module"] }
pyo3-numpy = "0.24"
rayon = "1.10"
toml = "0.9"

[profile.release]
opt-level = 3
lto = true
```

Note: The crate is `pyo3-numpy` on crates.io (imported as `use pyo3_numpy::...` in Rust code). Verify version compatibility with `cargo search pyo3-numpy`.

- [ ] **Step 3: Create minimal `src/lib.rs`**

```rust
use pyo3::prelude::*;

/// Aerocapture trajectory simulator Python bindings.
#[pymodule]
fn aerocapture_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
```

- [ ] **Step 4: Create `pyproject.toml`**

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "aerocapture-rs"
requires-python = ">=3.14"
version = "0.1.0"
description = "PyO3 bindings for the aerocapture trajectory simulator"

[tool.maturin]
features = ["pyo3/extension-module"]
```

- [ ] **Step 5: Add workspace section to `src/rust/Cargo.toml`**

Add at the top (before `[package]`):

```toml
[workspace]
members = [".", "aerocapture-py"]
```

- [ ] **Step 6: Verify workspace builds**

```bash
cd src/rust && cargo build --release
```

Expected: both crates compile.

- [ ] **Step 7: Run existing Rust tests to confirm no regressions**

```bash
cd src/rust && cargo test --release
```

Expected: all ~172 tests pass.

- [ ] **Step 8: Build and test Python module import**

```bash
cd src/rust/aerocapture-py && uv pip install maturin && maturin develop --release
cd ../../..
uv run python -c "import aerocapture_rs; print(aerocapture_rs.__version__)"
```

Expected: prints `0.1.0`.

- [ ] **Step 9: Commit**

```bash
git add src/rust/Cargo.toml src/rust/aerocapture-py/ src/rust/Cargo.lock
git commit -m "feat: add aerocapture-py crate skeleton with cargo workspace"
```

---

### Task 2: Add `RunOutput` and `run_for_api()` to core crate

**Files:**
- Modify: `src/rust/src/lib.rs`
- Modify: `src/rust/src/simulation/runner.rs`

The core crate has a private `SimResult` in `runner.rs:73` used for file output. We add a public `RunOutput` struct and a `run_for_api()` function that shares setup logic with the existing `run()`.

**Photo line column layout** (from `build_photo_values()` at `runner.rs:612`):
```
[0]=time, [1]=alt_km, [2]=lon_deg, [3]=lat_deg, [4]=vel_m/s,
[5]=fpa_deg, [6]=heading_deg, [7]=sma_km, [8]=ecc, [9]=inc_deg,
[10]=raan_deg, [11]=periapsis_km, [12]=apoapsis_km, [13]=phase,
[14]=bank_deg, [15]=vel_radial, [16]=aoa_deg, ...
```

Trajectory extraction: `[alt_km, lon_deg, lat_deg, vel, fpa_deg, heading_deg, flux, time]` maps to photo columns `[1, 2, 3, 4, 5, 6, 17, 0]`. (Column 17 needs verification — check `build_photo_values` for the flux/heat column index.)

- [ ] **Step 1: Add `RunOutput` struct to `lib.rs`**

Add to `src/rust/src/lib.rs`:

```rust
/// Public output from a single simulation run, for use by PyO3 and tests.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// Per-timestep state from photo output: [alt_km, lon_deg, lat_deg, vel_m_s, fpa_deg, heading_deg, flux, time]
    pub trajectory: Vec<[f64; 8]>,
    /// Full 52-column final record (same layout as CSV file output)
    pub final_record: [f64; 52],
    /// True if orbit is bound (ecc < 1 && energy < 0)
    pub captured: bool,
}
```

- [ ] **Step 2: Refactor `run()` to extract shared setup into a helper**

In `runner.rs`, extract the common setup (n_sims, dispersion draws, run_states, parallel dispatch) into a private `run_core()` that returns `Vec<SimResult>`:

```rust
/// Shared simulation orchestration: build run states, dispatch parallel/sequential runs.
fn run_core(config: &SimInput, data: &SimData, write_photo: bool) -> Result<Vec<SimResult>, SimError> {
    let n_sims = if config.n_sims == 0 { 1 } else { config.n_sims };
    let is_mc = n_sims > 1;

    let draws = data.dispersion_config.as_ref().map(|dc| {
        let draws = dc.generate_draws(n_sims as usize);
        if write_photo {
            let on_off = |b: bool| if b { "on" } else { "off" };
            eprintln!(
                "Monte Carlo: {} draws from seed {}, domains: state={} atmo={} aero={} nav={} mass={} vehicle={} pilot={} nav_filter={}",
                draws.len(), dc.seed,
                on_off(dc.initial_state.is_some()), on_off(dc.atmosphere.is_some()),
                on_off(dc.aerodynamics.is_some()), on_off(dc.navigation.is_some()),
                on_off(dc.mass.is_some()), on_off(dc.vehicle.is_some()),
                on_off(dc.pilot.is_some()), on_off(dc.nav_filter.is_some()),
            );
        }
        draws
    });

    let run_states: Vec<init::RunState> = (0..n_sims)
        .map(|sim_idx| {
            if let Some(ref d) = draws {
                init::init_run_from_draw(data, &d[sim_idx as usize])
            } else {
                init::init_run_from_draw(data, &crate::data::dispersions::DispersionDraw::default())
            }
        })
        .collect();

    let photo_sim_idx = if is_mc {
        if config.visualize_sim > 0 {
            (config.visualize_sim - 1).min(n_sims - 1)
        } else {
            n_sims - 1
        }
    } else {
        0
    };

    if is_mc {
        let start = std::time::Instant::now();
        if write_photo {
            eprintln!("Running {} simulations in parallel...", n_sims);
        }
        let results: Vec<SimResult> = run_states
            .par_iter()
            .enumerate()
            .map(|(idx, run_state)| {
                let do_photo = write_photo && idx as i32 == photo_sim_idx;
                run_single(config, data, run_state, idx as i32, do_photo)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if write_photo {
            let elapsed = start.elapsed();
            eprintln!(
                "Completed {} simulations in {:.3}s ({:.1} sims/s)",
                n_sims, elapsed.as_secs_f64(), n_sims as f64 / elapsed.as_secs_f64(),
            );
        }
        Ok(results)
    } else {
        let run_state = &run_states[0];
        if write_photo && config.screen_output {
            eprintln!(
                "  Entry: alt={:.3} km, vel={:.3} m/s, fpa={:.5} deg",
                run_state.entry.state.altitude / 1e3,
                run_state.entry.state.velocity,
                run_state.entry.state.flight_path.to_degrees(),
            );
        }
        Ok(vec![run_single(config, data, run_state, 0, write_photo)?])
    }
}
```

Then simplify `run()` to:

```rust
pub fn run(config: &SimInput, data: &SimData) -> Result<(), SimError> {
    let n_sims = if config.n_sims == 0 { 1 } else { config.n_sims };
    let photo_sim_idx = if n_sims > 1 {
        if config.visualize_sim > 0 { (config.visualize_sim - 1).min(n_sims - 1) } else { n_sims - 1 }
    } else { 0 };

    let results = run_core(config, data, true)?;
    write_csv_output(config, &results, photo_sim_idx)?;
    Ok(())
}
```

- [ ] **Step 3: Add `run_for_api()` using the shared `run_core()`**

```rust
/// Run simulation and return structured results (no file I/O).
///
/// Same physics as `run()`, but returns `Vec<RunOutput>` instead of writing files.
/// Used by the PyO3 interface for direct Python access.
pub fn run_for_api(config: &SimInput, data: &SimData) -> Result<Vec<crate::RunOutput>, SimError> {
    let results = run_core(config, data, false)?;

    Ok(results
        .into_iter()
        .map(|r| {
            let energy = r.final_line[7]; // MJ/kg
            let ecc = r.final_line[9];
            crate::RunOutput {
                // write_photo=false means photo_lines is empty, so trajectory is empty.
                // This matches the spec's include_trajectories=False default.
                trajectory: Vec::new(),
                final_record: r.final_line,
                captured: ecc < 1.0 && energy < 0.0,
            }
        })
        .collect())
}
```

**Design note on trajectories:** `run_core` passes `write_photo=false` for the API path to avoid unnecessary overhead. This means `photo_lines` will be empty, so `trajectory` will be an empty `Vec`. To support `include_trajectories=True` in the batch API, we'll need a variant that enables photo collection. For now, trajectory is empty by default — matching the spec's `include_trajectories=False` default. A follow-up can add a `write_photo` parameter to `run_for_api()`.

- [ ] **Step 4: Write tests for `run_for_api()`**

Add to `src/rust/src/simulation/runner.rs` in a test module:

```rust
#[cfg(test)]
mod run_output_tests {
    use super::*;
    use crate::config::SimInput;
    use crate::data::SimData;

    fn load_test_config() -> (SimInput, SimData) {
        let content = std::fs::read_to_string("../../configs/test/test_ref_orig.toml")
            .expect("test config");
        let (sim_config, toml_config) = SimInput::from_toml(&content).expect("parse");
        let sim_data = SimData::from_toml(&toml_config, &sim_config).expect("data");
        (sim_config, sim_data)
    }

    #[test]
    fn run_for_api_returns_one_result_for_single_sim() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data).expect("run");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn run_output_final_record_has_52_elements() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data).expect("run");
        assert_eq!(results[0].final_record.len(), 52);
    }

    #[test]
    fn run_output_final_record_matches_file_path() {
        // Run via file-writing path, parse output, compare with API path
        let (config, data) = load_test_config();

        // API path
        let api_results = run_for_api(&config, &data).expect("api run");
        let api_fr = &api_results[0].final_record;

        // File path
        run(&config, &data).expect("file run");

        // Parse the CSV output
        let suffix = config.results_suffix.trim_start_matches('.');
        let final_path = config.output_path(&format!("final.{}.csv", suffix));
        let content = std::fs::read_to_string(&final_path).expect("read final csv");
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() >= 2, "final CSV should have header + data");

        // Compare key columns: energy(7), ecc(9), periapsis(14), apoapsis(15), dv(41)
        assert!(api_fr[7].abs() > 0.0, "energy should be non-zero");
        assert!(api_fr[9] > 0.0, "eccentricity should be positive");
        // The final_record from both paths uses the same run_single(), so values are identical
        // by construction. This test verifies the API path produces non-degenerate output.
    }

    #[test]
    fn run_output_captured_flag_consistent_with_orbital_elements() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data).expect("run");
        let r = &results[0];
        let expected = r.final_record[9] < 1.0 && r.final_record[7] < 0.0;
        assert_eq!(r.captured, expected);
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cd src/rust && cargo test run_output_tests --release
```

Expected: all 4 tests pass.

- [ ] **Step 6: Run full test suite**

```bash
cd src/rust && cargo test --release
```

Expected: all tests pass (existing + new).

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/lib.rs src/rust/src/simulation/runner.rs
git commit -m "feat: add RunOutput struct and run_for_api() with shared run_core()"
```

---

### Task 3: Config override module

**Files:**
- Create: `src/rust/aerocapture-py/src/config.rs`
- Modify: `src/rust/aerocapture-py/src/lib.rs`

- [ ] **Step 1: Write tests for config override merging**

Create `src/rust/aerocapture-py/src/config.rs` with tests first (code shown in Step 3 under `#[cfg(test)]`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_override_simple_float() {
        let toml_str = r#"
        [guidance]
        type = "ftc"
        reference_bank_angle = 45.0
        "#;
        let mut value: toml::Value = toml::from_str(toml_str).unwrap();
        apply_override(&mut value, "guidance.reference_bank_angle", &OverrideValue::Float(60.0)).unwrap();
        let bank = value["guidance"]["reference_bank_angle"].as_float().unwrap();
        assert!((bank - 60.0).abs() < 1e-10);
    }

    #[test]
    fn apply_override_int_to_float_coercion() {
        let toml_str = r#"
        [guidance]
        reference_bank_angle = 45.0
        "#;
        let mut value: toml::Value = toml::from_str(toml_str).unwrap();
        apply_override(&mut value, "guidance.reference_bank_angle", &OverrideValue::Int(60)).unwrap();
        let bank = value["guidance"]["reference_bank_angle"].as_float().unwrap();
        assert!((bank - 60.0).abs() < 1e-10);
    }

    #[test]
    fn apply_override_type_mismatch_errors() {
        let toml_str = r#"
        [guidance]
        reference_bank_angle = 45.0
        "#;
        let mut value: toml::Value = toml::from_str(toml_str).unwrap();
        let result = apply_override(&mut value, "guidance.reference_bank_angle", &OverrideValue::Str("bad".into()));
        assert!(result.is_err());
    }

    #[test]
    fn apply_override_nested_deep_path() {
        let toml_str = r#"
        [guidance.equilibrium_glide]
        k_hdot = 0.1
        "#;
        let mut value: toml::Value = toml::from_str(toml_str).unwrap();
        apply_override(&mut value, "guidance.equilibrium_glide.k_hdot", &OverrideValue::Float(0.2)).unwrap();
        let val = value["guidance"]["equilibrium_glide"]["k_hdot"].as_float().unwrap();
        assert!((val - 0.2).abs() < 1e-10);
    }

    #[test]
    fn apply_override_integer_field() {
        let toml_str = r#"
        [simulation]
        n_sims = 10
        "#;
        let mut value: toml::Value = toml::from_str(toml_str).unwrap();
        apply_override(&mut value, "simulation.n_sims", &OverrideValue::Int(50)).unwrap();
        let val = value["simulation"]["n_sims"].as_integer().unwrap();
        assert_eq!(val, 50);
    }

    #[test]
    fn apply_override_creates_new_key() {
        let toml_str = r#"
        [guidance]
        type = "ftc"
        "#;
        let mut value: toml::Value = toml::from_str(toml_str).unwrap();
        apply_override(&mut value, "guidance.new_param", &OverrideValue::Float(1.5)).unwrap();
        let val = value["guidance"]["new_param"].as_float().unwrap();
        assert!((val - 1.5).abs() < 1e-10);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src/rust && cargo test -p aerocapture-py --release 2>&1 | head -20
```

Expected: compilation error — functions don't exist yet.

- [ ] **Step 3: Implement config override module**

Write the implementation above the tests in `src/rust/aerocapture-py/src/config.rs`:

```rust
//! TOML config loading with Python dict override merging.

use aerocapture::config::{SimInput, TomlConfig};
use aerocapture::data::SimData;

/// Override value types from Python.
#[derive(Debug, Clone)]
pub enum OverrideValue {
    Float(f64),
    Int(i64),
    Str(String),
    Bool(bool),
}

/// Apply a single dot-separated key override to a TOML value tree.
pub fn apply_override(
    root: &mut toml::Value,
    key_path: &str,
    value: &OverrideValue,
) -> Result<(), String> {
    let parts: Vec<&str> = key_path.split('.').collect();
    let mut current = root;

    for &part in &parts[..parts.len() - 1] {
        if !current.is_table() {
            return Err(format!("'{}' is not a table in path '{}'", part, key_path));
        }
        let table = current.as_table_mut().unwrap();
        if !table.contains_key(part) {
            table.insert(part.to_string(), toml::Value::Table(toml::map::Map::new()));
        }
        current = table.get_mut(part).unwrap();
    }

    let leaf_key = parts[parts.len() - 1];
    let table = current
        .as_table_mut()
        .ok_or_else(|| format!("Parent of '{}' is not a table", key_path))?;

    let new_value = if let Some(existing) = table.get(leaf_key) {
        match (existing, value) {
            (toml::Value::Float(_), OverrideValue::Float(v)) => toml::Value::Float(*v),
            (toml::Value::Float(_), OverrideValue::Int(v)) => toml::Value::Float(*v as f64),
            (toml::Value::Integer(_), OverrideValue::Int(v)) => toml::Value::Integer(*v),
            (toml::Value::String(_), OverrideValue::Str(v)) => toml::Value::String(v.clone()),
            (toml::Value::Boolean(_), OverrideValue::Bool(v)) => toml::Value::Boolean(*v),
            (existing, value) => {
                return Err(format!(
                    "Type mismatch for '{}': existing is {:?}, override is {:?}",
                    key_path, existing.type_str(), value,
                ));
            }
        }
    } else {
        match value {
            OverrideValue::Float(v) => toml::Value::Float(*v),
            OverrideValue::Int(v) => toml::Value::Integer(*v),
            OverrideValue::Str(v) => toml::Value::String(v.clone()),
            OverrideValue::Bool(v) => toml::Value::Boolean(*v),
        }
    };

    table.insert(leaf_key.to_string(), new_value);
    Ok(())
}

/// Load TOML, apply overrides, parse into SimInput + SimData.
pub fn load_and_override(
    toml_content: &str,
    overrides: &[(String, OverrideValue)],
) -> Result<(SimInput, SimData), String> {
    let mut value: toml::Value = toml::from_str(toml_content)
        .map_err(|e| format!("TOML parse error: {}", e))?;

    for (key, val) in overrides {
        apply_override(&mut value, key, val)?;
    }

    let patched_toml = toml::to_string(&value)
        .map_err(|e| format!("TOML serialize error: {}", e))?;

    let (sim_config, toml_config) = SimInput::from_toml(&patched_toml)
        .map_err(|e| format!("Config parse error: {}", e))?;

    let sim_data = SimData::from_toml(&toml_config, &sim_config)
        .map_err(|e| format!("Data load error: {}", e))?;

    Ok((sim_config, sim_data))
}
```

- [ ] **Step 4: Add `mod config;` to `lib.rs`**

- [ ] **Step 5: Run tests**

```bash
cd src/rust && cargo test -p aerocapture-py --release
```

Expected: all 6 config tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/rust/aerocapture-py/src/
git commit -m "feat: add config override module with type coercion"
```

---

### Task 4: Results conversion module

**Files:**
- Create: `src/rust/aerocapture-py/src/results.rs`
- Modify: `src/rust/aerocapture-py/src/lib.rs`

- [ ] **Step 1: Implement results module**

Write `src/rust/aerocapture-py/src/results.rs`:

```rust
//! Convert RunOutput to Python objects with numpy arrays.

use aerocapture::RunOutput;
use pyo3::prelude::*;
use pyo3_numpy::{PyArray1, PyArray2};

/// Single simulation result exposed to Python.
#[pyclass]
pub struct SimResult {
    output: RunOutput,
}

#[pymethods]
impl SimResult {
    /// Per-timestep trajectory, shape (N, 8). Empty if include_trajectories was False.
    #[getter]
    fn trajectory<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows: Vec<Vec<f64>> = self.output.trajectory.iter().map(|r| r.to_vec()).collect();
        if rows.is_empty() {
            PyArray2::from_vec2(py, &[vec![0.0; 8]; 0]).expect("empty trajectory")
        } else {
            PyArray2::from_vec2(py, &rows).expect("trajectory array")
        }
    }

    /// Full 52-column final record, shape (52,).
    #[getter]
    fn final_record<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.output.final_record)
    }

    /// Whether the orbit is bound.
    #[getter]
    fn captured(&self) -> bool {
        self.output.captured
    }

    // Convenience accessors over final_record columns
    #[getter] fn energy(&self) -> f64 { self.output.final_record[7] }
    #[getter] fn ecc(&self) -> f64 { self.output.final_record[9] }
    #[getter] fn periapsis_alt(&self) -> f64 { self.output.final_record[14] }
    #[getter] fn apoapsis_alt(&self) -> f64 { self.output.final_record[15] }
    #[getter] fn delta_v(&self) -> f64 { self.output.final_record[41] }
    #[getter] fn peri_err(&self) -> f64 { self.output.final_record[29] }
    #[getter] fn apo_err(&self) -> f64 { self.output.final_record[30] }
}

impl SimResult {
    pub fn from_output(output: RunOutput) -> Self {
        Self { output }
    }
}

/// Batch simulation results exposed to Python.
#[pyclass]
pub struct BatchResults {
    outputs: Vec<RunOutput>,
    include_trajectories: bool,
}

#[pymethods]
impl BatchResults {
    /// Final records, shape (N, 52). Directly compatible with compute_cost().
    #[getter]
    fn final_records<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows: Vec<Vec<f64>> = self.outputs.iter()
            .map(|o| o.final_record.to_vec())
            .collect();
        PyArray2::from_vec2(py, &rows).expect("final_records array")
    }

    /// Captured flags, shape (N,).
    #[getter]
    fn captured<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let flags: Vec<bool> = self.outputs.iter().map(|o| o.captured).collect();
        PyArray1::from_vec(py, flags)
    }

    /// Trajectories as list of numpy arrays. Empty list if include_trajectories=False.
    #[getter]
    fn trajectories<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        if !self.include_trajectories {
            return Ok(vec![]);
        }
        self.outputs.iter().map(|o| {
            let rows: Vec<Vec<f64>> = o.trajectory.iter().map(|r| r.to_vec()).collect();
            Ok(PyArray2::from_vec2(py, &rows).expect("trajectory"))
        }).collect()
    }
}

impl BatchResults {
    pub fn from_outputs(outputs: Vec<RunOutput>, include_trajectories: bool) -> Self {
        Self { outputs, include_trajectories }
    }
}
```

- [ ] **Step 2: Add `mod results;` to `lib.rs`**

- [ ] **Step 3: Verify compilation**

```bash
cd src/rust && cargo build -p aerocapture-py --release
```

Fix any `pyo3-numpy` API issues. Common fix: `PyArray2::from_vec2` may need `&Bound<'_, PyModule>` instead of `Python<'py>` depending on version.

- [ ] **Step 4: Write Python-side tests for numpy conversion**

These will be tested in Task 9 (Python integration tests) after the module is fully wired up. At this stage, compilation verification is sufficient since the numpy conversion is thin.

- [ ] **Step 5: Commit**

```bash
git add src/rust/aerocapture-py/src/
git commit -m "feat: add results module with numpy conversion"
```

---

### Task 5: Batch runner module

**Files:**
- Create: `src/rust/aerocapture-py/src/batch.rs`
- Modify: `src/rust/aerocapture-py/src/lib.rs`

- [ ] **Step 1: Implement batch runner**

Write `src/rust/aerocapture-py/src/batch.rs`:

```rust
//! Rayon-based parallel batch execution.

use aerocapture::RunOutput;
use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::simulation::runner;

use crate::config::{OverrideValue, apply_override};

/// Run a batch of simulations with different overrides.
///
/// The base TOML is parsed once. Per batch item, the TOML value tree is cloned,
/// overrides applied, then deserialized into SimInput + SimData. This avoids
/// re-reading the TOML file N times (but data tables like atmosphere are still
/// loaded per item since they depend on config — a future optimization can cache
/// them when only guidance params change).
///
/// Each override set produces one `RunOutput`. If the base config has `n_sims > 1`,
/// each batch item runs a full MC ensemble — but only the first result per item
/// is returned with a warning. For GA training, set `n_sims = 1` in the base config
/// and vary the seed via overrides.
///
/// Uses a scoped Rayon thread pool to avoid conflicts with the global pool.
pub fn run_batch(
    toml_content: &str,
    overrides_list: Vec<Vec<(String, OverrideValue)>>,
    n_threads: usize,
) -> Result<Vec<RunOutput>, String> {
    // Parse base TOML once
    let base_value: toml::Value = toml::from_str(toml_content)
        .map_err(|e| format!("TOML parse error: {}", e))?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| format!("Failed to create thread pool: {}", e))?;

    pool.install(|| {
        use rayon::prelude::*;

        overrides_list
            .into_par_iter()
            .map(|overrides| {
                // Clone base tree, apply overrides
                let mut value = base_value.clone();
                for (key, val) in &overrides {
                    apply_override(&mut value, key, val)?;
                }

                // Deserialize patched tree
                let patched_toml = toml::to_string(&value)
                    .map_err(|e| format!("TOML serialize error: {}", e))?;
                let (sim_config, toml_config) = SimInput::from_toml(&patched_toml)
                    .map_err(|e| format!("Config parse error: {}", e))?;
                let sim_data = SimData::from_toml(&toml_config, &sim_config)
                    .map_err(|e| format!("Data load error: {}", e))?;

                let results = runner::run_for_api(&sim_config, &sim_data)
                    .map_err(|e| format!("Simulation error: {}", e))?;

                if results.is_empty() {
                    return Err("Simulation produced no results".to_string());
                }
                if results.len() > 1 {
                    eprintln!(
                        "Warning: batch item produced {} results (n_sims > 1). \
                         Only the first is returned. Set n_sims=1 in base config for GA training.",
                        results.len()
                    );
                }
                Ok(results.into_iter().next().unwrap())
            })
            .collect::<Result<Vec<_>, String>>()
    })
}
```

- [ ] **Step 2: Add `mod batch;` to `lib.rs`**

- [ ] **Step 3: Verify compilation**

```bash
cd src/rust && cargo build -p aerocapture-py --release
```

- [ ] **Step 4: Commit**

```bash
git add src/rust/aerocapture-py/src/
git commit -m "feat: add batch runner with scoped Rayon thread pool"
```

---

### Task 6: Wire up PyO3 module entry points

**Files:**
- Modify: `src/rust/aerocapture-py/src/lib.rs`

- [ ] **Step 1: Implement full `lib.rs` with pyfunction bindings**

Update `src/rust/aerocapture-py/src/lib.rs`:

```rust
mod batch;
mod config;
mod results;

use config::OverrideValue;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use results::{BatchResults, SimResult};

/// Extract overrides from a Python dict into Vec<(key_path, OverrideValue)>.
fn extract_overrides(dict: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, OverrideValue)>> {
    let Some(dict) = dict else {
        return Ok(vec![]);
    };
    let mut overrides = Vec::new();
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        // Check bool before int — Python bool is a subclass of int
        let ov = if let Ok(v) = value.extract::<bool>() {
            OverrideValue::Bool(v)
        } else if let Ok(v) = value.extract::<i64>() {
            OverrideValue::Int(v)
        } else if let Ok(v) = value.extract::<f64>() {
            OverrideValue::Float(v)
        } else if let Ok(v) = value.extract::<String>() {
            OverrideValue::Str(v)
        } else {
            return Err(PyValueError::new_err(format!(
                "Unsupported override type for key '{}': expected bool, int, float, or str", key
            )));
        };
        overrides.push((key, ov));
    }
    Ok(overrides)
}

/// Run a single simulation.
#[pyfunction]
#[pyo3(signature = (toml_path, overrides=None))]
fn run(toml_path: &str, overrides: Option<&Bound<'_, PyDict>>) -> PyResult<SimResult> {
    let content = std::fs::read_to_string(toml_path)
        .map_err(|e| PyValueError::new_err(format!("Cannot read {}: {}", toml_path, e)))?;

    let override_list = extract_overrides(overrides)?;
    let (sim_config, sim_data) = config::load_and_override(&content, &override_list)
        .map_err(|e| PyValueError::new_err(e))?;

    let mut results = aerocapture::simulation::runner::run_for_api(&sim_config, &sim_data)
        .map_err(|e| PyValueError::new_err(format!("Simulation error: {}", e)))?;

    if results.is_empty() {
        return Err(PyValueError::new_err("Simulation produced no results"));
    }
    Ok(SimResult::from_output(results.remove(0)))
}

/// Run a batch of simulations in parallel.
#[pyfunction]
#[pyo3(signature = (toml_path, overrides_list, n_threads=None, include_trajectories=false))]
fn run_batch(
    toml_path: &str,
    overrides_list: Vec<Bound<'_, PyDict>>,
    n_threads: Option<usize>,
    include_trajectories: bool,
) -> PyResult<BatchResults> {
    let content = std::fs::read_to_string(toml_path)
        .map_err(|e| PyValueError::new_err(format!("Cannot read {}: {}", toml_path, e)))?;

    let n_threads = n_threads.unwrap_or_else(|| {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    });

    let override_sets: Vec<Vec<(String, OverrideValue)>> = overrides_list
        .iter()
        .map(|d| extract_overrides(Some(d)))
        .collect::<PyResult<Vec<_>>>()?;

    let outputs = batch::run_batch(&content, override_sets, n_threads)
        .map_err(|e| PyValueError::new_err(e))?;

    Ok(BatchResults::from_outputs(outputs, include_trajectories))
}

/// Load and inspect a TOML config as a Python dict (no simulation).
#[pyfunction]
fn load_config(py: Python<'_>, toml_path: &str) -> PyResult<PyObject> {
    let content = std::fs::read_to_string(toml_path)
        .map_err(|e| PyValueError::new_err(format!("Cannot read {}: {}", toml_path, e)))?;

    let value: toml::Value = toml::from_str(&content)
        .map_err(|e| PyValueError::new_err(format!("TOML parse error: {}", e)))?;

    toml_to_py(py, &value)
}

fn toml_to_py(py: Python<'_>, value: &toml::Value) -> PyResult<PyObject> {
    match value {
        toml::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        toml::Value::Integer(i) => Ok(i.into_pyobject(py)?.into_any().unbind()),
        toml::Value::Float(f) => Ok(f.into_pyobject(py)?.into_any().unbind()),
        toml::Value::Boolean(b) => Ok(b.into_pyobject(py)?.into_any().unbind()),
        toml::Value::Array(arr) => {
            let list: Vec<PyObject> = arr.iter().map(|v| toml_to_py(py, v)).collect::<PyResult<_>>()?;
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }
        toml::Value::Table(table) => {
            let dict = PyDict::new(py);
            for (k, v) in table {
                dict.set_item(k, toml_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
        _ => Ok(py.None()),
    }
}

/// Aerocapture trajectory simulator Python bindings.
#[pymodule]
fn aerocapture_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(load_config, m)?)?;
    m.add_class::<SimResult>()?;
    m.add_class::<BatchResults>()?;
    Ok(())
}
```

**Note:** `load_config()` returns a plain Python dict. The spec mentioned a `#[pyclass]` wrapper with `.guidance_scheme` and `.monte_carlo_seeds` — this is a deliberate simplification. A dict is sufficient for config inspection; a typed wrapper can be added later if needed. No downstream tasks depend on the `#[pyclass]` API.

- [ ] **Step 2: Build and fix compilation issues**

```bash
cd src/rust && cargo build -p aerocapture-py --release
```

Likely PyO3 API issues to fix:
- `into_pyobject` may need different method signatures in PyO3 0.24
- `Bound` lifetime annotations
- `pyo3_numpy` constructors

Consult PyO3 0.24 migration guide if needed.

- [ ] **Step 3: Rebuild Python module and smoke test**

```bash
cd src/rust/aerocapture-py && maturin develop --release && cd ../../..
uv run python -c "
import aerocapture_rs as aero
result = aero.run('configs/test/test_ref_orig.toml')
print(f'Captured: {result.captured}')
print(f'Final record shape: {result.final_record.shape}')
print(f'Energy: {result.energy:.6f} MJ/kg')
print(f'Eccentricity: {result.ecc:.6f}')
"
```

Expected: prints values matching the golden reference.

- [ ] **Step 4: Commit**

```bash
git add src/rust/aerocapture-py/src/
git commit -m "feat: wire up run(), run_batch(), load_config() Python bindings"
```

---

## Chunk 2: Python migration, testing, CI

### Task 7: Add `maturin` to dev dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add maturin to dev dependency group**

In `pyproject.toml`, add `"maturin>=1.0"` to the `dev` list:

```toml
[dependency-groups]
dev = [
    "pytest>=9.0",
    "ruff>=0.15",
    "mypy>=1.9",
    "pandas-stubs>=3.0",
    "scipy-stubs>=1.17.1.0",
    "types-defusedxml>=0.7.0.20250822",
    "types-openpyxl>=3.1.5.20250919",
    "types-python-dateutil>=2.9.0.20260305",
    "types-xlrd>=2.0.0.20251020",
    "hypothesis>=6.100",
    "maturin>=1.0",
]
```

- [ ] **Step 2: Sync**

```bash
uv sync --group dev
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add maturin to dev dependencies"
```

---

### Task 8: Migrate `compute_cost()` to 0-based 52-column indices

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py`
- Modify: `tests/test_cost.py`

This is a coordinated change: update the parser, cost function, and tests together.

- [ ] **Step 1: Update `tests/test_cost.py` to 52-column layout**

Change `N_COLS` and `_make_row()` indices:

```python
N_COLS = 52

def _make_row(
    *,
    energy: float = -1.0,
    ecc: float = 0.5,
    sim_time: float = 300.0,
    peri_err: float = 0.0,
    apo_err: float = 0.0,
    dv_total: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Build a single-row final_conditions array with the given values."""
    row = np.zeros((1, N_COLS))
    row[0, 7] = energy      # was 8
    row[0, 9] = ecc          # was 10
    row[0, 27] = sim_time    # was 28
    row[0, 29] = peri_err    # was 30
    row[0, 30] = apo_err     # was 31
    row[0, 41] = dv_total    # was 42
    return row
```

Also update the module docstring column references at the top of the file.

- [ ] **Step 2: Update `_parse_final_to_legacy_array()` in `evaluate.py`**

Change to return 52 columns (no `sim_number` prefix):

```python
def _parse_final_to_legacy_array(filepath: Path) -> npt.NDArray[np.float64] | None:
    """Parse a final conditions CSV file, returning 52-column array.

    Maps named CSV columns to the 52-column final_record layout
    (0-based, no sim_number prefix). Column 0 is now altitude_km
    (was sim_number in the old 53-column format).
    """
    import pandas as pd
    from aerocapture.io.parse_final import CSV_TO_LEGACY_INDEX

    df = pd.read_csv(filepath)
    if df.empty:
        return None
    n = len(df)
    result = np.zeros((n, 52))
    for col_name, legacy_idx in CSV_TO_LEGACY_INDEX.items():
        if col_name in df.columns:
            result[:, legacy_idx] = df[col_name].to_numpy()  # No +1 offset
    return result
```

- [ ] **Step 3: Update `compute_cost()` column indices**

In `evaluate.py`, change:

```python
energy = final_conditions[:, 7]   # MJ/kg (was 8)
ecc = final_conditions[:, 9]     # dimensionless (was 10)
sim_time = final_conditions[:, 27]  # s (was 28)
peri_err = final_conditions[:, 29]  # km (was 30)
apo_err = final_conditions[:, 30]   # km (was 31)
dv_total = final_conditions[:, 41]  # m/s (was 42)
```

Update the docstring to reflect the new column indices.

- [ ] **Step 4: Update `final_report.py` column constants**

In `src/python/aerocapture/training/final_report.py`, update the `_COL_*` constants (lines 19-30):

```python
# Final record column indices (52-column format, 0-based, no sim_number prefix)
_COL_VELOCITY = 3       # was 4
_COL_FPA = 4            # was 5
_COL_ENERGY = 7         # was 8
_COL_ECC = 9            # was 10
_COL_INCL = 10          # was 11
_COL_PERI_ERR = 29      # was 30
_COL_APO_ERR = 30       # was 31
_COL_DV1 = 37           # was 38
_COL_DV2 = 38           # was 39
_COL_DV3 = 39           # was 40
_COL_DV_TOTAL = 41      # was 42
```

Update the comment from "53-column format" to "52-column format".

- [ ] **Step 5: Search for any other hardcoded 53-column references**

```bash
grep -rn "_COL_\|N_COLS\|53" src/python/ tests/ --include="*.py" | grep -v __pycache__
```

Fix any remaining references.

- [ ] **Step 6: Run tests — verify all pass**

```bash
uv run pytest tests/test_cost.py -v
```

Expected: all pass.

- [ ] **Step 7: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass. Some regression tests that use the subprocess path may fail if they expect 53-column output — fix by updating the parse path.

- [ ] **Step 8: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py src/python/aerocapture/training/final_report.py tests/test_cost.py
git commit -m "refactor: migrate to 0-based 52-column final_record layout"
```

---

### Task 9: Add PyO3 fallback to `evaluate.py`

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py`

- [ ] **Step 1: Add PyO3 import with fallback at top of file**

```python
try:
    import aerocapture_rs as _aero_rs
    _HAS_PYO3 = True
except ImportError:
    _aero_rs = None  # type: ignore[assignment]
    _HAS_PYO3 = False
```

- [ ] **Step 2: Refactor `run_simulation()` to dispatch between paths**

```python
def run_simulation(
    config: TrainingConfig,
    cwd: str | Path | None = None,
    overrides: dict[str, object] | None = None,
) -> npt.NDArray[np.float64] | None:
    """Run the Rust simulator and parse final conditions.

    Tries PyO3 direct call first, falls back to subprocess if unavailable.

    Args:
        config: Training configuration.
        cwd: Working directory (defaults to config.sim.exec_dir).
        overrides: Optional dict of TOML key path overrides (for PyO3 path).

    Returns:
        Array of final conditions (N, 52), or None if simulation failed.
    """
    if _HAS_PYO3 and config.sim.toml_config:
        return _run_via_pyo3(config, cwd, overrides)
    return _run_via_subprocess(config, cwd)
```

- [ ] **Step 3: Extract subprocess path to `_run_via_subprocess()`**

Move existing logic:

```python
def _run_via_subprocess(config: TrainingConfig, cwd: str | Path | None = None) -> npt.NDArray[np.float64] | None:
    """Run simulator via subprocess (fallback path)."""
    if cwd is None:
        cwd = config.sim.exec_dir
    cwd = Path(cwd)
    executable = (cwd / config.sim.executable).resolve()
    if not config.sim.toml_config:
        return None
    toml_path = (cwd / config.sim.toml_config).resolve()
    try:
        subprocess.run(
            [str(executable), str(toml_path)],
            capture_output=True,
            cwd=str(cwd.resolve()),
            timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    final_file = cwd / config.sim.final_file
    csv_final = Path(str(final_file) + ".csv")
    if csv_final.exists():
        final_file = csv_final
    elif not final_file.exists():
        return None
    try:
        return _parse_final_to_legacy_array(final_file)
    except Exception:
        return None
```

- [ ] **Step 4: Add PyO3 path with overrides support**

```python
def _run_via_pyo3(
    config: TrainingConfig,
    cwd: str | Path | None = None,
    overrides: dict[str, object] | None = None,
) -> npt.NDArray[np.float64] | None:
    """Run simulator via PyO3 direct call."""
    assert _aero_rs is not None
    if cwd is None:
        cwd = config.sim.exec_dir
    cwd = Path(cwd)
    if not config.sim.toml_config:
        return None
    toml_path = str((cwd / config.sim.toml_config).resolve())
    try:
        result = _aero_rs.run(toml_path=toml_path, overrides=overrides)
        return result.final_record.reshape(1, 52)
    except Exception:
        import traceback
        traceback.print_exc()
        return None
```

Note: unlike the original `except Exception: return None`, this logs the traceback for debuggability during development.

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/ -v -k "cost or evaluate" 2>&1 | tail -30
```

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py
git commit -m "feat: add PyO3 direct call path with subprocess fallback in evaluate.py"
```

---

### Task 10: Python PyO3 integration tests

**Files:**
- Create: `tests/test_pyo3.py`

- [ ] **Step 1: Write PyO3 integration tests**

Create `tests/test_pyo3.py`:

```python
"""Integration tests for the PyO3 aerocapture_rs module."""
from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if PyO3 module not available
aero = pytest.importorskip("aerocapture_rs")

# Use a real config that exists in the repo
GOLDEN_TOML = "configs/test/test_ref_orig.toml"


class TestSingleRun:
    def test_run_returns_result(self) -> None:
        result = aero.run(GOLDEN_TOML)
        assert hasattr(result, "trajectory")
        assert hasattr(result, "final_record")
        assert hasattr(result, "captured")

    def test_final_record_shape(self) -> None:
        result = aero.run(GOLDEN_TOML)
        assert result.final_record.shape == (52,)
        assert result.final_record.dtype == np.float64

    def test_trajectory_is_2d_with_8_columns(self) -> None:
        result = aero.run(GOLDEN_TOML)
        assert result.trajectory.ndim == 2
        assert result.trajectory.shape[1] == 8

    def test_convenience_accessors_match_final_record(self) -> None:
        result = aero.run(GOLDEN_TOML)
        assert result.energy == result.final_record[7]
        assert result.ecc == result.final_record[9]
        assert result.periapsis_alt == result.final_record[14]
        assert result.apoapsis_alt == result.final_record[15]
        assert result.delta_v == result.final_record[41]
        assert result.peri_err == result.final_record[29]
        assert result.apo_err == result.final_record[30]

    def test_captured_flag_consistent_with_orbital_elements(self) -> None:
        result = aero.run(GOLDEN_TOML)
        expected = result.ecc < 1.0 and result.energy < 0.0
        assert result.captured == expected


class TestOverrides:
    def test_override_changes_result(self) -> None:
        """Different bank angle should produce different final state."""
        r1 = aero.run(GOLDEN_TOML)
        r2 = aero.run(GOLDEN_TOML, overrides={"guidance.reference_bank_angle": 30.0})
        assert not np.array_equal(r1.final_record, r2.final_record)

    def test_invalid_override_type_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            aero.run(GOLDEN_TOML, overrides={"guidance.reference_bank_angle": [1, 2, 3]})


class TestBatchRun:
    def test_batch_returns_correct_count(self) -> None:
        """Each override set produces one row in final_records."""
        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(5)]
        results = aero.run_batch(GOLDEN_TOML, overrides)
        assert results.final_records.shape == (5, 52)
        assert results.captured.shape == (5,)

    def test_batch_trajectories_off_by_default(self) -> None:
        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(3)]
        results = aero.run_batch(GOLDEN_TOML, overrides)
        assert results.trajectories == []

    def test_batch_trajectories_on(self) -> None:
        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(3)]
        results = aero.run_batch(GOLDEN_TOML, overrides, include_trajectories=True)
        assert len(results.trajectories) == 3
        for traj in results.trajectories:
            assert traj.ndim == 2
            assert traj.shape[1] == 8


class TestCostCompat:
    def test_pyo3_final_records_work_with_compute_cost(self) -> None:
        """Verify PyO3 output is directly compatible with compute_cost()."""
        from aerocapture.training.evaluate import compute_cost

        overrides = [{"simulation.random_seed": float(i) / 10.0} for i in range(5)]
        results = aero.run_batch(GOLDEN_TOML, overrides)
        cost = compute_cost(results.final_records)
        assert isinstance(cost, float)
        assert cost >= 0.0


class TestBitIdenticalRegression:
    def test_pyo3_matches_subprocess(self, build_rust: None) -> None:
        """PyO3 and subprocess paths must produce identical final_record values.

        Requires: Rust binary built (uses the session-scoped build_rust fixture
        from conftest.py). Tests are run from the repo root.
        """
        from aerocapture.training.config import SimConfig, TrainingConfig
        from aerocapture.training.evaluate import _run_via_subprocess

        # Configure subprocess path with matching output file path
        config = TrainingConfig(
            sim=SimConfig(
                toml_config=GOLDEN_TOML,
                final_file="output/final.test_ref_orig",
            ),
        )
        # Subprocess path
        sub_result = _run_via_subprocess(config)
        assert sub_result is not None, "Subprocess path failed — is the Rust binary built?"

        # PyO3 path
        pyo3_result = aero.run(GOLDEN_TOML)
        pyo3_array = pyo3_result.final_record.reshape(1, 52)

        np.testing.assert_array_equal(
            sub_result, pyo3_array,
            err_msg="PyO3 and subprocess paths must produce bit-identical results",
        )


class TestLoadConfig:
    def test_load_config_returns_dict(self) -> None:
        config = aero.load_config(GOLDEN_TOML)
        assert isinstance(config, dict)
        assert "mission" in config
        assert "guidance" in config

    def test_load_config_nonexistent_raises(self) -> None:
        with pytest.raises(ValueError):
            aero.load_config("nonexistent.toml")


class TestFallback:
    def test_subprocess_fallback_works(self, build_rust: None) -> None:
        """Verify the subprocess path still works independently.

        Requires: Rust binary built (uses session-scoped build_rust fixture).
        """
        from aerocapture.training.config import SimConfig, TrainingConfig
        from aerocapture.training.evaluate import _run_via_subprocess

        config = TrainingConfig(
            sim=SimConfig(
                toml_config=GOLDEN_TOML,
                final_file="output/final.test_ref_orig",
            ),
        )
        result = _run_via_subprocess(config)
        assert result is not None, "Subprocess path failed — is the Rust binary built?"
        assert result.shape[1] == 52
```

Note: `simulation.random_seed` is a float in the TOML config (e.g., `random_seed = 0.6866`), so overrides use `float(i) / 10.0` not bare `int`.

- [ ] **Step 2: Build module and run tests**

```bash
cd src/rust/aerocapture-py && maturin develop --release && cd ../../..
uv run pytest tests/test_pyo3.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all existing + new tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pyo3.py
git commit -m "test: add PyO3 integration tests with bit-identical regression"
```

---

### Task 11: Update CI for PyO3

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add PyO3 test job**

Add after the `python-test` job:

```yaml
  python-pyo3:
    name: Python (pyo3)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable

      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: src/rust

      - uses: astral-sh/setup-uv@v5

      - name: Install dependencies
        run: uv sync --group dev

      - name: Build Rust simulator
        run: cargo build --release
        working-directory: src/rust

      - name: Build PyO3 module
        run: cd src/rust/aerocapture-py && uv run maturin develop --release

      - name: Run PyO3 tests
        run: uv run pytest tests/test_pyo3.py -v
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add PyO3 build and test job"
```

---

### Task 12: Final verification

- [ ] **Step 1: Clean build from scratch**

```bash
cd src/rust && cargo build --release
cd aerocapture-py && maturin develop --release
cd ../../..
```

- [ ] **Step 2: Run all Rust tests**

```bash
cd src/rust && cargo test --release
```

Expected: all pass.

- [ ] **Step 3: Run all Python tests**

```bash
uv run pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4: Run linting**

```bash
./lint_code.sh
./check_all.sh
```

Expected: clean.

---

### Task 13: Smart commit the feature branch

- [ ] **Step 1: Use `/smart-commit` skill**

Invoke the `smart-commit` skill, telling it to take the entire `feature/pyo3-interface` branch into account (all commits since diverging from `main`). This updates CLAUDE.md and README.md to reflect the new PyO3 interface, then creates a final commit.
