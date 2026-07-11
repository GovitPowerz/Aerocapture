//! Parallel batch runner.
//!
//! Parses the base TOML once (resolving `base` inheritance), then applies
//! per-run overrides in parallel via Rayon (a scoped thread pool when
//! `n_threads` is provided, otherwise the global pool).

use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use aerocapture::RunOutput;
use aerocapture::config::SimInput;
use aerocapture::data::dispersions::DispersionDraw;
use aerocapture::data::{SharedTables, SimData};
use aerocapture::simulation::runner::SimError;
use toml::{Table, Value};

use crate::config::{OverrideValue, apply_override};

/// Typed error for `run_batch`: distinguishes caller contract violations
/// (should surface as `PyValueError`) from runtime failures (should surface
/// as `PyRuntimeError`).
#[derive(Debug)]
pub enum BatchError {
    /// Caller violated the `run_batch` contract (e.g. `n_sims > 1`).
    Contract(String),
    /// Config load, TOML parse, data load, or simulation failure.
    Runtime(String),
}

/// Run a batch of simulations with per-run TOML overrides.
///
/// 1. Read and parse the TOML file, resolving `base` inheritance.
/// 2. For each entry in `overrides_list`, clone the resolved tree, apply
///    overrides, serialize back, parse via `SimInput::from_toml` +
///    `SimData::from_toml`, and run `run_for_api`.
/// 3. Returns exactly one `RunOutput` per batch item.
///
/// **Contract:** `n_sims` must equal 1 for every resolved override set.
/// `run_batch` is designed for one-trajectory-per-override evaluation (e.g.
/// the GA training loop). Use `run_mc` for multi-sim Monte Carlo per config.
/// Passing `n_sims > 1` in the base config or an override returns an error.
///
/// When `n_threads` is `Some(n)`, runs inside a scoped Rayon thread pool with
/// `n` threads; when `None`, reuses the global Rayon pool (no per-call build).
pub fn run_batch(
    toml_path: &Path,
    overrides_list: Vec<Vec<(String, OverrideValue)>>,
    n_threads: Option<usize>,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<RunOutput>, BatchError> {
    // Read and parse the base config once.
    let toml_content = std::fs::read_to_string(toml_path).map_err(|e| {
        BatchError::Runtime(format!("Cannot read '{}': {}", toml_path.display(), e))
    })?;
    let base_table: Table = toml::from_str(&toml_content)
        .map_err(|e| BatchError::Runtime(format!("TOML parse error: {}", e)))?;
    let base_value = Value::Table(base_table);

    // Resolve base inheritance once.
    let mut visited = HashSet::new();
    let base_value = aerocapture::config::resolve_toml_bases(base_value, toml_path, &mut visited)
        .map_err(|e| BatchError::Runtime(format!("Base resolution error: {}", e)))?;

    // Preload the shared tables (atmosphere/wind/ref) from the base config so
    // per-override SimData construction skips the disk reads -- run_batch is
    // called with thousands of overrides that differ only in guidance params /
    // NN path. Best-effort: a base that doesn't form a complete config (or an
    // override that retargets data.atmosphere / data.wind_table) falls back to
    // the full per-override load below. The reference-trajectory override is
    // handled inside `from_toml_with_tables` (joint ref_bank path).
    let base_shared: Option<(SharedTables, Option<String>, Option<String>)> =
        toml::to_string(&base_value).ok().and_then(|base_str| {
            let (base_input, base_toml) = SimInput::from_toml(&base_str).ok()?;
            let shared = SharedTables::from_toml(&base_toml, &base_input).ok()?;
            Some((
                shared,
                base_toml.data.atmosphere.clone(),
                base_toml.data.wind_table.clone(),
            ))
        });

    let run = || -> Result<Vec<RunOutput>, BatchError> {
        use rayon::prelude::*;

        let results: Vec<Result<RunOutput, BatchError>> = overrides_list
            .into_par_iter()
            .map(|overrides| {
                // Clone base tree, apply overrides.
                let mut patched = base_value.clone();
                for (key, value) in &overrides {
                    apply_override(&mut patched, key, value).map_err(BatchError::Runtime)?;
                }

                let toml_str = toml::to_string(&patched)
                    .map_err(|e| BatchError::Runtime(format!("TOML serialize error: {}", e)))?;

                let (sim_input, toml_config) = SimInput::from_toml(&toml_str)
                    .map_err(|e| BatchError::Runtime(format!("Config parse error: {}", e)))?;
                let shared_ok = base_shared.as_ref().is_some_and(|(_, atm, wind)| {
                    toml_config.data.atmosphere == *atm && toml_config.data.wind_table == *wind
                });
                let sim_data = if shared_ok {
                    let (shared, _, _) = base_shared.as_ref().unwrap();
                    SimData::from_toml_with_tables(&toml_config, &sim_input, shared, None)
                } else {
                    SimData::from_toml(&toml_config, &sim_input)
                }
                .map_err(|e| BatchError::Runtime(format!("Data load error: {}", e)))?;

                if sim_input.n_sims > 1 {
                    return Err(BatchError::Contract(format!(
                        "run_batch expects one sim per override (n_sims must be 1 per override); \
                         got n_sims={} — use run_mc for multi-sim per config",
                        sim_input.n_sims
                    )));
                }

                let outputs = aerocapture::simulation::runner::run_for_api(
                    &sim_input,
                    &sim_data,
                    include_trajectories,
                    wall_timeout,
                )
                .map_err(|e: SimError| BatchError::Runtime(format!("Simulation error: {}", e)))?;

                outputs.into_iter().next().ok_or_else(|| {
                    BatchError::Runtime("Simulation produced no results".to_string())
                })
            })
            .collect();

        // Collect results, returning the first error encountered.
        results.into_iter().collect()
    };

    match n_threads {
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| BatchError::Runtime(format!("Failed to create thread pool: {}", e)))?;
            pool.install(run)
        }
        None => run(), // reuse the global Rayon pool (no per-call build)
    }
}

/// Run simulations with pre-computed dispersion draws supplied by the caller.
///
/// Loads config once, converts `[f64; 26]` arrays to `DispersionDraw`, then
/// delegates to `run_for_api_with_draws()` which runs in parallel via Rayon.
pub fn run_with_external_draws(
    toml_path: &Path,
    overrides: Vec<(String, crate::config::OverrideValue)>,
    draws: Vec<[f64; 26]>,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<RunOutput>, String> {
    let (sim_input, sim_data) = crate::config::load_and_override(toml_path, &overrides)?;

    let dispersion_draws: Vec<DispersionDraw> =
        draws.into_iter().map(DispersionDraw::from_array).collect();

    aerocapture::simulation::runner::run_for_api_with_draws(
        &sim_input,
        &sim_data,
        dispersion_draws,
        include_trajectories,
        wall_timeout,
    )
    .map_err(|e: SimError| format!("Simulation error: {}", e))
}
