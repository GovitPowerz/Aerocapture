//! Parallel batch runner.
//!
//! Parses the base TOML once (resolving `base` inheritance), then applies
//! per-run overrides in parallel using a scoped Rayon thread pool.

use std::collections::HashSet;
use std::path::Path;

use aerocapture::RunOutput;
use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::simulation::runner::SimError;
use toml::{Table, Value};

use crate::config::{OverrideValue, apply_override};

/// Run a batch of simulations with per-run TOML overrides.
///
/// 1. Read and parse the TOML file, resolving `base` inheritance.
/// 2. For each entry in `overrides_list`, clone the resolved tree, apply
///    overrides, serialize back, parse via `SimInput::from_toml` +
///    `SimData::from_toml`, and run `run_for_api`.
/// 3. Returns the first `RunOutput` per batch item (if `n_sims > 1` in
///    the config, only the first result is kept and a warning is printed).
///
/// Uses a scoped Rayon thread pool with `n_threads` threads.
pub fn run_batch(
    toml_path: &Path,
    overrides_list: Vec<Vec<(String, OverrideValue)>>,
    n_threads: usize,
    include_trajectories: bool,
) -> Result<Vec<RunOutput>, String> {
    // Read and parse the base config once.
    let toml_content = std::fs::read_to_string(toml_path)
        .map_err(|e| format!("Cannot read '{}': {}", toml_path.display(), e))?;
    let base_table: Table =
        toml::from_str(&toml_content).map_err(|e| format!("TOML parse error: {}", e))?;
    let base_value = Value::Table(base_table);

    // Resolve base inheritance once.
    let mut visited = HashSet::new();
    let base_value = aerocapture::config::resolve_toml_bases(base_value, toml_path, &mut visited)
        .map_err(|e| format!("Base resolution error: {}", e))?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| format!("Failed to create thread pool: {}", e))?;

    pool.install(|| {
        use rayon::prelude::*;

        let results: Vec<Result<RunOutput, String>> = overrides_list
            .into_par_iter()
            .map(|overrides| {
                // Clone base tree, apply overrides.
                let mut patched = base_value.clone();
                for (key, value) in &overrides {
                    apply_override(&mut patched, key, value)?;
                }

                let toml_str = toml::to_string(&patched)
                    .map_err(|e| format!("TOML serialize error: {}", e))?;

                let (sim_input, toml_config) = SimInput::from_toml(&toml_str)
                    .map_err(|e| format!("Config parse error: {}", e))?;
                let sim_data = SimData::from_toml(&toml_config, &sim_input)
                    .map_err(|e| format!("Data load error: {}", e))?;

                if sim_input.n_sims > 1 {
                    eprintln!(
                        "Warning: n_sims={} in config, but batch runner only keeps first result",
                        sim_input.n_sims
                    );
                }

                let outputs = aerocapture::simulation::runner::run_for_api(
                    &sim_input,
                    &sim_data,
                    include_trajectories,
                )
                .map_err(|e: SimError| format!("Simulation error: {}", e))?;

                outputs
                    .into_iter()
                    .next()
                    .ok_or_else(|| "Simulation produced no results".to_string())
            })
            .collect();

        // Collect results, returning the first error encountered.
        results.into_iter().collect()
    })
}
