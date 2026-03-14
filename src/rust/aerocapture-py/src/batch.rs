//! Parallel batch runner.
//!
//! Parses the base TOML once, then applies per-run overrides in parallel
//! using a scoped Rayon thread pool.

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::simulation::runner::SimError;
use aerocapture::RunOutput;
use toml::{Table, Value};

use crate::config::{apply_override, OverrideValue};

/// Run a batch of simulations with per-run TOML overrides.
///
/// 1. Parse `toml_content` into a TOML value tree (the "base config").
/// 2. For each entry in `overrides_list`, clone the base tree, apply
///    overrides, serialize back, parse via `SimInput::from_toml` +
///    `SimData::from_toml`, and run `run_for_api`.
/// 3. Returns the first `RunOutput` per batch item (if `n_sims > 1` in
///    the config, only the first result is kept and a warning is printed).
///
/// Uses a scoped Rayon thread pool with `n_threads` threads.
pub fn run_batch(
    toml_content: &str,
    overrides_list: Vec<Vec<(String, OverrideValue)>>,
    n_threads: usize,
) -> Result<Vec<RunOutput>, String> {
    // Parse the base config once.
    let base_table: Table =
        toml::from_str(toml_content).map_err(|e| format!("TOML parse error: {}", e))?;
    let base_value = Value::Table(base_table);

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

                let outputs = aerocapture::simulation::runner::run_for_api(&sim_input, &sim_data)
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
