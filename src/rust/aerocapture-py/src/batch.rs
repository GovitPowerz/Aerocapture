//! Deploy-side runners over the kernel: `run_batch` (one override set per
//! simulation), `run_seeds` (one override set on a seed axis) and
//! `run_with_external_draws`.

use std::path::Path;
use std::time::Duration;

use aerocapture::RunOutput;
use aerocapture::data::dispersions::DispersionDraw;
use aerocapture::simulation::runner::SimError;

use crate::config::OverrideValue;
use crate::kernel::{self, BaseConfig};

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
/// The base TOML is resolved and its shared tables loaded once; each entry of
/// `overrides_list` is built into its own `SimData` (`BaseConfig::build`) and
/// flown on its own `[monte_carlo] seed`. Returns one `RunOutput` per entry.
/// A seed sweep of ONE override set belongs on `run_seeds`, which builds once.
///
/// **Contract:** `n_sims` must equal 1 for every resolved override set. Use
/// `run_mc` for multi-sim Monte Carlo per config.
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
    let base = BaseConfig::load(toml_path).map_err(BatchError::Runtime)?;

    let run = || -> Result<Vec<RunOutput>, BatchError> {
        use rayon::prelude::*;

        let results: Vec<Result<RunOutput, BatchError>> = overrides_list
            .into_par_iter()
            .map(|overrides| {
                let (sim_input, sim_data) =
                    base.build(&overrides, None).map_err(BatchError::Runtime)?;

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

    kernel::in_pool(n_threads, run).map_err(BatchError::Runtime)?
}

/// Fly ONE override set on a seed axis: the config is built once and each seed
/// is a `run_for_api_cell`. Returns one `RunOutput` per seed, in `seeds` order,
/// byte-identical to `run_batch` with one override set per seed
/// (`tests/test_run_seeds.py`).
///
/// The config is built as the per-seed path builds it: one sim, and a
/// `[monte_carlo]` section even when the TOML has none (the per-seed path's
/// `monte_carlo.seed` override creates it). The seed value written here is never
/// read: every cell re-seeds its own draw.
pub fn run_seeds(
    toml_path: &Path,
    mut overrides: Vec<(String, OverrideValue)>,
    seeds: Vec<u64>,
    n_threads: Option<usize>,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<RunOutput>, String> {
    let base = BaseConfig::load(toml_path)?;
    if seeds.is_empty() {
        return Ok(Vec::new());
    }
    overrides.push(("simulation.n_sims".to_string(), OverrideValue::Int(1)));
    overrides.push(("monte_carlo.seed".to_string(), OverrideValue::Int(0)));

    kernel::in_pool(n_threads, || {
        let cell = base.build(&overrides, None)?;
        kernel::fly(
            std::slice::from_ref(&cell),
            &seeds,
            include_trajectories,
            wall_timeout,
            |output| output,
        )
    })?
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
