//! The cell-by-seeds kernel behind `run_grid`, `run_seeds` and `run_batch`.
//!
//! The two rules those three entries share live here:
//!
//! - an override set becomes a `SimData` (`BaseConfig::build`): patch the resolved
//!   base tree, serialize, parse, validate, build over the base config's shared
//!   tables;
//! - a built config is flown on a seed axis (`fly`): one `run_for_api_cell` per
//!   (config, seed), the bit-identity invariant of ADR-0004.
//!
//! `run_grid` is N configs x K seeds keeping the final records, `run_seeds` is
//! 1 config x K seeds keeping the whole `RunOutput`, and `run_batch` builds one
//! config per override set and flies each on its own `[monte_carlo] seed`.
//! Not yet routed here: `run_mc`, `run_with_draws`, the env and `collect_*` build
//! through `config::load_and_override`, and `collect_*` keeps its own seed loop
//! (`lib.rs::collect_trace`).

use std::path::Path;
use std::time::Duration;

use aerocapture::RunOutput;
use aerocapture::config::SimInput;
use aerocapture::data::neural::NeuralNetModel;
use aerocapture::data::{SharedTables, SimData};
use rayon::prelude::*;
use toml::Value;

use crate::config::{OverrideValue, apply_override, resolve_toml};

/// The disk-backed tables of the base config and the paths they were loaded from.
struct BaseTables {
    tables: SharedTables,
    atmosphere: Option<String>,
    wind_table: Option<String>,
}

/// A TOML config with `base` inheritance resolved and its shared tables
/// (atmosphere / wind / reference trajectory) loaded once.
pub struct BaseConfig {
    value: Value,
    /// `Err` when the base alone does not form a complete config; `build` then
    /// loads the tables per override set.
    tables: Result<BaseTables, String>,
}

impl BaseConfig {
    pub fn load(toml_path: &Path) -> Result<Self, String> {
        let value = resolve_toml(toml_path)?;
        let tables = Self::load_tables(&value);
        Ok(Self { value, tables })
    }

    fn load_tables(value: &Value) -> Result<BaseTables, String> {
        let base_str =
            toml::to_string(value).map_err(|e| format!("TOML serialize error: {}", e))?;
        let (base_input, base_toml) =
            SimInput::from_toml(&base_str).map_err(|e| format!("Config parse error: {}", e))?;
        let tables = SharedTables::from_toml(&base_toml, &base_input)
            .map_err(|e| format!("Shared table load error: {}", e))?;
        Ok(BaseTables {
            tables,
            atmosphere: base_toml.data.atmosphere.clone(),
            wind_table: base_toml.data.wind_table.clone(),
        })
    }

    /// The base config's own table-load error, for callers that share the tables
    /// across every override set and cannot fall back to a per-set load.
    pub fn require_shared_tables(&self) -> Result<(), String> {
        self.tables.as_ref().map(|_| ()).map_err(Clone::clone)
    }

    /// One override set -> `(SimInput, SimData)`.
    ///
    /// The shared tables are reused when the patched config still points at the
    /// base's atmosphere and wind table; otherwise every table is loaded for this
    /// set. A `data.reference_trajectory` override is handled inside
    /// `from_toml_with_tables` (the joint `ref_bank` path). `injected_nn` replaces
    /// the on-disk model load and needs the shared tables.
    pub fn build(
        &self,
        overrides: &[(String, OverrideValue)],
        injected_nn: Option<NeuralNetModel>,
    ) -> Result<(SimInput, SimData), String> {
        let mut patched = self.value.clone();
        for (key, value) in overrides {
            apply_override(&mut patched, key, value)?;
        }
        let toml_str =
            toml::to_string(&patched).map_err(|e| format!("TOML serialize error: {}", e))?;
        let (sim_input, toml_config) =
            SimInput::from_toml(&toml_str).map_err(|e| format!("Config parse error: {}", e))?;

        let shared = self.tables.as_ref().ok().filter(|base| {
            toml_config.data.atmosphere == base.atmosphere
                && toml_config.data.wind_table == base.wind_table
        });
        let sim_data = match shared {
            Some(base) => {
                SimData::from_toml_with_tables(&toml_config, &sim_input, &base.tables, injected_nn)
            }
            None if injected_nn.is_some() => {
                return Err(
                    "Data load error: an injected NN model needs the base config's shared tables"
                        .to_string(),
                );
            }
            None => SimData::from_toml(&toml_config, &sim_input),
        }
        .map_err(|e| format!("Data load error: {}", e))?;
        Ok((sim_input, sim_data))
    }
}

/// Fly every (config, seed) pair in parallel and keep `keep(output)` of each,
/// row-major: index `i * seeds.len() + k` is config `i` on `seeds[k]`.
///
/// Each cell is `run_for_api_cell`: the static draw of `seed` and `sim_idx = 0`,
/// bit-identical to a per-seed run of the same config with `monte_carlo.seed =
/// seed` and `n_sims = 1`.
pub fn fly<T: Send>(
    cells: &[(SimInput, SimData)],
    seeds: &[u64],
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
    keep: impl Fn(RunOutput) -> T + Sync,
) -> Result<Vec<T>, String> {
    let n_seeds = seeds.len();
    let flown: Vec<Result<T, String>> = (0..cells.len() * n_seeds)
        .into_par_iter()
        .map(|flat| {
            let (sim_input, sim_data) = &cells[flat / n_seeds];
            aerocapture::simulation::runner::run_for_api_cell(
                sim_input,
                sim_data,
                seeds[flat % n_seeds],
                include_trajectories,
                wall_timeout,
            )
            .map(&keep)
            .map_err(|e| format!("Simulation error: {}", e))
        })
        .collect();
    flown.into_iter().collect()
}

/// Run `f` in a scoped Rayon pool of `n_threads`, or on the global pool (no
/// per-call build) when `None`.
pub fn in_pool<R: Send>(
    n_threads: Option<usize>,
    f: impl FnOnce() -> R + Send,
) -> Result<R, String> {
    match n_threads {
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| format!("Failed to create thread pool: {}", e))?;
            Ok(pool.install(f))
        }
        None => Ok(f()),
    }
}
