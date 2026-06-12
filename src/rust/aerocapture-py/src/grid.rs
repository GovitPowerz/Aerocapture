//! `run_grid`: build N SimData once (Arc-shared tables), run the N x K
//! (individual x seed) grid in parallel under a released GIL, return the
//! (n_pop, n_seeds, FINAL_RECORD_LEN) final-records array.
//!
//! Bit-identical to the per-seed `run_batch` path: each cell runs via
//! `run_for_api_cell` (static draw from `seed_k`, `sim_idx = 0`, constant base
//! `simulation.random_seed`), reproducing both the dispersion draw and the
//! per-sim EKF / Gauss-Markov RNG stream.

use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use aerocapture::config::SimInput;
use aerocapture::data::neural::{LayerSpec, NormSpec, OutputParam};
use aerocapture::data::{SharedTables, SimData};
use aerocapture::simulation::final_record::FINAL_RECORD_LEN;
use rayon::prelude::*;
use toml::{Table, Value};

use crate::config::{OverrideValue, apply_override};

/// Per-individual NN weight payload built once, before `py.detach`.
pub struct NnSpec {
    pub specs: Vec<LayerSpec>,
    pub input_mask: Option<Vec<usize>>,
    pub output_param: OutputParam,
    pub scaled_pi_n: f64,
    pub delta_max: f64,
    pub normalization: Option<Vec<NormSpec>>,
    pub weights: Vec<Vec<f64>>, // (n_pop, n_weights)
}

/// Run the (n_pop x n_seeds) grid. `nn` carries per-individual weights + the
/// shared architecture when the scheme is NN; `None` for non-NN schemes.
/// Returns final records flattened row-major as (i * n_seeds + k) -> [f64; LEN].
#[allow(clippy::too_many_arguments)]
pub fn run_grid(
    toml_path: &Path,
    overrides_list: Vec<Vec<(String, OverrideValue)>>,
    seeds: Vec<u64>,
    nn: Option<NnSpec>,
    n_threads: Option<usize>,
    wall_timeout: Option<Duration>,
) -> Result<(usize, usize, Vec<[f64; FINAL_RECORD_LEN]>), String> {
    let n_pop = overrides_list.len();
    let n_seeds = seeds.len();
    if n_pop == 0 || n_seeds == 0 {
        return Ok((n_pop, n_seeds, Vec::new()));
    }
    if let Some(ref s) = nn
        && s.weights.len() != n_pop
    {
        return Err(format!(
            "run_grid: weights rows ({}) must match overrides_list len ({})",
            s.weights.len(),
            n_pop
        ));
    }

    // Read + parse + resolve base inheritance ONCE.
    let toml_content = std::fs::read_to_string(toml_path)
        .map_err(|e| format!("Cannot read '{}': {}", toml_path.display(), e))?;
    let base_table: Table =
        toml::from_str(&toml_content).map_err(|e| format!("TOML parse error: {}", e))?;
    let mut visited = HashSet::new();
    let base_value =
        aerocapture::config::resolve_toml_bases(Value::Table(base_table), toml_path, &mut visited)
            .map_err(|e| format!("Base resolution error: {}", e))?;

    // Atmosphere / wind tables are loaded once from the base config and shared
    // across the grid — a per-individual override of their paths would be
    // silently ignored (the reference trajectory is the one table
    // `from_toml_with_tables` reloads per individual, for the joint ref_bank gene).
    for ovs in &overrides_list {
        for (k, _) in ovs {
            if k == "data.atmosphere" || k == "data.wind_table" {
                return Err(format!(
                    "run_grid: '{}' override is not supported (atmosphere/wind tables are shared across the grid; use run_batch)",
                    k
                ));
            }
        }
    }

    let run = || -> Result<Vec<[f64; FINAL_RECORD_LEN]>, String> {
        // SharedTables once from the base config. Per-individual overrides may
        // retarget `data.reference_trajectory` (joint ref_bank); SimData's
        // `from_toml_with_tables` reloads that table when the patched path
        // differs from the shared load.
        let base_str =
            toml::to_string(&base_value).map_err(|e| format!("TOML serialize error: {}", e))?;
        let (base_input, base_toml) =
            SimInput::from_toml(&base_str).map_err(|e| format!("Config parse error: {}", e))?;
        let shared = SharedTables::from_toml(&base_toml, &base_input)
            .map_err(|e| format!("Shared table load error: {}", e))?;

        // Build N SimData (in parallel) — each gets its own NN model when nn is Some.
        let built: Vec<Result<(SimInput, SimData), String>> = (0..n_pop)
            .into_par_iter()
            .map(|i| {
                let mut patched = base_value.clone();
                for (k, v) in &overrides_list[i] {
                    apply_override(&mut patched, k, v)?;
                }
                let toml_str = toml::to_string(&patched)
                    .map_err(|e| format!("TOML serialize error: {}", e))?;
                let (sim_input, toml_config) = SimInput::from_toml(&toml_str)
                    .map_err(|e| format!("Config parse error: {}", e))?;

                let injected = match &nn {
                    Some(spec) => Some(
                        crate::build_model_from_flat(
                            &spec.weights[i],
                            &spec.specs,
                            spec.input_mask.clone(),
                            spec.output_param,
                            spec.scaled_pi_n,
                            spec.delta_max,
                            spec.normalization.as_deref(),
                        )
                        .map_err(|e| format!("NN build error: {}", e))?,
                    ),
                    None => None,
                };

                let sim_data =
                    SimData::from_toml_with_tables(&toml_config, &sim_input, &shared, injected)
                        .map_err(|e| format!("Data load error: {}", e))?;
                Ok((sim_input, sim_data))
            })
            .collect();

        let mut siminputs: Vec<SimInput> = Vec::with_capacity(n_pop);
        let mut simdatas: Vec<SimData> = Vec::with_capacity(n_pop);
        for r in built {
            let (si, sd) = r?;
            siminputs.push(si);
            simdatas.push(sd);
        }

        // Run the N x K grid in parallel, row-major flat index = i * n_seeds + k.
        let cells: Vec<Result<[f64; FINAL_RECORD_LEN], String>> = (0..n_pop * n_seeds)
            .into_par_iter()
            .map(|flat| {
                let i = flat / n_seeds;
                let k = flat % n_seeds;
                aerocapture::simulation::runner::run_for_api_cell(
                    &siminputs[i],
                    &simdatas[i],
                    seeds[k],
                    false,
                    wall_timeout,
                )
                .map(|o| o.final_record)
                .map_err(|e| format!("Simulation error: {}", e))
            })
            .collect();

        cells.into_iter().collect()
    };

    let records = match n_threads {
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| format!("Failed to create thread pool: {}", e))?;
            pool.install(run)?
        }
        None => run()?,
    };

    Ok((n_pop, n_seeds, records))
}
