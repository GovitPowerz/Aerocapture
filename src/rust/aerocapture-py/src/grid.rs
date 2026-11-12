//! `run_grid`: build N SimData once (Arc-shared tables), run the N x K
//! (individual x seed) grid in parallel under a released GIL, return the
//! (n_pop, n_seeds, FINAL_RECORD_LEN) final-records array.
//!
//! Bit-identical to the per-seed `run_batch` path: each cell runs via
//! `run_for_api_cell` (static draw from `seed_k`, `sim_idx = 0`, constant base
//! `simulation.random_seed`), reproducing both the dispersion draw and the
//! per-sim EKF / Gauss-Markov RNG stream.

use std::path::Path;
use std::time::Duration;

use aerocapture::data::neural::{LayerSpec, NormSpec, OutputParam};
use aerocapture::simulation::final_record::FINAL_RECORD_LEN;
use rayon::prelude::*;

use crate::config::OverrideValue;
use crate::kernel::{self, BaseConfig};

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

    // Atmosphere / wind tables are loaded once from the base config and shared
    // across the grid: an override that retargets them is refused here instead
    // of reloading both tables per individual (the reference trajectory is the one
    // table `from_toml_with_tables` reloads per individual, for the joint ref_bank gene).
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

    let base = BaseConfig::load(toml_path)?;
    base.require_shared_tables()?;

    let run = || -> Result<Vec<[f64; FINAL_RECORD_LEN]>, String> {
        // Build N SimData (in parallel): each gets its own NN model when nn is Some.
        let built: Vec<Result<_, String>> = (0..n_pop)
            .into_par_iter()
            .map(|i| {
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
                base.build(&overrides_list[i], injected)
            })
            .collect();
        let cells = built.into_iter().collect::<Result<Vec<_>, String>>()?;

        // The N x K grid, row-major flat index = i * n_seeds + k.
        kernel::fly(&cells, &seeds, false, wall_timeout, |output| {
            output.final_record
        })
    };

    let records = kernel::in_pool(n_threads, run)??;

    Ok((n_pop, n_seeds, records))
}
