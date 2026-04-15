//! BatchedSimulation: per-step vectorized env for RL training.
//!
//! Holds N independent SimStates sharing one Arc<SimData>. step() advances
//! each env one outer guidance tick via Rayon, auto-resets on done, and
//! returns the stacked (obs, reward, done, info) payload.
//!
//! step() is implemented in Task 1.4. Calling it now raises AttributeError.

use std::path::Path;
use std::sync::Arc;

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::data::dispersions::DispersionDraw;
use aerocapture::simulation::runner::{SimState, build_sim_state};

use crate::config;
use crate::extract_overrides;

/// Vectorized step-based simulator for RL training.
#[pyclass(unsendable)]
pub struct BatchedSimulation {
    #[pyo3(get)]
    pub n_envs: usize,
    #[pyo3(get)]
    pub obs_dim: usize,
    sim_input: SimInput,
    sim_data: Arc<SimData>,
    envs: Vec<SimState>,
    seed_base: u64,
    episode_counter: Vec<u64>,
    episode_ids: Vec<u64>,
    step_counts: Vec<u64>,
}

#[pymethods]
impl BatchedSimulation {
    #[new]
    #[pyo3(signature = (toml_path, n_envs, overrides=None, seed_base=3_000_000))]
    fn new(
        toml_path: &str,
        n_envs: usize,
        overrides: Option<&Bound<'_, PyDict>>,
        seed_base: u64,
    ) -> PyResult<Self> {
        if n_envs == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("n_envs must be > 0"));
        }
        let overrides = extract_overrides(overrides)?;
        let (sim_input, sim_data) = config::load_and_override(Path::new(toml_path), &overrides)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let nn = sim_data
            .neural_net
            .as_ref()
            .expect("neural_net model required for RL env");
        let obs_dim = nn.input_mask.as_ref().map(|m: &Vec<usize>| m.len()).unwrap_or(16);

        let sim_data = Arc::new(sim_data);
        let mut envs = Vec::with_capacity(n_envs);
        let mut episode_ids = Vec::with_capacity(n_envs);
        let mut episode_counter = Vec::with_capacity(n_envs);

        for i in 0..n_envs {
            let seed = seed_base + i as u64;
            let draw = draw_from_seed(&sim_data, seed);
            let run_state = aerocapture::simulation::init::init_run_from_draw(&sim_data, &draw);
            let state = build_sim_state(&sim_input, &sim_data, run_state, seed);
            envs.push(state);
            episode_ids.push(seed);
            episode_counter.push(i as u64);
        }

        Ok(Self {
            n_envs,
            obs_dim,
            sim_input,
            sim_data,
            envs,
            seed_base,
            episode_counter,
            episode_ids,
            step_counts: vec![0u64; n_envs],
        })
    }

    #[pyo3(signature = (seeds=None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seeds: Option<PyReadonlyArray1<'py, i64>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let explicit_seeds = seeds.is_some();
        let seeds_vec: Vec<u64> = match seeds {
            Some(arr) => {
                let n = arr.len();
                if n != self.n_envs {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "seeds length {} does not match n_envs {}",
                        n, self.n_envs
                    )));
                }
                arr.as_array().iter().map(|&s| s as u64).collect()
            }
            None => (0..self.n_envs)
                .map(|i| self.seed_base + self.episode_counter[i])
                .collect(),
        };

        for (i, &seed) in seeds_vec.iter().enumerate() {
            let draw = draw_from_seed(&self.sim_data, seed);
            let run_state =
                aerocapture::simulation::init::init_run_from_draw(&self.sim_data, &draw);
            self.envs[i] = build_sim_state(&self.sim_input, &self.sim_data, run_state, seed);
            self.episode_ids[i] = seed;
            self.step_counts[i] = 0;
            if !explicit_seeds {
                // Advance so next default-seed reset draws a fresh, distinct seed per env.
                self.episode_counter[i] += self.n_envs as u64;
            }
        }

        Ok(self.build_obs(py))
    }

    fn close(&mut self) {
        self.envs.clear();
    }

    /// Current episode seed for each env slot (advances each default-seed reset).
    fn current_seeds<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u64>> {
        numpy::PyArray1::from_slice(py, &self.episode_ids)
    }
}

impl BatchedSimulation {
    fn build_obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let arr = PyArray2::<f32>::zeros(py, [self.n_envs, self.obs_dim], false);
        let mut view = unsafe { arr.as_array_mut() };
        for (i, env) in self.envs.iter().enumerate() {
            let obs = build_obs_for_env(env, &self.sim_data, &self.sim_input);
            for (j, &v) in obs.iter().enumerate() {
                view[[i, j]] = v as f32;
            }
        }
        arr
    }
}

/// Build the observation vector for a single env state.
///
/// Delegates to `build_nn_input` so the RL observation matches exactly what the
/// runtime NN guidance sees. Panics if no neural_net model is loaded -- callers
/// must use a config with `[data] neural_network` set.
fn build_obs_for_env(state: &SimState, data: &Arc<SimData>, config: &SimInput) -> Vec<f64> {
    let nav = state.last_nav_output();
    let planet = &config.planet;
    let target_inclination = data.target_orbit.inclination;
    let ref_velocity_latched = state.guidance_state.reference_velocity;
    let nn = data
        .neural_net
        .as_ref()
        .expect("neural_net model required for RL env");

    aerocapture::gnc::guidance::neural::build_nn_input(
        &nav,
        nn,
        data,
        planet,
        target_inclination,
        ref_velocity_latched,
    )
}

/// Generate a deterministic dispersion draw for a given seed.
///
/// For RL training we want each env to start with a reproducible but varied
/// scenario. We use the seed to pick from the configured dispersion distribution.
/// When no dispersion config is present (nominal runs), returns the zero draw.
fn draw_from_seed(data: &SimData, _seed: u64) -> DispersionDraw {
    // Use the zero draw for now; Task 1.4 can add seeded MC draws.
    let _ = data;
    DispersionDraw::default()
}
