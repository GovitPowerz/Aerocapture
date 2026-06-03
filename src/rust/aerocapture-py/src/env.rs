//! BatchedSimulation: per-step vectorized env for RL training.
//!
//! Holds N independent SimStates sharing one Arc<SimData>. step() advances
//! each env one outer guidance tick via Rayon, auto-resets on done, and
//! returns the stacked (obs, reward, done, info) payload.
//!
//! step() is implemented in Task 1.4.

use std::path::Path;
use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::data::dispersions::DispersionDraw;
use aerocapture::integration::events::{EventContext, EventDef};
use aerocapture::simulation::runner::{
    SimState, TermReason, build_final_record, build_sim_state, ifinal_for,
};

use crate::config;
use crate::extract_overrides;

// Type aliases to satisfy clippy::type_complexity on #[pymethods] return types.
type ResetObs<'py> = PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)>;
type StepReturn<'py> = PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<bool>>,
    Vec<Py<PyDict>>,
    Bound<'py, PyArray2<f32>>,
)>;

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
    event_defs: Vec<EventDef>,
    event_ctx: EventContext,
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
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_envs must be > 0",
            ));
        }
        let overrides = extract_overrides(overrides)?;
        let (sim_input, sim_data) = config::load_and_override(Path::new(toml_path), &overrides)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let nn = sim_data.neural_net.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "RL env requires a neural_network model ([data] neural_network)",
            )
        })?;
        let obs_dim = nn
            .input_mask
            .as_ref()
            .map(|m: &Vec<usize>| m.len())
            .unwrap_or(16);

        let sim_data = Arc::new(sim_data);

        let event_defs = aerocapture::integration::events::build_aerocapture_events();
        let event_ctx = EventContext {
            planet_radius: sim_input.planet.equatorial_radius,
            polar_radius: sim_input.planet.polar_radius,
            exit_altitude: sim_data.final_conditions.altitude,
            exit_velocity_threshold: sim_data.guidance.exit_velocity_threshold,
        };

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
            event_defs,
            event_ctx,
        })
    }

    #[pyo3(signature = (seeds=None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seeds: Option<PyReadonlyArray1<'py, i64>>,
    ) -> ResetObs<'py> {
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

        let obs = self.build_obs(py);
        let aux = self.build_aux(py);
        Ok((obs, aux))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, f32>,
    ) -> StepReturn<'py> {
        if actions.len() != self.n_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "actions length {} does not match n_envs {}",
                actions.len(),
                self.n_envs
            )));
        }
        let actions_vec: Vec<f64> = actions.as_array().iter().map(|&v| v as f64).collect();

        let sim_input = &self.sim_input;
        let sim_data = &self.sim_data;
        let event_defs = &self.event_defs;
        let event_ctx = &self.event_ctx;

        // Advance all envs one tick in parallel; collect terminal info where done.
        // Also capture (energy, pdyn) from nav output BEFORE auto-reset so PBRS
        // gets the pre-reset values for terminal steps.
        // Release the GIL during the Rayon block so other Python threads can run
        // and Ctrl-C is responsive.
        let outcomes: Vec<(bool, Option<TerminalOutcome>, [f64; 2])> = py.detach(|| {
            self.envs
                .par_iter_mut()
                .zip(actions_vec.par_iter())
                .map(|(state, &action)| {
                    let bank = action.clamp(-std::f64::consts::PI, std::f64::consts::PI);
                    aerocapture::simulation::tick::step_one_tick(
                        state,
                        sim_input,
                        sim_data,
                        &sim_input.planet,
                        Some(bank),
                        event_defs,
                        event_ctx,
                    );
                    // Capture aux (energy, pdyn) from nav output before potential reset.
                    let nav = state.last_nav_output();
                    let aux = [nav.energy_estimated, nav.dynamic_pressure_estimated];
                    if state.term() != TermReason::None {
                        // Capture terminal obs BEFORE the env state is reset.
                        let terminal_obs = build_obs_for_env(state, sim_data, sim_input);
                        let fr = build_final_record(state, sim_data, &sim_input.planet);
                        // Guarded by the enclosing `if state.term() != TermReason::None`;
                        // ifinal_for's None arm (unreachable!) cannot fire here.
                        let ifinal = ifinal_for(state.term());
                        let ecc = fr[9];
                        let energy = fr[7]; // MJ/kg; negative = captured
                        let captured = ifinal == 3 && ecc < 1.0 && energy < 0.0;
                        let violated = state.any_constraint_violated(sim_data);
                        // Truncation vs termination: ifinal=2 (Timeout) is a max_time cutoff
                        // where the trajectory is still physically valid -- the value
                        // function should bootstrap V(terminal_obs), not 0.
                        let truncated = ifinal == 2;
                        let term = TerminalOutcome {
                            ifinal,
                            captured,
                            ecc,
                            dv_m_s: fr[41],
                            peak_heat_flux_kw_m2: fr[16],
                            peak_g_load: fr[17],
                            peak_heat_load_kj_m2: fr[28] * 1e3, // MJ/m2 -> kJ/m2
                            violated_constraints: violated,
                            truncated,
                            final_record: fr,
                            terminal_obs,
                        };
                        (true, Some(term), aux)
                    } else {
                        (false, None, aux)
                    }
                })
                .collect()
        });

        // Auto-reset terminated envs with advancing seeds.
        for (i, (done, _, _)) in outcomes.iter().enumerate() {
            if *done {
                self.episode_counter[i] += self.n_envs as u64;
                let seed = self.seed_base + self.episode_counter[i];
                let draw = draw_from_seed(&self.sim_data, seed);
                let run_state =
                    aerocapture::simulation::init::init_run_from_draw(&self.sim_data, &draw);
                self.envs[i] = build_sim_state(&self.sim_input, &self.sim_data, run_state, seed);
                self.episode_ids[i] = seed;
                self.step_counts[i] = 0;
            } else {
                self.step_counts[i] += 1;
            }
        }

        // Build return arrays.
        let obs = self.build_obs(py);
        let reward_arr = PyArray1::<f32>::from_iter(py, outcomes.iter().map(|_| 0.0f32));
        let done_arr = PyArray1::<bool>::from_iter(py, outcomes.iter().map(|(d, _, _)| *d));

        // Aux array: (n_envs, 2) with [energy_estimated, dynamic_pressure_estimated].
        // Values are from the pre-reset nav output (terminal steps get their final-tick values).
        let aux = PyArray2::<f32>::zeros(py, [self.n_envs, 2], false);
        {
            let mut aux_view = unsafe { aux.as_array_mut() };
            for (i, (_, _, a)) in outcomes.iter().enumerate() {
                aux_view[[i, 0]] = a[0] as f32;
                aux_view[[i, 1]] = a[1] as f32;
            }
        }

        let mut info_list: Vec<Py<PyDict>> = Vec::with_capacity(self.n_envs);
        for (_, term, _) in &outcomes {
            let dict = PyDict::new(py);
            if let Some(t) = term {
                dict.set_item("ifinal", t.ifinal)?;
                dict.set_item("captured", t.captured)?;
                dict.set_item("ecc", t.ecc)?;
                dict.set_item("dv_m_s", t.dv_m_s)?;
                dict.set_item("peak_heat_flux_kW_m2", t.peak_heat_flux_kw_m2)?;
                dict.set_item("peak_g_load", t.peak_g_load)?;
                dict.set_item("peak_heat_load_kJ_m2", t.peak_heat_load_kj_m2)?;
                dict.set_item("violated_constraints", t.violated_constraints)?;
                dict.set_item("truncated", t.truncated)?;
                dict.set_item("final_record", t.final_record.to_vec())?;
                // Pre-reset obs of the terminated episode; PPO needs this for value bootstrap.
                let term_obs: Vec<f32> = t.terminal_obs.iter().map(|&v| v as f32).collect();
                dict.set_item("terminal_observation", term_obs)?;
            }
            info_list.push(dict.unbind());
        }

        Ok((obs, reward_arr, done_arr, info_list, aux))
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

    /// Auxiliary array (n_envs, 2): [energy_estimated, dynamic_pressure_estimated] per env.
    fn build_aux<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let arr = PyArray2::<f32>::zeros(py, [self.n_envs, 2], false);
        let mut view = unsafe { arr.as_array_mut() };
        for (i, env) in self.envs.iter().enumerate() {
            let nav = env.last_nav_output();
            view[[i, 0]] = nav.energy_estimated as f32;
            view[[i, 1]] = nav.dynamic_pressure_estimated as f32;
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
        .expect("invariant: neural_net validated in BatchedSimulation::new");

    let time_since_flip = state.sim_time() - state.guidance_state.last_sign_flip_time_for_nn;
    aerocapture::gnc::guidance::neural::build_nn_input(
        &nav,
        nn.input_mask.as_deref(),
        nn.ablated_input,
        nn.ablated_value,
        data,
        planet,
        target_inclination,
        ref_velocity_latched,
        state.guidance_state.prev_inclination_error_for_nn,
        state.guidance_state.prev_bank_for_nn,
        time_since_flip,
        state.guidance_state.inclination_error_integral,
        state.guidance_state.prev_realized_bank_for_nn,
    )
}

/// Generate a deterministic dispersion draw for a given seed.
///
/// Clones the DispersionConfig with the given seed so each env gets a
/// distinct, reproducible MC scenario. Falls back to the zero draw when
/// no dispersion config is present (nominal runs).
fn draw_from_seed(data: &SimData, seed: u64) -> DispersionDraw {
    match &data.dispersion_config {
        Some(cfg) => {
            let mut seeded = cfg.clone();
            seeded.seed = seed;
            seeded
                .generate_draws(1)
                .into_iter()
                .next()
                .unwrap_or_default()
        }
        None => DispersionDraw::default(),
    }
}

/// Terminal step payload for one env slot.
struct TerminalOutcome {
    ifinal: i32,
    captured: bool,
    ecc: f64,
    dv_m_s: f64,
    peak_heat_flux_kw_m2: f64,
    peak_g_load: f64,
    peak_heat_load_kj_m2: f64,
    violated_constraints: bool,
    truncated: bool,
    final_record: [f64; 52],
    /// Last observation of the terminated episode (pre-reset), for PPO value bootstrap.
    terminal_obs: Vec<f64>,
}
