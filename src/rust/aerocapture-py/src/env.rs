//! BatchedSimulation: per-step vectorized env for RL training.
//!
//! Holds N independent SimStates sharing one Arc<SimData>. step() advances
//! each env one outer guidance tick via Rayon, auto-resets on done, and
//! returns the stacked (obs, reward, done, info, aux) payload. For an
//! auto-reset env, obs and aux rows are the new episode's s_0; the ended
//! episode's s_T is in info["terminal_observation"] / info["terminal_aux"].
//!
//! `auto_reset=False` (fixed-pool consumers: the world-model plant) freezes a
//! done slot instead: no new seed is drawn, later step() calls skip its physics,
//! its obs / aux rows stay the terminal state (the ending step's info payload),
//! `done` stays True and `info` is empty, until `reset(seeds)` re-seeds it.
//!
//! Observation timing matches deploy: every env is held at a tick's sense point
//! (`tick::sense_tick` done, `last_nav` = nav(t_k)). step(a_k) runs
//! `tick::act_tick(Some(a_k))` then the next tick's `sense_tick`, so the returned
//! obs is nav(t_{k+1}), the input the deployed NN reads when choosing a_{k+1}.

use aerocapture::gnc::guidance::neural::{NnInputContext, NnModelView, build_nn_input};
use std::path::Path;
use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use aerocapture::config::{GuidanceType, SimInput};
use aerocapture::data::SimData;
use aerocapture::data::guidance_params::NeuralNetMode;
use aerocapture::integration::events::{EventContext, EventDef};
use aerocapture::orbit::{elements, maneuver};
use aerocapture::simulation::final_record::{
    FINAL_RECORD_LEN, FR_DV_TOTAL_MS, FR_ECC, FR_ENERGY_MJKG, FR_G_LOAD, FR_HEAT_FLUX_KW_M2,
    FR_HEAT_LOAD_MJM2,
};
use aerocapture::simulation::runner::{
    SimState, SimStateOptions, TermReason, build_final_record, build_sim_state, ifinal_for,
};
use aerocapture::simulation::tick;

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
/// Per-env step result: (done, terminal payload, aux row).
type StepOutcome = (bool, Option<TerminalOutcome>, [f64; AUX_WIDTH]);

/// Width of the per-env auxiliary array returned by `reset` / `step`:
/// `[energy_estimated, dynamic_pressure_estimated, predicted_dv1, predicted_dv2,
/// predicted_dv3, heat_flux_fraction, heat_load_fraction]`. The thermal fractions
/// are delivered RAW (fraction of the configured limit) so the RL reward never has
/// to invert the NN input normalization (which is model-dependent).
pub const AUX_WIDTH: usize = 7;

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
    auto_reset: bool,
    /// Under `auto_reset=false`, a done slot's terminal (obs, aux) rows, copied
    /// out on every later step instead of rebuilt from the frozen state.
    frozen: Vec<Option<(Vec<f64>, [f64; AUX_WIDTH])>>,
    episode_counter: Vec<u64>,
    episode_ids: Vec<u64>,
    step_counts: Vec<u64>,
    event_defs: Vec<EventDef>,
    event_ctx: EventContext,
}

#[pymethods]
impl BatchedSimulation {
    #[new]
    #[pyo3(signature = (toml_path, n_envs, overrides=None, seed_base=3_000_000, auto_reset=true))]
    fn new(
        toml_path: &str,
        n_envs: usize,
        overrides: Option<&Bound<'_, PyDict>>,
        seed_base: u64,
        auto_reset: bool,
    ) -> PyResult<Self> {
        if n_envs == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_envs must be > 0",
            ));
        }
        let overrides = extract_overrides(overrides)?;
        let (sim_input, sim_data) = config::load_and_override(Path::new(toml_path), &overrides)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        // The action replaces the NN forward pass inside guidance: under any other
        // scheme it would be silently ignored, and under magnitude_only it would
        // lose its sign and go unused in the exit phase.
        if sim_input.guidance_type != GuidanceType::NeuralNetwork
            || sim_data.guidance.neural_network.mode != NeuralNetMode::FullNeural
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "RL env requires guidance.type = \"neural_network\" in full_neural mode",
            ));
        }

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
            envs.push(fresh_env(&sim_input, &sim_data, seed));
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
            auto_reset,
            frozen: vec![None; n_envs],
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
            self.envs[i] = fresh_env(&self.sim_input, &self.sim_data, seed);
            self.episode_ids[i] = seed;
            self.step_counts[i] = 0;
            self.frozen[i] = None;
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
        let frozen = &self.frozen;

        // Advance all envs one tick in parallel; collect terminal info where done.
        // The aux is captured BEFORE auto-reset: a done env's row is its terminal
        // aux until the reset below replaces it.
        // Release the GIL during the Rayon block so other Python threads can run
        // and Ctrl-C is responsive.
        let mut outcomes: Vec<StepOutcome> = py.detach(|| {
            self.envs
                .par_iter_mut()
                .zip(actions_vec.par_iter())
                .zip(frozen.par_iter())
                .map(|((state, &action), frozen)| {
                    // A frozen slot (done under auto_reset=false) is left untouched:
                    // its rows repeat the terminal state, its payload was delivered once.
                    if let Some((_, aux)) = frozen {
                        return (true, None, *aux);
                    }
                    let bank = action.clamp(-std::f64::consts::PI, std::f64::consts::PI);
                    tick::act_tick(
                        state,
                        sim_input,
                        sim_data,
                        &sim_input.planet,
                        Some(bank),
                        event_defs,
                        event_ctx,
                    );
                    // Sense the post-integration state: nav(t_{k+1}) is the next
                    // observation. Terminal states are sensed too, so terminal_obs
                    // and the terminal aux (the PBRS Phi(s_T)) describe the state
                    // the episode ended in; a non-finite (NaN-crash) state keeps
                    // the previous tick's navigation instead of poisoning both.
                    if state.physics_state().iter().all(|x| x.is_finite()) {
                        tick::sense_tick(state, sim_input, sim_data, &sim_input.planet);
                    }
                    let aux = aux_for_env(state, sim_data, sim_input);
                    if state.term() != TermReason::None {
                        // Capture terminal obs BEFORE the env state is reset.
                        let terminal_obs = build_obs_for_env(state, sim_data, sim_input);
                        let fr = build_final_record(state, sim_data, &sim_input.planet);
                        // Guarded by the enclosing `if state.term() != TermReason::None`;
                        // ifinal_for's None arm (unreachable!) cannot fire here.
                        let ifinal = ifinal_for(state.term());
                        let ecc = fr[FR_ECC];
                        let energy = fr[FR_ENERGY_MJKG]; // MJ/kg; negative = captured
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
                            dv_m_s: fr[FR_DV_TOTAL_MS],
                            peak_heat_flux_kw_m2: fr[FR_HEAT_FLUX_KW_M2],
                            peak_g_load: fr[FR_G_LOAD],
                            peak_heat_load_kj_m2: fr[FR_HEAT_LOAD_MJM2] * 1e3, // MJ/m2 -> kJ/m2
                            violated_constraints: violated,
                            truncated,
                            final_record: fr,
                            terminal_obs,
                            terminal_aux: aux,
                        };
                        (true, Some(term), aux)
                    } else {
                        (false, None, aux)
                    }
                })
                .collect()
        });

        // Auto-reset terminated envs with advancing seeds. The returned aux must
        // describe the returned obs, so a reset env's row becomes its new
        // episode's s_0 aux; the terminal aux travels in info["terminal_aux"].
        // Without auto-reset the slot keeps its terminal state and seed.
        for (i, (done, term, aux)) in outcomes.iter_mut().enumerate() {
            if !*done {
                self.step_counts[i] += 1;
            } else if !self.auto_reset {
                if let Some(t) = term {
                    self.frozen[i] = Some((t.terminal_obs.clone(), *aux));
                }
            } else {
                self.episode_counter[i] += self.n_envs as u64;
                let seed = self.seed_base + self.episode_counter[i];
                self.envs[i] = fresh_env(&self.sim_input, &self.sim_data, seed);
                self.episode_ids[i] = seed;
                self.step_counts[i] = 0;
                *aux = aux_for_env(&self.envs[i], &self.sim_data, &self.sim_input);
            }
        }

        // Build return arrays.
        let obs = self.build_obs(py);
        let reward_arr = PyArray1::<f32>::from_iter(py, outcomes.iter().map(|_| 0.0f32));
        let done_arr = PyArray1::<bool>::from_iter(py, outcomes.iter().map(|(d, _, _)| *d));

        // Aux array: (n_envs, AUX_WIDTH), see `AUX_WIDTH` for the column order.
        // Row i matches obs row i (a reset env's row is its new episode's s_0).
        let aux = PyArray2::<f32>::zeros(py, [self.n_envs, AUX_WIDTH], false);
        {
            let mut aux_view = unsafe { aux.as_array_mut() };
            for (i, (_, _, a)) in outcomes.iter().enumerate() {
                for j in 0..AUX_WIDTH {
                    aux_view[[i, j]] = a[j] as f32;
                }
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
                // Pre-reset aux of the terminated episode; PBRS needs this for Phi(s_T).
                let term_aux: Vec<f32> = t.terminal_aux.iter().map(|&v| v as f32).collect();
                dict.set_item("terminal_aux", term_aux)?;
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
            let built;
            let obs = match &self.frozen[i] {
                Some((obs, _)) => obs,
                None => {
                    built = build_obs_for_env(env, &self.sim_data, &self.sim_input);
                    &built
                }
            };
            for (j, &v) in obs.iter().enumerate() {
                view[[i, j]] = v as f32;
            }
        }
        arr
    }

    /// Auxiliary array (n_envs, AUX_WIDTH): [energy_estimated, dynamic_pressure_estimated,
    /// predicted_dv1, predicted_dv2, predicted_dv3, heat_flux_fraction, heat_load_fraction]
    /// per env. The 3 DV components are the raw m/s correction-budget estimate consumed
    /// by the DV-reward potential; the thermal fractions are raw (fraction of limit).
    fn build_aux<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let arr = PyArray2::<f32>::zeros(py, [self.n_envs, AUX_WIDTH], false);
        let mut view = unsafe { arr.as_array_mut() };
        for (i, env) in self.envs.iter().enumerate() {
            let aux = aux_for_env(env, &self.sim_data, &self.sim_input);
            for j in 0..AUX_WIDTH {
                view[[i, j]] = aux[j] as f32;
            }
        }
        arr
    }
}

/// One env's aux row from its current navigation, see `AUX_WIDTH` for the columns.
fn aux_for_env(state: &SimState, data: &Arc<SimData>, config: &SimInput) -> [f64; AUX_WIDTH] {
    let nav = state.last_nav_output();
    let dv = predicted_dv_for_state(state, data, config);
    [
        nav.energy_estimated,
        nav.dynamic_pressure_estimated,
        dv[0],
        dv[1],
        dv[2],
        nav.heat_flux_fraction,
        nav.heat_load_fraction,
    ]
}

/// A new episode for `seed`, sensed at its first tick: `last_nav` = nav(t_0),
/// the navigation the deployed NN reads on tick 0.
fn fresh_env(sim_input: &SimInput, sim_data: &SimData, seed: u64) -> SimState {
    let draw = sim_data.draw_from_seed(seed);
    let run_state = aerocapture::simulation::init::init_run_from_draw(sim_data, &draw);
    let mut state = build_sim_state(
        sim_input,
        sim_data,
        run_state,
        seed,
        SimStateOptions::default(),
    );
    tick::sense_tick(&mut state, sim_input, sim_data, &sim_input.planet);
    state
}

/// Build the observation vector for a single env state.
///
/// Delegates to `build_nn_input` so the RL observation matches exactly what the
/// runtime NN guidance sees. Panics if no neural_net model is loaded -- callers
/// must use a config with `[data] neural_network` set.
fn build_obs_for_env(state: &SimState, data: &Arc<SimData>, config: &SimInput) -> Vec<f64> {
    let nav = state.last_nav_output();
    let planet = &config.planet;
    let nn = data
        .neural_net
        .as_ref()
        .expect("invariant: neural_net validated in BatchedSimulation::new");
    let ctx = NnInputContext::from_guidance_state(
        &state.guidance_state,
        state.sim_time(),
        data.target_orbit.inclination,
    );
    build_nn_input(&nav, NnModelView::of(nn), data, planet, &ctx)
}

/// Predicted correction delta-v [dv1, dv2, dv3] (raw m/s) on the current
/// osculating orbit for one env. Mirrors `build_obs_for_env`'s orbit
/// construction so the aux DV equals candidate inputs 32-34 that
/// `build_nn_input` produces (pre-normalization). Consumed by the DV-reward
/// potential (`StepRewardCalculator`, potential = "dv").
fn predicted_dv_for_state(state: &SimState, data: &Arc<SimData>, config: &SimInput) -> [f64; 3] {
    let nav = state.last_nav_output();
    let orbit = elements::from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &config.planet,
    );
    maneuver::predicted_dv_for_nn(
        &orbit,
        &data.target_orbit,
        &data.parking_orbit,
        &config.planet,
    )
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
    final_record: [f64; FINAL_RECORD_LEN],
    /// Last observation of the terminated episode (pre-reset), for PPO value bootstrap.
    terminal_obs: Vec<f64>,
    /// Aux row of `terminal_obs` (pre-reset), for the PBRS Phi(s_T).
    terminal_aux: [f64; AUX_WIDTH],
}
