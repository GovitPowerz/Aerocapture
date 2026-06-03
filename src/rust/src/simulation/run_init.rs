//! Per-env `SimState` construction.
//!
//! `build_sim_state` builds a fresh `SimState` (entry state, GNC subsystem init,
//! per-env RNG seeding, bias-mode `last_nav` priming) without running the
//! simulation loop. Used by the CLI path (`runner::run_single`) and by
//! `BatchedSimulation` to initialize and reset individual RL environments.

use crate::config::SimInput;
use crate::data::SimData;
use crate::gnc::control::pilot::PilotState;
use crate::gnc::guidance::dispatch::GuidanceState;
use crate::gnc::navigation::estimator::{self, NavigationFilter};
use crate::integration::dopri45::Dopri45State;
use crate::integration::sequencer::SequencerState;
use crate::simulation::init;
use crate::simulation::runner::navigate_from_state;
use crate::simulation::sim_types::{BOUNCE_ALT_UNSET, SimState, TermReason};
use std::time::Instant;

/// Construct a fresh `SimState` for env `i` without running the simulation loop.
///
/// Used by `BatchedSimulation` to initialize and reset individual environments.
/// The `sim_idx` is set to `env_idx as i32` for per-env RNG seeding.
pub fn build_sim_state(
    config: &SimInput,
    data: &SimData,
    run_state: init::RunState,
    env_idx: u64,
) -> SimState {
    let planet = &config.planet;
    let req = planet.equatorial_radius;

    let r0 = run_state.entry.state.altitude + req;
    let entry_longitude = run_state.entry.state.longitude;
    let entry_latitude = run_state.entry.state.latitude;
    let entry_velocity = run_state.entry.state.velocity;
    let entry_flight_path = run_state.entry.state.flight_path;
    let entry_azimuth = run_state.entry.state.azimuth;
    let entry_initial_date = run_state.entry.initial_date;
    let entry_initial_bank = run_state.entry.initial_bank;
    let entry_initial_aoa = run_state.entry.initial_aoa;

    let reference_bank_angle = config.reference_bank_angle.to_radians();
    let initial_bank_angle = if config.reference_trajectory {
        reference_bank_angle
    } else {
        entry_initial_bank
    };

    let dt = data.periods.integration;
    let max_time = config.max_time;
    let exit_altitude = data.final_conditions.altitude;

    let nav_filter = match data.nav_mode {
        crate::data::NavMode::Bias => NavigationFilter::new_bias(),
        crate::data::NavMode::Ekf => {
            let nav_toml = data
                .nav_config
                .as_ref()
                .expect("EKF mode requires [navigation] config");
            let (imu_cfg, st_cfg, ekf_cfg) = estimator::build_ekf_configs(nav_toml);
            let seed = config.random_seed as u64 + env_idx * 10_000;
            NavigationFilter::new_ekf(imu_cfg, st_cfg, ekf_cfg, seed)
        }
    };

    let nav_biases = run_state.nav_biases;
    let gm_config = data.density_perturbation.filter(|g| !g.is_disabled());
    let (gm_rng, gm_normal) = if gm_config.is_some() {
        use rand::SeedableRng;
        let rng = rand::rngs::StdRng::seed_from_u64(
            config.random_seed as u64 + env_idx * 10_000 + 0xDE45,
        );
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        (Some(rng), Some(normal))
    } else {
        (None, None)
    };

    let guidance_state = GuidanceState::new(
        entry_initial_bank,
        entry_initial_aoa,
        data.neural_net.as_ref(),
    );
    assert_eq!(
        data.neural_net.is_some(),
        guidance_state.nn_state.is_some(),
        "nn_state presence must match neural_net presence",
    );
    let pilot_state = PilotState {
        bank_angle: initial_bank_angle,
        bank_rate: 0.0,
    };
    let sequencer = SequencerState::new();

    let mut s = SimState {
        state: [
            r0,
            entry_longitude,
            entry_latitude,
            entry_velocity,
            entry_flight_path,
            entry_azimuth,
            0.0,
            entry_initial_date,
        ],
        accumulator: [0.0; 8],
        gill_toggle: 0,
        dopri: Dopri45State::new(),
        bank_angle: initial_bank_angle,
        aoa: entry_initial_aoa,
        bounced: false,
        bounce_alt: BOUNCE_ALT_UNSET,
        bounce_time: 1e30,
        max_heat_flux: 0.0,
        max_load_factor: 0.0,
        max_dyn_pressure: 0.0,
        alt_max_flux: 0.0,
        alt_max_load: 0.0,
        alt_max_pdyn: 0.0,
        time_max_flux: 0.0,
        time_max_load: 0.0,
        time_max_pdyn: 0.0,
        event_records: Vec::new(),
        nav_filter,
        guidance_state,
        pilot_state,
        sequencer,
        sim_time: entry_initial_date,
        term: TermReason::None,
        step: 0,
        first_iter: true,
        run_state,
        nav_biases,
        supervised_trace: Vec::new(),
        photo_lines: Vec::new(),
        cumulative_bank_change_deg: 0.0,
        dynamic_pressure_for_photo: 0.0,
        density_estimate_for_photo: 0.0,
        guidance_phase_for_photo: 1,
        gm_config,
        gm_rng,
        gm_normal,
        last_nav: crate::gnc::navigation::estimator::NavigationOutput::default(),
        dt,
        max_time,
        exit_altitude,
        reference_bank_angle,
        write_photo: false,
        sim_idx: env_idx as i32,
        wall_timeout: None,
        wall_start: Instant::now(),
        is_single: false,
    };

    // Prime last_nav so the RL env's reset() returns a valid initial observation
    // instead of a zeroed-out NavigationOutput. Bias mode is stateless (the call is
    // a pure function of the truth state + biases), so priming costs nothing. EKF
    // mode advances the filter via `ekf.predict(nav_dt, ...)` on every call; since
    // tick.rs also navigates on first_iter, priming there would predict the filter
    // twice before any physics advance. Skip priming for EKF; the first tick will
    // populate `last_nav` before the policy's second action. The initial RL action
    // (step 0) is based on a default NavigationOutput under EKF mode.
    // CLI path: benign — at nominal entry (>100 km) density_gain is force-reset,
    // bounce_flag/phase are unchanged, and last_nav is overwritten on the first tick.
    if matches!(data.nav_mode, crate::data::NavMode::Bias) {
        s.last_nav = navigate_from_state(&mut s, data, planet);
    }
    s
}
