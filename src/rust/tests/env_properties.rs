//! Property tests for `step_one_tick`.
//!
//! Sweeps (seed, action_rad, n_steps) to verify that the state vector
//! remains finite after any sequence of steps, regardless of bank command.

mod common;

use proptest::prelude::*;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::data::dispersions::DispersionDraw;
use aerocapture::simulation::{init, runner, tick};

fn load() -> (SimInput, SimData) {
    let repo = common::repo_root();
    std::env::set_current_dir(&repo).expect("set cwd to repo root");
    let path = repo.join("configs/test/test_ref_orig.toml");
    let (sim_input, toml_config) =
        SimInput::from_toml_file(&path).unwrap_or_else(|e| panic!("Failed to load config: {}", e));
    let sim_data = SimData::from_toml(&toml_config, &sim_input)
        .unwrap_or_else(|e| panic!("Failed to build SimData: {}", e));
    (sim_input, sim_data)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// State vector must stay finite under any (seed, action, n_steps) triple.
    ///
    /// Uses reference_trajectory mode (from test_ref_orig.toml) so the forced_bank
    /// argument has no effect on guidance but the integrator still runs a full tick.
    /// The important invariant is: `step_one_tick` never produces NaN/Inf in the
    /// physics state regardless of what bank command is passed.
    #[test]
    fn step_returns_finite_state_for_any_action_and_seed(
        seed in 0u64..10_000u64,
        action_rad in -std::f64::consts::PI..std::f64::consts::PI,
        n_steps in 1usize..50usize,
    ) {
        let (config, data) = load();
        let planet = config.planet.clone();

        // Use seed as env_idx — dispersions are off for this config but the
        // RNG seeding path still exercises the build_sim_state init code.
        let draw = DispersionDraw::default();
        let run_state = init::init_run_from_draw(&data, &draw);
        let mut state = runner::build_sim_state(&config, &data, run_state, seed);

        let event_defs = runner::build_event_defs();
        let event_ctx = runner::build_event_ctx(&config, &data);

        for _ in 0..n_steps {
            let out = tick::step_one_tick(
                &mut state,
                &config,
                &data,
                &planet,
                Some(action_rad),
                &event_defs,
                &event_ctx,
            );
            let phys = state.physics_state();
            prop_assert!(
                phys.iter().all(|v: &f64| v.is_finite()),
                "state contains non-finite value after tick: {:?}",
                phys,
            );
            if out.done {
                break;
            }
        }
    }
}
