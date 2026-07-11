//! The RL env injects its policy action via `forced_bank`; the NN-input
//! telemetry (prev commanded bank, sign-flip clock) must track that action,
//! not the discarded internal guidance command -- otherwise the policy's
//! observation of its own previous action is wrong during training and
//! right at deploy (train/deploy mismatch on inputs 22/27/28).

mod common;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::data::dispersions::DispersionDraw;
use aerocapture::simulation::init;
use aerocapture::simulation::runner::{build_event_ctx, build_event_defs, build_sim_state};
use aerocapture::simulation::tick::step_one_tick;

#[test]
fn forced_bank_drives_nn_telemetry() {
    let repo = common::repo_root();
    std::env::set_current_dir(&repo).expect("set cwd to repo root");
    let path = repo.join("configs/test/test_neural_golden.toml");
    let (config, toml_config) = SimInput::from_toml_file(&path).expect("load config");
    let data = SimData::from_toml(&toml_config, &config).expect("build sim data");

    let draw = DispersionDraw::default();
    let run_state = init::init_run_from_draw(&data, &draw);
    let mut state = build_sim_state(&config, &data, run_state, 0);
    let event_defs = build_event_defs();
    let event_ctx = build_event_ctx(&config, &data);

    let forced = 0.7_f64;
    step_one_tick(
        &mut state,
        &config,
        &data,
        &config.planet,
        Some(forced),
        &event_defs,
        &event_ctx,
    );

    assert_eq!(
        state.guidance_state.prev_bank_for_nn, forced,
        "prev_bank_for_nn must track the forced RL action, not the internal guidance command"
    );

    // A forced sign flip must stamp the sign-flip clock.
    let t_before = state.guidance_state.last_sign_flip_time_for_nn;
    step_one_tick(
        &mut state,
        &config,
        &data,
        &config.planet,
        Some(-forced),
        &event_defs,
        &event_ctx,
    );
    assert_eq!(state.guidance_state.prev_bank_for_nn, -forced);
    assert!(
        state.guidance_state.last_sign_flip_time_for_nn > t_before,
        "sign-flip clock must advance on a forced sign reversal"
    );
}
