//! Verify that per-env NnState is independent across BatchedSimulation envs.
//!
//! Phase 0 has only `LayerState::None` (stateless) so the test is structural:
//! two envs constructed from the same NN-scheme config both have `Some(NnState)`
//! with the expected number of layer states. Task 4 guarantees this via
//! `GuidanceState::new(..., data.neural_net.as_ref())` in `build_sim_state`.

mod common;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::data::dispersions::DispersionDraw;
use aerocapture::simulation::init;
use aerocapture::simulation::runner::build_sim_state;

#[test]
fn guidance_state_nn_state_is_per_env() {
    let repo = common::repo_root();
    std::env::set_current_dir(&repo).expect("set cwd to repo root");
    let path = repo.join("configs/test/test_neural_golden.toml");
    let (config, toml_config) = SimInput::from_toml_file(&path).expect("load config");
    let data = SimData::from_toml(&toml_config, &config).expect("build sim data");

    assert!(
        data.neural_net.is_some(),
        "test_neural_golden.toml must produce a loaded neural net model"
    );

    let draw = DispersionDraw::default();
    let run_state_0 = init::init_run_from_draw(&data, &draw);
    let run_state_1 = init::init_run_from_draw(&data, &draw);

    let s0 = build_sim_state(&config, &data, run_state_0, 0);
    let s1 = build_sim_state(&config, &data, run_state_1, 1);

    assert!(
        s0.guidance_state.nn_state.is_some(),
        "env 0 must have nn_state"
    );
    assert!(
        s1.guidance_state.nn_state.is_some(),
        "env 1 must have nn_state"
    );

    let n_layers = data.neural_net.as_ref().unwrap().layers.len();
    assert_eq!(
        s0.guidance_state
            .nn_state
            .as_ref()
            .unwrap()
            .layer_states
            .len(),
        n_layers,
        "env 0 nn_state must have one LayerState per NN layer"
    );
    assert_eq!(
        s1.guidance_state
            .nn_state
            .as_ref()
            .unwrap()
            .layer_states
            .len(),
        n_layers,
        "env 1 nn_state must have one LayerState per NN layer"
    );

    // Sanity: the two NnStates are independent objects (not aliasing).
    let p0 = s0.guidance_state.nn_state.as_ref().unwrap() as *const _;
    let p1 = s1.guidance_state.nn_state.as_ref().unwrap() as *const _;
    assert_ne!(p0, p1, "per-env nn_state must be independent owned copies");
}
