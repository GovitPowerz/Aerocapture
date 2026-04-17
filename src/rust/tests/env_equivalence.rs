//! Equivalence: stepping a `SimState` via `step_one_tick` in reference-trajectory
//! mode must produce a bit-identical final record to `run_single_collect` on the
//! same config.
//!
//! Reference-trajectory mode (`config.reference_trajectory = true`) bypasses the
//! full GNC chain: bank angle is held constant at `reference_bank_angle` and no
//! navigation filter runs. This makes the two paths trivially comparable without
//! worrying about pilot dynamics or guidance divergence.

mod common;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::data::dispersions::DispersionDraw;
use aerocapture::simulation::{init, runner, tick};

fn load() -> (SimInput, SimData) {
    let repo = common::repo_root();
    // Data file paths in the TOML are relative to repo root — set cwd so they resolve.
    std::env::set_current_dir(&repo).expect("set cwd to repo root");
    let path = repo.join("configs/test/test_ref_orig.toml");
    let (sim_input, toml_config) =
        SimInput::from_toml_file(&path).unwrap_or_else(|e| panic!("Failed to load config: {}", e));
    let sim_data = SimData::from_toml(&toml_config, &sim_input)
        .unwrap_or_else(|e| panic!("Failed to build SimData: {}", e));
    (sim_input, sim_data)
}

/// `step_one_tick` loop must reproduce `run_single_collect` bit-identically
/// for a reference-trajectory (constant bank) run with default dispersions.
#[test]
fn step_matches_run_single_collect_reference_trajectory() {
    let (config, data) = load();

    // Sanity: the test config uses reference_trajectory mode.
    assert!(
        config.reference_trajectory,
        "test_ref_orig.toml must have reference_trajectory = true"
    );

    // Reference path: full run via run_single_collect.
    let ref_final = runner::run_single_collect(&config, &data).unwrap();

    // Step path: manual loop via step_one_tick.
    let planet = config.planet.clone();
    let draw = DispersionDraw::default();
    let run_state = init::init_run_from_draw(&data, &draw);
    let mut state = runner::build_sim_state(&config, &data, run_state, 0);

    let event_defs = runner::build_event_defs();
    let event_ctx = runner::build_event_ctx(&config, &data);

    loop {
        let out = tick::step_one_tick(
            &mut state,
            &config,
            &data,
            &planet,
            None,
            &event_defs,
            &event_ctx,
        );
        if out.done {
            break;
        }
    }

    let step_final = runner::build_final_record(&state, &data, &planet);

    for i in 0..52 {
        let diff = (ref_final[i] - step_final[i]).abs();
        assert!(
            diff < 1e-9,
            "final_record[{}] mismatch: run_single={:.15e} step={:.15e} diff={:.3e}",
            i,
            ref_final[i],
            step_final[i],
            diff,
        );
    }
}
