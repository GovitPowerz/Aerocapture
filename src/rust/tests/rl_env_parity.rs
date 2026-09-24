//! The RL env flies exactly the deployed NN. A policy that runs the model on the
//! observation `BatchedSimulation` would return (built from `last_nav_output()` after
//! `sense_tick`, as `env.rs::build_obs_for_env` does) and injects the action through
//! `act_tick` must reproduce the deploy loop (`step_one_tick(None)`) bit for bit.
//! A one-tick observation lag, or an action injected after the command shaper
//! (so shaping and the prev-bank telemetry see the raw action), breaks it.

mod common;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::data::nn_state::NnState;
use aerocapture::gnc::guidance::neural::{NnInputContext, nn_bank_angle};
use aerocapture::simulation::init;
use aerocapture::simulation::runner::{
    SimState, SimStateOptions, TermReason, build_event_ctx, build_event_defs, build_final_record,
    build_sim_state,
};
use aerocapture::simulation::tick::{act_tick, sense_tick, step_one_tick};

fn fresh(config: &SimInput, data: &SimData, seed: u64) -> SimState {
    let draw = data.draw_from_seed(seed);
    let run_state = init::init_run_from_draw(data, &draw);
    build_sim_state(config, data, run_state, seed, SimStateOptions::default())
}

fn assert_env_matches_deploy(config_name: &str) {
    let repo = common::repo_root();
    std::env::set_current_dir(&repo).expect("set cwd to repo root");
    let (config, toml_config) =
        SimInput::from_toml_file(&repo.join("configs/test").join(config_name))
            .expect("load config");
    let data = SimData::from_toml(&toml_config, &config).expect("build sim data");
    let nn = data.neural_net.as_ref().expect("config loads an NN");
    let planet = &config.planet;
    let (defs, ctx) = (build_event_defs(), build_event_ctx(&config, &data));

    for seed in [3_000_000_u64, 3_000_001, 3_000_002] {
        let mut deploy = fresh(&config, &data, seed);
        let mut deploy_banks = Vec::new();
        while deploy.term() == TermReason::None {
            step_one_tick(&mut deploy, &config, &data, planet, None, &defs, &ctx);
            deploy_banks.push(deploy.guidance_state.prev_bank_for_nn);
        }

        let mut env = fresh(&config, &data, seed);
        let mut policy_state = NnState::for_model(nn);
        let mut env_banks = Vec::new();
        sense_tick(&mut env, &config, &data, planet);
        loop {
            let nn_ctx = NnInputContext::from_guidance_state(
                &env.guidance_state,
                env.sim_time(),
                data.target_orbit.inclination,
            );
            // Same input build as `env.rs::build_obs_for_env`, decoded per the
            // model's own `output_param`.
            let action = nn_bank_angle(
                &env.last_nav_output(),
                nn,
                &mut policy_state,
                &data,
                planet,
                &nn_ctx,
            );
            act_tick(&mut env, &config, &data, planet, Some(action), &defs, &ctx);
            env_banks.push(env.guidance_state.prev_bank_for_nn);
            if env.term() != TermReason::None {
                break;
            }
            sense_tick(&mut env, &config, &data, planet);
        }

        if let Some(k) = (0..deploy_banks.len().min(env_banks.len()))
            .find(|&k| deploy_banks[k].to_bits() != env_banks[k].to_bits())
        {
            panic!(
                "{config_name} seed {seed}: commanded bank diverges at tick {k}: deploy {} env {}",
                deploy_banks[k], env_banks[k]
            );
        }
        assert_eq!(
            deploy_banks.len(),
            env_banks.len(),
            "{config_name} seed {seed}: tick count"
        );
        let (fd, fe) = (
            build_final_record(&deploy, &data, planet),
            build_final_record(&env, &data, planet),
        );
        assert!(
            fd.iter()
                .zip(fe.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{config_name} seed {seed}: final records differ"
        );
    }
}

#[test]
fn env_matches_deploy_bias_nav() {
    assert_env_matches_deploy("test_rl_env_parity.toml");
}

#[test]
fn env_matches_deploy_ekf_nav() {
    assert_env_matches_deploy("test_rl_env_parity_ekf.toml");
}
