//! Entry-fan agreement: every public runner entry point is a draw-source adapter
//! over one fan-out (`run_core`), so the same draw must produce the same numbers
//! whichever adapter delivers it. Exact (`to_bits`) equality on the final record
//! and the echoed dispersions, over the six golden configs (3 dispersed sims each):
//!
//! (a) `run_for_api` == `run_for_api_with_draws` fed the config's own draws;
//! (b) `run_for_api_cell(seed)` == `run_for_api` on a config re-seeded to `seed`
//!     with `n_sims = 1` -- ADR-0004's `run_grid` bit-identity invariant (the cell
//!     is sim_idx 0 on both sides, NOT the k-th sim of the batch, whose legacy noise
//!     stream is seeded `random_seed + k * 10_000`);
//! (c) `run_single_collect` == `run_for_api_with_draws` fed one default draw.
//!
//! Run under the default legacy noise regime and under `noise_seeding = "per_draw"`
//! (ADR-0003: the regime is part of every number).

mod common;

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use aerocapture::data::dispersions::DispersionDraw;
use aerocapture::simulation::runner;
use rstest::rstest;

fn load(name: &str, seed: Option<u64>, n_sims: Option<i32>, per_draw: bool) -> (SimInput, SimData) {
    let repo = common::repo_root();
    std::env::set_current_dir(&repo).expect("set cwd to repo root");
    let path = repo.join("configs/test").join(name);
    let (mut sim_input, mut toml) =
        SimInput::from_toml_file(&path).unwrap_or_else(|e| panic!("load {name}: {e}"));
    let mc = toml
        .monte_carlo
        .as_mut()
        .expect("golden configs declare [monte_carlo]");
    if let Some(s) = seed {
        mc.seed = s;
    }
    if per_draw {
        mc.noise_seeding = Some("per_draw".to_string());
    }
    if let Some(n) = n_sims {
        sim_input.n_sims = n;
    }
    let data =
        SimData::from_toml(&toml, &sim_input).unwrap_or_else(|e| panic!("SimData {name}: {e}"));
    (sim_input, data)
}

fn bits(fr: &[f64]) -> Vec<u64> {
    fr.iter().map(|v| v.to_bits()).collect()
}

fn assert_same(label: &str, a: &aerocapture::RunOutput, b: &aerocapture::RunOutput) {
    assert_eq!(
        bits(&a.final_record),
        bits(&b.final_record),
        "{label}: final_record differs"
    );
    assert_eq!(
        bits(&a.dispersions),
        bits(&b.dispersions),
        "{label}: dispersions differ"
    );
}

#[rstest]
#[case("test_eqglide_golden.toml")]
#[case("test_energy_ctrl_golden.toml")]
#[case("test_pred_guid_golden.toml")]
#[case("test_fnpag_golden.toml")]
#[case("test_ftc_golden.toml")]
#[case("test_neural_golden.toml")]
fn entry_points_agree(#[case] config: &str, #[values(false, true)] per_draw: bool) {
    let (cfg, data) = load(config, None, None, per_draw);
    let regime = if per_draw { "per_draw" } else { "legacy" };
    let n_sims = cfg.n_sims as usize;
    assert!(n_sims > 1, "{config}: the fan-out case needs a batch");

    // (a) the config's own draws through the external-draw adapter.
    let batch = runner::run_for_api(&cfg, &data, false, None).expect("run_for_api");
    let dc = data.dispersion_config.as_ref().expect("dispersions on");
    let draws = dc.generate_draws(n_sims);
    let with_draws =
        runner::run_for_api_with_draws(&cfg, &data, draws, false, None).expect("with_draws");
    assert_eq!(batch.len(), n_sims);
    assert_eq!(with_draws.len(), n_sims);
    for (k, (x, y)) in batch.iter().zip(&with_draws).enumerate() {
        assert_same(&format!("{config} [{regime}] (a) sim {k}"), x, y);
    }

    // (b) one grid cell == the per-seed batch path (monte_carlo.seed = seed, n_sims = 1).
    for seed in [dc.seed, 7, 12_345] {
        let cell = runner::run_for_api_cell(&cfg, &data, seed, false, None).expect("cell");
        let (cfg1, data1) = load(config, Some(seed), Some(1), per_draw);
        let per_seed = runner::run_for_api(&cfg1, &data1, false, None).expect("per-seed");
        assert_eq!(per_seed.len(), 1);
        assert_same(
            &format!("{config} [{regime}] (b) seed {seed}"),
            &cell,
            &per_seed[0],
        );
    }

    // (c) the in-memory default-draw run == the external-draw adapter fed one default draw.
    let collected = runner::run_single_collect(&cfg, &data).expect("run_single_collect");
    let default_run =
        runner::run_for_api_with_draws(&cfg, &data, vec![DispersionDraw::default()], false, None)
            .expect("default draw");
    assert_eq!(
        bits(&collected),
        bits(&default_run[0].final_record),
        "{config} [{regime}] (c) run_single_collect vs default draw"
    );
}
