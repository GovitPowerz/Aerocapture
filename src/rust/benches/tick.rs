//! Per-scheme onboard guidance cost: the `dispatch::guidance_step` calls of one
//! nominal flight, replayed.
//!
//! Setup flies each scheme's config once on the undispersed draw and snapshots,
//! before every outer tick, the exact inputs of that tick's guidance call: the
//! navigation output, the pilot-realized bank, the sim time and the pre-call
//! `GuidanceState` (recurrent NN state, FNPAG replan clock, lateral logic). A
//! replay of every snapshot must reproduce the flight's commanded bank bit for
//! bit before anything is timed. One iteration replays the whole flight's
//! guidance calls from clones of the snapshots (cloned outside the timed region),
//! so the reported time is the guidance-only cost of one flight: navigation,
//! pilot and plant integration are excluded, and FNPAG's replan duty cycle is
//! averaged in rather than sampled. The throughput line counts guidance calls:
//! its inverse is the mean cost of one guidance tick.
//!
//! Classical schemes fly their golden test configs (the pinned reference
//! configurations, legacy noise regime); piecewise constant and the two NN rows
//! fly training configs (per_draw regime), the NN rows with the paper's deployed
//! models from the committed bundle (articles/paper/data/runs/headline/).
//!
//! Extra rows come from `AEROCAPTURE_TICK_CONFIGS` (`id=config.toml;...`): the
//! throughput driver passes the four deployed cells that way, their GA parameters
//! routed by the Python deploy rule rather than a second copy of it here.
//!
//! Run: cargo bench --bench tick --manifest-path src/rust/Cargo.toml

use std::collections::HashSet;
use std::hint::black_box;
use std::path::Path;

use aerocapture::config::{SimInput, resolve_toml_bases};
use aerocapture::data::SimData;
use aerocapture::data::dispersions::DispersionDraw;
use aerocapture::gnc::guidance::dispatch::{GuidanceState, guidance_step};
use aerocapture::gnc::navigation::estimator::NavigationOutput;
use aerocapture::simulation::init::init_run_from_draw;
use aerocapture::simulation::runner::{
    SimStateOptions, TermReason, build_event_ctx, build_event_defs, build_sim_state,
};
use aerocapture::simulation::tick::step_one_tick;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

// (bench id, config, NN model pinned over the config's [data] neural_network)
const SCHEMES: &[(&str, &str, Option<&str>)] = &[
    ("ftc", "configs/test/test_ftc_golden.toml", None),
    (
        "equilibrium_glide",
        "configs/test/test_eqglide_golden.toml",
        None,
    ),
    (
        "energy_controller",
        "configs/test/test_energy_ctrl_golden.toml",
        None,
    ),
    ("pred_guid", "configs/test/test_pred_guid_golden.toml", None),
    ("fnpag", "configs/test/test_fnpag_golden.toml", None),
    (
        "piecewise_constant",
        "configs/training/msr_aller_piecewise_constant_train.toml",
        None,
    ),
    (
        "nn_dense_515",
        "configs/training/msr_aller_nn_atan2_best_paper.toml",
        Some("articles/paper/data/runs/headline/dense_p515/best_model.json"),
    ),
    (
        "nn_mamba_962",
        "configs/training/sweep/mamba_p962.toml",
        Some("articles/paper/data/runs/headline/mamba_p962/best_model.json"),
    ),
];

fn extra_rows() -> Vec<(String, String)> {
    std::env::var("AEROCAPTURE_TICK_CONFIGS")
        .unwrap_or_default()
        .split(';')
        .filter(|row| !row.is_empty())
        .map(|row| {
            let (id, config) = row
                .split_once('=')
                .expect("AEROCAPTURE_TICK_CONFIGS rows are id=path");
            (id.to_string(), config.to_string())
        })
        .collect()
}

struct Call {
    nav: NavigationOutput,
    bank: f64,
    time: f64,
    state: GuidanceState,
}

struct Flight {
    input: SimInput,
    data: SimData,
    reference_bank: f64,
    calls: Vec<Call>,
}

impl Flight {
    fn call(&self, c: &Call, state: &mut GuidanceState) -> f64 {
        guidance_step(
            &c.nav,
            c.bank,
            c.time,
            self.reference_bank,
            state,
            &self.data,
            &self.input.planet,
            self.input.reference_trajectory,
            self.input.guidance_type,
        )
        .bank_angle_commanded
    }
}

fn load(config: &str, model: Option<&str>) -> (SimInput, SimData) {
    let path = Path::new(config);
    let root: toml::Value =
        toml::from_str(&std::fs::read_to_string(path).expect("read config")).expect("parse");
    let mut resolved = resolve_toml_bases(root, path, &mut HashSet::new()).expect("bases");
    if let Some(m) = model {
        resolved["data"]["neural_network"] = toml::Value::String(m.to_string());
    }
    let (input, toml_cfg) =
        SimInput::from_toml(&toml::to_string(&resolved).expect("serialize")).expect("config");
    let data = SimData::from_toml(&toml_cfg, &input).expect("data");
    (input, data)
}

fn fly(config: &str, model: Option<&str>) -> Flight {
    let (input, data) = load(config, model);
    let run_state = init_run_from_draw(&data, &DispersionDraw::default());
    let mut sim = build_sim_state(&input, &data, run_state, 0, SimStateOptions::default());
    let (defs, ctx) = (build_event_defs(), build_event_ctx(&input, &data));
    let mut calls = Vec::new();
    let mut flown = Vec::new();
    while sim.term() == TermReason::None {
        let mut state = sim.guidance_state.clone();
        let (bank, time) = (sim.bank_angle(), sim.sim_time());
        step_one_tick(&mut sim, &input, &data, &input.planet, None, &defs, &ctx);
        let nav = sim.last_nav_output();
        // The tick latches the reference velocity into the guidance state just before the call.
        if nav.phase_transition_flag == 1 {
            state.reference_velocity = nav.reference_velocity;
        }
        flown.push(sim.guidance_state.bank_angle_commanded);
        calls.push(Call {
            nav,
            bank,
            time,
            state,
        });
    }
    let flight = Flight {
        reference_bank: sim.reference_bank_angle(),
        input,
        data,
        calls,
    };
    for (c, bank) in flight.calls.iter().zip(&flown) {
        let replayed = flight.call(c, &mut c.state.clone());
        assert_eq!(
            replayed.to_bits(),
            bank.to_bits(),
            "{config}: replay diverged from the flight at t={} s",
            c.time
        );
    }
    flight
}

fn bench_guidance(c: &mut Criterion) {
    std::env::set_current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        .expect("cd to the repo root (configs and data paths are root-relative)");
    let mut g = c.benchmark_group("guidance_per_flight");
    g.sample_size(20);
    let rows = SCHEMES
        .iter()
        .map(|(id, config, model)| {
            (
                id.to_string(),
                config.to_string(),
                model.map(str::to_string),
            )
        })
        .chain(
            extra_rows()
                .into_iter()
                .map(|(id, config)| (id, config, None)),
        );
    for (id, config, model) in rows {
        let flight = fly(&config, model.as_deref());
        eprintln!(
            "{id}: {} guidance calls in the nominal flight of {config}",
            flight.calls.len()
        );
        g.throughput(Throughput::Elements(flight.calls.len() as u64));
        g.bench_function(&id, |b| {
            b.iter_batched_ref(
                || {
                    flight
                        .calls
                        .iter()
                        .map(|c| c.state.clone())
                        .collect::<Vec<_>>()
                },
                |states| {
                    for (c, s) in flight.calls.iter().zip(states.iter_mut()) {
                        black_box(flight.call(black_box(c), s));
                    }
                },
                BatchSize::LargeInput,
            )
        });
    }
    g.finish();
}

criterion_group!(benches, bench_guidance);
criterion_main!(benches);
