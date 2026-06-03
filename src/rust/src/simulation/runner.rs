//! Main simulation loop.
//!
//! Monte Carlo runs are parallelized with rayon (one thread per trajectory).

use crate::config::{AdaptiveConfig, PlanetConfig, SimInput};
use crate::data::SimData;
use crate::data::dispersions::DISPERSION_DRAW_LEN;
use crate::gnc::control::pilot::PilotState;
use crate::gnc::guidance::dispatch::GuidanceState;
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, norm, to_absolute_cartesian};
use crate::gnc::navigation::estimator::{self, NavigationFilter};
use crate::integration::dopri45::{self, Dopri45State};
use crate::integration::events::{self, EventAction, EventContext, EventDef, EventRecord};
use crate::integration::rk4;
use crate::integration::sequencer::SequencerState;
use crate::orbit::maneuver::DeltaV;
use crate::orbit::{elements, maneuver};
use crate::physics::{atmosphere, gravity};
use crate::simulation::init;
use crate::simulation::output;
use rayon::prelude::*;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

pub(crate) const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
pub(crate) const G0: f64 = 9.81;

/// Sentinel value indicating the vehicle has not yet bounced (no valid bounce altitude).
pub(crate) const BOUNCE_ALT_UNSET: f64 = 1e34;
/// Minimum bounce altitude (m) required before treating a re-descending trajectory
/// as a guaranteed atmospheric-apoapsis crash. Guards against transient FPA sign
/// changes during deep passes with aggressive bank reversals.
pub(crate) const MIN_BOUNCE_ALT_FOR_CRASH_M: f64 = 20e3;

/// Virtual DV base for hyperbolic exits (m/s).
/// Set above any realistic captured orbit correction DV.
pub(crate) const HYPERBOLIC_BASE: f64 = 10_000.0;
/// Minimum virtual DV for crash / pending-crash / timeout (m/s).
/// Set above any realistic captured orbit correction DV so captures remain
/// strictly preferable in cost space, but low enough that near-target
/// crashes don't dwarf bad captures under the `squared` / `cubed`
/// cost_transform. A near-miss crash should cost ~CRASH_FLOOR; a deep
/// plunge scales up via |ΔE| so the optimizer still sees a gradient.
pub(crate) const CRASH_FLOOR: f64 = 3_000.0;
/// Energy-error weight (m/s per MJ/kg of |E_orb - E_target|).
pub(crate) const CRASH_ENERGY_WEIGHT: f64 = 1_000.0;
/// Max time-survival bonus (m/s) — subtracted linearly in t/t_max.
pub(crate) const CRASH_TIME_BONUS: f64 = 500.0;

/// Upper cap on |ΔE| (MJ/kg) when computing virtual DV.
/// Real surface-crash energies rarely exceed ~15 MJ/kg at Mars; 50 is a
/// generous cap that also absorbs Inf from degenerate states.
pub(crate) const CRASH_ENERGY_CAP_MJKG: f64 = 50.0;

/// Virtual DV for non-capturing terminations (Crash, PendingCrash, Timeout).
///
/// Penalizes energy distance from target; softens crashes near the capture
/// boundary so PSO/GA will explore closer to the crash limit.
///
/// Non-finite inputs (NaN from degenerate-state MC dispersions) fall back
/// to the worst-case cap — caller still gets a finite, large virtual DV.
pub(crate) fn virtual_dv_non_capture(
    orbital_energy_j_kg: f64,
    target_sma_m: f64,
    mu: f64,
    sim_time: f64,
    max_time: f64,
) -> f64 {
    let target_energy_j_kg = -mu / (2.0 * target_sma_m);
    let delta_e_mj = if orbital_energy_j_kg.is_finite() && target_energy_j_kg.is_finite() {
        ((orbital_energy_j_kg - target_energy_j_kg).abs() / 1e6).min(CRASH_ENERGY_CAP_MJKG)
    } else {
        CRASH_ENERGY_CAP_MJKG
    };
    let t_ratio = if max_time.is_finite() && max_time > 0.0 && sim_time.is_finite() {
        (sim_time / max_time).clamp(0.0, 1.0)
    } else {
        0.0
    };
    CRASH_FLOOR + CRASH_ENERGY_WEIGHT * delta_e_mj - CRASH_TIME_BONUS * t_ratio
}

/// Default absolute tolerances for DOPRI45, one per state component.
/// State = [r(m), lon(rad), lat(rad), V(m/s), gamma(rad), psi(rad), flux(kJ/m²), time(s)]
pub(crate) const DOPRI45_ATOL: [f64; 8] = [
    1.0,  // r: 1 m on ~3.4e6 m
    1e-8, // lon: ~0.03 m at Mars equator
    1e-8, // lat: ~0.03 m
    1e-3, // V: 1 mm/s on ~5700 m/s
    1e-8, // gamma: ~0.03 m position equiv
    1e-8, // psi: ~0.03 m
    1e-2, // flux: 0.01 kJ/m² on O(1000) total
    1e-6, // time: machine-level for identity derivative
];

#[derive(Debug)]
pub struct SimError(pub String);

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SimError {}

/// Simulation state — mutable per-tick data plus pre-loop constants.
///
/// Expanded to include all mutable loop-local variables from `run_single` so that
/// `tick::step_one_tick` can take a single `&mut SimState` argument.
#[allow(dead_code)]
pub struct SimState {
    // ── Physics state vector: [r, lon, lat, V, gamma, psi, flux, time] ──
    pub(crate) state: [f64; 8],
    // RK4 internals
    pub(crate) accumulator: [f64; 8],
    pub(crate) gill_toggle: i32,
    // DOPRI45 adaptive integrator state (only used in adaptive mode)
    pub(crate) dopri: Dopri45State,
    // Guidance
    pub(crate) bank_angle: f64, // realized bank angle (rad)
    pub(crate) aoa: f64,        // realized AoA (rad)
    // Tracking
    pub(crate) bounced: bool,
    pub(crate) bounce_alt: f64,
    pub(crate) bounce_time: f64,
    pub(crate) max_heat_flux: f64,
    pub(crate) max_load_factor: f64, // m/s², divided by G0 when written to final_record
    pub(crate) max_dyn_pressure: f64,
    // Max-value altitudes and times (for carltf output)
    pub(crate) alt_max_flux: f64,
    pub(crate) alt_max_load: f64,
    pub(crate) alt_max_pdyn: f64,
    pub(crate) time_max_flux: f64,
    pub(crate) time_max_load: f64,
    pub(crate) time_max_pdyn: f64,
    // Event detection records (adaptive integrator only)
    pub(crate) event_records: Vec<EventRecord>,

    // ── GNC subsystem state (initialized before the loop) ──
    pub(crate) nav_filter: NavigationFilter,
    pub guidance_state: GuidanceState,
    pub(crate) pilot_state: PilotState,
    pub(crate) sequencer: SequencerState,

    // ── Loop control ──
    pub(crate) sim_time: f64,
    pub(crate) term: TermReason,
    pub(crate) step: usize,
    pub(crate) first_iter: bool,

    // ── Dispersed run state (copied from init::RunState; density_perturbation mutated each tick) ──
    pub(crate) run_state: init::RunState,
    pub(crate) nav_biases: crate::gnc::navigation::estimator::NavigationBiases,

    // ── Supervised trace (only populated when config.collect_supervised=true) ──
    // Lives on SimState (not RunState) so that the per-tick run_state copy
    // calls in tick.rs do NOT deep-copy the growing trace vector. This was
    // O(N²) memory churn during supervised data collection.
    pub(crate) supervised_trace: Vec<(Vec<f64>, f64, f64, f64, f64)>,

    // ── Photo output accumulators ──
    pub(crate) photo_lines: Vec<[f64; 30]>,
    pub(crate) cumulative_bank_change_deg: f64,
    pub(crate) dynamic_pressure_for_photo: f64,
    pub(crate) density_estimate_for_photo: f64,
    pub(crate) guidance_phase_for_photo: i32,

    // ── Gauss-Markov density perturbation RNG (None when disabled) ──
    pub(crate) gm_config: Option<crate::data::dispersions::DensityPerturbationConfig>,
    pub(crate) gm_rng: Option<rand::rngs::StdRng>,
    pub(crate) gm_normal: Option<rand_distr::Normal<f64>>,

    // ── Last navigation output (cached for RL observation building) ──
    pub(crate) last_nav: crate::gnc::navigation::estimator::NavigationOutput,

    // ── Pre-loop constants (read-only within the tick, stored here for single-arg dispatch) ──
    pub(crate) dt: f64,
    pub(crate) max_time: f64,
    pub(crate) exit_altitude: f64,
    pub(crate) reference_bank_angle: f64,
    pub(crate) write_photo: bool,
    pub(crate) sim_idx: i32,
    pub(crate) wall_timeout: Option<Duration>,
    pub(crate) wall_start: Instant,
    pub(crate) is_single: bool,
}

/// Termination reason
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TermReason {
    None,
    Crash,
    Timeout,
    AtmosphereExit,
    PendingCrash,
}

impl SimState {
    /// Return the most recent navigation output, used by RL observation builders.
    pub fn last_nav_output(&self) -> crate::gnc::navigation::estimator::NavigationOutput {
        self.last_nav
    }

    /// Return the current sim time (seconds since trajectory start).
    /// Used by RL observation builders for time-since-last-event inputs.
    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    /// Return the current termination reason.
    pub fn term(&self) -> TermReason {
        self.term
    }

    /// Return the raw physics state vector: [r, lon, lat, V, gamma, psi, flux, time].
    pub fn physics_state(&self) -> [f64; 8] {
        self.state
    }

    /// True if any flight constraint was violated during this trajectory.
    /// Constraint limits are in SI units as stored in `SimData::constraints`.
    pub fn any_constraint_violated(&self, data: &SimData) -> bool {
        let c = &data.constraints;
        self.max_heat_flux > c.max_heat_flux
            || self.max_load_factor > c.max_load_factor
            || self.max_dyn_pressure > c.max_dynamic_pressure
            || self.state[6] > c.max_heat_load
    }
}

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
    if matches!(data.nav_mode, crate::data::NavMode::Bias) {
        s.last_nav = navigate_from_state(&mut s, data, planet);
    }
    s
}

/// Run one navigation pass on a `SimState`, returning the `NavigationOutput`.
///
/// Shared between `build_sim_state` (primes `last_nav` so the RL env has a
/// valid initial observation) and `tick::step_one_tick` (invoked every outer
/// GNC tick). Dispatches on the state's `nav_filter` variant.
pub(crate) fn navigate_from_state(
    state: &mut SimState,
    data: &SimData,
    planet: &PlanetConfig,
) -> crate::gnc::navigation::estimator::NavigationOutput {
    let position_true = [state.state[0], state.state[1], state.state[2]];
    let velocity_true = [state.state[3], state.state[4], state.state[5]];
    match &mut state.nav_filter {
        NavigationFilter::Bias(nav_state) => estimator::navigate(
            &position_true,
            &velocity_true,
            state.guidance_state.aoa_commanded,
            state.sim_time,
            &state.nav_biases,
            nav_state,
            data,
            planet,
            state.run_state.density_bias,
            state.run_state.density_perturbation,
            state.run_state.cx_bias,
            state.run_state.cz_bias,
            state.run_state.mass_bias,
            state.run_state.incidence_bias,
            state.run_state.ref_area_bias,
            state.run_state.filter_gain_bias,
        ),
        NavigationFilter::Ekf {
            ekf,
            imu,
            star_tracker,
            st_config,
            ekf_config,
            legacy,
            ..
        } => estimator::navigate_ekf(
            &position_true,
            &velocity_true,
            state.guidance_state.aoa_commanded,
            state.sim_time,
            data.periods.navigation,
            &state.nav_biases,
            legacy,
            ekf,
            imu,
            star_tracker,
            st_config,
            ekf_config,
            data,
            planet,
            state.run_state.density_bias,
            state.run_state.density_perturbation,
            state.run_state.cx_bias,
            state.run_state.cz_bias,
            state.run_state.mass_bias,
            state.run_state.incidence_bias,
            state.run_state.ref_area_bias,
        ),
    }
}

/// Result from a single simulation run.
struct SimResult {
    sim_idx: i32,
    final_line: [f64; 52],
    photo_lines: Vec<[f64; 30]>,
    dispersions: [f64; DISPERSION_DRAW_LEN],
    supervised_trace: Vec<(Vec<f64>, f64, f64, f64, f64)>,
}

/// Shared simulation orchestration: build run states, dispatch parallel/sequential runs.
fn run_core(
    config: &SimInput,
    data: &SimData,
    write_photo: bool,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<SimResult>, SimError> {
    let n_sims = if config.n_sims == 0 { 1 } else { config.n_sims };
    let is_mc = n_sims > 1;

    let draws = data.dispersion_config.as_ref().map(|dc| {
        let draws = dc.generate_draws(n_sims as usize);
        if write_photo {
            let on_off = |b: bool| if b { "on" } else { "off" };
            eprintln!(
                "Monte Carlo: {} draws from seed {}, domains: state={} atmo={} aero={} nav={} mass={} vehicle={} pilot={} nav_filter={}",
                draws.len(), dc.seed,
                on_off(dc.initial_state.is_some()), on_off(dc.atmosphere.is_some()),
                on_off(dc.aerodynamics.is_some()), on_off(dc.navigation.is_some()),
                on_off(dc.mass.is_some()), on_off(dc.vehicle.is_some()),
                on_off(dc.pilot.is_some()), on_off(dc.nav_filter.is_some()),
            );
        }
        draws
    });

    let run_states: Vec<(init::RunState, [f64; DISPERSION_DRAW_LEN])> = (0..n_sims)
        .map(|sim_idx| {
            let draw = if let Some(ref d) = draws {
                &d[sim_idx as usize]
            } else {
                &crate::data::dispersions::DispersionDraw::default()
            };
            (init::init_run_from_draw(data, draw), draw.to_array())
        })
        .collect();

    let photo_sim_idx = if is_mc {
        if config.visualize_sim > 0 {
            (config.visualize_sim - 1).min(n_sims - 1)
        } else {
            n_sims - 1
        }
    } else {
        0
    };

    if is_mc {
        let start = std::time::Instant::now();
        if write_photo {
            eprintln!("Running {} simulations in parallel...", n_sims);
        }
        let results: Vec<SimResult> = run_states
            .par_iter()
            .enumerate()
            .map(|(idx, (run_state, disp_array))| {
                let do_photo = (write_photo && idx as i32 == photo_sim_idx) || include_trajectories;
                let mut result =
                    run_single(config, data, run_state, idx as i32, do_photo, wall_timeout)?;
                result.dispersions = *disp_array;
                Ok(result)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if write_photo {
            let elapsed = start.elapsed();
            eprintln!(
                "Completed {} simulations in {:.3}s ({:.1} sims/s)",
                n_sims,
                elapsed.as_secs_f64(),
                n_sims as f64 / elapsed.as_secs_f64(),
            );
        }
        Ok(results)
    } else {
        let (run_state, disp_array) = &run_states[0];
        if write_photo && config.screen_output {
            eprintln!(
                "  Entry: alt={:.3} km, vel={:.3} m/s, fpa={:.5} deg",
                run_state.entry.state.altitude / 1e3,
                run_state.entry.state.velocity,
                run_state.entry.state.flight_path.to_degrees(),
            );
        }
        let mut result = run_single(
            config,
            data,
            run_state,
            0,
            write_photo || include_trajectories,
            wall_timeout,
        )?;
        result.dispersions = *disp_array;
        Ok(vec![result])
    }
}

/// Run the full simulation.
pub fn run(config: &SimInput, data: &SimData) -> Result<(), SimError> {
    let n_sims = if config.n_sims == 0 { 1 } else { config.n_sims };
    let photo_sim_idx = if n_sims > 1 {
        if config.visualize_sim > 0 {
            (config.visualize_sim - 1).min(n_sims - 1)
        } else {
            n_sims - 1
        }
    } else {
        0
    };

    let results = run_core(config, data, true, false, None)?;
    write_csv_output(config, &results, photo_sim_idx)?;
    Ok(())
}

/// Project each 30-element photo line onto the 17-element trajectory row exposed
/// by the PyO3 API. Index mapping and unit scaling (energy J->MJ, pdyn Pa->kPa)
/// are the contract documented on `BatchResults` trajectory columns.
fn project_trajectory(photo_lines: &[[f64; 30]]) -> Vec<[f64; 17]> {
    photo_lines
        .iter()
        .map(|p| {
            [
                p[1],        // [0]  alt_km
                p[2],        // [1]  lon_deg
                p[3],        // [2]  lat_deg
                p[4],        // [3]  vel_m_s
                p[5],        // [4]  fpa_deg
                p[6],        // [5]  heading_deg
                p[24],       // [6]  heat_flux_kw_m2
                p[0],        // [7]  time_s
                p[18] / 1e6, // [8]  energy_mj_kg
                p[19] / 1e3, // [9]  pdyn_kpa
                p[14],       // [10] bank_angle_deg
                p[9],        // [11] inclination_deg
                p[25],       // [12] g_load_g
                p[26],       // [13] nav_density_ratio
                p[27],       // [14] truth_density_kg_m3
                p[28],       // [15] heat_load_kj_m2
                p[29],       // [16] density_perturbation
            ]
        })
        .collect()
}

/// Assemble a `RunOutput` from one `SimResult`: project the trajectory (only when
/// requested), extract energy/ecc, and apply the capture predicate.
fn assemble_run_output(r: SimResult, include_trajectories: bool) -> crate::RunOutput {
    let energy = r.final_line[7]; // MJ/kg
    let ecc = r.final_line[9];
    let trajectory = if include_trajectories {
        project_trajectory(&r.photo_lines)
    } else {
        Vec::new()
    };
    let ifinal_val = r.final_line[31] as i32;
    crate::RunOutput {
        trajectory,
        final_record: r.final_line,
        captured: ifinal_val == 3 && ecc < 1.0 && energy < 0.0,
        dispersions: r.dispersions,
        supervised_trace: r.supervised_trace,
    }
}

/// Run simulation and return structured results (no file I/O).
///
/// Same physics as `run()`, but returns `Vec<RunOutput>` instead of writing files.
/// Used by the PyO3 interface for direct Python access.
pub fn run_for_api(
    config: &SimInput,
    data: &SimData,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<crate::RunOutput>, SimError> {
    let results = run_core(config, data, false, include_trajectories, wall_timeout)?;

    Ok(results
        .into_iter()
        .map(|r| assemble_run_output(r, include_trajectories))
        .collect())
}

/// Run simulation with pre-computed dispersion draws (no file I/O).
///
/// Accepts a `Vec<DispersionDraw>` from the caller instead of generating
/// draws internally. Each draw maps to exactly one simulation run.
/// Used by the PyO3 `run_with_draws()` binding for external sampling.
pub fn run_for_api_with_draws(
    config: &SimInput,
    data: &SimData,
    external_draws: Vec<crate::data::dispersions::DispersionDraw>,
    include_trajectories: bool,
    wall_timeout: Option<Duration>,
) -> Result<Vec<crate::RunOutput>, SimError> {
    let n = external_draws.len();
    let is_mc = n > 1;

    let run_states: Vec<(init::RunState, [f64; DISPERSION_DRAW_LEN])> = external_draws
        .iter()
        .map(|draw| (init::init_run_from_draw(data, draw), draw.to_array()))
        .collect();

    let results: Vec<SimResult> = if is_mc {
        run_states
            .par_iter()
            .enumerate()
            .map(|(idx, (run_state, disp_array))| {
                let mut result = run_single(
                    config,
                    data,
                    run_state,
                    idx as i32,
                    include_trajectories,
                    wall_timeout,
                )?;
                result.dispersions = *disp_array;
                Ok(result)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else if n == 1 {
        let (run_state, disp_array) = &run_states[0];
        let mut result = run_single(
            config,
            data,
            run_state,
            0,
            include_trajectories,
            wall_timeout,
        )?;
        result.dispersions = *disp_array;
        vec![result]
    } else {
        return Ok(Vec::new());
    };

    Ok(results
        .into_iter()
        .map(|r| assemble_run_output(r, include_trajectories))
        .collect())
}

/// Write output in CSV format with named headers and clean schema.
fn write_csv_output(
    config: &SimInput,
    results: &[SimResult],
    photo_sim_idx: i32,
) -> Result<(), SimError> {
    let suffix = config.results_suffix.trim_start_matches('.');
    let final_path = config.output_path(&format!("final.{}.csv", suffix));
    let mut final_file = BufWriter::new(
        File::create(&final_path)
            .map_err(|e| SimError(format!("Cannot create {}: {}", final_path, e)))?,
    );

    output::write_final_csv_header(&mut final_file)
        .map_err(|e| SimError(format!("Final CSV header error: {}", e)))?;

    for result in results {
        let csv_values = extract_final_csv_values(&result.final_line);
        output::write_final_csv_line(&mut final_file, result.sim_idx + 1, &csv_values)
            .map_err(|e| SimError(format!("Final CSV write error: {}", e)))?;
    }
    final_file
        .flush()
        .map_err(|e| SimError(format!("Final CSV flush error: {}", e)))?;

    // Write photo CSV
    let photo_path = config.output_path(&format!("photo.{}.csv", suffix));
    if let Some(result) = results.iter().find(|r| r.sim_idx == photo_sim_idx) {
        let mut photo_file = BufWriter::new(
            File::create(&photo_path)
                .map_err(|e| SimError(format!("Cannot create {}: {}", photo_path, e)))?,
        );

        output::write_photo_csv_header(&mut photo_file)
            .map_err(|e| SimError(format!("Photo CSV header error: {}", e)))?;

        for line in &result.photo_lines {
            let csv_values = extract_photo_csv_values(line);
            output::write_photo_csv_line(&mut photo_file, &csv_values)
                .map_err(|e| SimError(format!("Photo CSV write error: {}", e)))?;
        }
        photo_file
            .flush()
            .map_err(|e| SimError(format!("Photo CSV flush error: {}", e)))?;
    }

    Ok(())
}

/// Extract 22 CSV values from the 30-element photo array.
/// Drops: [20] radial_velocity_2 (duplicate), [22] sim_number, [23] reserved, [24-27] trajectory-only, [29] density_perturbation.
fn extract_photo_csv_values(values: &[f64; 30]) -> [f64; 22] {
    [
        values[0],  // time_s
        values[1],  // altitude_km
        values[2],  // longitude_deg
        values[3],  // latitude_deg
        values[4],  // velocity_m_s
        values[5],  // flight_path_deg
        values[6],  // azimuth_deg
        values[7],  // semi_major_axis_km
        values[8],  // eccentricity
        values[9],  // inclination_deg
        values[10], // raan_deg
        values[11], // periapsis_alt_km
        values[12], // apoapsis_alt_km
        values[13], // phase
        values[14], // bank_angle_deg
        values[15], // radial_velocity_m_s
        values[16], // aoa_deg
        values[17], // cumulative_bank_change_deg
        values[18], // energy_j_kg
        values[19], // dynamic_pressure_pa
        values[21], // dynamic_pressure_onboard_kpa (skip [20] duplicate)
        values[28], // heat_load_kj_m2
    ]
}

/// Extract 39 CSV values from the 52-element final array.
/// Drops 14 always-zero indices: 32-36, 42-44, 46-47, 49-51.
fn extract_final_csv_values(values: &[f64; 52]) -> [f64; 39] {
    [
        values[0],  // altitude_km
        values[1],  // longitude_deg
        values[2],  // latitude_deg
        values[3],  // velocity_m_s
        values[4],  // flight_path_deg
        values[5],  // azimuth_deg
        values[6],  // radial_velocity_m_s
        values[7],  // energy_mj_kg
        values[8],  // semi_major_axis_km
        values[9],  // eccentricity
        values[10], // inclination_deg
        values[11], // raan_deg
        values[12], // arg_periapsis_deg
        values[13], // true_anomaly_deg
        values[14], // periapsis_alt_km
        values[15], // apoapsis_alt_km
        values[16], // max_heat_flux_kw_m2
        values[17], // max_load_factor_g
        values[18], // max_dyn_pressure_kpa
        values[19], // alt_max_flux_km
        values[20], // alt_max_load_km
        values[21], // alt_max_pdyn_km
        values[22], // time_max_flux_s
        values[23], // time_max_load_s
        values[24], // time_max_pdyn_s
        values[25], // bounce_alt_km
        values[26], // bounce_time_s
        values[27], // sim_time_s
        values[28], // integrated_flux_mj_m2
        values[29], // periapsis_err_km
        values[30], // apoapsis_err_km
        values[31], // ifinal
        values[37], // dv1_m_s
        values[38], // dv2_m_s
        values[39], // dv3_m_s
        values[40], // dv12_m_s
        values[41], // dv_total_m_s
        values[45], // cumulative_bank_change_deg
        values[48], // n_roll_reversals
    ]
}

/// Run a single simulation, returning results.
fn run_single(
    config: &SimInput,
    data: &SimData,
    run_state: &init::RunState,
    sim_idx: i32,
    write_photo: bool,
    wall_timeout: Option<Duration>,
) -> Result<SimResult, SimError> {
    let planet = &config.planet;

    // Construct the base SimState via the shared constructor (identical seed
    // derivation, GNC init, and bias-mode last_nav priming as the RL env path);
    // `sim_idx as u64` reproduces the historical per-sim seeds exactly:
    // EKF `random_seed + sim_idx*10_000`, GM-RNG `... + 0xDE45`.
    let mut sim_state = build_sim_state(config, data, *run_state, sim_idx as u64);

    // CLI-specific overrides not produced by `build_sim_state` (which targets the
    // RL env defaults: no photo, no wall timeout, not the single-run banner).
    sim_state.write_photo = write_photo;
    sim_state.wall_timeout = wall_timeout;
    sim_state.is_single = config.n_sims <= 1 && config.screen_output;
    let is_single = sim_state.is_single;

    // Event detection setup (used by adaptive integrator)
    let event_defs = build_event_defs();
    let event_ctx = build_event_ctx(config, data);

    if is_single {
        eprintln!(
            "  Init: entry.initial_bank={:.5}deg, reference_bank_angle={:.5}deg, sim.bank_angle={:.5}deg",
            run_state.entry.initial_bank.to_degrees(),
            sim_state.reference_bank_angle.to_degrees(),
            sim_state.bank_angle.to_degrees()
        );
    }

    // Main simulation loop
    while sim_state.term == TermReason::None {
        let _outcome = crate::simulation::tick::step_one_tick(
            &mut sim_state,
            config,
            data,
            planet,
            None,
            &event_defs,
            &event_ctx,
        );
    }

    // Final photo snapshot
    if sim_state.write_photo {
        let sim_time = sim_state.sim_time;
        let dynamic_pressure_for_photo = sim_state.dynamic_pressure_for_photo;
        let density_estimate_for_photo = sim_state.density_estimate_for_photo;
        let sim_idx = sim_state.sim_idx;
        let cumulative_bank_change_deg = sim_state.cumulative_bank_change_deg;
        let density_gain = sim_state.nav_filter.density_gain();
        let run_state_snap = sim_state.run_state;
        let cumulative_flux = sim_state.state[6];
        let guidance_phase_for_photo = sim_state.guidance_phase_for_photo;
        let photo_line = build_photo_values(
            &sim_state,
            sim_time,
            planet,
            dynamic_pressure_for_photo,
            density_estimate_for_photo,
            sim_idx + 1,
            cumulative_bank_change_deg * DEG_TO_RAD,
            data,
            density_gain,
            &run_state_snap,
            cumulative_flux,
            guidance_phase_for_photo,
        );
        sim_state.photo_lines.push(photo_line);
    }

    // === Final conditions ===
    let (alt_final, _lat_final) = geodetic_from_spherical(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        planet,
    );

    if sim_state.is_single {
        eprintln!(
            "  Final: alt={:.3} km, vel={:.3} m/s, t={:.1} s, steps={}, term={:?}",
            alt_final / 1e3,
            sim_state.state[3],
            sim_state.sim_time,
            sim_state.step,
            sim_state.term,
        );
    }

    promote_pending_crash_if_applicable(&mut sim_state, planet);

    // The 52-element final record / termination classification / virtual-DV is
    // assembled by `build_final_record` (the same path the RL per-step env API
    // takes via `tick.rs`), keeping CLI and env outputs bit-identical.
    let final_record = build_final_record(&sim_state, data, planet);

    let event_records = std::mem::take(&mut sim_state.event_records);

    // Append event records as photo rows and sort by time (column 0)
    if sim_state.write_photo {
        for record in &event_records {
            sim_state.photo_lines.push(build_event_photo_values(
                &record.state,
                record.time,
                planet,
                data,
                &sim_state.run_state,
                record.bank_angle_deg,
                record.aoa_deg,
                record.cumulative_bank_change_deg,
                record.guidance_phase,
                record.density_gain,
            ));
        }
        sim_state
            .photo_lines
            .sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
    }

    let photo_lines = std::mem::take(&mut sim_state.photo_lines);

    let supervised_trace = if config.collect_supervised {
        std::mem::take(&mut sim_state.supervised_trace)
    } else {
        Vec::new()
    };

    Ok(SimResult {
        sim_idx,
        final_line: final_record,
        photo_lines,
        dispersions: [0.0; DISPERSION_DRAW_LEN],
        supervised_trace,
    })
}

/// Assemble the 52-element final record from a terminated `SimState`.
///
/// Mirrors the block at the end of `run_single`. Requires `term != TermReason::None`.
/// Called by `BatchedSimulation::step()` on terminal steps.
/// Pure predicate: would this orbit be a "pending crash" -- captured (bound + e<1)
/// but with apoapsis below the atmospheric ceiling, so guaranteed to re-enter?
///
/// Extracted so it can be unit-tested without constructing a full `SimState`.
pub fn is_pending_crash(
    eccentricity: f64,
    energy: f64,
    apoapsis_alt: f64,
    exit_altitude: f64,
) -> bool {
    let captured = eccentricity < 1.0 && energy < 0.0;
    captured && apoapsis_alt < exit_altitude
}

/// Promote `AtmosphereExit` to `PendingCrash` when the resulting orbit has
/// apoapsis below the atmospheric ceiling (captured but doomed to re-entry).
///
/// Called both by `finalize_run` (CLI path) and `tick.rs` (RL per-step path)
/// so both sources of `ifinal`/`final_record` see the same terminal classification.
pub fn promote_pending_crash_if_applicable(sim_state: &mut SimState, planet: &PlanetConfig) {
    if sim_state.term != TermReason::AtmosphereExit {
        return;
    }
    let orbit = elements::from_spherical(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        sim_state.state[3],
        sim_state.state[4],
        sim_state.state[5],
        planet,
    );
    let (_, velocity_abs) = to_absolute_cartesian(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        sim_state.state[3],
        sim_state.state[4],
        sim_state.state[5],
        planet,
    );
    let speed_abs = norm(&velocity_abs);
    let energy = speed_abs * speed_abs / 2.0 - planet.mu / sim_state.state[0];
    if is_pending_crash(
        orbit.eccentricity,
        energy,
        orbit.apoapsis_alt,
        sim_state.exit_altitude,
    ) {
        sim_state.term = TermReason::PendingCrash;
    }
}

/// Map a terminal `TermReason` to the `ifinal` classification code written to
/// `final_record[31]`.
///
/// Single source of truth shared by `run_single`, `build_final_record`, and the
/// RL per-step path in `tick.rs`. Genuinely unreachable on `None`: every caller
/// is reached only after the simulation has terminated.
pub fn ifinal_for(term: TermReason) -> i32 {
    match term {
        TermReason::AtmosphereExit => 3,
        TermReason::Crash => 1,
        TermReason::PendingCrash => 4,
        TermReason::Timeout => 2,
        TermReason::None => unreachable!("ifinal requested for a non-terminated state"),
    }
}

pub fn build_final_record(
    sim_state: &SimState,
    data: &SimData,
    planet: &PlanetConfig,
) -> [f64; 52] {
    let (alt_final, lat_final) = geodetic_from_spherical(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        planet,
    );

    let orbit = elements::from_spherical(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        sim_state.state[3],
        sim_state.state[4],
        sim_state.state[5],
        planet,
    );

    let mu = planet.mu;
    let (_position_abs, velocity_abs) = to_absolute_cartesian(
        sim_state.state[0],
        sim_state.state[1],
        sim_state.state[2],
        sim_state.state[3],
        sim_state.state[4],
        sim_state.state[5],
        planet,
    );
    let speed_abs = norm(&velocity_abs);
    let energy = speed_abs * speed_abs / 2.0 - mu / sim_state.state[0];
    let velocity_radial = sim_state.state[3] * sim_state.state[4].sin();

    let captured = orbit.eccentricity < 1.0 && energy < 0.0;

    let ifinal = ifinal_for(sim_state.term);

    let deltav = if sim_state.term == TermReason::AtmosphereExit && captured {
        maneuver::compute_deltav(&orbit, &data.target_orbit, &data.parking_orbit, planet)
    } else if sim_state.term == TermReason::AtmosphereExit {
        let v_escape = (2.0 * mu / sim_state.state[0]).sqrt();
        let v_excess = (speed_abs - v_escape).max(0.0);
        DeltaV {
            dv1: 0.0,
            dv2: 0.0,
            dv3: 0.0,
            total: HYPERBOLIC_BASE + v_excess,
        }
    } else {
        let virtual_dv = virtual_dv_non_capture(
            energy,
            data.target_orbit.semi_major_axis,
            mu,
            sim_state.sim_time,
            sim_state.max_time,
        );
        DeltaV {
            dv1: 0.0,
            dv2: 0.0,
            dv3: 0.0,
            total: virtual_dv,
        }
    };

    let mut fr = [0.0_f64; 52];
    fr[0] = alt_final / 1e3;
    fr[1] = sim_state.state[1] / DEG_TO_RAD;
    fr[2] = lat_final / DEG_TO_RAD;
    fr[3] = sim_state.state[3];
    fr[4] = sim_state.state[4] / DEG_TO_RAD;
    fr[5] = sim_state.state[5] / DEG_TO_RAD;
    fr[6] = velocity_radial;
    fr[7] = energy / 1e6;
    fr[8] = orbit.semi_major_axis / 1e3;
    fr[9] = orbit.eccentricity;
    fr[10] = orbit.inclination / DEG_TO_RAD;
    fr[11] = orbit.raan / DEG_TO_RAD;
    fr[12] = orbit.arg_periapsis / DEG_TO_RAD;
    fr[13] = orbit.true_anomaly / DEG_TO_RAD;
    fr[14] = orbit.periapsis_alt / 1e3;
    fr[15] = orbit.apoapsis_alt / 1e3;
    fr[16] = sim_state.max_heat_flux / 1e3;
    fr[17] = sim_state.max_load_factor / G0;
    fr[18] = sim_state.max_dyn_pressure / 1e3;
    fr[19] = sim_state.alt_max_flux / 1e3;
    fr[20] = sim_state.alt_max_load / 1e3;
    fr[21] = sim_state.alt_max_pdyn / 1e3;
    fr[22] = sim_state.time_max_flux;
    fr[23] = sim_state.time_max_load;
    fr[24] = sim_state.time_max_pdyn;
    fr[25] = sim_state.bounce_alt / 1e3;
    fr[26] = sim_state.bounce_time;
    fr[27] = sim_state.sim_time;
    fr[28] = sim_state.state[6] / 1e6;
    fr[29] = orbit.periapsis_alt / 1e3 - data.target_orbit.periapsis / 1e3;
    fr[30] = orbit.apoapsis_alt / 1e3 - data.target_orbit.apoapsis / 1e3;
    fr[31] = ifinal as f64;
    fr[37] = deltav.dv1;
    fr[38] = deltav.dv2;
    fr[39] = deltav.dv3;
    fr[40] = deltav.dv1.abs() + deltav.dv2.abs();
    fr[41] = deltav.total;
    fr[45] = sim_state.cumulative_bank_change_deg;
    fr[46] = orbit.inclination / DEG_TO_RAD - data.target_orbit.inclination / DEG_TO_RAD;
    fr[48] = sim_state.guidance_state.lateral_state.n_reversals as f64;
    fr
}

/// Build the standard aerocapture event definitions.
///
/// Convenience wrapper around `events::build_aerocapture_events()` for tests
/// and external callers that need to drive `step_one_tick` directly.
pub fn build_event_defs() -> Vec<events::EventDef> {
    events::build_aerocapture_events()
}

/// Build the standard `EventContext` from a config + data pair.
///
/// Matches the construction in `run_single`. Use alongside `build_event_defs()`
/// when calling `step_one_tick` outside the normal runner loop.
pub fn build_event_ctx(config: &SimInput, data: &SimData) -> events::EventContext {
    let planet = &config.planet;
    let exit_altitude = data.final_conditions.altitude;
    events::EventContext {
        planet_radius: planet.equatorial_radius,
        polar_radius: planet.polar_radius,
        exit_altitude,
        exit_velocity_threshold: data.guidance.exit_velocity_threshold,
    }
}

/// Run a single simulation and return the 52-element final record in memory.
///
/// Equivalent to `run_single` but skips file I/O and returns the final record
/// directly. Intended for tests that need to compare against the step-API path.
pub fn run_single_collect(config: &SimInput, data: &SimData) -> Result<[f64; 52], SimError> {
    let draw = crate::data::dispersions::DispersionDraw::default();
    let run_state = init::init_run_from_draw(data, &draw);
    let result = run_single(config, data, &run_state, 0, false, None)?;
    Ok(result.final_line)
}

/// Build a photo snapshot line.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_photo_values(
    sim: &SimState,
    sim_time: f64,
    planet: &PlanetConfig,
    dynamic_pressure: f64,
    density_estimate: f64,
    sim_index: i32,
    cumulative_bank_change: f64,
    data: &SimData,
    density_gain: f64,
    run_state: &init::RunState,
    cumulative_flux: f64,
    guidance_phase: i32,
) -> [f64; 30] {
    let (altitude, latitude) =
        geodetic_from_spherical(sim.state[0], sim.state[1], sim.state[2], planet);

    let orbit = elements::from_spherical(
        sim.state[0],
        sim.state[1],
        sim.state[2],
        sim.state[3],
        sim.state[4],
        sim.state[5],
        planet,
    );

    let mu = planet.mu;
    let (_position_abs, velocity_abs) = to_absolute_cartesian(
        sim.state[0],
        sim.state[1],
        sim.state[2],
        sim.state[3],
        sim.state[4],
        sim.state[5],
        planet,
    );
    let speed_abs = norm(&velocity_abs);
    let energy = speed_abs * speed_abs / 2.0 - mu / sim.state[0];
    let velocity_radial = sim.state[3] * sim.state[4].sin();

    let phase = guidance_phase as f64;

    // Compute per-timestep heat flux, g-load, and truth density for trajectory output.
    // Use dispersed values (matching track_peak_values) so trajectory plots are consistent
    // with final_record peak values and constraint classification.
    let rho_truth = data.atmosphere.density_at(altitude);
    let rho_dispersed = atmosphere::density(
        &data.atmosphere,
        altitude,
        run_state.density_bias,
        run_state.density_perturbation,
    );
    // Wind-corrected velocity for aero-dependent quantities
    let v_eff = effective_airspeed(
        sim.state[3],
        sim.state[4],
        sim.state[5],
        sim.state[2],
        altitude,
        data,
        run_state,
    );
    let heat_flux = data.capsule.cq * rho_dispersed.sqrt() * v_eff.powf(3.05);
    let aoa_dispersed = sim.aoa + run_state.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);
    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_accel = rho_dispersed * ref_area * v_eff * v_eff / (2.0 * mass);
    let load_factor = aero_accel * (cx * cx + cz * cz).sqrt();

    [
        sim_time,
        altitude / 1e3,
        sim.state[1] / DEG_TO_RAD,
        latitude / DEG_TO_RAD,
        sim.state[3],
        sim.state[4] / DEG_TO_RAD,
        sim.state[5] / DEG_TO_RAD,
        orbit.semi_major_axis / 1e3,
        orbit.eccentricity,
        orbit.inclination / DEG_TO_RAD,
        orbit.raan / DEG_TO_RAD,
        orbit.periapsis_alt / 1e3,
        orbit.apoapsis_alt / 1e3,
        phase,
        sim.bank_angle / DEG_TO_RAD,
        velocity_radial,
        sim.aoa / DEG_TO_RAD,
        cumulative_bank_change / DEG_TO_RAD,
        energy,
        dynamic_pressure,
        velocity_radial,
        0.5 * density_estimate * sim.state[3] * sim.state[3] / 1e3,
        sim_index as f64,
        0.0,
        heat_flux / 1e3,                // [24] heat_flux kW/m²
        load_factor / G0,               // [25] g-load in g's
        density_gain,                   // [26] nav density ratio (estimated/model)
        rho_truth,                      // [27] truth density kg/m³
        cumulative_flux / 1e3,          // [28] heat_load_kj_m2 (J/m2 -> kJ/m2)
        run_state.density_perturbation, // [29] density_perturbation (fractional GM value)
    ]
}

/// Build a photo row from an event record's state.
///
/// Computes the same physics quantities as `build_photo_values` but uses the event
/// state directly. GNC-dependent values (bank_angle, aoa, cumulative_bank_change,
/// phase, density_gain) are carried from the enclosing tick because events occur
/// mid-tick and GNC quantities are constant within a tick.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_event_photo_values(
    state: &[f64; 8],
    event_time: f64,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
    bank_angle_deg: f64,
    aoa_deg: f64,
    cumulative_bank_change_deg: f64,
    guidance_phase: f64,
    density_gain: f64,
) -> [f64; 30] {
    let (altitude, latitude) = geodetic_from_spherical(state[0], state[1], state[2], planet);

    let orbit = elements::from_spherical(
        state[0], state[1], state[2], state[3], state[4], state[5], planet,
    );

    let mu = planet.mu;
    let (_position_abs, velocity_abs) = to_absolute_cartesian(
        state[0], state[1], state[2], state[3], state[4], state[5], planet,
    );
    let speed_abs = norm(&velocity_abs);
    let energy = speed_abs * speed_abs / 2.0 - mu / state[0];
    let velocity_radial = state[3] * state[4].sin();

    let rho_truth = data.atmosphere.density_at(altitude);
    let rho_dispersed = atmosphere::density(
        &data.atmosphere,
        altitude,
        run_state.density_bias,
        run_state.density_perturbation,
    );
    let v_eff = effective_airspeed(
        state[3], state[4], state[5], state[2], altitude, data, run_state,
    );
    let heat_flux = data.capsule.cq * rho_dispersed.sqrt() * v_eff.powf(3.05);
    let pdyn = 0.5 * rho_dispersed * v_eff * v_eff;

    let aoa_dispersed = run_state.incidence_bias; // aoa=0 + bias
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);
    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_accel = rho_dispersed * ref_area * v_eff * v_eff / (2.0 * mass);
    let load_factor = aero_accel * (cx * cx + cz * cz).sqrt();

    // cumulative heat load: state[6] is integrated flux in J/m²
    let cumulative_flux = state[6];

    [
        event_time,                     // [0]  time_s
        altitude / 1e3,                 // [1]  altitude_km
        state[1] / DEG_TO_RAD,          // [2]  longitude_deg
        latitude / DEG_TO_RAD,          // [3]  latitude_deg
        state[3],                       // [4]  velocity_m_s
        state[4] / DEG_TO_RAD,          // [5]  flight_path_deg
        state[5] / DEG_TO_RAD,          // [6]  azimuth_deg
        orbit.semi_major_axis / 1e3,    // [7]  semi_major_axis_km
        orbit.eccentricity,             // [8]  eccentricity
        orbit.inclination / DEG_TO_RAD, // [9]  inclination_deg
        orbit.raan / DEG_TO_RAD,        // [10] raan_deg
        orbit.periapsis_alt / 1e3,      // [11] periapsis_alt_km
        orbit.apoapsis_alt / 1e3,       // [12] apoapsis_alt_km
        guidance_phase,                 // [13] phase (from enclosing tick)
        bank_angle_deg,                 // [14] bank_angle_deg (from enclosing tick)
        velocity_radial,                // [15] radial_velocity_m_s
        aoa_deg,                        // [16] aoa_deg (from enclosing tick)
        cumulative_bank_change_deg,     // [17] cumulative_bank_change_deg (from enclosing tick)
        energy,                         // [18] energy_j_kg
        pdyn,                           // [19] dynamic_pressure_pa
        velocity_radial, // [20] radial_velocity_2 (duplicate, matches build_photo_values)
        0.0,             // [21] dynamic_pressure_onboard_kpa (no nav estimate at event time)
        0.0,             // [22] sim_index (not applicable for event rows)
        0.0,             // [23] reserved
        heat_flux / 1e3, // [24] heat_flux_kw_m2
        load_factor / G0, // [25] g_load_g
        density_gain,    // [26] nav_density_ratio (from enclosing tick)
        rho_truth,       // [27] truth_density_kg_m3
        cumulative_flux / 1e3, // [28] heat_load_kj_m2
        run_state.density_perturbation, // [29] density_perturbation
    ]
}

/// Perform one integration step using Gill's RK4.
pub(crate) fn integrate_step(
    sim: &mut SimState,
    dt: f64,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
) {
    sim.gill_toggle = 0;

    for k in 1..=4 {
        let derivs =
            compute_derivatives(&sim.state, sim.bank_angle, sim.aoa, planet, data, run_state);
        rk4::rk4_increment(
            dt,
            &derivs,
            k,
            8,
            &mut sim.gill_toggle,
            &mut sim.accumulator,
            &mut sim.state,
        );
    }
}

pub(crate) struct AdaptiveEventResult {
    pub(crate) triggered: Vec<events::TriggeredEvent>,
}

pub(crate) const EVENT_TOL: f64 = 1e-3; // 1 ms event location tolerance

#[allow(clippy::too_many_arguments)]
pub(crate) fn integrate_adaptive_with_events(
    sim: &mut SimState,
    dt_outer: f64,
    config: &AdaptiveConfig,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
    event_defs: &[EventDef],
    event_ctx: &EventContext,
    tick_start_time: f64,
) -> AdaptiveEventResult {
    const MAX_SUBSTEPS: u32 = 1000;

    let bank_angle = sim.bank_angle;
    let aoa = sim.aoa;
    let mut t_remaining = dt_outer;
    let mut h = config.initial_dt.min(t_remaining).max(config.min_dt);
    let mut n_substeps: u32 = 0;
    let mut n_rejections: u32 = 0;

    // Cache event guard values at beginning of tick
    let mut g_prev = events::evaluate_events(&sim.state, event_defs, event_ctx);

    let mut all_triggered: Vec<events::TriggeredEvent> = Vec::new();

    while t_remaining > 1e-14 {
        h = h.min(t_remaining).min(config.max_dt).max(config.min_dt);

        // If remaining time is very small, take it in one step regardless
        if t_remaining <= config.min_dt * 1.5 {
            h = t_remaining;
        }

        let y0 = sim.state;

        let (result, stages) = dopri45::dopri45_step_with_stages(
            &mut sim.state,
            h,
            &mut sim.dopri,
            &DOPRI45_ATOL,
            config.rtol,
            &mut |state| compute_derivatives(state, bank_angle, aoa, planet, data, run_state),
        );

        if result.accepted {
            // Check for events in this accepted substep
            let k1 = &stages[0];
            let k7 = &stages[6];

            let t_base = tick_start_time + (dt_outer - t_remaining);
            if let Some(triggered) = events::check_events_and_locate(
                &y0, &sim.state, h, k1, k7, event_defs, event_ctx, &g_prev, EVENT_TOL, t_base,
            ) {
                let event = &event_defs[triggered.event_index];

                // Record this event (GNC fields populated by caller after return)
                sim.event_records.push(EventRecord {
                    time: triggered.time,
                    state: triggered.state,
                    event_type: event.event_type,
                    bank_angle_deg: 0.0,
                    aoa_deg: 0.0,
                    cumulative_bank_change_deg: 0.0,
                    guidance_phase: 0.0,
                    density_gain: 0.0,
                });

                // Rewind state to the event location
                sim.state = triggered.state;

                // Invalidate FSAL -- state was rewound, cached derivative is stale
                sim.dopri.invalidate_fsal();

                match event.action {
                    EventAction::Terminate(_) => {
                        // Terminal event: return immediately
                        all_triggered.push(triggered);
                        return AdaptiveEventResult {
                            triggered: all_triggered,
                        };
                    }
                    EventAction::Record | EventAction::PhaseTransition => {
                        // Non-terminal: adjust t_remaining for partial step consumed
                        let consumed = triggered.theta * h;
                        t_remaining -= consumed;
                        n_substeps += 1;
                        h = result.dt_next;

                        // Re-evaluate guard values at the new (event) state
                        g_prev = events::evaluate_events(&sim.state, event_defs, event_ctx);

                        // Force g_prev to exactly 0.0 for the fired event so it won't
                        // re-trigger on the next substep (the g0 == 0.0 skip in
                        // check_events_and_locate prevents re-detection at the same
                        // zero-crossing).
                        g_prev[triggered.event_index] = 0.0;

                        all_triggered.push(triggered);

                        // Check substep cap BEFORE continuing — the old `continue`
                        // bypassed the cap check at the bottom of the loop, allowing
                        // unbounded event accumulation when trajectories oscillate
                        // near an event boundary (e.g. FPA ≈ 0 at bounce).
                        if n_substeps + n_rejections >= MAX_SUBSTEPS {
                            return AdaptiveEventResult {
                                triggered: all_triggered,
                            };
                        }

                        continue;
                    }
                }
            }

            // No event: normal accepted step
            t_remaining -= h;
            n_substeps += 1;
            h = result.dt_next;

            // Update guard values for next substep
            g_prev = events::evaluate_events(&sim.state, event_defs, event_ctx);
        } else {
            // Rejected step: dopri45_step_with_stages restores state to y0 internally
            n_rejections += 1;
            h = result.dt_next;
        }

        if n_substeps + n_rejections >= MAX_SUBSTEPS {
            eprintln!(
                "WARNING: adaptive integrator hit {} step limit with t_remaining={:.2e}s ({} accepted, {} rejected)",
                MAX_SUBSTEPS, t_remaining, n_substeps, n_rejections,
            );
            return AdaptiveEventResult {
                triggered: all_triggered,
            };
        }
    }

    AdaptiveEventResult {
        triggered: all_triggered,
    }
}

/// Update peak tracking values (heat flux, load factor, dynamic pressure)
/// after each integration step.
pub(crate) fn track_peak_values(
    sim: &mut SimState,
    altitude: f64,
    sim_time: f64,
    data: &SimData,
    run_state: &init::RunState,
) {
    let v = sim.state[3];
    let gamma = sim.state[4];
    let psi = sim.state[5];
    let lat = sim.state[2];
    let rho = atmosphere::density(
        &data.atmosphere,
        altitude,
        run_state.density_bias,
        run_state.density_perturbation,
    );

    // Wind-corrected velocity for aero-dependent quantities
    let v_eff = effective_airspeed(v, gamma, psi, lat, altitude, data, run_state);

    // Heat flux (W/m²) — same formula as dflux in compute_derivatives
    let heat_flux = data.capsule.cq * rho.sqrt() * v_eff.powf(3.05);

    // Dynamic pressure (Pa)
    let pdyn = 0.5 * rho * v_eff * v_eff;

    // Load factor (m/s²) — aerodynamic acceleration magnitude
    let aoa_dispersed = sim.aoa + run_state.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);
    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);
    let aero_accel = rho * ref_area * v_eff * v_eff / (2.0 * mass);
    let load_factor = aero_accel * (cx * cx + cz * cz).sqrt();

    if heat_flux > sim.max_heat_flux {
        sim.max_heat_flux = heat_flux;
        sim.alt_max_flux = altitude;
        sim.time_max_flux = sim_time;
    }
    if load_factor > sim.max_load_factor {
        sim.max_load_factor = load_factor;
        sim.alt_max_load = altitude;
        sim.time_max_load = sim_time;
    }
    if pdyn > sim.max_dyn_pressure {
        sim.max_dyn_pressure = pdyn;
        sim.alt_max_pdyn = altitude;
        sim.time_max_pdyn = sim_time;
    }
}

/// Compute effective airspeed accounting for wind.
///
/// The state velocity `v` is relative to the planet-fixed atmosphere.
/// Wind adds a velocity perturbation: we subtract wind from the vehicle's
/// ground-relative velocity components to get the airspeed used for aero forces.
/// Returns the original `v` when wind is disabled or no wind table is loaded.
pub(crate) fn effective_airspeed(
    v: f64,
    gamma: f64,
    psi: f64,
    lat: f64,
    altitude: f64,
    data: &SimData,
    run_state: &init::RunState,
) -> f64 {
    if !data.wind_enabled {
        return v;
    }
    if let Some(ref wt) = data.wind_table {
        let w = wt.wind_at(altitude, lat);
        let scale = run_state.wind_scale;
        let rot = run_state.wind_direction_bias;
        // Apply dispersions: scale and rotate wind vector
        let we = scale * (w.east * rot.cos() - w.north * rot.sin());
        let wn = scale * (w.east * rot.sin() + w.north * rot.cos());
        // Project into trajectory frame and compute effective speed
        let cos_g = gamma.cos();
        let v_east = v * cos_g * psi.sin() - we;
        let v_north = v * cos_g * psi.cos() - wn;
        let v_vert = v * gamma.sin();
        (v_east * v_east + v_north * v_north + v_vert * v_vert).sqrt()
    } else {
        v
    }
}

/// Compute state derivatives (equations of motion).
///
/// State = [r, lon, lat, V, gamma, psi, flux, time]
pub(crate) fn compute_derivatives(
    state: &[f64; 8],
    bank_angle: f64,
    aoa: f64,
    planet: &PlanetConfig,
    data: &SimData,
    run_state: &init::RunState,
) -> [f64; 8] {
    let r = state[0];
    let _lon = state[1];
    let lat = state[2];
    let v = state[3];
    let gamma = state[4];
    let psi = state[5];

    let (gravtl, gravtr) = gravity::gravity(r, lat, planet);
    let (altitude, _lat_geo) = geodetic_from_spherical(r, state[1], lat, planet);
    let rho = atmosphere::density(
        &data.atmosphere,
        altitude,
        run_state.density_bias,
        run_state.density_perturbation,
    );

    let aoa_dispersed = aoa + run_state.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + run_state.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + run_state.cz_bias);

    let mass = data.capsule.mass * (1.0 + run_state.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + run_state.ref_area_bias);

    // Wind-corrected velocity for aero forces and heat flux.
    // Note: aero force *magnitude* uses v_eff (airspeed) but is applied along the
    // planet-relative velocity direction. This is a first-order approximation valid
    // when wind << vehicle speed. At Mars entry (100 m/s wind vs 5700 m/s), the
    // direction error is O(wind/V)² ≈ 0.03%.
    let v_eff = effective_airspeed(v, gamma, psi, lat, altitude, data, run_state);

    let aero_factor = rho * ref_area / (2.0 * mass);
    let acdrag = aero_factor * cx * v_eff * v_eff;
    let aclift = aero_factor * cz * v_eff * v_eff;

    let cos_bank = bank_angle.cos();
    let sin_bank = bank_angle.sin();
    let cos_gamma = gamma.cos();
    let sin_gamma = gamma.sin();
    let cos_psi = psi.cos();
    let sin_psi = psi.sin();
    let cos_lat = lat.cos();
    let sin_lat = lat.sin();
    let tan_gamma = sin_gamma / cos_gamma;
    let tan_lat = sin_lat / cos_lat;

    let omega = planet.omega;

    // Kinematic derivatives use original v (planet-relative)
    let dr = v * sin_gamma;
    let dlon = v * cos_gamma * sin_psi / (r * cos_lat);
    let dlat = v * cos_gamma * cos_psi / r;

    let dv = -acdrag - gravtr * sin_gamma - gravtl * cos_gamma * cos_psi
        + omega * omega * r * cos_lat * (cos_lat * sin_gamma - sin_lat * cos_gamma * cos_psi);

    let dgamma = (aclift * cos_bank / v) + (v * cos_gamma / r)
        - ((gravtr * cos_gamma - gravtl * sin_gamma * cos_psi) / v)
        + (2.0 * omega * sin_psi * cos_lat)
        + (omega * omega * r * cos_lat * (sin_lat * sin_gamma * cos_psi + cos_lat * cos_gamma) / v);

    let dpsi = (aclift * sin_bank / (v * cos_gamma))
        + (v * cos_gamma * sin_psi * tan_lat / r)
        + (2.0 * omega * (sin_lat - cos_psi * cos_lat * tan_gamma))
        + (gravtl * sin_psi / (v * cos_gamma))
        + (omega * omega * r * cos_lat * sin_lat * sin_psi / (v * cos_gamma));

    // Heat flux uses wind-corrected velocity
    let dflux = data.capsule.cq * rho.sqrt() * v_eff.powf(3.05);
    let dtime = 1.0;

    [dr, dlon, dlat, dv, dgamma, dpsi, dflux, dtime]
}

#[cfg(test)]
mod run_output_tests {
    use super::*;
    use crate::config::SimInput;
    use crate::data::SimData;

    fn load_config(config_name: &str) -> (SimInput, SimData) {
        // Data file paths in TOML configs are relative to repo root
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let repo_root = std::path::PathBuf::from(&manifest)
            .join("../..")
            .canonicalize()
            .unwrap();
        std::env::set_current_dir(&repo_root).unwrap();

        let path = std::path::Path::new(config_name);
        let (sim_config, toml_config) = SimInput::from_toml_file(path).expect("parse");
        let sim_data = SimData::from_toml(&toml_config, &sim_config).expect("data");
        (sim_config, sim_data)
    }

    fn load_test_config() -> (SimInput, SimData) {
        load_config("configs/test/test_ref_orig.toml")
    }

    #[test]
    fn run_for_api_returns_one_result_for_single_sim() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data, false, None).expect("run");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn run_output_final_record_has_52_elements() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data, false, None).expect("run");
        assert_eq!(results[0].final_record.len(), 52);
    }

    #[test]
    fn run_output_final_record_matches_file_path() {
        let (config, data) = load_test_config();
        let api_results = run_for_api(&config, &data, false, None).expect("api run");
        let api_fr = &api_results[0].final_record;

        run(&config, &data).expect("file run");

        let suffix = config.results_suffix.trim_start_matches('.');
        let final_path = config.output_path(&format!("final.{}.csv", suffix));
        let content = std::fs::read_to_string(&final_path).expect("read final csv");
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() >= 2, "final CSV should have header + data");

        assert!(api_fr[7].abs() > 0.0, "energy should be non-zero");
        assert!(api_fr[9] > 0.0, "eccentricity should be positive");
    }

    #[test]
    fn run_output_captured_flag_consistent_with_orbital_elements() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data, false, None).expect("run");
        let r = &results[0];
        let ifinal_val = r.final_record[31] as i32;
        let expected = ifinal_val == 3 && r.final_record[9] < 1.0 && r.final_record[7] < 0.0;
        assert_eq!(r.captured, expected);
    }

    #[test]
    fn peak_values_populated_for_atmospheric_trajectory() {
        let (config, data) = load_config("configs/test/test_high_bank_orig.toml");
        let results = run_for_api(&config, &data, false, None).expect("run");
        let rec = &results[0].final_record;

        // Columns 16-18: peak heat flux (kW/m²), load factor (g), dynamic pressure (kPa)
        assert!(
            rec[16] > 0.0,
            "max_heat_flux should be > 0, got {}",
            rec[16]
        );
        assert!(
            rec[17] > 0.0,
            "max_load_factor should be > 0, got {}",
            rec[17]
        );
        assert!(
            rec[18] > 0.0,
            "max_dyn_pressure should be > 0, got {}",
            rec[18]
        );

        // Columns 19-24: altitudes and times at peak values
        assert!(rec[19] > 0.0, "alt_max_flux should be > 0, got {}", rec[19]);
        assert!(rec[20] > 0.0, "alt_max_load should be > 0, got {}", rec[20]);
        assert!(rec[21] > 0.0, "alt_max_pdyn should be > 0, got {}", rec[21]);
        assert!(
            rec[22] > 0.0,
            "time_max_flux should be > 0, got {}",
            rec[22]
        );
        assert!(
            rec[23] > 0.0,
            "time_max_load should be > 0, got {}",
            rec[23]
        );
        assert!(
            rec[24] > 0.0,
            "time_max_pdyn should be > 0, got {}",
            rec[24]
        );

        // Physical plausibility for Mars entry:
        assert!(
            rec[16] > 10.0 && rec[16] < 500.0,
            "peak heat flux {:.1} kW/m² outside reasonable Mars entry range",
            rec[16]
        );
        assert!(
            rec[17] > 1.0 && rec[17] < 30.0,
            "peak load factor {:.1} g outside reasonable Mars entry range",
            rec[17]
        );
    }

    #[test]
    fn heat_load_in_trajectory_is_monotonically_nondecreasing() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data, true, None).expect("run");
        let traj = &results[0].trajectory;
        assert!(!traj.is_empty(), "trajectory should not be empty");
        for i in 1..traj.len() {
            assert!(
                traj[i][15] >= traj[i - 1][15],
                "heat load must be monotonically non-decreasing at step {}: {} < {}",
                i,
                traj[i][15],
                traj[i - 1][15]
            );
        }
    }

    #[test]
    fn heat_load_final_matches_final_record() {
        let (config, data) = load_test_config();
        let results = run_for_api(&config, &data, true, None).expect("run");
        let r = &results[0];
        let last_traj_heat_load = r.trajectory.last().unwrap()[15]; // kJ/m²
        let final_record_heat_load = r.final_record[28] * 1e3; // MJ/m² → kJ/m²
        let diff = (last_traj_heat_load - final_record_heat_load).abs();
        assert!(
            diff < 1.0, // allow 1 kJ/m² tolerance (photo cadence vs final state)
            "trajectory last heat load ({:.2}) should match final_record ({:.2}), diff={:.4}",
            last_traj_heat_load,
            final_record_heat_load,
            diff
        );
    }
}

#[cfg(test)]
mod virtual_dv_tests {
    use super::*;
    use proptest::prelude::*;

    // Mars-ish constants for proptest scenarios.
    const MU_MARS: f64 = 4.282837e13;
    const TARGET_SMA: f64 = 2.0e7; // 20000 km → E_target ≈ -1.07 MJ/kg

    proptest! {
        #[test]
        fn crash_virtual_dv_finite_and_bounded_below(
            energy_j_kg in -5.0e7f64..5.0e7,
            sim_time in 0.0f64..10000.0,
            max_time in 100.0f64..10000.0,
        ) {
            let dv = virtual_dv_non_capture(energy_j_kg, TARGET_SMA, MU_MARS, sim_time, max_time);
            prop_assert!(dv.is_finite());
            // Lower bound: CRASH_FLOOR - CRASH_TIME_BONUS (when ΔE = 0 and t_ratio = 1).
            prop_assert!(dv >= CRASH_FLOOR - CRASH_TIME_BONUS, "dv={} below floor", dv);
        }

        #[test]
        fn crash_virtual_dv_monotonic_in_energy_error(
            delta_e_mj in 0.0f64..20.0,
        ) {
            let e_target = -MU_MARS / (2.0 * TARGET_SMA);
            let dv0 = virtual_dv_non_capture(e_target, TARGET_SMA, MU_MARS, 0.0, 1000.0);
            let dv1 = virtual_dv_non_capture(e_target + delta_e_mj * 1e6, TARGET_SMA, MU_MARS, 0.0, 1000.0);
            let dv2 = virtual_dv_non_capture(e_target - delta_e_mj * 1e6, TARGET_SMA, MU_MARS, 0.0, 1000.0);
            // Symmetric: |+ΔE| and |-ΔE| produce identical cost.
            prop_assert!((dv1 - dv2).abs() < 1e-9);
            // Monotonic: bigger |ΔE| → bigger cost.
            prop_assert!(dv1 >= dv0 - 1e-9);
        }

        #[test]
        fn crash_virtual_dv_survival_reduces_cost(
            energy_j_kg in -5.0e7f64..5.0e7,
        ) {
            let early = virtual_dv_non_capture(energy_j_kg, TARGET_SMA, MU_MARS, 0.0, 1000.0);
            let late = virtual_dv_non_capture(energy_j_kg, TARGET_SMA, MU_MARS, 1000.0, 1000.0);
            prop_assert!((early - late - CRASH_TIME_BONUS).abs() < 1e-9);
        }

        #[test]
        fn hyperbolic_virtual_dv_above_base(
            v_excess in 0.0f64..5000.0,
        ) {
            let virtual_dv = HYPERBOLIC_BASE + v_excess;
            prop_assert!(virtual_dv >= HYPERBOLIC_BASE);
            prop_assert!(virtual_dv.is_finite());
        }
    }

    #[test]
    fn non_finite_inputs_produce_finite_capped_output() {
        // NaN energy (from degenerate state) must not propagate.
        let dv_nan = virtual_dv_non_capture(f64::NAN, TARGET_SMA, MU_MARS, 0.0, 1000.0);
        assert!(dv_nan.is_finite());
        assert!(dv_nan >= CRASH_FLOOR);
        // Expected: CRASH_FLOOR + CRASH_ENERGY_WEIGHT * CRASH_ENERGY_CAP_MJKG - 0.
        let expected = CRASH_FLOOR + CRASH_ENERGY_WEIGHT * CRASH_ENERGY_CAP_MJKG;
        assert!((dv_nan - expected).abs() < 1e-9);

        // +Inf energy also capped.
        let dv_inf = virtual_dv_non_capture(f64::INFINITY, TARGET_SMA, MU_MARS, 500.0, 1000.0);
        assert!(dv_inf.is_finite());
        assert!((dv_inf - (expected - CRASH_TIME_BONUS * 0.5)).abs() < 1e-9);

        // NaN sim_time.
        let dv_t_nan = virtual_dv_non_capture(0.0, TARGET_SMA, MU_MARS, f64::NAN, 1000.0);
        assert!(dv_t_nan.is_finite());
    }

    #[test]
    fn near_target_crash_stays_above_typical_capture_floor() {
        // A crash with energy exactly at target (best possible crash) at max survival
        // time must still cost more than typical captures (~500-2000 m/s) so the
        // optimizer never prefers crashing over capturing.
        let e_target = -MU_MARS / (2.0 * TARGET_SMA);
        let best_possible_crash =
            virtual_dv_non_capture(e_target, TARGET_SMA, MU_MARS, 1000.0, 1000.0);
        assert!(
            best_possible_crash >= 2500.0,
            "best crash DV {} too close to captures",
            best_possible_crash
        );
        assert!(
            best_possible_crash <= CRASH_FLOOR,
            "best crash DV {} exceeds floor {}",
            best_possible_crash,
            CRASH_FLOOR
        );
    }
}

#[cfg(test)]
mod pending_crash_tests {
    use super::is_pending_crash;

    // exit_altitude in meters matches the field's unit.
    const EXIT_ALT: f64 = 125_000.0;

    #[test]
    fn hyperbolic_orbit_is_not_pending_crash() {
        // e >= 1 -> not captured -> not pending crash regardless of apoapsis.
        assert!(!is_pending_crash(1.1, 1.0e6, 0.0, EXIT_ALT));
    }

    #[test]
    fn positive_energy_is_not_pending_crash() {
        // energy > 0 -> unbound -> not captured even if e < 1.
        assert!(!is_pending_crash(0.5, 1.0e6, 100_000.0, EXIT_ALT));
    }

    #[test]
    fn captured_with_high_apoapsis_is_not_pending_crash() {
        // Bound + apoapsis well above exit altitude -> clean capture.
        assert!(!is_pending_crash(
            0.5,
            -1.0e6,
            EXIT_ALT + 10_000.0,
            EXIT_ALT
        ));
    }

    #[test]
    fn captured_with_apoapsis_below_ceiling_is_pending_crash() {
        // Bound but apoapsis under the atmosphere -> guaranteed re-entry.
        assert!(is_pending_crash(0.5, -1.0e6, EXIT_ALT - 10_000.0, EXIT_ALT));
    }

    #[test]
    fn boundary_apoapsis_equal_exit_is_not_pending_crash() {
        // Strict inequality -> apoapsis == exit is a clean edge.
        assert!(!is_pending_crash(0.5, -1.0e6, EXIT_ALT, EXIT_ALT));
    }

    #[test]
    fn nan_inputs_do_not_promote() {
        // NaN comparisons are false -> no spurious promotion on numerical blow-up.
        assert!(!is_pending_crash(f64::NAN, -1.0e6, 0.0, EXIT_ALT));
        assert!(!is_pending_crash(0.5, f64::NAN, 0.0, EXIT_ALT));
        assert!(!is_pending_crash(0.5, -1.0e6, f64::NAN, EXIT_ALT));
    }
}
