//! Foundational simulation types and constants.
//!
//! Leaf module owning the per-tick `SimState`, the `TermReason` / `SimError`
//! enums, and the shared virtual-DV / integrator constants. Both `runner`
//! (main loop) and `finalize` (termination classification) depend on this
//! module rather than on each other, breaking the prior inside-out coupling.

use crate::data::SimData;
use crate::gnc::control::pilot::PilotState;
use crate::gnc::guidance::dispatch::GuidanceState;
use crate::gnc::navigation::estimator::NavigationFilter;
use crate::integration::dopri45::Dopri45State;
use crate::integration::events::EventRecord;
use crate::integration::sequencer::SequencerState;
use crate::simulation::init;
use std::fmt;
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

/// Default absolute tolerances for DOPRI45, one per state component.
/// State = [r(m), lon(rad), lat(rad), V(m/s), gamma(rad), psi(rad), flux(J/m²), time(s)]
pub(crate) const DOPRI45_ATOL: [f64; 8] = [
    1.0,  // r: 1 m on ~3.4e6 m
    1e-8, // lon: ~0.03 m at Mars equator
    1e-8, // lat: ~0.03 m
    1e-3, // V: 1 mm/s on ~5700 m/s
    1e-8, // gamma: ~0.03 m position equiv
    1e-8, // psi: ~0.03 m
    1e-2, // flux: 0.01 J/m² on O(1e6-1e7) total (rtol dominates)
    1e-6, // time: machine-level for identity derivative
];

pub(crate) const EVENT_TOL: f64 = 1e-3; // 1 ms event location tolerance

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
