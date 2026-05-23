//! Per-tick advance function, extracted from `runner::run_single`.
//!
//! `step_one_tick` advances `SimState` by exactly one outer GNC tick.
//! `runner::run_single` loops over it; `BatchedSimulation::step` (Task 1.3+) will
//! call it with `forced_bank = Some(policy_action)`.

use crate::config::{IntegrationMode, PlanetConfig, SimInput};
use crate::data::SimData;
use crate::gnc::control::angle_utils::{shortest_angle_diff, wrap_to_pi};
use crate::gnc::control::pilot;
use crate::gnc::guidance::dispatch;
use crate::gnc::navigation::coordinates::geodetic_from_spherical;
use crate::integration::events::{self, EventContext, EventDef, EventType};
use crate::physics::atmosphere;
use crate::simulation::runner::{
    DEG_TO_RAD, SimState, TermReason, build_photo_values, effective_airspeed,
    integrate_adaptive_with_events, integrate_step, navigate_from_state,
    promote_pending_crash_if_applicable, track_peak_values,
};

/// Outcome of one outer guidance tick.
///
/// Events triggered during a tick (bounce, atmosphere_exit, crash, phase_transition) are
/// accumulated in `SimState::event_records`. Consumers (e.g. `BatchedSimulation`) can drain
/// them between ticks via `std::mem::take(&mut state.event_records)`.
#[allow(dead_code)]
pub struct TickOutcome {
    /// Commanded bank angle used this tick (rad). Echoed from caller for BatchedSimulation;
    /// computed from guidance dispatch for the existing runner path.
    pub bank_commanded: f64,
    /// True if simulation should terminate after this tick (atmosphere exit, crash, pending
    /// crash, NaN/Inf, or max_time reached).
    pub done: bool,
    /// Termination code matching ifinal semantics (see runner.rs constants).
    pub ifinal: Option<i32>,
}

/// Advance `state` by exactly one outer GNC tick.
///
/// Verbatim extraction of `run_single`'s loop body. The loop invariant is:
/// on entry, `state.term == TermReason::None`; on exit, either `state.term`
/// is still `None` (tick completed normally) or it is set to a terminal reason
/// and `outcome.done` is `true`.
///
/// `forced_bank`: when `Some(radians)`, overrides guidance output with this bank
/// command. Used by `BatchedSimulation` to inject RL policy actions. The existing
/// `run_single` call site always passes `None`.
#[allow(clippy::too_many_arguments)]
pub fn step_one_tick(
    state: &mut SimState,
    config: &SimInput,
    data: &SimData,
    planet: &PlanetConfig,
    forced_bank: Option<f64>,
    event_defs: &[EventDef],
    event_ctx: &EventContext,
) -> TickOutcome {
    let dt = state.dt;

    if !state.first_iter {
        state.sim_time += dt;
    }
    state.first_iter = false;

    let flags = state.sequencer.update(state.sim_time, &data.periods);

    // Step Gauss-Markov density perturbation
    if let Some(gm) = state.gm_config {
        use rand_distr::Distribution;
        let z: f64 = state
            .gm_normal
            .as_ref()
            .unwrap()
            .sample(state.gm_rng.as_mut().unwrap());
        state.run_state.density_perturbation = crate::data::dispersions::step_density_perturbation(
            state.run_state.density_perturbation,
            dt,
            gm.tau,
            gm.sigma,
            z,
        );
    }

    // === Navigation + Guidance + Pilot ===
    if !config.reference_trajectory {
        let mut nav_out = navigate_from_state(state, data, planet);

        state.dynamic_pressure_for_photo = nav_out.dynamic_pressure_estimated;
        state.density_estimate_for_photo = nav_out.density_guidance;
        state.guidance_phase_for_photo = nav_out.guidance_phase;

        // Latch reference velocity at the phase 1→2 transition
        if nav_out.phase_transition_flag == 1 {
            state.guidance_state.reference_velocity = nav_out.reference_velocity;
        }

        // Compute thermal fractions for guidance limiter + NN inputs.
        // Instantaneous heat flux uses the same formula as track_peak_values.
        {
            let (alt_for_thermal, _) =
                geodetic_from_spherical(state.state[0], state.state[1], state.state[2], planet);
            let rho_thermal = atmosphere::density(
                &data.atmosphere,
                alt_for_thermal,
                state.run_state.density_bias,
                state.run_state.density_perturbation,
            );
            let v_eff_thermal = effective_airspeed(
                state.state[3],
                state.state[4],
                state.state[5],
                state.state[2],
                alt_for_thermal,
                data,
                &state.run_state,
            );
            let heat_flux_now = data.capsule.cq * rho_thermal.sqrt() * v_eff_thermal.powf(3.05);

            nav_out.heat_flux_fraction = if data.constraints.max_heat_flux > 0.0 {
                heat_flux_now / data.constraints.max_heat_flux
            } else {
                0.0
            };
            nav_out.heat_load_fraction = if data.constraints.max_heat_load > 0.0 {
                state.state[6] / data.constraints.max_heat_load
            } else {
                0.0
            };
        }

        // Cache for RL observation building (build_nn_input reads this via last_nav_output())
        state.last_nav = nav_out;

        let guidance_out = dispatch::guidance_step(
            &nav_out,
            state.bank_angle,
            state.sim_time,
            state.reference_bank_angle,
            &mut state.guidance_state,
            data,
            planet,
            config.reference_trajectory,
            config.guidance_type,
        );

        // Supervised-trace push gates on guidance.longitudinal_active=1 so we
        // don't pollute the dataset with inhibited-guidance ticks where the
        // recorded magnitude is just |reference_bank_angle| (constant per
        // config). Active-only rows give the regression target real signal.
        if config.collect_supervised && guidance_out.longitudinal_active == 1 {
            // Explicit full mask: select all 21 inputs.
            // Passing None would trigger the backward-compat default (first 16 only).
            const FULL_MASK: [usize; 21] = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            ];
            let nn_input = crate::gnc::guidance::neural::build_nn_input(
                &nav_out,
                Some(&FULL_MASK),
                None, // no ablation
                data,
                planet,
                data.target_orbit.inclination,
                state.guidance_state.reference_velocity,
            );
            // Supervised target is the pre-lateral, pre-shaper magnitude so
            // the warm-start cloned NN replaces ONLY the predictor-corrector.
            // Under magnitude_only deploy the NN's output is fed BACK INTO
            // lateral / thermal_limiter / command_shaper, so capturing the
            // post-shaper signed command here would cause double-shaping (the
            // shaper runs again at deploy time on the NN's own output).
            state
                .supervised_trace
                .push((nn_input, guidance_out.pre_lateral_magnitude));
        }

        let bank_angle_commanded = forced_bank.unwrap_or(guidance_out.bank_angle_commanded);

        let max_rate = data.capsule.max_bank_rate * (1.0 + state.run_state.max_bank_rate_bias);
        state.pilot_state = pilot::apply_pilot(
            &data.pilot,
            bank_angle_commanded,
            &state.pilot_state,
            data.periods.pilot,
            max_rate,
            &state.run_state.pilot_biases,
        );

        let bank_change = shortest_angle_diff(state.bank_angle, state.pilot_state.bank_angle).abs();
        if bank_change > 1e-10 {
            state.cumulative_bank_change_deg += bank_change / DEG_TO_RAD;
        }

        state.bank_angle = wrap_to_pi(state.pilot_state.bank_angle);
        state.aoa = guidance_out.aoa_commanded;

        if state.is_single && (state.step < 5 || state.step.is_multiple_of(50)) {
            let (dbg_alt, _) =
                geodetic_from_spherical(state.state[0], state.state[1], state.state[2], planet);
            eprintln!(
                "  step={} t={:.1} bank={:.3}deg aoa={:.3}deg longitudinal={} alt={:.1}km vel={:.1}",
                state.step,
                state.sim_time,
                state.bank_angle.to_degrees(),
                state.aoa.to_degrees(),
                guidance_out.longitudinal_active,
                dbg_alt / 1e3,
                state.state[3],
            );
        }
    } else {
        // Reference trajectory mode: compute pdyn from truth state for photo output
        let (alt_truth, _) =
            geodetic_from_spherical(state.state[0], state.state[1], state.state[2], planet);
        let rho_truth = atmosphere::density(
            &data.atmosphere,
            alt_truth,
            state.run_state.density_bias,
            state.run_state.density_perturbation,
        );
        state.dynamic_pressure_for_photo = 0.5 * rho_truth * state.state[3] * state.state[3];
        state.density_estimate_for_photo = rho_truth;
    }

    // === Photo snapshot ===
    if state.write_photo && flags.photo {
        let sim_time = state.sim_time;
        let dynamic_pressure_for_photo = state.dynamic_pressure_for_photo;
        let density_estimate_for_photo = state.density_estimate_for_photo;
        let sim_idx = state.sim_idx;
        let cumulative_bank_change_deg = state.cumulative_bank_change_deg;
        let density_gain = state.nav_filter.density_gain();
        let run_state_snap = state.run_state.clone();
        let cumulative_flux = state.state[6];
        let guidance_phase_for_photo = state.guidance_phase_for_photo;
        let photo_line = build_photo_values(
            state,
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
        state.photo_lines.push(photo_line);
    }

    // === Integration step ===
    let mut adaptive_events: Vec<events::TriggeredEvent> = Vec::new();
    match &data.integration_mode {
        IntegrationMode::FixedGill => {
            let run_state_snap = state.run_state.clone();
            integrate_step(state, dt, planet, data, &run_state_snap);
        }
        IntegrationMode::AdaptiveDopri45(adaptive_config) => {
            let run_state_snap = state.run_state.clone();
            let sim_time = state.sim_time;
            let result = integrate_adaptive_with_events(
                state,
                dt,
                adaptive_config,
                planet,
                data,
                &run_state_snap,
                event_defs,
                event_ctx,
                sim_time,
            );
            adaptive_events = result.triggered;
        }
    }

    // Populate GNC context on event records from this tick's state
    // (GNC quantities are constant within a tick, so the current values apply)
    let n_prev_events = state.event_records.len() - adaptive_events.len();
    let density_gain = state.nav_filter.density_gain();
    for rec in state.event_records[n_prev_events..].iter_mut() {
        rec.bank_angle_deg = state.bank_angle / DEG_TO_RAD;
        rec.aoa_deg = state.aoa / DEG_TO_RAD;
        rec.cumulative_bank_change_deg = state.cumulative_bank_change_deg;
        rec.guidance_phase = state.guidance_phase_for_photo as f64;
        rec.density_gain = density_gain;
    }

    let (altitude, _lat_geo) =
        geodetic_from_spherical(state.state[0], state.state[1], state.state[2], planet);

    let run_state_snap = state.run_state.clone();
    let sim_time = state.sim_time;
    track_peak_values(state, altitude, sim_time, data, &run_state_snap);

    // === Process adaptive integrator events (in chronological order) ===
    for triggered in &adaptive_events {
        let event_type = event_defs[triggered.event_index].event_type;
        match event_type {
            EventType::Bounce => {
                if !state.bounced {
                    state.bounced = true;
                    state.bounce_alt = triggered.state[0] - planet.equatorial_radius;
                    state.bounce_time = triggered.time;
                }
            }
            EventType::AtmosphereExit => {
                if state.bounced {
                    state.sim_time = triggered.time;
                    state.term = TermReason::AtmosphereExit;
                }
            }
            EventType::Crash => {
                state.sim_time = triggered.time;
                state.term = TermReason::Crash;
            }
            EventType::PhaseTransition => {
                // Phase transition: nav layer picks up on next tick
            }
        }
    }

    // NaN/Inf safety net: extreme GA parameters can blow up numerically.
    // All termination checks evaluate to false on NaN, so the loop would spin forever.
    let mut early_break = false;
    if state.state.iter().any(|x| !x.is_finite()) {
        state.term = TermReason::Crash;
        early_break = true;
    }

    if !early_break
        && let Some(timeout) = state.wall_timeout
        && state.wall_start.elapsed() > timeout
    {
        state.term = TermReason::Timeout;
        early_break = true;
    }

    if !early_break {
        // === Termination checks (simple event-based: FixedGill only) ===
        if matches!(data.integration_mode, IntegrationMode::FixedGill) {
            if altitude <= 0.0 {
                state.term = TermReason::Crash;
            }
            if state.bounced && altitude >= state.exit_altitude {
                state.term = TermReason::AtmosphereExit;
            }

            // Bounce detection
            if !state.bounced && state.state[4].sin() >= 0.0 {
                state.bounced = true;
                state.bounce_alt = altitude;
                state.bounce_time = state.sim_time;
            }
        }

        // === Checks that run for both integration modes ===
        if state.sim_time >= state.max_time {
            state.term = TermReason::Timeout;
        }

        // Atmospheric apoapsis crash: bounced, now descending again, still inside atmosphere
        // → the apoapsis is below the atmospheric ceiling, guaranteed re-entry crash.
        // Guard: bounce altitude must be above 20 km to exclude transient FPA sign changes
        // during the deep pass (aggressive bank reversals can momentarily push FPA positive).
        if state.bounced
            && state.bounce_alt > 20e3
            && state.state[4].sin() < 0.0
            && altitude < state.exit_altitude
            && state.term == TermReason::None
        {
            state.term = TermReason::Crash;
        }

        // Trapped orbit detection: after bounce, if the osculating semi-major axis
        // implies the orbit fits entirely within the atmosphere, the vehicle is trapped.
        // Uses vis-viva (a = -mu/(2E)) with inertial velocity — no FPA dependency,
        // catches oscillating trajectories that the FPA-based check above misses.
        if state.bounced && state.bounce_alt > 20e3 && state.term == TermReason::None {
            use crate::gnc::navigation::coordinates::{norm, to_absolute_cartesian};
            let (_, v_abs) = to_absolute_cartesian(
                state.state[0],
                state.state[1],
                state.state[2],
                state.state[3],
                state.state[4],
                state.state[5],
                planet,
            );
            let speed_abs = norm(&v_abs);
            let energy_abs = speed_abs * speed_abs / 2.0 - planet.mu / state.state[0];
            // Bound orbit with semi-major axis small enough that apoapsis < atmosphere ceiling
            // a*(1+e) < r_exit. Conservative: use a*2 < r_exit (assumes e<1, so a*(1+e) < 2a).
            if energy_abs < 0.0 {
                let sma = -planet.mu / (2.0 * energy_abs);
                let r_exit = planet.equatorial_radius + state.exit_altitude;
                if 2.0 * sma < r_exit {
                    state.term = TermReason::Crash;
                }
            }
        }
    }

    state.step += 1;

    let done = state.term != TermReason::None;
    if done {
        promote_pending_crash_if_applicable(state, planet);
    }
    let ifinal = if done {
        Some(match state.term {
            TermReason::AtmosphereExit => 3,
            TermReason::Crash => 1,
            TermReason::PendingCrash => 4,
            TermReason::Timeout => 2,
            TermReason::None => unreachable!(),
        })
    } else {
        None
    };

    TickOutcome {
        bank_commanded: state.bank_angle,
        done,
        ifinal,
    }
}
