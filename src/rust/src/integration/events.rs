//! Event detection framework for the adaptive integrator.
//!
//! Provides Brent's root-finding method and a typed event system for locating
//! zero-crossings (bounce, atmosphere exit, crash, phase transition) within
//! DOPRI45 sub-steps using dense output.

use crate::integration::dopri45::dopri45_dense;
use crate::simulation::runner::TermReason;

// ── Brent's root-finder ──────────────────────────────────────────────────────

/// Find a root of `f` in [a, b] using Brent's method.
///
/// Combines inverse quadratic interpolation with bisection for guaranteed
/// convergence. 50-iteration cap. `tol` is the half-width tolerance on the
/// root interval.
///
/// Panics if `f(a)` and `f(b)` have the same sign.
pub fn brent(mut a: f64, mut b: f64, tol: f64, f: &mut impl FnMut(f64) -> f64) -> f64 {
    let mut fa = f(a);
    let mut fb = f(b);

    assert!(
        fa * fb <= 0.0,
        "brent: f(a) and f(b) must have opposite signs (f(a)={fa}, f(b)={fb})"
    );

    // Ensure |f(b)| <= |f(a)| so b is always the better end
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    #[allow(unused_assignments)]
    let mut s = b;
    let mut d = 0.0_f64;

    for _ in 0..50 {
        if fb.abs() < tol || (b - a).abs() < tol {
            return b;
        }

        if fa != fc && fb != fc {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions under which we fall back to bisection
        let bisect = {
            let between = if a < b { a..=b } else { b..=a };
            !between.contains(&s)
                || (mflag && (s - b).abs() >= (b - c).abs() / 2.0)
                || (!mflag && (s - b).abs() >= (c - d).abs() / 2.0)
                || (mflag && (b - c).abs() < tol)
                || (!mflag && (c - d).abs() < tol)
        };

        if bisect {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    b
}

// ── Event framework types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventAction {
    Terminate(TermReason),
    Record,
    PhaseTransition,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventType {
    Bounce,
    AtmosphereExit,
    Crash,
    PhaseTransition,
}

/// Contextual constants passed to event evaluation functions.
#[derive(Debug, Clone)]
pub struct EventContext {
    pub planet_radius: f64,
    pub polar_radius: f64,
    pub exit_altitude: f64,
    pub exit_velocity_threshold: f64,
}

/// Geocentric radius of the reference ellipsoid at geocentric latitude `lat`.
/// For a spherical planet (equatorial == polar), returns equatorial_radius.
fn ellipsoid_radius(lat: f64, equatorial: f64, polar: f64) -> f64 {
    let c = lat.cos();
    let s = lat.sin();
    let a2c = equatorial * equatorial * c;
    let b2s = polar * polar * s;
    let ac = equatorial * c;
    let bs = polar * s;
    ((a2c * a2c + b2s * b2s) / (ac * ac + bs * bs)).sqrt()
}

/// Definition of a trackable event.
pub struct EventDef {
    /// Sign function: positive on one side, negative on the other.
    pub eval: fn(&[f64; 8], &EventContext) -> f64,
    /// +1 = only trigger on rising zero-crossing, -1 = falling, 0 = both.
    pub direction: i8,
    pub action: EventAction,
    pub event_type: EventType,
}

/// A located event with its time, state, and type.
#[derive(Debug, Clone)]
pub struct EventRecord {
    pub time: f64,
    pub state: [f64; 8],
    pub event_type: EventType,
}

/// A located event with precise zero-crossing position from Brent's root-finding.
pub struct TriggeredEvent {
    pub event_index: usize,
    /// Fractional step position in [0, 1] at the zero-crossing.
    pub theta: f64,
    pub state: [f64; 8],
    /// Absolute simulation time at the zero-crossing.
    pub time: f64,
}

// ── Event functions ──────────────────────────────────────────────────────────

/// sin(gamma): rising through zero means FPA crosses 0 upward (bounce).
fn event_bounce(state: &[f64; 8], _ctx: &EventContext) -> f64 {
    state[4].sin()
}

/// Altitude above atmosphere exit threshold (latitude-corrected for oblateness).
fn event_atmosphere_exit(state: &[f64; 8], ctx: &EventContext) -> f64 {
    let r_surface = ellipsoid_radius(state[2], ctx.planet_radius, ctx.polar_radius);
    (state[0] - r_surface) - ctx.exit_altitude
}

/// Altitude above planet surface (latitude-corrected); falling through zero = impact.
fn event_crash(state: &[f64; 8], ctx: &EventContext) -> f64 {
    let r_surface = ellipsoid_radius(state[2], ctx.planet_radius, ctx.polar_radius);
    state[0] - r_surface
}

/// threshold - V: rising through zero = V dropped below threshold.
fn event_phase_transition(state: &[f64; 8], ctx: &EventContext) -> f64 {
    ctx.exit_velocity_threshold - state[3]
}

/// Build the standard aerocapture event set.
pub fn build_aerocapture_events() -> Vec<EventDef> {
    vec![
        EventDef {
            eval: event_bounce,
            direction: 1,
            action: EventAction::Record,
            event_type: EventType::Bounce,
        },
        EventDef {
            eval: event_atmosphere_exit,
            direction: 1,
            action: EventAction::Terminate(TermReason::AtmosphereExit),
            event_type: EventType::AtmosphereExit,
        },
        EventDef {
            eval: event_crash,
            direction: -1,
            action: EventAction::Terminate(TermReason::Crash),
            event_type: EventType::Crash,
        },
        EventDef {
            eval: event_phase_transition,
            direction: 1,
            action: EventAction::PhaseTransition,
            event_type: EventType::PhaseTransition,
        },
    ]
}

// ── Event evaluation helpers ─────────────────────────────────────────────────

/// Evaluate all event functions at `state` and return cached values.
pub fn evaluate_events(state: &[f64; 8], events: &[EventDef], ctx: &EventContext) -> Vec<f64> {
    events.iter().map(|e| (e.eval)(state, ctx)).collect()
}

/// Detect and locate the earliest zero-crossing in a DOPRI45 substep.
///
/// After each accepted substep `[y0, y5]` with step size `h`, evaluates all events,
/// checks for sign changes that match direction filters, and uses Brent's method on
/// the dense output interpolant to locate the earliest crossing.
///
/// - `g_start`: cached event values at `y0` (from `evaluate_events`)
/// - `tol`: root-finding tolerance in seconds (e.g. 1e-3); converted internally to theta
///
/// Returns `None` if no event triggered, or `Some(TriggeredEvent)` for the earliest.
#[allow(clippy::too_many_arguments)]
pub fn check_events_and_locate(
    y0: &[f64; 8],
    y5: &[f64; 8],
    h: f64,
    k1: &[f64; 8],
    k7: &[f64; 8],
    events: &[EventDef],
    ctx: &EventContext,
    g_start: &[f64],
    tol: f64,
    t_base: f64,
) -> Option<TriggeredEvent> {
    let tol_theta = tol / h;
    let mut earliest: Option<TriggeredEvent> = None;

    for (i, event) in events.iter().enumerate() {
        let g0 = g_start[i];
        let g1 = (event.eval)(y5, ctx);

        // No crossing if same sign; also skip if already at zero at start
        if g0 == 0.0 || g0 * g1 > 0.0 {
            continue;
        }

        // Direction filter
        let rising = g0 < 0.0 && g1 >= 0.0;
        let falling = g0 > 0.0 && g1 <= 0.0;
        let triggered = match event.direction {
            1 => rising,
            -1 => falling,
            _ => rising || falling,
        };
        if !triggered {
            continue;
        }

        let theta = brent(0.0, 1.0, tol_theta, &mut |th| {
            let y = dopri45_dense(y0, y5, h, k1, k7, th);
            (event.eval)(&y, ctx)
        });

        let is_earlier = earliest.as_ref().is_none_or(|e| theta < e.theta);
        if is_earlier {
            let state = dopri45_dense(y0, y5, h, k1, k7, theta);
            earliest = Some(TriggeredEvent {
                event_index: i,
                theta,
                state,
                time: t_base + theta * h,
            });
        }
    }

    earliest
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // Brent tests

    #[test]
    fn brent_finds_sin_root() {
        let root = brent(3.0, 3.5, 1e-12, &mut |x| x.sin());
        assert!((root - PI).abs() < 1e-11);
    }

    #[test]
    fn brent_finds_linear_root() {
        let root = brent(0.0, 5.0, 1e-12, &mut |x| x - 2.5);
        assert!((root - 2.5).abs() < 1e-11);
    }

    #[test]
    fn brent_root_at_endpoint_a() {
        // f(0)=0: bracket [0,1], sign condition 0*1 <= 0 is satisfied
        let root = brent(0.0, 1.0, 1e-12, &mut |x| x);
        assert!(root.abs() < 1e-11);
    }

    #[test]
    fn brent_root_at_endpoint_b() {
        let root = brent(-1.0, 0.0, 1e-12, &mut |x| x);
        assert!(root.abs() < 1e-11);
    }

    #[test]
    fn brent_converges_on_tight_bracket() {
        let a = PI - 5e-5;
        let b = PI + 5e-5;
        let root = brent(a, b, 1e-12, &mut |x| x.sin());
        assert!((root - PI).abs() < 1e-11);
    }

    #[test]
    #[should_panic]
    fn brent_panics_on_same_sign() {
        brent(1.0, 2.0, 1e-12, &mut |x| x * x + 1.0);
    }

    // Event function tests

    #[test]
    fn event_functions_sign_correctness() {
        let ctx = EventContext {
            planet_radius: 3_396_200.0,
            polar_radius: 3_376_200.0,
            exit_altitude: 200_000.0,
            exit_velocity_threshold: 5_500.0,
        };

        // Descending state: gamma < 0, V above threshold, alt at exit altitude, above ground
        let descending: [f64; 8] = [
            ctx.planet_radius + ctx.exit_altitude, // r = planet_radius + exit_alt
            0.0,                                   // lon
            0.0,                                   // lat
            6_000.0,                               // V > threshold
            -0.1_f64,                              // gamma (negative = descending)
            0.0,                                   // psi
            0.0,                                   // flux
            0.0,                                   // time
        ];

        // Ascending state: gamma > 0, V below threshold, alt inside atmosphere, above ground
        let ascending: [f64; 8] = [
            ctx.planet_radius + 50_000.0, // r = inside atmosphere
            0.0,
            0.0,
            5_000.0, // V < threshold
            0.1_f64, // gamma (positive = ascending)
            0.0,
            0.0,
            0.0,
        ];

        // Bounce (sin(gamma)): negative when descending, positive when ascending
        assert!(event_bounce(&descending, &ctx) < 0.0);
        assert!(event_bounce(&ascending, &ctx) > 0.0);

        // AtmosphereExit: zero when alt == exit_altitude; positive above, negative below
        let at_exit = descending; // r == planet_radius + exit_altitude => value == 0
        assert_eq!(event_atmosphere_exit(&at_exit, &ctx), 0.0);
        // Inside atmosphere: negative
        assert!(event_atmosphere_exit(&ascending, &ctx) < 0.0);

        // Crash: positive above ground (both states are above ground)
        assert!(event_crash(&descending, &ctx) > 0.0);
        assert!(event_crash(&ascending, &ctx) > 0.0);

        // PhaseTransition: negative when V > threshold (descending), positive when V < threshold (ascending)
        assert!(event_phase_transition(&descending, &ctx) < 0.0);
        assert!(event_phase_transition(&ascending, &ctx) > 0.0);
    }

    #[test]
    fn build_aerocapture_events_has_four_events() {
        let events = build_aerocapture_events();
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].event_type, EventType::Bounce);
        assert_eq!(events[1].event_type, EventType::AtmosphereExit);
        assert_eq!(events[2].event_type, EventType::Crash);
        assert_eq!(events[3].event_type, EventType::PhaseTransition);
    }

    // check_events_and_locate tests

    #[test]
    fn check_events_locates_zero_crossing() {
        use crate::integration::dopri45::{Dopri45State, dopri45_step_with_stages};

        let ctx = EventContext {
            planet_radius: 0.0,
            polar_radius: 0.0,
            exit_altitude: 1.5, // event at state[0] = 1.5
            exit_velocity_threshold: 0.0,
        };
        let events = vec![EventDef {
            eval: |state: &[f64; 8], ctx: &EventContext| state[0] - ctx.exit_altitude,
            direction: 1,
            action: EventAction::Terminate(TermReason::AtmosphereExit),
            event_type: EventType::AtmosphereExit,
        }];

        let atol = [1e-12; 8];
        let rtol = 1e-12;
        let mut state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y0 = state;
        let h = 1.0;
        let (result, stages) = dopri45_step_with_stages(
            &mut state,
            h,
            &mut Dopri45State::new(),
            &atol,
            rtol,
            &mut |_| {
                let mut d = [0.0; 8];
                d[0] = 1.0;
                d[7] = 1.0;
                d
            },
        );
        assert!(result.accepted);
        let y5 = state;
        let k1 = &stages[0];
        let k7 = &stages[6];

        let g_start = evaluate_events(&y0, &events, &ctx);
        let triggered =
            check_events_and_locate(&y0, &y5, h, k1, k7, &events, &ctx, &g_start, 1e-3, 0.0);
        assert!(triggered.is_some());
        let t = triggered.unwrap();
        assert_eq!(t.event_index, 0);
        assert!(
            (t.theta - 0.5).abs() < 0.01,
            "theta={} expected ~0.5",
            t.theta
        );
        assert!(
            (t.state[0] - 1.5).abs() < 0.01,
            "state[0]={} expected ~1.5",
            t.state[0]
        );
    }

    #[test]
    fn check_events_respects_direction_filter() {
        // Rising crossing (state[0] goes from 1.0 to 2.0), but direction=-1 means only falling.
        use crate::integration::dopri45::{Dopri45State, dopri45_step_with_stages};

        let ctx = EventContext {
            planet_radius: 0.0,
            polar_radius: 0.0,
            exit_altitude: 1.5,
            exit_velocity_threshold: 0.0,
        };
        let events = vec![EventDef {
            eval: |state: &[f64; 8], ctx: &EventContext| state[0] - ctx.exit_altitude,
            direction: -1, // only falling -- should NOT fire on a rising crossing
            action: EventAction::Terminate(TermReason::Crash),
            event_type: EventType::Crash,
        }];

        let atol = [1e-12; 8];
        let rtol = 1e-12;
        let mut state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y0 = state;
        let h = 1.0;
        let (result, stages) = dopri45_step_with_stages(
            &mut state,
            h,
            &mut Dopri45State::new(),
            &atol,
            rtol,
            &mut |_| {
                let mut d = [0.0; 8];
                d[0] = 1.0;
                d[7] = 1.0;
                d
            },
        );
        assert!(result.accepted);
        let y5 = state;
        let k1 = &stages[0];
        let k7 = &stages[6];

        let g_start = evaluate_events(&y0, &events, &ctx);
        let triggered =
            check_events_and_locate(&y0, &y5, h, k1, k7, &events, &ctx, &g_start, 1e-3, 0.0);
        assert!(
            triggered.is_none(),
            "direction=-1 should not trigger on a rising crossing"
        );
    }

    #[test]
    fn check_events_picks_earliest_of_two() {
        // Two events: one at state[0] = 1.3 (theta~0.3), one at state[0] = 1.7 (theta~0.7).
        // Both rising. Should pick the first (theta~0.3).
        use crate::integration::dopri45::{Dopri45State, dopri45_step_with_stages};

        let ctx = EventContext {
            planet_radius: 0.0,
            polar_radius: 0.0,
            exit_altitude: 1.3,           // event 0 fires here
            exit_velocity_threshold: 1.7, // event 1 fires when state[0] reaches 1.7
        };
        let events = vec![
            EventDef {
                eval: |state: &[f64; 8], ctx: &EventContext| state[0] - ctx.exit_altitude,
                direction: 1,
                action: EventAction::Terminate(TermReason::AtmosphereExit),
                event_type: EventType::AtmosphereExit,
            },
            EventDef {
                eval: |state: &[f64; 8], ctx: &EventContext| state[0] - ctx.exit_velocity_threshold,
                direction: 1,
                action: EventAction::PhaseTransition,
                event_type: EventType::PhaseTransition,
            },
        ];

        let atol = [1e-12; 8];
        let rtol = 1e-12;
        let mut state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y0 = state;
        let h = 1.0;
        let (result, stages) = dopri45_step_with_stages(
            &mut state,
            h,
            &mut Dopri45State::new(),
            &atol,
            rtol,
            &mut |_| {
                let mut d = [0.0; 8];
                d[0] = 1.0;
                d[7] = 1.0;
                d
            },
        );
        assert!(result.accepted);
        let y5 = state;
        let k1 = &stages[0];
        let k7 = &stages[6];

        let g_start = evaluate_events(&y0, &events, &ctx);
        let triggered =
            check_events_and_locate(&y0, &y5, h, k1, k7, &events, &ctx, &g_start, 1e-3, 0.0);
        assert!(triggered.is_some());
        let t = triggered.unwrap();
        assert_eq!(t.event_index, 0, "should pick the earlier event (index 0)");
        assert!(
            (t.theta - 0.3).abs() < 0.01,
            "theta={} expected ~0.3",
            t.theta
        );
        assert!(
            (t.state[0] - 1.3).abs() < 0.01,
            "state[0]={} expected ~1.3",
            t.state[0]
        );
    }
}
