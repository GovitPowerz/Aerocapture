//! Lateral guidance -- predictive roll reversal via inclination error projection.
//!
//! Shared by all unsigned-magnitude guidance schemes (EqGlide, EnergyController,
//! PredGuid, FNPAG). Schemes that produce signed bank angles (NeuralNetwork,
//! PiecewiseConstant) bypass this entirely.

use crate::config::PlanetConfig;
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;

/// Predictive lateral guidance configuration (TOML-configurable, GA-tunable).
#[derive(Debug, Clone)]
pub struct LateralParams {
    /// Lookahead horizon for inclination error projection (seconds).
    pub tau: f64,
    /// Projected inclination error threshold for reversal trigger (radians).
    pub threshold: f64,
    /// Minimum time between consecutive reversals (seconds).
    pub min_reversal_interval: f64,
    /// Energy at which lateral guidance arms (J/kg). Upper bound of the active window.
    pub lateral_activation: f64,
    /// Energy below which lateral guidance disarms (J/kg). Lower bound of the active window.
    pub lateral_inhibition: f64,
    /// Maximum number of roll reversals per trajectory.
    pub max_reversals: i32,
}

impl Default for LateralParams {
    /// Default produces **inactive** lateral guidance: `tau = 0.0` triggers
    /// the early-return guard. Use explicit values (or TOML `[guidance.lateral]`)
    /// to activate.
    fn default() -> Self {
        Self {
            tau: 0.0,
            threshold: 0.0,
            min_reversal_interval: 0.0,
            lateral_activation: 0.0,
            lateral_inhibition: 0.0,
            max_reversals: 0,
        }
    }
}

/// Lateral guidance mutable state (per-run).
#[derive(Debug, Clone)]
pub struct LateralState {
    /// Current roll direction sign (+-1.0).
    pub roll_sign: f64,
    /// Number of roll reversals executed so far.
    pub n_reversals: i32,
    /// Previous tick's inclination error (None on first tick).
    pub prev_inclination_error: Option<f64>,
    /// Previous tick's guidance time (seconds).
    pub prev_time: f64,
    /// Time of most recent reversal (seconds).
    pub last_reversal_time: f64,
}

impl LateralState {
    pub fn new(initial_bank: f64) -> Self {
        Self {
            roll_sign: if initial_bank >= 0.0 { 1.0 } else { -1.0 },
            n_reversals: 0,
            prev_inclination_error: None,
            prev_time: 0.0,
            last_reversal_time: f64::NEG_INFINITY,
        }
    }
}

/// Compute roll sign based on projected inclination error.
///
/// Projects the inclination error forward by `tau` seconds using finite-difference
/// rate estimation. Reverses when the projected error exceeds `threshold` and the
/// minimum reversal interval has elapsed.
///
/// Returns `true` if a reversal was triggered this step.
#[allow(clippy::too_many_arguments)]
pub fn lateral_guidance(
    params: &LateralParams,
    state: &mut LateralState,
    nav: &NavigationOutput,
    target_inclination: f64,
    energy: f64,
    bank_magnitude: f64,
    sim_time: f64,
    planet: &PlanetConfig,
) -> bool {
    // Guard: tau must be positive to activate predictive lateral guidance
    if params.tau <= 0.0 {
        return false;
    }

    // Energy window gate: lateral_inhibition <= energy <= lateral_activation
    if energy > params.lateral_activation || energy < params.lateral_inhibition {
        return false;
    }

    // Skip degenerate bank angles (near 0 or pi, where roll sign is physically meaningless)
    let pi = std::f64::consts::PI;
    if bank_magnitude.abs() < 1e-10 || (bank_magnitude.abs() - pi).abs() < 1e-10 {
        return false;
    }

    // Compute current orbital inclination
    let orbit = elements::from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    let inclination_error = target_inclination - orbit.inclination;
    let current_time = sim_time;

    // First tick: store state and return (no rate available yet)
    if state.prev_inclination_error.is_none() {
        state.prev_inclination_error = Some(inclination_error);
        state.prev_time = current_time;
        return false;
    }

    // Compute inclination error rate via finite difference
    let dt = current_time - state.prev_time;
    let di_err_dt = if dt > 1e-12 {
        (inclination_error - state.prev_inclination_error.unwrap()) / dt
    } else {
        0.0
    };

    // Update history for next tick
    state.prev_inclination_error = Some(inclination_error);
    state.prev_time = current_time;

    // Project inclination error forward by tau seconds
    let i_err_projected = inclination_error + di_err_dt * params.tau;

    // Check if projected error exceeds threshold
    if i_err_projected.abs() <= params.threshold {
        return false;
    }

    // Enforce reversal budget
    if state.n_reversals >= params.max_reversals {
        return false;
    }

    // Enforce minimum reversal interval (anti-chatter)
    if current_time - state.last_reversal_time < params.min_reversal_interval {
        return false;
    }

    // Determine desired roll sign from projected error (same convention as legacy)
    let desired_sign = if i_err_projected > 0.0 { -1.0 } else { 1.0 };

    // Only reverse if sign actually changes
    if desired_sign * state.roll_sign < 0.0 {
        state.roll_sign = desired_sign;
        state.n_reversals += 1;
        state.last_reversal_time = current_time;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PlanetConfig;
    use crate::gnc::navigation::estimator::NavigationOutput;

    fn test_nav() -> NavigationOutput {
        let r = PlanetConfig::mars().equatorial_radius + 50_000.0;
        NavigationOutput {
            position_estimated: [r, 0.0, 0.0],
            velocity_estimated: [5000.0, -0.15, 0.6],
            acceleration_estimated: [50.0, -8.0],
            aero_coefficients: [1.269, -0.205],
            density_guidance: 0.001,
            ..Default::default()
        }
    }

    fn active_params() -> LateralParams {
        LateralParams {
            tau: 15.0,
            threshold: 0.01, // ~0.57 deg
            min_reversal_interval: 5.0,
            lateral_activation: 0.0,
            lateral_inhibition: -1e12,
            max_reversals: 5,
        }
    }

    /// Helper: run two guidance ticks to seed the finite difference, then
    /// return a state ready for the third (decision) tick.
    fn seeded_state(params: &LateralParams, target: f64, t0: f64, t1: f64) -> (LateralState, f64) {
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        lateral_guidance(
            params,
            &mut state,
            &nav,
            target,
            -1e6,
            1.0,
            t0,
            &PlanetConfig::mars(),
        );
        assert!(state.prev_inclination_error.is_some());
        (state, t1)
    }

    #[test]
    fn no_reversal_on_first_tick() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        let reversed = lateral_guidance(
            &params,
            &mut state,
            &nav,
            10.0,
            -1e6,
            1.0,
            0.0,
            &PlanetConfig::mars(),
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
        assert!(state.prev_inclination_error.is_some());
    }

    #[test]
    fn no_reversal_when_error_converging() {
        let params = active_params();
        let planet = PlanetConfig::mars();
        let nav = test_nav();
        let orbit = elements::from_spherical(
            nav.position_estimated[0],
            nav.position_estimated[1],
            nav.position_estimated[2],
            nav.velocity_estimated[0],
            nav.velocity_estimated[1],
            nav.velocity_estimated[2],
            &planet,
        );
        // Target inclination very close to actual: error ~ 0 < threshold
        let target = orbit.inclination + 0.001; // 0.001 rad < threshold 0.01
        let (mut state, time) = seeded_state(&params, target, 0.0, 1.0);
        let reversed =
            lateral_guidance(&params, &mut state, &nav, target, -1e6, 1.0, time, &planet);
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn reversal_when_projected_error_exceeds_threshold() {
        let params = active_params();
        let nav = test_nav();
        let (mut state, time) = seeded_state(&params, 10.0, 0.0, 1.0);
        let reversed = lateral_guidance(
            &params,
            &mut state,
            &nav,
            10.0,
            -1e6,
            1.0,
            time,
            &PlanetConfig::mars(),
        );
        assert!(reversed);
        assert_eq!(state.roll_sign, -1.0); // positive error -> negative sign
        assert_eq!(state.n_reversals, 1);
    }

    #[test]
    fn reversal_negative_projected_error() {
        let params = active_params();
        let nav = test_nav();
        let (mut state, time) = seeded_state(&params, -10.0, 0.0, 1.0);
        state.roll_sign = -1.0; // start negative
        let reversed = lateral_guidance(
            &params,
            &mut state,
            &nav,
            -10.0,
            -1e6,
            1.0,
            time,
            &PlanetConfig::mars(),
        );
        assert!(reversed);
        assert_eq!(state.roll_sign, 1.0); // negative error -> positive sign
        assert_eq!(state.n_reversals, 1);
    }

    #[test]
    fn respects_min_reversal_interval() {
        let params = active_params(); // min_reversal_interval = 5.0
        let nav = test_nav();
        let (mut state, time) = seeded_state(&params, 10.0, 0.0, 1.0);
        // First reversal at t=1
        let r1 = lateral_guidance(
            &params,
            &mut state,
            &nav,
            10.0,
            -1e6,
            1.0,
            time,
            &PlanetConfig::mars(),
        );
        assert!(r1);
        assert_eq!(state.last_reversal_time, 1.0);

        // Try second reversal at t=3 (only 2s after first, < 5s interval)
        let r2 = lateral_guidance(
            &params,
            &mut state,
            &nav,
            -10.0,
            -1e6,
            1.0,
            3.0,
            &PlanetConfig::mars(),
        );
        assert!(!r2);
        assert_eq!(state.n_reversals, 1);

        // Try at t=7 (6s after first reversal, > 5s interval)
        let r3 = lateral_guidance(
            &params,
            &mut state,
            &nav,
            -10.0,
            -1e6,
            1.0,
            7.0,
            &PlanetConfig::mars(),
        );
        assert!(r3);
        assert_eq!(state.n_reversals, 2);
    }

    #[test]
    fn respects_max_reversals() {
        let params = LateralParams {
            max_reversals: 1,
            min_reversal_interval: 0.0, // disable interval for this test
            ..active_params()
        };
        let nav = test_nav();
        let (mut state, time) = seeded_state(&params, 10.0, 0.0, 1.0);
        let r1 = lateral_guidance(
            &params,
            &mut state,
            &nav,
            10.0,
            -1e6,
            1.0,
            time,
            &PlanetConfig::mars(),
        );
        assert!(r1);
        assert_eq!(state.n_reversals, 1);

        // Budget exhausted: second reversal blocked
        let r2 = lateral_guidance(
            &params,
            &mut state,
            &nav,
            -10.0,
            -1e6,
            1.0,
            10.0,
            &PlanetConfig::mars(),
        );
        assert!(!r2);
        assert_eq!(state.n_reversals, 1);
    }

    #[test]
    fn no_reversal_outside_energy_window() {
        let active = active_params();
        let params_narrow = LateralParams {
            lateral_activation: -1e12,
            lateral_inhibition: -1e12,
            ..active_params()
        };
        let nav = test_nav();
        // Seed state using active params (wide energy window) so prev_inclination_error is set
        let (mut state, time) = seeded_state(&active, 10.0, 0.0, 1.0);
        // Now call with narrow-window params and energy=1e6 (outside window)
        let reversed = lateral_guidance(
            &params_narrow,
            &mut state,
            &nav,
            10.0,
            1e6,
            1.0,
            time,
            &PlanetConfig::mars(),
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn no_reversal_when_bank_near_zero() {
        let params = active_params();
        let nav = test_nav();
        let (mut state, time) = seeded_state(&params, 10.0, 0.0, 1.0);
        let reversed = lateral_guidance(
            &params,
            &mut state,
            &nav,
            10.0,
            -1e6,
            1e-15,
            time,
            &PlanetConfig::mars(),
        );
        assert!(!reversed);
    }

    #[test]
    fn no_reversal_when_bank_near_pi() {
        let params = active_params();
        let nav = test_nav();
        let (mut state, time) = seeded_state(&params, 10.0, 0.0, 1.0);
        let reversed = lateral_guidance(
            &params,
            &mut state,
            &nav,
            10.0,
            -1e6,
            std::f64::consts::PI,
            time,
            &PlanetConfig::mars(),
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn tau_zero_produces_inactive() {
        let params = LateralParams::default(); // tau = 0.0
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        let reversed = lateral_guidance(
            &params,
            &mut state,
            &nav,
            10.0,
            -1e6,
            1.0,
            0.0,
            &PlanetConfig::mars(),
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn no_same_sign_reversal() {
        // If desired_sign == current sign, no reversal fires
        let params = LateralParams {
            min_reversal_interval: 0.0,
            ..active_params()
        };
        let nav = test_nav();
        let (mut state, time) = seeded_state(&params, 10.0, 0.0, 1.0);
        // Positive error -> desired sign -1.0. Pre-set roll_sign = -1.0
        state.roll_sign = -1.0;
        let reversed = lateral_guidance(
            &params,
            &mut state,
            &nav,
            10.0,
            -1e6,
            1.0,
            time,
            &PlanetConfig::mars(),
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn arb_nav() -> impl Strategy<Value = NavigationOutput> {
            (
                3.4e6_f64..3.6e6,
                -std::f64::consts::PI..std::f64::consts::PI,
                -1.0_f64..1.0,
                3000.0_f64..7000.0,
                -0.3_f64..0.1,
                -std::f64::consts::PI..std::f64::consts::PI,
            )
                .prop_map(|(r, lon, lat, v, fpa, hdg)| NavigationOutput {
                    position_estimated: [r, lon, lat],
                    velocity_estimated: [v, fpa, hdg],
                    acceleration_estimated: [50.0, -8.0],
                    aero_coefficients: [1.269, -0.205],
                    density_guidance: 0.001,
                    ..Default::default()
                })
        }

        proptest! {
            #[test]
            fn roll_sign_is_pm_one(nav in arb_nav(), target in -2.0_f64..2.0) {
                let params = active_params();
                let mut state = LateralState::new(1.0);
                lateral_guidance(&params, &mut state, &nav, target, -1e6, 1.0, 0.0, &PlanetConfig::mars());
                lateral_guidance(&params, &mut state, &nav, target, -1e6, 1.0, 1.0, &PlanetConfig::mars());
                prop_assert!(state.roll_sign == 1.0 || state.roll_sign == -1.0);
            }

            #[test]
            fn n_reversals_monotonic(
                nav in arb_nav(),
                targets in proptest::collection::vec(-2.0_f64..2.0, 5..20),
            ) {
                let params = LateralParams {
                    min_reversal_interval: 0.0,
                    ..active_params()
                };
                let mut state = LateralState::new(1.0);
                let mut prev_n = 0;
                for (i, t) in targets.iter().enumerate() {
                    lateral_guidance(&params, &mut state, &nav, *t, -1e6, 1.0, i as f64, &PlanetConfig::mars());
                    prop_assert!(state.n_reversals >= prev_n);
                    prev_n = state.n_reversals;
                }
            }

            #[test]
            fn n_reversals_bounded(
                nav in arb_nav(),
                targets in proptest::collection::vec(-2.0_f64..2.0, 5..30),
                max_rev in 1_i32..10,
            ) {
                let params = LateralParams {
                    max_reversals: max_rev,
                    min_reversal_interval: 0.0,
                    ..active_params()
                };
                let mut state = LateralState::new(1.0);
                for (i, t) in targets.iter().enumerate() {
                    lateral_guidance(&params, &mut state, &nav, *t, -1e6, 1.0, i as f64, &PlanetConfig::mars());
                }
                prop_assert!(state.n_reversals <= max_rev);
            }

            #[test]
            fn projected_error_finite(
                nav in arb_nav(),
                target in -2.0_f64..2.0,
                tau in 0.1_f64..100.0,
            ) {
                let params = LateralParams {
                    tau,
                    min_reversal_interval: 0.0,
                    ..active_params()
                };
                let mut state = LateralState::new(1.0);
                lateral_guidance(&params, &mut state, &nav, target, -1e6, 1.0, 0.0, &PlanetConfig::mars());
                lateral_guidance(&params, &mut state, &nav, target, -1e6, 1.0, 1.0, &PlanetConfig::mars());
                // If we got here without panic, the projected error was finite
                prop_assert!(state.roll_sign.is_finite());
                prop_assert!(state.n_reversals >= 0);
            }
        }
    }
}
