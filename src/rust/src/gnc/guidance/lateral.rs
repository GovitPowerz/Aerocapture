//! Lateral guidance — inclination corridor roll reversal logic.
//!
//! Shared by all unsigned-magnitude guidance schemes (EqGlide, EnergyController,
//! PredGuid, FNPAG). Schemes that produce signed bank angles (NeuralNetwork,
//! PiecewiseConstant) bypass this entirely.

use crate::config::Planet;
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;

/// Lateral guidance configuration (TOML-configurable, per-scheme tunable).
#[derive(Debug, Clone)]
pub struct LateralParams {
    /// Velocity scaling for corridor width (m/s).
    pub corridor_slope: f64,
    /// Baseline corridor width at low velocity (rad).
    pub corridor_intercept: f64,
    /// Energy at which lateral guidance arms (J/kg). Upper bound of the active window.
    pub lateral_activation: f64,
    /// Energy below which lateral guidance disarms (J/kg). Lower bound of the active window.
    pub lateral_inhibition: f64,
    /// Maximum number of roll reversals per trajectory.
    pub max_reversals: i32,
}

impl Default for LateralParams {
    fn default() -> Self {
        Self {
            corridor_slope: 0.0,
            corridor_intercept: 0.0,
            lateral_activation: 0.0,
            lateral_inhibition: 0.0,
            max_reversals: 0,
        }
    }
}

/// Lateral guidance mutable state (per-run).
#[derive(Debug, Clone)]
pub struct LateralState {
    /// Current roll direction sign (±1.0).
    pub roll_sign: f64,
    /// Number of roll reversals executed so far.
    pub n_reversals: i32,
}

impl LateralState {
    pub fn new(initial_bank: f64) -> Self {
        Self {
            roll_sign: if initial_bank >= 0.0 { 1.0 } else { -1.0 },
            n_reversals: 0,
        }
    }
}

/// Compute roll sign based on inclination error and corridor boundary.
///
/// Returns `true` if a reversal was triggered this step.
pub fn lateral_guidance(
    params: &LateralParams,
    state: &mut LateralState,
    nav: &NavigationOutput,
    target_inclination: f64,
    energy: f64,
    bank_magnitude: f64,
    planet: &Planet,
) -> bool {
    // Guard: corridor_slope must be positive to avoid division by zero
    if params.corridor_slope <= 0.0 {
        return false;
    }

    // Energy window gate: lateral_inhibition <= energy <= lateral_activation
    if energy > params.lateral_activation || energy < params.lateral_inhibition {
        return false;
    }

    // Skip degenerate bank angles (near 0 or π, where roll sign is physically meaningless)
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
    let velocity = nav.velocity_estimated[0];

    // Corridor boundary: narrows with decreasing velocity (clamped to non-negative)
    let corridor_width =
        ((velocity / params.corridor_slope).powi(4) + params.corridor_intercept).max(0.0);

    // Check reversal conditions
    if inclination_error.abs() < corridor_width {
        return false;
    }
    if state.n_reversals >= params.max_reversals {
        return false;
    }

    let previous_sign = state.roll_sign;

    if inclination_error > corridor_width {
        state.roll_sign = -1.0;
    } else if inclination_error < -corridor_width {
        state.roll_sign = 1.0;
    }

    // Check if sign actually changed
    if state.roll_sign * previous_sign < 0.0 {
        state.n_reversals += 1;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Planet;
    use crate::gnc::navigation::estimator::NavigationOutput;

    fn test_nav() -> NavigationOutput {
        let r = Planet::Mars.equatorial_radius() + 50_000.0;
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
            corridor_slope: 13080.458,
            corridor_intercept: 0.0,
            lateral_activation: 0.0,
            lateral_inhibition: -1e12,
            max_reversals: 5,
        }
    }

    #[test]
    fn no_reversal_when_outside_energy_window() {
        let params = LateralParams {
            lateral_activation: -1e12,
            lateral_inhibition: -1e12,
            ..active_params()
        };
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        let reversed = lateral_guidance(&params, &mut state, &nav, 1.0, 1e6, 1.0, &Planet::Mars);
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn no_reversal_when_inclination_within_corridor() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        let orbit = elements::from_spherical(
            nav.position_estimated[0],
            nav.position_estimated[1],
            nav.position_estimated[2],
            nav.velocity_estimated[0],
            nav.velocity_estimated[1],
            nav.velocity_estimated[2],
            &Planet::Mars,
        );
        let reversed = lateral_guidance(
            &params,
            &mut state,
            &nav,
            orbit.inclination,
            -1e6,
            1.0,
            &Planet::Mars,
        );
        assert!(!reversed);
        assert_eq!(state.n_reversals, 0);
    }

    #[test]
    fn reversal_when_inclination_exceeds_corridor() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        assert_eq!(state.roll_sign, 1.0);
        let nav = test_nav();
        let reversed = lateral_guidance(&params, &mut state, &nav, 10.0, -1e6, 1.0, &Planet::Mars);
        assert!(reversed);
        assert_eq!(state.roll_sign, -1.0);
        assert_eq!(state.n_reversals, 1);
    }

    #[test]
    fn reversal_negative_error() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        state.roll_sign = -1.0;
        let nav = test_nav();
        let reversed = lateral_guidance(&params, &mut state, &nav, -10.0, -1e6, 1.0, &Planet::Mars);
        assert!(reversed);
        assert_eq!(state.roll_sign, 1.0);
        assert_eq!(state.n_reversals, 1);
    }

    #[test]
    fn respects_max_reversals() {
        let params = LateralParams {
            max_reversals: 1,
            ..active_params()
        };
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        let r1 = lateral_guidance(&params, &mut state, &nav, 10.0, -1e6, 1.0, &Planet::Mars);
        assert!(r1);
        assert_eq!(state.n_reversals, 1);
        assert_eq!(state.roll_sign, -1.0);
        let r2 = lateral_guidance(&params, &mut state, &nav, -10.0, -1e6, 1.0, &Planet::Mars);
        assert!(!r2);
        assert_eq!(state.n_reversals, 1);
        assert_eq!(state.roll_sign, -1.0);
    }

    #[test]
    fn no_reversal_when_bank_near_zero() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        let reversed =
            lateral_guidance(&params, &mut state, &nav, 10.0, -1e6, 1e-15, &Planet::Mars);
        assert!(!reversed);
    }

    #[test]
    fn roll_sign_always_pm_one() {
        let params = active_params();
        let mut state = LateralState::new(1.0);
        let nav = test_nav();
        lateral_guidance(&params, &mut state, &nav, 10.0, -1e6, 1.0, &Planet::Mars);
        assert!(state.roll_sign == 1.0 || state.roll_sign == -1.0);
        lateral_guidance(&params, &mut state, &nav, -10.0, -1e6, 1.0, &Planet::Mars);
        assert!(state.roll_sign == 1.0 || state.roll_sign == -1.0);
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
                let params = LateralParams {
                    corridor_slope: 13080.458,
                    corridor_intercept: 0.0,
                    lateral_activation: 0.0,
                    lateral_inhibition: -1e12,
                    max_reversals: 5,
                };
                let mut state = LateralState::new(1.0);
                lateral_guidance(&params, &mut state, &nav, target, -1e6, 1.0, &Planet::Mars);
                prop_assert!(state.roll_sign == 1.0 || state.roll_sign == -1.0);
            }

            #[test]
            fn n_reversals_monotonic(
                nav in arb_nav(),
                targets in proptest::collection::vec(-2.0_f64..2.0, 5..20),
            ) {
                let params = LateralParams {
                    corridor_slope: 13080.458,
                    corridor_intercept: 0.0,
                    lateral_activation: 0.0,
                    lateral_inhibition: -1e12,
                    max_reversals: 100,
                };
                let mut state = LateralState::new(1.0);
                let mut prev_n = 0;
                for t in &targets {
                    lateral_guidance(&params, &mut state, &nav, *t, -1e6, 1.0, &Planet::Mars);
                    prop_assert!(state.n_reversals >= prev_n);
                    prev_n = state.n_reversals;
                }
            }

            #[test]
            fn corridor_width_positive(v in 1000.0_f64..8000.0) {
                let slope = 13080.458_f64;
                let intercept = 0.01_f64;
                let width = (v / slope).powi(4) + intercept;
                prop_assert!(width > 0.0);
            }
        }
    }
}
