//! Thermal safety limiter -- bank angle override near heat flux / heat load limits.
//!
//! Smooth ramp from guidance-commanded bank angle toward full lift-up (cos_bank=1.0)
//! as thermal quantities approach constraint limits. GA-tunable activation thresholds
//! and ramp exponents per scheme. Applied to unsigned-magnitude schemes only.

/// GA-tunable thermal limiter parameters.
#[derive(Debug, Clone, Copy)]
pub struct ThermalLimiterParams {
    /// Fraction of max_heat_flux at which ramp begins (0.6--0.95).
    pub heat_flux_activation: f64,
    /// Fraction of max_heat_load at which ramp begins (0.6--0.95).
    pub heat_load_activation: f64,
    /// Ramp shape for heat flux (1.0=linear, 2.0=quadratic).
    pub heat_flux_ramp_exponent: f64,
    /// Ramp shape for heat load (1.0=linear, 2.0=quadratic).
    pub heat_load_ramp_exponent: f64,
}

impl Default for ThermalLimiterParams {
    fn default() -> Self {
        Self {
            heat_flux_activation: 1.0,   // 1.0 = never activates
            heat_load_activation: 1.0,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 1.0,
        }
    }
}

/// Apply thermal safety limit to a commanded cos(bank) value.
///
/// Blends `cos_bank_cmd` toward 1.0 (full lift-up) as thermal fractions
/// approach 1.0. Returns the limited cos(bank), always in [cos_bank_cmd, 1.0].
///
/// Both limiters are evaluated independently; the most restrictive wins.
/// If both fractions are below their activation thresholds, returns `cos_bank_cmd` unchanged.
pub fn apply_thermal_limit(
    cos_bank_cmd: f64,
    heat_flux_fraction: f64,
    heat_load_fraction: f64,
    params: &ThermalLimiterParams,
) -> f64 {
    let alpha_flux = compute_alpha(heat_flux_fraction, params.heat_flux_activation, params.heat_flux_ramp_exponent);
    let alpha_load = compute_alpha(heat_load_fraction, params.heat_load_activation, params.heat_load_ramp_exponent);
    let alpha = alpha_flux.max(alpha_load);
    (1.0 - alpha) * cos_bank_cmd + alpha * 1.0
}

/// Compute ramp blending factor alpha for a single thermal quantity.
///
/// Returns 0.0 below activation, 1.0 at or above 100%, smooth ramp in between.
fn compute_alpha(fraction: f64, activation: f64, exponent: f64) -> f64 {
    if fraction <= activation {
        0.0
    } else if fraction >= 1.0 {
        1.0
    } else {
        ((fraction - activation) / (1.0 - activation)).powf(exponent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn active_params() -> ThermalLimiterParams {
        ThermalLimiterParams {
            heat_flux_activation: 0.8,
            heat_load_activation: 0.85,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 2.0,
        }
    }

    #[test]
    fn below_activation_returns_unchanged() {
        let p = active_params();
        let cos_cmd = 0.3;
        let result = apply_thermal_limit(cos_cmd, 0.5, 0.5, &p);
        assert_relative_eq!(result, cos_cmd, epsilon = 1e-12);
    }

    #[test]
    fn at_limit_returns_full_lift_up() {
        let p = active_params();
        let result = apply_thermal_limit(-0.5, 1.0, 0.0, &p);
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn heat_load_at_limit_returns_full_lift_up() {
        let p = active_params();
        let result = apply_thermal_limit(-0.5, 0.0, 1.0, &p);
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn above_limit_returns_full_lift_up() {
        let p = active_params();
        let result = apply_thermal_limit(-0.5, 1.5, 0.0, &p);
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn mid_ramp_linear() {
        let p = ThermalLimiterParams {
            heat_flux_activation: 0.8,
            heat_load_activation: 1.0, // disabled
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 1.0,
        };
        let cos_cmd = 0.0;
        // fraction=0.9, activation=0.8 => alpha = (0.9-0.8)/(1.0-0.8) = 0.5
        // result = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        let result = apply_thermal_limit(cos_cmd, 0.9, 0.0, &p);
        assert_relative_eq!(result, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn mid_ramp_quadratic() {
        let p = ThermalLimiterParams {
            heat_flux_activation: 1.0, // disabled
            heat_load_activation: 0.8,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 2.0,
        };
        let cos_cmd = 0.0;
        // fraction=0.9, activation=0.8, exponent=2.0
        // alpha = ((0.9-0.8)/(1.0-0.8))^2 = 0.5^2 = 0.25
        // result = 0.75 * 0.0 + 0.25 * 1.0 = 0.25
        let result = apply_thermal_limit(cos_cmd, 0.0, 0.9, &p);
        assert_relative_eq!(result, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn most_restrictive_wins() {
        let p = ThermalLimiterParams {
            heat_flux_activation: 0.8,
            heat_load_activation: 0.8,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 1.0,
        };
        // flux fraction = 0.85 => alpha_flux = (0.85-0.8)/0.2 = 0.25
        // load fraction = 0.95 => alpha_load = (0.95-0.8)/0.2 = 0.75
        // alpha_max = 0.75
        let cos_cmd = 0.0;
        let result = apply_thermal_limit(cos_cmd, 0.85, 0.95, &p);
        let expected = 0.25 * 0.0 + 0.75 * 1.0;
        assert_relative_eq!(result, expected, epsilon = 1e-12);
    }

    #[test]
    fn zero_fractions_no_intervention() {
        let p = active_params();
        let cos_cmd = -0.7;
        let result = apply_thermal_limit(cos_cmd, 0.0, 0.0, &p);
        assert_relative_eq!(result, cos_cmd, epsilon = 1e-12);
    }

    #[test]
    fn default_params_never_activate() {
        let p = ThermalLimiterParams::default();
        let cos_cmd = -1.0;
        // activation=1.0 means fraction must exceed 1.0 to trigger
        let result = apply_thermal_limit(cos_cmd, 0.99, 0.99, &p);
        assert_relative_eq!(result, cos_cmd, epsilon = 1e-12);
    }

    #[test]
    fn negative_cos_bank_pushed_toward_one() {
        let p = ThermalLimiterParams {
            heat_flux_activation: 0.5,
            heat_load_activation: 1.0,
            heat_flux_ramp_exponent: 1.0,
            heat_load_ramp_exponent: 1.0,
        };
        let cos_cmd = -1.0; // full lift-down
        // fraction=1.0 => alpha=1.0
        let result = apply_thermal_limit(cos_cmd, 1.0, 0.0, &p);
        assert_relative_eq!(result, 1.0, epsilon = 1e-12);
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn output_between_cmd_and_one(
                cos_cmd in -1.0..=1.0_f64,
                flux_frac in 0.0..2.0_f64,
                load_frac in 0.0..2.0_f64,
                flux_act in 0.5..1.0_f64,
                load_act in 0.5..1.0_f64,
                flux_exp in 0.5..3.0_f64,
                load_exp in 0.5..3.0_f64,
            ) {
                let p = ThermalLimiterParams {
                    heat_flux_activation: flux_act,
                    heat_load_activation: load_act,
                    heat_flux_ramp_exponent: flux_exp,
                    heat_load_ramp_exponent: load_exp,
                };
                let result = apply_thermal_limit(cos_cmd, flux_frac, load_frac, &p);
                prop_assert!(result.is_finite(), "result not finite: {}", result);
                prop_assert!(result >= cos_cmd - 1e-12, "result {} < cos_cmd {}", result, cos_cmd);
                prop_assert!(result <= 1.0 + 1e-12, "result {} > 1.0", result);
            }

            #[test]
            fn monotonic_in_fraction(
                cos_cmd in -1.0..=1.0_f64,
                frac_lo in 0.0..1.0_f64,
                frac_hi in 0.0..1.0_f64,
                activation in 0.5..0.99_f64,
                exponent in 0.5..3.0_f64,
            ) {
                let p = ThermalLimiterParams {
                    heat_flux_activation: activation,
                    heat_load_activation: 1.0,
                    heat_flux_ramp_exponent: exponent,
                    heat_load_ramp_exponent: 1.0,
                };
                let lo = frac_lo.min(frac_hi);
                let hi = frac_lo.max(frac_hi);
                let r_lo = apply_thermal_limit(cos_cmd, lo, 0.0, &p);
                let r_hi = apply_thermal_limit(cos_cmd, hi, 0.0, &p);
                // Higher fraction => more intervention => result closer to 1.0
                prop_assert!(r_hi >= r_lo - 1e-12, "not monotonic: r_hi={} < r_lo={}", r_hi, r_lo);
            }
        }
    }
}
