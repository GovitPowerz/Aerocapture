//! Equilibrium Glide guidance.
//!
//! Computes bank angle to maintain equilibrium between gravitational,
//! centrifugal, and aerodynamic lift forces. The vehicle "glides" along
//! a path where the vertical acceleration is zero (d²h/dt² ≈ 0).
//!
//! The equilibrium condition gives:
//!   cos(bank) = (g - V²/r) * m / L
//!
//! where g is local gravity, V is velocity, r is radius, m is mass,
//! and L is lift force. This is the simplest closed-form aerocapture
//! guidance law — no reference trajectory or prediction needed.
//!
//! For aerocapture, pure equilibrium glide tends to dissipate too much
//! energy (crashes) or too little (skip-out). We augment it with a
//! radial velocity feedback term to dampen altitude oscillations and
//! a velocity-dependent bias to control energy dissipation rate.

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::geodetic_from_spherical;
use crate::gnc::navigation::estimator::NavigationOutput;

/// Compute equilibrium glide bank angle.
///
/// Returns the bank angle magnitude (always positive) in radians.
/// The caller handles roll sign via lateral guidance.
pub fn equilibrium_glide_bank(
    nav: &NavigationOutput,
    data: &SimData,
    planet: &PlanetConfig,
) -> f64 {
    let r = nav.position_estimated[0];
    let v = nav.velocity_estimated[0];

    // Local gravity (simplified — use mu/r² for the dominant term)
    let mu = planet.mu;
    let g = mu / (r * r);

    // Centrifugal acceleration
    let v2_over_r = v * v / r;

    // Lift force per unit mass: L/m = 0.5 * rho * V² * S * Cz / m
    let (altitude, _) = geodetic_from_spherical(
        r,
        nav.position_estimated[1],
        nav.position_estimated[2],
        planet,
    );
    let rho = data
        .atmosphere_onboard
        .density_at(altitude, &data.atmosphere);
    let cz = nav.aero_coefficients[1]; // lift coefficient from navigation
    let sref = data.capsule.reference_area;
    let mass = data.capsule.mass;
    let lift_accel = 0.5 * rho * v * v * sref * cz.abs() / mass;

    if lift_accel.abs() < 1e-10 {
        // No lift available — hold moderate bank angle
        return 60.0_f64.to_radians();
    }

    // Base equilibrium: cos(bank) = (g - V²/r) / lift_accel
    let cos_eq = (g - v2_over_r) / lift_accel;

    let params = &data.guidance.eq_glide;

    // Radial velocity damping: if sinking, reduce bank (more lift-up);
    // if rising, increase bank (more lift-down toward equilibrium)
    let hdot = v * nav.velocity_estimated[1].sin();
    let k_hdot = params.k_hdot_scale / v.max(100.0);
    let cos_hdot_correction = -k_hdot * hdot;

    // Velocity-dependent bias: at hyperbolic velocities, we want more drag
    // (higher bank angle) to dissipate energy. As velocity drops toward
    // circular, reduce bank to stop dissipation.
    let v_circular = (mu / r).sqrt();
    let v_ratio = v / v_circular;
    let velocity_bias = if v_ratio > params.v_ratio_threshold {
        -params.velocity_bias_high * (v_ratio - params.v_ratio_threshold).min(1.0)
    } else {
        params.velocity_bias_low * (params.v_ratio_threshold - v_ratio).min(0.5)
    };

    // Altitude-dependent correction: if too low, bias toward lift-up
    let alt_km = altitude / 1e3;
    let alt_bias = if alt_km < params.alt_bias_threshold {
        params.velocity_bias_low * (1.0 - alt_km / params.alt_bias_threshold)
    } else {
        0.0
    };

    let cos_bank = (cos_eq + cos_hdot_correction + velocity_bias + alt_bias)
        .clamp(params.cos_bank_min, params.cos_bank_max);
    let bank = cos_bank.acos();

    // Safety clamp: never go below 15° (skip-out) or above 120° (crash risk)
    bank.clamp(15.0_f64.to_radians(), 120.0_f64.to_radians())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    use crate::data::aerodynamics::AeroTables;
    use crate::data::atmosphere::{AtmosphereModel, DensityProfile};
    use crate::data::capsule::Capsule;
    use crate::data::guidance_params::GuidanceParams;
    use crate::data::incidence::IncidenceProfile;
    use crate::data::pilot::{PilotModel, PilotType};
    use crate::data::{
        Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit, SimData,
        SphericalState, SuccessCriteria, TimePeriods,
    };

    fn test_nav(velocity: f64) -> NavigationOutput {
        let r = 3_396_200.0 + 50_000.0; // Mars radius + 50 km
        NavigationOutput {
            position_estimated: [r, 0.0, 0.0],
            velocity_estimated: [velocity, -0.15, 0.6],
            acceleration_estimated: [50.0, -8.0],
            aero_coefficients: [1.269, -0.205],
            density_guidance: 0.001,
            density_exit: 1e-6,
            dynamic_pressure_estimated: 0.5 * 0.001 * velocity * velocity,
            energy_estimated: -1e6,
            ..Default::default()
        }
    }

    fn test_sim_data() -> SimData {
        SimData {
            capsule: Capsule {
                mass: 1089.0,
                reference_area: 14.7,
                cq: 0.00008242,
                max_bank_rate: 15.0_f64.to_radians(),
                periods: TimePeriods::default(),
            },
            aero: AeroTables {
                n_points: 2,
                incidence: vec![-0.5, 0.0],
                cx: vec![1.269, 1.269],
                cz: vec![-0.205, -0.205],
                equilibrium_aoa: -0.48,
                ..Default::default()
            },
            atmosphere: AtmosphereModel {
                n_points: 3,
                altitudes: vec![0.0, 50_000.0, 130_000.0],
                densities: vec![0.02, 0.001, 1e-8],
                ref_density: 1e-8,
                scale_factor: 1e-4,
                ref_altitude: 130_000.0,
                gas_constant: 1.3,
                density_profile: DensityProfile::default(),
            },
            atmosphere_onboard: crate::data::atmosphere::OnboardAtmosphereModel::Identical,
            entry: EntryConditions {
                state: SphericalState {
                    altitude: 130_000.0,
                    velocity: 5687.0,
                    flight_path: -10.8_f64.to_radians(),
                    ..Default::default()
                },
                initial_bank: 64.77_f64.to_radians(),
                initial_aoa: -27.5_f64.to_radians(),
                initial_date: 0.0,
            },
            guidance: GuidanceParams {
                density_filter_gain: 0.8,
                exit_velocity_threshold: 4400.0,
                exit_altitude_threshold: 60_000.0,
                ..Default::default()
            },
            incidence: IncidenceProfile {
                n_points: 2,
                altitudes: vec![-10_000.0, 150_000.0],
                incidences: vec![-0.48, -0.48],
            },
            periods: TimePeriods::default(),
            pilot: PilotModel {
                pilot_type: PilotType::Perfect,
                time_constant: 0.0,
                damping: 0.0,
                frequency: 0.0,
            },
            target_orbit: OrbitalTarget {
                semi_major_axis: 3_649_622.0,
                eccentricity: 0.067,
                inclination: 50.0_f64.to_radians(),
                raan: -7.612_f64.to_radians(),
                apoapsis: 500_130.0,
                periapsis: 11_233.0,
            },
            final_conditions: FinalConditions::default(),
            parking_orbit: ParkingOrbit::default(),
            constraints: Constraints::default(),
            success: SuccessCriteria::default(),
            wind_enabled: false,
            wind_table: None,
            neural_net: None,
            dispersion_config: None,
            nav_mode: crate::data::NavMode::Bias,
            nav_config: None,
            integration_mode: crate::config::IntegrationMode::FixedGill,
        }
    }

    #[rstest]
    #[case(3000.0)]
    #[case(4500.0)]
    #[case(5687.0)]
    fn bank_angle_in_valid_range(#[case] velocity: f64) {
        let nav = test_nav(velocity);
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let bank = equilibrium_glide_bank(&nav, &data, &planet);

        let min_bank = 15.0_f64.to_radians();
        let max_bank = 120.0_f64.to_radians();
        assert!(
            bank >= min_bank && bank <= max_bank,
            "bank={:.4} rad ({:.1}°) outside [{:.4}, {:.4}] for V={} m/s",
            bank,
            bank.to_degrees(),
            min_bank,
            max_bank,
            velocity,
        );
    }

    #[test]
    fn zero_lift_returns_default() {
        let mut nav = test_nav(4500.0);
        nav.aero_coefficients = [1.269, 0.0]; // Cz = 0

        let mut data = test_sim_data();
        data.aero.cz = vec![0.0, 0.0];

        let planet = PlanetConfig::mars();
        let bank = equilibrium_glide_bank(&nav, &data, &planet);

        assert_relative_eq!(bank, 60.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn higher_velocity_gives_larger_bank() {
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let bank_slow = equilibrium_glide_bank(&test_nav(3000.0), &data, &planet);
        let bank_fast = equilibrium_glide_bank(&test_nav(5687.0), &data, &planet);

        assert!(
            bank_fast > bank_slow,
            "expected bank at V=5687 ({:.4} rad) > bank at V=3000 ({:.4} rad)",
            bank_fast,
            bank_slow,
        );
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn output_always_finite_and_bounded(
                alt in 10_000.0..130_000.0_f64,
                vel in 2000.0..7000.0_f64,
                fpa in -0.2..0.05_f64,
                rho in 1e-6..0.05_f64,
            ) {
                let mut nav = test_nav(vel);
                let r = PlanetConfig::mars().equatorial_radius + alt;
                nav.position_estimated[0] = r;
                nav.velocity_estimated[1] = fpa;
                nav.density_guidance = rho;
                nav.dynamic_pressure_estimated = 0.5 * rho * vel * vel;

                let data = test_sim_data();
                let planet = PlanetConfig::mars();
                let bank = equilibrium_glide_bank(&nav, &data, &planet);

                let min_bank = 15.0_f64.to_radians();
                let max_bank = 120.0_f64.to_radians();
                prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
                prop_assert!(bank >= min_bank - 1e-10, "bank below 15°: {} rad", bank);
                prop_assert!(bank <= max_bank + 1e-10, "bank above 120°: {} rad", bank);
            }
        }
    }
}
