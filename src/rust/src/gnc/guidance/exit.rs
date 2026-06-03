//! Exit-phase longitudinal guidance.
//!
//! Stateless bank angle controller for the exit phase (guidance_phase == 2).
//! Uses dynamic pressure feedback with radial velocity damping to steer the
//! vehicle to an acceptable apoapsis after the atmospheric bounce.
//!
//! The controller computes:
//!   cos(bank) = (pdyn_current - pdyn_target) / pdyn_current
//!             + K_rv * (v_r - v_r_ref) / pdyn_current
//!
//! where `pdyn_target` is the target dynamic pressure at exit altitude and
//! `v_r_ref` is the radial velocity latched at the phase 1→2 transition.
//! The result is clamped to [-1, 1] then converted to a bank angle magnitude.
//!

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::estimator::NavigationOutput;

/// Compute exit-phase bank angle magnitude.
///
/// Returns a bank angle magnitude in [0, π] radians.
/// The caller handles roll sign via lateral guidance (or bypasses it for schemes
/// that produce signed bank angles directly).
///
/// # Arguments
/// * `nav` - Current navigation output (estimated state).
/// * `data` - Simulation data (guidance parameters).
/// * `_planet` - Planet configuration (unused; present for a uniform call signature).
/// * `reference_velocity` - Radial velocity (m/s) latched at the phase 1→2 transition.
pub fn exit_guidance(
    nav: &NavigationOutput,
    data: &SimData,
    _planet: &PlanetConfig,
    reference_velocity: f64,
) -> f64 {
    let velocity = nav.velocity_estimated[0];
    let fpa = nav.velocity_estimated[1];
    let velocity_radial = velocity * fpa.sin();

    // Target dynamic pressure: q_exit * margin, where q_exit = 0.5 * rho_exit * V^2.
    // `density_exit` is computed every nav step from the onboard model
    // evaluated at `exit_altitude_threshold`.
    let pdyn_target = 0.5 * nav.density_exit * velocity * velocity * data.guidance.exit_pdyn_margin;
    // Current dynamic pressure from nav estimate.
    let pdyn_current = 0.5 * nav.density_guidance * velocity * velocity;

    // Safe denominator — avoids division-by-zero at extreme altitudes.
    let pdyn_safe = if pdyn_current.abs() > 1e-10 {
        pdyn_current
    } else {
        1e-10
    };

    // Dynamic pressure correction: positive when we're deeper than target (bank up).
    let pdyn_correction = (pdyn_current - pdyn_target) / pdyn_safe;

    // Radial velocity damping: positive when ascending faster than reference.
    let radial_vel_correction =
        data.guidance.exit_radial_vel_gain * (velocity_radial - reference_velocity) / pdyn_safe;

    // Predictor-corrector sum, clamped and converted to angle.
    let cos_bank = pdyn_correction + radial_vel_correction;
    cos_bank.clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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
                exit_pdyn_margin: 1.75,
                exit_altitude_threshold: 60_000.0,
                exit_radial_vel_gain: 10.0,
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
            sim_phase: crate::config::SimPhase::Full,
            density_perturbation: None,
            nn_normalization_override: None,
        }
    }

    /// Build a NavigationOutput for a post-bounce ascending state.
    fn ascending_nav(
        velocity: f64,
        fpa: f64,
        density_guidance: f64,
        density_exit: f64,
    ) -> NavigationOutput {
        let r = PlanetConfig::mars().equatorial_radius + 50_000.0;
        NavigationOutput {
            position_estimated: [r, 0.0, 0.0],
            velocity_estimated: [velocity, fpa, 0.6],
            acceleration_estimated: [50.0, -8.0],
            aero_coefficients: [1.269, -0.205],
            density_guidance,
            density_exit,
            dynamic_pressure_estimated: 0.5 * density_guidance * velocity * velocity,
            energy_estimated: -1e6,
            guidance_phase: 2,
            bounce_flag: 1,
            ..Default::default()
        }
    }

    #[test]
    fn exit_guidance_returns_finite_bounded_bank() {
        let nav = ascending_nav(4800.0, 0.05, 1e-5, 1e-7);
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let bank = exit_guidance(&nav, &data, &planet, -50.0);

        assert!(bank.is_finite(), "bank angle must be finite, got {}", bank);
        assert!(
            (0.0..=std::f64::consts::PI).contains(&bank),
            "bank={:.4} rad ({:.1}°) outside [0, π]",
            bank,
            bank.to_degrees(),
        );
    }

    #[test]
    fn exit_guidance_higher_pdyn_gives_different_bank() {
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        // Nearly horizontal flight so the radial velocity term is small, letting the
        // pdyn feedback dominate. Reference velocity matched to near-zero radial vel.
        let fpa = 0.001_f64; // ~0.057°
        let ref_vel = 4800.0 * fpa.sin(); // ≈ 4.8 m/s — latched at phase transition

        // Low density → small pdyn_current relative to pdyn_target → cos_bank < 1 interior solution
        let nav_low = ascending_nav(4800.0, fpa, 1e-6, 1e-7);
        let bank_low = exit_guidance(&nav_low, &data, &planet, ref_vel);

        // High density → large pdyn_current → different interior solution
        let nav_high = ascending_nav(4800.0, fpa, 1e-3, 1e-7);
        let bank_high = exit_guidance(&nav_high, &data, &planet, ref_vel);

        assert_ne!(
            bank_low, bank_high,
            "different density_guidance values should produce different bank angles \
             (got bank_low={:.4} rad, bank_high={:.4} rad)",
            bank_low, bank_high,
        );
    }

    #[test]
    fn exit_guidance_zero_density_exit_gives_clamped_result() {
        // density_exit = 0 → pdyn_target = 0 → pdyn_correction = 1.0 → cos_bank clamped to 1.0
        // acos(1.0) = 0.0, so bank should be 0 radians (lift-up).
        let nav = ascending_nav(4800.0, 0.05, 1e-5, 0.0);
        let data = test_sim_data();
        let planet = PlanetConfig::mars();

        let bank = exit_guidance(&nav, &data, &planet, 0.0);

        assert!(
            bank.is_finite(),
            "bank must be finite with zero density_exit, got {}",
            bank,
        );
        assert!(
            (0.0..=std::f64::consts::PI).contains(&bank),
            "bank={:.4} rad outside [0, π] with zero density_exit",
            bank,
        );
    }

    proptest! {
        #[test]
        fn output_always_finite_and_bounded(
            vel in 2000.0..6000.0_f64,
            fpa in 0.01..0.3_f64,
            density_guidance in 1e-8..1e-2_f64,
            density_exit in 0.0..1e-4_f64,
            ref_vel in -200.0..200.0_f64,
        ) {
            let nav = ascending_nav(vel, fpa, density_guidance, density_exit);
            let data = test_sim_data();
            let planet = PlanetConfig::mars();

            let bank = exit_guidance(&nav, &data, &planet, ref_vel);

            prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
            prop_assert!(bank >= 0.0 - 1e-10, "bank below 0: {} rad", bank);
            prop_assert!(
                bank <= std::f64::consts::PI + 1e-10,
                "bank above π: {} rad",
                bank,
            );
        }
    }
}
