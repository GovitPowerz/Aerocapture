//! PredGuid — Apollo/Shuttle-heritage drag tracking guidance.
//!
//! Tracks a reference drag acceleration profile as a function of velocity.
//! The reference drag profile is derived from the reference trajectory
//! (tables_energie_gains), which stores dynamic pressure vs energy. Since
//! drag = q * S * Cx / m, and energy is monotonically related to velocity
//! during atmospheric flight, we can map between them.
//!
//! The control law modulates bank angle to match the reference drag level:
//!
//!   D_cmd = D_ref + K * (D - D_ref)   [drag error feedback]
//!   cos(bank) = D_cmd / L              [bank angle from drag command]
//!
//! In practice, we use dynamic pressure as a proxy for drag (proportional
//! for constant ballistic coefficient) and compute:
//!
//!   cos(bank) = cos_ref + K_d * (D - D_ref) / (L/m)
//!
//! This is conceptually similar to FTC but uses a simpler feedback structure
//! without the altitude-rate damping term.

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::total_energy;
use crate::gnc::navigation::estimator::NavigationOutput;

/// PredGuid persistent state (runtime-only, no tunable params).
#[derive(Debug, Clone)]
pub struct PredGuidState {
    _placeholder: (),
}

impl Default for PredGuidState {
    fn default() -> Self {
        Self::new()
    }
}

impl PredGuidState {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

/// Compute PredGuid bank angle command.
///
/// Tracks reference drag acceleration profile using bank angle modulation.
/// Returns bank angle magnitude in radians.
pub fn predguid_bank(
    nav: &NavigationOutput,
    _state: &PredGuidState,
    data: &SimData,
    planet: &PlanetConfig,
) -> f64 {
    let ref_traj = &data.guidance.ref_trajectory;
    if ref_traj.n_points == 0 {
        return 60.0_f64.to_radians();
    }

    // Current energy for reference lookup
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    // Reference values
    let cos_bank_ref = ref_traj.interpolate(energy, &ref_traj.cos_bank);
    let pdyn_ref = ref_traj.interpolate(energy, &ref_traj.pressure);

    // Current drag and lift accelerations from navigation
    let drag_accel = nav.acceleration_estimated[0]; // D/m
    let lift_accel = nav.acceleration_estimated[1]; // L/m

    // Current dynamic pressure
    let v = nav.velocity_estimated[0];
    let pdyn = 0.5 * nav.density_guidance * v * v;

    // Reference drag acceleration: D_ref/m = pdyn_ref * S * Cx / m
    let cx = nav.aero_coefficients[0];
    let sref = data.capsule.reference_area;
    let mass = data.capsule.mass;
    let drag_ref = pdyn_ref * sref * cx / mass;

    // Drag error
    let drag_err = drag_accel - drag_ref;

    // Lift acceleration (absolute value for bank angle computation)
    let lift_abs = lift_accel.abs().max(1e-10);

    // PredGuid control law:
    // The reference cos(bank) already accounts for the nominal drag.
    // We add a correction proportional to the drag error, normalized by lift.
    //
    // If drag > ref: we're too deep → increase bank (more lift-down) → increase cos_bank
    // If drag < ref: we're too high → decrease bank (more lift-up) → decrease cos_bank
    //
    // The sign convention: cos(bank) = 1 means full lift-up (min drag exposure),
    // cos(bank) = -1 means full lift-down (max drag exposure).
    // So drag error should DECREASE cos_bank (bank toward lift-up to reduce drag).
    let params = &data.guidance.pred_guid;
    let k_drag = if pdyn > params.pdyn_threshold {
        params.k_drag_high
    } else {
        params.k_drag_low
    };
    let cos_bank = cos_bank_ref - k_drag * drag_err / lift_abs;

    let cos_bank = cos_bank.clamp(-1.0, 1.0);
    cos_bank.acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    use crate::data::aerodynamics::AeroTables;
    use crate::data::atmosphere::{AtmosphereModel, DensityProfile};
    use crate::data::capsule::Capsule;
    use crate::data::guidance_params::{GuidanceParams, ReferenceTrajectory};
    use crate::data::incidence::IncidenceProfile;
    use crate::data::pilot::{PilotModel, PilotType};
    use crate::data::{
        Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit, SimData,
        SphericalState, SuccessCriteria, TimePeriods,
    };

    fn test_nav(velocity: f64) -> NavigationOutput {
        let r = 3_396_200.0 + 50_000.0;
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
            sim_phase: crate::config::SimPhase::Full,
        }
    }

    fn test_sim_data_with_ref_traj() -> SimData {
        let mut data = test_sim_data();
        data.guidance.ref_trajectory = ReferenceTrajectory {
            n_points: 3,
            energy: vec![-1e6, -3e6, -5e6],
            pressure: vec![500.0, 800.0, 300.0],
            radial_vel: vec![-100.0, -50.0, 50.0],
            altitude_rate: vec![-500.0, -200.0, 100.0],
            inclination: vec![0.87, 0.87, 0.87],
            time: vec![0.0, 300.0, 600.0],
            cos_bank: vec![0.4, 0.3, 0.5],
        };
        data
    }

    #[test]
    fn no_ref_trajectory_returns_default() {
        let nav = test_nav(4500.0);
        let state = PredGuidState::new();
        let data = test_sim_data(); // ref_trajectory.n_points == 0
        let planet = PlanetConfig::mars();

        let bank = predguid_bank(&nav, &state, &data, &planet);

        assert_relative_eq!(bank, 60.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[rstest]
    #[case(3000.0)]
    #[case(4500.0)]
    #[case(5687.0)]
    fn output_is_finite(#[case] velocity: f64) {
        let nav = test_nav(velocity);
        let state = PredGuidState::new();
        let data = test_sim_data_with_ref_traj();
        let planet = PlanetConfig::mars();

        let bank = predguid_bank(&nav, &state, &data, &planet);

        assert!(
            bank.is_finite(),
            "bank angle must be finite for V={}",
            velocity
        );
        assert!(
            (0.0..=std::f64::consts::PI).contains(&bank),
            "bank={:.4} rad outside [0, pi] for V={}",
            bank,
            velocity,
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

                let state = PredGuidState::new();
                let data = test_sim_data_with_ref_traj();
                let planet = PlanetConfig::mars();
                let bank = predguid_bank(&nav, &state, &data, &planet);

                prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
                prop_assert!(bank >= 0.0 - 1e-10, "bank negative: {}", bank);
                prop_assert!(bank <= std::f64::consts::PI + 1e-10, "bank > pi: {}", bank);
            }
        }
    }
}
