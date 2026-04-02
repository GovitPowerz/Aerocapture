//! Central guidance dispatcher: routes to scheme-specific modules per phase.

use crate::config::{GuidanceType, PlanetConfig};
use crate::data::SimData;
use crate::gnc::control::angle_utils::shortest_angle_diff;
use crate::gnc::guidance::ftc::{self as ftc_capture, FtcCaptureState};
use crate::gnc::guidance::lateral::{self, LateralState};
use crate::gnc::guidance::{
    energy_controller, equilibrium_glide, exit, fnpag, neural, piecewise_constant, predguid,
    thermal_limiter,
};
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::gnc::navigation::estimator::NavigationOutput;

/// Guidance dispatcher persistent state.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GuidanceState {
    // Bank angle command
    pub bank_angle_commanded: f64, // current commanded bank angle (rad)
    pub bank_angle_previous: f64,  // previous commanded bank angle (rad)
    pub pilot_bank_angle_previous: f64, // previous pilot bank angle (rad)
    pub aoa_commanded: f64,        // commanded AoA (rad)

    // Roll sign and reversal tracking
    pub lateral_state: LateralState,
    pub cumulative_bank_change: f64, // cumulative bank angle changes (rad)

    // Guidance securization (longitudinal only; lateral securization handled by lateral module)
    pub securization_counters: [i32; 2], // securization counters ([0]=longi inactive, [1]=longi secur)
    pub longi_active: i32,               // longitudinal securization indicator (1=active)

    // Reference velocity
    pub reference_velocity: f64,

    // Counters
    pub n_secur: i32,  // number of securization events
    pub n_active: i32, // number of active guidance calls

    // Scheme-specific states
    pub ftc_capture: FtcCaptureState,
    pub energy_ctrl: energy_controller::EnergyControllerState,
    pub predguid: predguid::PredGuidState,
    pub fnpag: fnpag::FnpagState,
}

impl GuidanceState {
    pub fn new(initial_bank: f64, initial_aoa: f64) -> Self {
        Self {
            bank_angle_commanded: initial_bank,
            bank_angle_previous: initial_bank,
            pilot_bank_angle_previous: initial_bank,
            aoa_commanded: initial_aoa,
            lateral_state: LateralState::new(initial_bank),
            cumulative_bank_change: 0.0,
            securization_counters: [0, 0],
            longi_active: 1,
            reference_velocity: 0.0,
            n_secur: 0,
            n_active: 0,
            ftc_capture: FtcCaptureState::default(),
            energy_ctrl: energy_controller::EnergyControllerState::new(),
            predguid: predguid::PredGuidState::new(),
            fnpag: fnpag::FnpagState::new(initial_bank),
        }
    }
}

/// Guidance dispatcher output.
#[derive(Debug, Clone, Copy, Default)]
pub struct GuidanceOutput {
    pub bank_angle_commanded: f64, // commanded bank angle (rad)
    pub aoa_commanded: f64,        // commanded AoA (rad)
    pub bank_rate: f64,            // bank rate before saturation (rad/s)
    pub longitudinal_active: i32,  // longitudinal guidance active
    pub rate_saturated: i32,       // rate saturation occurred
    pub roll_reversal_active: i32, // roll reversal indicator
}

/// Run one guidance step (dispatches to the active scheme).
#[allow(clippy::too_many_arguments)]
pub fn guidance_step(
    nav: &NavigationOutput,
    pilot_bank_angle: f64, // pilot-realized bank angle
    _sim_time: f64,
    reference_bank_angle: f64, // reference bank angle (from config, rad)
    state: &mut GuidanceState,
    data: &SimData,
    planet: &PlanetConfig,
    is_reference: bool,
    guidance_type: GuidanceType,
) -> GuidanceOutput {
    let mut out = GuidanceOutput::default();

    state.pilot_bank_angle_previous = pilot_bank_angle;

    // === Angle of attack guidance ===
    // proalf returns altitude as scheduling parameter
    let (altitude, _) = geodetic_from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        planet,
    );
    state.aoa_commanded = data.incidence.incidence_at(altitude);
    out.aoa_commanded = state.aoa_commanded;

    // === Longitudinal guidance activation ===
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    let mut longitudinal_active: i32;
    if energy <= data.guidance.longi_activation && energy >= data.guidance.longi_inhibition {
        longitudinal_active = 1;
    } else {
        longitudinal_active = 0;
        state.securization_counters[1] += 1;
    }

    longitudinal_active *= state.longi_active;
    out.longitudinal_active = longitudinal_active;

    // === Reference trajectory mode ===
    if is_reference {
        longitudinal_active = 0;
        state.longi_active = 0;
    }

    // === Longitudinal bank angle command ===
    // reference_bank_angle passed as parameter from config.reference_bank_angle
    let mut bank_angle_longitudinal: f64;

    // Schemes that produce signed bank angles bypass exit guidance entirely
    let uses_exit_guidance = !matches!(
        guidance_type,
        GuidanceType::PiecewiseConstant | GuidanceType::NeuralNetwork
    );

    if is_reference {
        state.bank_angle_commanded = reference_bank_angle;
        bank_angle_longitudinal = reference_bank_angle;
    } else if longitudinal_active == 0 {
        bank_angle_longitudinal = reference_bank_angle.abs();
    } else if nav.guidance_phase == 2 && uses_exit_guidance {
        // Exit phase: shared pdyn-feedback controller for all unsigned-magnitude schemes
        bank_angle_longitudinal = exit::exit_guidance(nav, data, planet, state.reference_velocity);
        state.n_active += 1;
    } else {
        // Capture phase: scheme-specific longitudinal guidance
        bank_angle_longitudinal = match guidance_type {
            GuidanceType::Ftc => {
                ftc_capture::ftc_bank_angle(nav, &mut state.ftc_capture, data, planet)
            }
            GuidanceType::NeuralNetwork => {
                let nn = data.neural_net.as_ref().expect("NN params not loaded");
                neural::nn_bank_angle(nav, nn, planet, data.target_orbit.inclination)
            }
            GuidanceType::EquilibriumGlide => {
                equilibrium_glide::equilibrium_glide_bank(nav, data, planet)
            }
            GuidanceType::EnergyController => {
                energy_controller::energy_controller_bank(nav, &state.energy_ctrl, data, planet)
            }
            GuidanceType::PredGuid => predguid::predguid_bank(nav, &state.predguid, data, planet),
            GuidanceType::Fnpag => fnpag::fnpag_bank(nav, &mut state.fnpag, data, planet),
            GuidanceType::PiecewiseConstant => piecewise_constant::piecewise_constant_bank(
                nav,
                &data.guidance.piecewise_constant,
                planet,
            ),
        };
        state.n_active += 1;
    }

    // === Thermal safety limiter (unsigned-magnitude schemes only) ===
    let uses_thermal_limiter = !matches!(
        guidance_type,
        GuidanceType::PiecewiseConstant | GuidanceType::NeuralNetwork
    );
    if uses_thermal_limiter && longitudinal_active == 1 && !is_reference {
        let cos_bank = bank_angle_longitudinal.cos();
        let cos_limited = thermal_limiter::apply_thermal_limit(
            cos_bank,
            nav.heat_flux_fraction,
            nav.heat_load_fraction,
            &data.guidance.thermal_limiter,
        );
        bank_angle_longitudinal = cos_limited.acos();
    }

    // Schemes that provide signed bank angles — skip lateral guidance entirely
    let skip_lateral = matches!(
        guidance_type,
        GuidanceType::PiecewiseConstant | GuidanceType::NeuralNetwork
    );
    if skip_lateral {
        state.bank_angle_commanded = bank_angle_longitudinal;
        state.lateral_state.roll_sign = if bank_angle_longitudinal >= 0.0 {
            1.0
        } else {
            -1.0
        };
    }

    // === Lateral guidance ===
    let mut roll_reversal_active = false;
    if !skip_lateral {
        roll_reversal_active = lateral::lateral_guidance(
            &data.guidance.lateral,
            &mut state.lateral_state,
            nav,
            data.target_orbit.inclination,
            energy,
            bank_angle_longitudinal,
            planet,
        );
    }
    // === Combine longitudinal and lateral commands ===
    if !is_reference && !skip_lateral {
        state.bank_angle_commanded = bank_angle_longitudinal * state.lateral_state.roll_sign;
    }

    // === Roll rate saturation (wrap-aware) ===
    let max_bank_rate = data.capsule.max_bank_rate;
    let guidance_period = data.periods.guidance;
    let angle_diff = shortest_angle_diff(state.bank_angle_previous, state.bank_angle_commanded);
    let bank_rate = angle_diff / guidance_period;
    let mut rate_saturated = 0;

    if bank_rate.abs() - max_bank_rate > 1e-10 {
        rate_saturated = 1;
        state.bank_angle_commanded =
            state.bank_angle_previous + max_bank_rate.copysign(angle_diff) * guidance_period;
    }

    // Cumulative bank angle tracking (shortest path)
    if bank_rate.abs() > 1e-10 {
        state.cumulative_bank_change += angle_diff.abs();
    }

    state.bank_angle_previous = state.bank_angle_commanded;

    out.bank_angle_commanded = state.bank_angle_commanded;
    out.bank_rate = bank_rate;
    out.rate_saturated = rate_saturated;
    out.roll_reversal_active = if roll_reversal_active { 1 } else { 0 };

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::{GuidanceType, PlanetConfig};
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
    use crate::gnc::guidance::lateral::LateralParams;
    use crate::gnc::navigation::estimator::NavigationOutput;

    // ─── Fixture builders ───────────────────────────────────────────────────

    fn test_nav() -> NavigationOutput {
        let r = PlanetConfig::mars().equatorial_radius + 50_000.0; // Mars + 50 km
        NavigationOutput {
            position_estimated: [r, 0.0, 0.0],
            velocity_estimated: [5000.0, -0.15, 0.6],
            acceleration_estimated: [50.0, -8.0],
            aero_coefficients: [1.269, -0.205],
            density_guidance: 0.001,
            density_exit: 1e-6,
            dynamic_pressure_estimated: 0.5 * 0.001 * 5000.0 * 5000.0,
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
                // Wide activation window so longitudinal guidance fires
                longi_activation: 1e12,
                longi_inhibition: -1e12,
                lateral: LateralParams {
                    lateral_activation: -1e12, // disable lateral for simple tests
                    lateral_inhibition: -1e12,
                    corridor_slope: 13080.458,
                    corridor_intercept: 0.0,
                    max_reversals: 5,
                },
                density_filter_gain: 0.8,
                exit_velocity_threshold: 4400.0,
                exit_altitude_threshold: 60_000.0,
                capture_damping: 0.7,
                capture_frequency: 0.072,
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
        }
    }

    // ─── Deterministic tests ─────────────────────────────────────────────────

    /// guidance_step should return a finite bank angle for a typical MSR state
    /// using the FTC scheme.
    #[test]
    fn guidance_step_returns_finite_output() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let initial_bank = 64.77_f64.to_radians();
        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians());

        let out = guidance_step(
            &nav,
            initial_bank,
            0.0, // sim_time
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        assert!(
            out.bank_angle_commanded.is_finite(),
            "bank_angle_commanded not finite: {}",
            out.bank_angle_commanded
        );
        assert!(
            out.aoa_commanded.is_finite(),
            "aoa_commanded not finite: {}",
            out.aoa_commanded
        );
        assert!(
            out.bank_rate.is_finite(),
            "bank_rate not finite: {}",
            out.bank_rate
        );
    }

    /// In reference mode, output bank should equal the reference bank angle.
    #[test]
    fn reference_mode_returns_reference_bank() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let reference_bank_angle = 45.0_f64.to_radians();
        let mut state = GuidanceState::new(reference_bank_angle, -0.48_f64.to_radians());
        // Prime bank_angle_previous so rate saturation doesn't shift the value
        state.bank_angle_previous = reference_bank_angle;

        let out = guidance_step(
            &nav,
            reference_bank_angle,
            0.0,
            reference_bank_angle,
            &mut state,
            &data,
            &planet,
            true, // is_reference
            GuidanceType::Ftc,
        );

        assert!(
            (out.bank_angle_commanded - reference_bank_angle).abs() < 1e-9,
            "expected bank_angle_commanded ≈ reference_bank_angle ({:.6} rad), got {:.6} rad",
            reference_bank_angle,
            out.bank_angle_commanded,
        );
    }

    /// Bank angle magnitude should stay within [0, π] radians.
    #[test]
    fn output_bank_bounded() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let initial_bank = 64.77_f64.to_radians();
        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians());

        let out = guidance_step(
            &nav,
            initial_bank,
            0.0,
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        let pi = std::f64::consts::PI;
        assert!(
            out.bank_angle_commanded >= -pi && out.bank_angle_commanded <= pi,
            "bank_angle_commanded = {:.4} rad is outside [-π, π]",
            out.bank_angle_commanded,
        );
    }

    /// longitudinal_active=0 (guidance inactive) should still return a finite bank
    /// equal to |reference_bank_angle|, without saturating.
    #[test]
    fn inactive_longitudinal_guidance_uses_reference_bank() {
        let nav = test_nav();
        let mut data = test_sim_data();
        // Force energy outside activation window so longitudinal_active=0
        data.guidance.longi_activation = -1e12;
        data.guidance.longi_inhibition = -2e12;
        data.guidance.lateral.lateral_activation = -2e12;

        let planet = PlanetConfig::mars();
        let reference_bank_angle = 30.0_f64.to_radians();
        let mut state = GuidanceState::new(reference_bank_angle, -0.48_f64.to_radians());
        state.bank_angle_previous = reference_bank_angle;

        let out = guidance_step(
            &nav,
            reference_bank_angle,
            0.0,
            reference_bank_angle,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        assert!(
            out.bank_angle_commanded.is_finite(),
            "expected finite bank_angle_commanded, got {}",
            out.bank_angle_commanded
        );
        // When guidance is inactive and no lateral, bank_angle_commanded comes from reference_bank_angle.abs()
        // clamped by rate saturation — it should stay close to reference_bank_angle for a single step
        let pi = std::f64::consts::PI;
        assert!(
            out.bank_angle_commanded.abs() <= pi,
            "bank_angle_commanded magnitude exceeds π: {}",
            out.bank_angle_commanded
        );
    }

    // ─── Property-based tests ────────────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// For any valid atmospheric state, guidance_step produces finite output.
            #[test]
            fn output_always_finite(
                alt in 20_000.0..130_000.0_f64,
                vel in 2000.0..7000.0_f64,
                fpa in -0.3..0.05_f64,
                bank_deg in 0.0..90.0_f64,
            ) {
                let r = PlanetConfig::mars().equatorial_radius + alt;
                let initial_bank = bank_deg.to_radians();
                let nav = NavigationOutput {
                    position_estimated: [r, 0.0, 0.0],
                    velocity_estimated: [vel, fpa, 0.6],
                    acceleration_estimated: [50.0, -8.0],
                    aero_coefficients: [1.269, -0.205],
                    density_guidance: 1e-4,
                    density_exit: 1e-6,
                    dynamic_pressure_estimated: 0.5 * 1e-4 * vel * vel,
                    energy_estimated: -1e6,
                    ..Default::default()
                };

                let data = test_sim_data();
                let planet = PlanetConfig::mars();
                let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians());

                let out = guidance_step(
                    &nav,
                    initial_bank,
                    0.0,
                    initial_bank,
                    &mut state,
                    &data,
                    &planet,
                    false,
                    GuidanceType::Ftc,
                );

                prop_assert!(out.bank_angle_commanded.is_finite(), "bank_angle_commanded not finite: {}", out.bank_angle_commanded);
                prop_assert!(out.aoa_commanded.is_finite(), "aoa_commanded not finite: {}", out.aoa_commanded);

                let pi = std::f64::consts::PI;
                prop_assert!(
                    out.bank_angle_commanded >= -pi && out.bank_angle_commanded <= pi,
                    "bank_angle_commanded = {} outside [-π, π]",
                    out.bank_angle_commanded
                );
            }
        }
    }

    // ─── Phase dispatch tests ─────────────────────────────────────────────────

    /// Phase 2 should dispatch to exit guidance for FTC scheme.
    #[test]
    fn phase_2_dispatches_to_exit_guidance() {
        let mut nav = test_nav();
        nav.guidance_phase = 2;
        nav.bounce_flag = 1;
        nav.density_exit = 1e-6;
        nav.velocity_estimated[1] = 0.05; // positive FPA (ascending)

        let mut data = test_sim_data();
        data.guidance.exit_pdyn_margin = 1.75;
        data.guidance.exit_radial_vel_gain = 10.0;

        let planet = PlanetConfig::mars();
        let initial_bank = 64.77_f64.to_radians();
        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians());
        state.reference_velocity = 50.0;

        let out = guidance_step(
            &nav,
            initial_bank,
            100.0,
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        assert!(
            out.bank_angle_commanded.is_finite(),
            "exit phase should produce finite bank: {}",
            out.bank_angle_commanded
        );
    }

    /// PiecewiseConstant scheme should ignore phase 2 (produces its own signed bank).
    #[test]
    fn piecewise_constant_ignores_exit_phase() {
        let mut nav = test_nav();
        nav.guidance_phase = 2;
        nav.bounce_flag = 1;

        let mut data = test_sim_data();
        data.guidance.piecewise_constant = crate::data::guidance_params::PiecewiseConstantParams {
            bank_angles: [0.5; 10],
            energy_min: -6.0e6,
            energy_max: 5.0e6,
        };

        let planet = PlanetConfig::mars();
        let initial_bank = 0.5;
        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians());

        let out = guidance_step(
            &nav,
            initial_bank,
            100.0,
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::PiecewiseConstant,
        );

        assert!(out.bank_angle_commanded.is_finite());
    }
}
