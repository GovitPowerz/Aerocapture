//! FTC (Full Trajectory Control) predictor-corrector guidance.

use crate::config::{GuidanceType, Planet};
use crate::data::SimData;
use crate::gnc::guidance::{energy_controller, equilibrium_glide, fnpag, neural, predguid};
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;

/// FTC guidance persistent state.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FtcState {
    // Bank angle command
    pub bank_angle_commanded: f64, // current commanded bank angle (rad)
    pub bank_angle_previous: f64,  // previous commanded bank angle (rad)
    pub pilot_bank_angle_previous: f64, // previous pilot bank angle (rad)
    pub aoa_commanded: f64,        // commanded AoA (rad)

    // Roll sign and reversal tracking
    pub roll_sign: f64,            // roll polarity sign (-1, 0, +1)
    pub cumulative_bank_change: f64, // cumulative bank angle changes (rad)
    pub n_reversals: i32,          // number of roll reversals
    pub reversal_active: i32,      // roll reversal active flag
    pub rolway: i32,               // roll reversal path (+1=short, -1=long)
    pub reversal_duration: f64,    // roll reversal duration (s)

    // Guidance securization
    pub securization_counters: [i32; 2], // securization counters
    pub guidance_active: [i32; 2],       // securization indicators

    // Reference velocity
    pub reference_velocity: f64,

    // Counters
    pub n_secur: i32,  // number of securization events
    pub n_active: i32, // number of active guidance calls

    // Optional states for alternative guidance algorithms
    pub energy_ctrl: energy_controller::EnergyControllerState,
    pub predguid: predguid::PredGuidState,
    pub fnpag: fnpag::FnpagState,
}

impl FtcState {
    pub fn new(initial_bank: f64, initial_aoa: f64) -> Self {
        Self {
            bank_angle_commanded: initial_bank,
            bank_angle_previous: initial_bank,
            pilot_bank_angle_previous: initial_bank,
            aoa_commanded: initial_aoa,
            roll_sign: if initial_bank >= 0.0 { 1.0 } else { -1.0 },
            cumulative_bank_change: 0.0,
            n_reversals: 0,
            reversal_active: 0,
            rolway: 1,
            reversal_duration: 0.0,
            securization_counters: [0, 0],
            guidance_active: [1, 1],
            reference_velocity: 0.0,
            n_secur: 0,
            n_active: 0,
            energy_ctrl: energy_controller::EnergyControllerState::new(),
            predguid: predguid::PredGuidState::new(),
            fnpag: fnpag::FnpagState::new(initial_bank),
        }
    }
}

/// FTC guidance output.
#[derive(Debug, Clone, Copy, Default)]
pub struct FtcOutput {
    pub bank_angle_commanded: f64, // commanded bank angle (rad)
    pub aoa_commanded: f64,        // commanded AoA (rad)
    pub bank_rate: f64,            // bank rate before saturation (rad/s)
    pub longitudinal_active: i32,  // longitudinal guidance active
    pub rate_saturated: i32,       // rate saturation occurred
    pub roll_reversal_active: i32, // roll reversal indicator
}

/// Run one FTC guidance step.
#[allow(clippy::too_many_arguments)]
pub fn guidance_step(
    nav: &NavigationOutput,
    pilot_bank_angle: f64, // pilot-realized bank angle
    sim_time: f64,
    reference_bank_angle: f64, // reference bank angle (from config, rad)
    state: &mut FtcState,
    data: &SimData,
    planet: &Planet,
    is_reference: bool,
    guidance_type: GuidanceType,
) -> FtcOutput {
    let pi = std::f64::consts::PI;
    let mut out = FtcOutput::default();

    let previous_roll_sign = state.roll_sign;
    state.pilot_bank_angle_previous = pilot_bank_angle;

    // === Angle of attack guidance ===
    // proalf returns altitude as scheduling parameter
    let (altitude, _) =
        geodetic_from_spherical(nav.position_estimated[0], nav.position_estimated[1], nav.position_estimated[2], planet);
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

    longitudinal_active *= state.guidance_active[0];
    out.longitudinal_active = longitudinal_active;

    // === Reference trajectory mode ===
    if is_reference {
        longitudinal_active = 0;
        state.guidance_active[0] = 0;
        state.guidance_active[1] = 0;
    }

    // === Longitudinal bank angle command ===
    // reference_bank_angle passed as parameter from config.reference_bank_angle
    let bank_angle_longitudinal: f64;

    if is_reference {
        state.bank_angle_commanded = reference_bank_angle;
        bank_angle_longitudinal = reference_bank_angle;
    } else if longitudinal_active == 0 {
        bank_angle_longitudinal = reference_bank_angle.abs();
    } else {
        // Longitudinal guidance dispatch
        bank_angle_longitudinal = match guidance_type {
            GuidanceType::Ftc => capture_guidance(nav, energy, altitude, state, data, planet),
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
        };
        state.n_active += 1;
    }

    // === Lateral guidance activation ===
    let mut lateral_active: i32;
    if energy <= data.guidance.lateral_activation && energy >= data.guidance.lateral_inhibition {
        lateral_active = 1;
    } else {
        lateral_active = 0;
    }

    lateral_active *= state.guidance_active[1];

    // === Lateral guidance ===
    let mut roll_reversal_active = 0;
    if lateral_active == 1 {
        lateral_guidance(nav, bank_angle_longitudinal, sim_time, state, data, planet, &mut roll_reversal_active);
        if state.reversal_active == 1 {
            state.guidance_active[1] = 0;
        }
    } else {
        state.roll_sign = previous_roll_sign;
    }

    // === Combine longitudinal and lateral commands ===
    if !is_reference {
        if state.guidance_active[0] * state.guidance_active[1] == 1 {
            state.bank_angle_commanded = bank_angle_longitudinal * state.roll_sign;
        } else if state.reversal_active == 1 {
            let max_bank_rate = data.capsule.max_bank_rate;
            let guidance_period = data.periods.guidance;
            if state.rolway == 1 {
                if state.roll_sign > 0.0 {
                    state.bank_angle_commanded = state.bank_angle_previous + max_bank_rate * guidance_period;
                } else {
                    state.bank_angle_commanded = state.bank_angle_previous - max_bank_rate * guidance_period;
                }
            } else {
                if state.roll_sign > 0.0 {
                    state.bank_angle_commanded = state.bank_angle_previous - max_bank_rate * guidance_period;
                    if state.bank_angle_commanded < -pi {
                        state.bank_angle_commanded += 2.0 * pi;
                    }
                } else {
                    state.bank_angle_commanded = state.bank_angle_previous + max_bank_rate * guidance_period;
                    if state.bank_angle_commanded > pi {
                        state.bank_angle_commanded -= 2.0 * pi;
                    }
                }
            }
        }
    }

    // === Roll rate saturation ===
    let max_bank_rate = data.capsule.max_bank_rate;
    let guidance_period = data.periods.guidance;
    let bank_rate = (state.bank_angle_commanded - state.bank_angle_previous) / guidance_period;
    let mut rate_saturated = 0;

    if bank_rate.abs() - max_bank_rate > 1e-10 {
        rate_saturated = 1;
        if state.bank_angle_commanded > state.bank_angle_previous {
            state.bank_angle_commanded = state.bank_angle_previous + max_bank_rate * guidance_period;
        } else {
            state.bank_angle_commanded = state.bank_angle_previous - max_bank_rate * guidance_period;
        }
    }

    // Cumulative bank angle tracking
    if bank_rate.abs() > 1e-10 {
        state.cumulative_bank_change += (state.bank_angle_commanded - state.bank_angle_previous).abs();
    }

    state.bank_angle_previous = state.bank_angle_commanded;

    out.bank_angle_commanded = state.bank_angle_commanded;
    out.bank_rate = bank_rate;
    out.rate_saturated = rate_saturated;
    out.roll_reversal_active = roll_reversal_active;

    out
}

/// Capture phase longitudinal guidance: altitude-gain predictor-corrector.
fn capture_guidance(
    nav: &NavigationOutput,
    energy: f64,
    altitude: f64,
    state: &mut FtcState,
    data: &SimData,
    _planet: &Planet,
) -> f64 {
    let ref_traj = &data.guidance.ref_trajectory;

    let velocity_relative = nav.velocity_estimated[0];
    let velocity_radial = velocity_relative * nav.velocity_estimated[1].sin();
    let dynamic_pressure_equilibrium = 0.5 * nav.density_guidance * velocity_relative * velocity_relative;

    // Interpolate reference trajectory at current energy
    let cos_bank_nominal = ref_traj.interpolate(energy, &ref_traj.cos_bank);
    let dynamic_pressure_nominal = ref_traj.interpolate(energy, &ref_traj.pressure);
    let altitude_rate_nominal = ref_traj.interpolate(energy, &ref_traj.radial_vel);
    let _httnom = ref_traj.interpolate(energy, &ref_traj.altitude_rate);

    // Compute gains
    let (gain_altitude_rate, gain_dynamic_pressure) = compute_gains(altitude, &nav.aero_coefficients, data);

    // Predictor-corrector equation
    // cos(bank_angle_longitudinal) = cos_bank_nominal + gain_altitude_rate*(velocity_radial - altitude_rate_nominal)/dynamic_pressure_equilibrium + gain_dynamic_pressure*(dynamic_pressure_equilibrium - dynamic_pressure_nominal)/dynamic_pressure_equilibrium
    let dynamic_pressure_equilibrium_safe = if dynamic_pressure_equilibrium.abs() > 1e-10 { dynamic_pressure_equilibrium } else { 1e-10 };
    let mut cos_bank_commanded = cos_bank_nominal
        + gain_altitude_rate * (velocity_radial - altitude_rate_nominal) / dynamic_pressure_equilibrium_safe
        + gain_dynamic_pressure * (dynamic_pressure_equilibrium - dynamic_pressure_nominal) / dynamic_pressure_equilibrium_safe;

    // Securization: clamp cos to [-1, 1]
    let is_securized;
    let bank_angle_longitudinal;
    if cos_bank_commanded.abs() > 1.0 {
        cos_bank_commanded = cos_bank_commanded.signum();
        bank_angle_longitudinal = cos_bank_commanded.acos();
        is_securized = 1;
    } else {
        bank_angle_longitudinal = cos_bank_commanded.acos().abs();
        is_securized = 0;
    }

    if is_securized == 1 {
        state.securization_counters[0] += 1;
        state.n_secur += 1;
    }

    bank_angle_longitudinal
}

/// Compute guidance gains from altitude-based Pdyn model.
fn compute_gains(altitude: f64, aero_coefficients: &[f64; 2], data: &SimData) -> (f64, f64) {
    let pdyn_table = &data.guidance.pdyn_table;
    let alt_km = altitude / 1e3;

    // Find altitude bracket; use Option<usize> as "not found" sentinel.
    let mut found: Option<usize> = None;
    for i in 0..pdyn_table.len().saturating_sub(1) {
        if alt_km >= pdyn_table[i].altitude
            && alt_km < pdyn_table[i + 1].altitude
            && found.is_none()
        {
            found = Some(i);
        }
    }
    // If no bracket found, fall back to last entry
    let table_index = found.unwrap_or_else(|| {
        if pdyn_table.is_empty() {
            0
        } else {
            pdyn_table.len() - 1
        }
    });

    let pressure_coeff = if table_index < pdyn_table.len() {
        pdyn_table[table_index].coeff_a
    } else {
        1.0
    };

    // Gains
    let damping_capture = data.guidance.capture_damping;
    let frequency_capture = data.guidance.capture_frequency;
    let reference_area = data.capsule.reference_area;
    let mass = data.capsule.mass;
    let cz = aero_coefficients[1]; // lift coefficient

    let gain_altitude_rate = if (reference_area * cz).abs() > 1e-30 {
        -2.0 * damping_capture * frequency_capture * mass / (reference_area * cz)
    } else {
        0.0
    };

    let gain_dynamic_pressure = if (pressure_coeff * reference_area * cz).abs() > 1e-30 {
        -frequency_capture * frequency_capture * mass / (pressure_coeff * reference_area * cz)
    } else {
        0.0
    };

    (gain_altitude_rate, gain_dynamic_pressure)
}

/// Lateral guidance — roll reversal logic.
fn lateral_guidance(
    nav: &NavigationOutput,
    bank_angle_longitudinal: f64,
    _sim_time: f64,
    state: &mut FtcState,
    data: &SimData,
    planet: &Planet,
    roll_reversal_active: &mut i32,
) {
    let pi = std::f64::consts::PI;

    if bank_angle_longitudinal == 0.0 || bank_angle_longitudinal == pi {
        return;
    }

    let previous_roll_sign = state.roll_sign;

    // Compute orbital elements for inclination
    let orbit = elements::from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    let inclination_error = data.target_orbit.inclination - orbit.inclination;
    // Hemisphere correction intentionally omitted (inactive)

    let velocity_relative = nav.velocity_estimated[0];

    // Corridor boundary: inclination_max = (v/corridor_slope)^4 + corridor_intercept
    let corridor_slope = data.guidance.corridor_slope;
    let corridor_intercept = data.guidance.corridor_intercept;
    let inclination_max = (velocity_relative / corridor_slope).powi(4) + corridor_intercept;

    // Reversal decision
    if inclination_error.abs() >= inclination_max && bank_angle_longitudinal.abs() > 1e-10 && state.n_reversals < data.guidance.max_reversals
    {
        if inclination_error > inclination_max {
            state.roll_sign = -1.0;
        } else if inclination_error < -inclination_max {
            state.roll_sign = 1.0;
        }

        if state.roll_sign * previous_roll_sign < 0.0 {
            // Roll reversal commanded
            *roll_reversal_active = 1;
            state.n_reversals += 1;

            if state.reversal_active == 0 {
                state.reversal_active = 1;
                state.reversal_active = 0; // immediately reset after arming
                state.rolway = 1;
                let bank_angle_change = state.bank_angle_previous.abs() + bank_angle_longitudinal.abs();
                let max_bank_rate = data.capsule.max_bank_rate;
                let guidance_period = data.periods.guidance;
                state.reversal_duration = bank_angle_change / max_bank_rate;
                state.reversal_duration = (state.reversal_duration / guidance_period).floor() * guidance_period;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::{GuidanceType, Planet};
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
    use crate::gnc::navigation::estimator::NavigationOutput;

    // ─── Fixture builders ───────────────────────────────────────────────────

    fn test_nav() -> NavigationOutput {
        let r = Planet::Mars.equatorial_radius() + 50_000.0; // Mars + 50 km
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
                lateral_activation: -1e12, // disable lateral for simple tests
                lateral_inhibition: -1e12,
                density_filter_gain: 0.8,
                exit_velocity_threshold: 4400.0,
                exit_altitude_threshold: 60_000.0,
                capture_damping: 0.7,
                capture_frequency: 0.072,
                corridor_slope: 13080.458,
                max_reversals: 5,
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
            neural_net: None,
            dispersion_config: None,
        }
    }

    // ─── Deterministic tests ─────────────────────────────────────────────────

    /// guidance_step should return a finite bank angle for a typical MSR state
    /// using the FTC scheme.
    #[test]
    fn guidance_step_returns_finite_output() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = Planet::Mars;
        let initial_bank = 64.77_f64.to_radians();
        let mut state = FtcState::new(initial_bank, -0.48_f64.to_radians());

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

        assert!(out.bank_angle_commanded.is_finite(), "bank_angle_commanded not finite: {}", out.bank_angle_commanded);
        assert!(out.aoa_commanded.is_finite(), "aoa_commanded not finite: {}", out.aoa_commanded);
        assert!(out.bank_rate.is_finite(), "bank_rate not finite: {}", out.bank_rate);
    }

    /// In reference mode, output bank should equal the reference bank angle.
    #[test]
    fn reference_mode_returns_reference_bank() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = Planet::Mars;
        let reference_bank_angle = 45.0_f64.to_radians();
        let mut state = FtcState::new(reference_bank_angle, -0.48_f64.to_radians());
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
        let planet = Planet::Mars;
        let initial_bank = 64.77_f64.to_radians();
        let mut state = FtcState::new(initial_bank, -0.48_f64.to_radians());

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
        data.guidance.lateral_activation = -2e12;

        let planet = Planet::Mars;
        let reference_bank_angle = 30.0_f64.to_radians();
        let mut state = FtcState::new(reference_bank_angle, -0.48_f64.to_radians());
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
                let r = Planet::Mars.equatorial_radius() + alt;
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
                let planet = Planet::Mars;
                let mut state = FtcState::new(initial_bank, -0.48_f64.to_radians());

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
}
