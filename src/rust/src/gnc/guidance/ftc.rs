//! FTC capture-phase longitudinal guidance: altitude-gain predictor-corrector.

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::gnc::navigation::estimator::NavigationOutput;

/// FTC capture-phase persistent state.
#[derive(Debug, Clone, Default)]
pub struct FtcCaptureState {
    pub securization_counters: [i32; 2],
    pub n_secur: i32,
}

/// Compute FTC capture-phase bank angle (unsigned magnitude).
pub fn ftc_bank_angle(
    nav: &NavigationOutput,
    capture_state: &mut FtcCaptureState,
    data: &SimData,
    planet: &PlanetConfig,
) -> f64 {
    let (altitude, _) = geodetic_from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        planet,
    );
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        planet,
    );

    let ref_traj = &data.guidance.ref_trajectory;

    let velocity_relative = nav.velocity_estimated[0];
    let velocity_radial = velocity_relative * nav.velocity_estimated[1].sin();
    let dynamic_pressure_equilibrium =
        0.5 * nav.density_guidance * velocity_relative * velocity_relative;

    // Interpolate reference trajectory at current energy
    let cos_bank_nominal = ref_traj.interpolate(energy, &ref_traj.cos_bank);
    let dynamic_pressure_nominal = ref_traj.interpolate(energy, &ref_traj.pressure);
    let altitude_rate_nominal = ref_traj.interpolate(energy, &ref_traj.radial_vel);
    let _httnom = ref_traj.interpolate(energy, &ref_traj.altitude_rate);

    // Compute gains
    let (gain_altitude_rate, gain_dynamic_pressure) =
        compute_gains(altitude, &nav.aero_coefficients, data);

    // Predictor-corrector equation
    let dynamic_pressure_equilibrium_safe = if dynamic_pressure_equilibrium.abs() > 1e-10 {
        dynamic_pressure_equilibrium
    } else {
        1e-10
    };
    let mut cos_bank_commanded = cos_bank_nominal
        + gain_altitude_rate * (velocity_radial - altitude_rate_nominal)
            / dynamic_pressure_equilibrium_safe
        + gain_dynamic_pressure * (dynamic_pressure_equilibrium - dynamic_pressure_nominal)
            / dynamic_pressure_equilibrium_safe;

    // Securization: clamp cos to [-1, 1]
    let is_securized;
    let bank_angle_longitudinal;
    if cos_bank_commanded.abs() > 1.0 {
        cos_bank_commanded = cos_bank_commanded.signum();
        bank_angle_longitudinal = cos_bank_commanded.acos();
        is_securized = 1;
    } else {
        bank_angle_longitudinal = cos_bank_commanded.acos();
        is_securized = 0;
    }

    if is_securized == 1 {
        capture_state.securization_counters[0] += 1;
        capture_state.n_secur += 1;
    }

    bank_angle_longitudinal
}

/// Cosine fade: 1.0 below `start`, 0.0 above `end`, smooth cosine taper between.
/// Degenerate case: if `end <= start`, returns 1.0 (no fade).
fn cosine_fade(alt_km: f64, start: f64, end: f64) -> f64 {
    if end <= start {
        return 1.0;
    }
    let t = ((alt_km - start) / (end - start)).clamp(0.0, 1.0);
    0.5 * (1.0 + (std::f64::consts::PI * t).cos())
}

/// Compute guidance gains using analytical exponential decay model.
fn compute_gains(altitude: f64, aero_coefficients: &[f64; 2], data: &SimData) -> (f64, f64) {
    let alt_km = altitude / 1e3;

    // Exponential decay pressure coefficient
    let pressure_coeff = data.guidance.pressure_coeff_base
        * (-alt_km / data.guidance.pressure_coeff_scale_height).exp();

    // Cosine fade: both gains taper to zero above the sensible atmosphere
    let fade = cosine_fade(
        alt_km,
        data.guidance.gain_fade_start_km,
        data.guidance.gain_fade_end_km,
    );

    // Gains
    let damping_capture = data.guidance.capture_damping;
    let frequency_capture = data.guidance.capture_frequency;
    let reference_area = data.capsule.reference_area;
    let mass = data.capsule.mass;
    let cz = aero_coefficients[1]; // lift coefficient

    let gain_altitude_rate = if (reference_area * cz).abs() > 1e-30 {
        fade * -2.0 * damping_capture * frequency_capture * mass / (reference_area * cz)
    } else {
        0.0
    };

    let gain_dynamic_pressure = if (pressure_coeff * reference_area * cz).abs() > 1e-30 {
        fade * -frequency_capture * frequency_capture * mass
            / (pressure_coeff * reference_area * cz)
    } else {
        0.0
    };

    (gain_altitude_rate, gain_dynamic_pressure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
                pressure_coeff_base: -134.4,
                pressure_coeff_scale_height: 6.9,
                gain_fade_start_km: 80.0,
                gain_fade_end_km: 100.0,
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
            nn_normalization_override: None,
        }
    }

    #[test]
    fn cosine_fade_below_start_is_one() {
        let (start, end) = (80.0, 100.0);
        assert_eq!(cosine_fade(50.0, start, end), 1.0);
        assert_eq!(cosine_fade(0.0, start, end), 1.0);
        assert_eq!(cosine_fade(79.99, start, end), 1.0);
    }

    #[test]
    fn cosine_fade_above_end_is_zero() {
        let (start, end) = (80.0, 100.0);
        assert_relative_eq!(cosine_fade(100.0, start, end), 0.0, epsilon = 1e-15);
        assert_relative_eq!(cosine_fade(150.0, start, end), 0.0, epsilon = 1e-15);
        assert_relative_eq!(cosine_fade(500.0, start, end), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn cosine_fade_midpoint_is_half() {
        assert_relative_eq!(cosine_fade(90.0, 80.0, 100.0), 0.5, epsilon = 1e-15);
    }

    #[test]
    fn cosine_fade_monotonically_decreasing() {
        let (start, end) = (80.0, 100.0);
        let n = 100;
        let mut prev = cosine_fade(start, start, end);
        for i in 1..=n {
            let alt = start + (end - start) * (i as f64) / (n as f64);
            let val = cosine_fade(alt, start, end);
            assert!(
                val <= prev,
                "cosine_fade not monotonically decreasing: f({})={} > f({})={}",
                alt,
                val,
                alt - (end - start) / (n as f64),
                prev,
            );
            prev = val;
        }
    }

    #[test]
    fn cosine_fade_degenerate_end_le_start() {
        // end == start
        assert_eq!(cosine_fade(90.0, 100.0, 100.0), 1.0);
        // end < start
        assert_eq!(cosine_fade(90.0, 100.0, 50.0), 1.0);
    }

    #[test]
    fn pressure_coeff_decreases_with_altitude() {
        let base = -0.001_f64;
        let scale_height = 10.0_f64;

        let coeff_at = |alt_km: f64| base * (-alt_km / scale_height).exp();

        // All coefficients are negative; magnitude decreases with altitude.
        assert!(
            coeff_at(0.0).abs() > coeff_at(50.0).abs(),
            "|coeff(0)| should > |coeff(50)|"
        );
        assert!(
            coeff_at(50.0).abs() > coeff_at(100.0).abs(),
            "|coeff(50)| should > |coeff(100)|"
        );
    }

    #[test]
    fn gains_zero_when_fade_is_zero() {
        let data = test_sim_data();
        // 120 km is above fade_end=100 km, so fade = 0
        let aero_coefficients = [1.269, -0.205];
        let altitude_m = 120_000.0;

        let (g_alt, g_pdyn) = compute_gains(altitude_m, &aero_coefficients, &data);
        assert_relative_eq!(g_alt, 0.0, epsilon = 1e-30);
        assert_relative_eq!(g_pdyn, 0.0, epsilon = 1e-30);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn gains_are_finite_for_any_altitude(
                alt_km in 0.0_f64..500.0,
                cz in -2.0_f64..2.0,
            ) {
                let data = test_sim_data();
                let aero_coefficients = [0.5, cz];
                let altitude_m = alt_km * 1e3;

                let (g_alt, g_pdyn) = compute_gains(altitude_m, &aero_coefficients, &data);
                prop_assert!(g_alt.is_finite(), "gain_altitude_rate is not finite at alt={alt_km} km");
                prop_assert!(g_pdyn.is_finite(), "gain_dynamic_pressure is not finite at alt={alt_km} km");
            }
        }
    }
}
