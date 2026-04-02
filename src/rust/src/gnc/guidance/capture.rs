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
        bank_angle_longitudinal = cos_bank_commanded.acos().abs();
        is_securized = 0;
    }

    if is_securized == 1 {
        capture_state.securization_counters[0] += 1;
        capture_state.n_secur += 1;
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
