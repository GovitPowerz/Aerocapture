//! Shared test fixture builders for integration tests.
//!
//! Provides ready-made instances of `NavigationOutput`, `SimData`, and
//! `NavigationBiases` so integration tests don't duplicate boilerplate.
//! Values mirror the Mars MSR defaults used in the guidance unit tests
//! (see `gnc/guidance/equilibrium_glide.rs` lines 107–194 for the source
//! of truth).

use aerocapture::data::aerodynamics::AeroTables;
use aerocapture::data::atmosphere::{AtmosphereModel, DensityProfile};
use aerocapture::data::capsule::Capsule;
use aerocapture::data::guidance_params::GuidanceParams;
use aerocapture::data::incidence::IncidenceProfile;
use aerocapture::data::pilot::{PilotModel, PilotType};
use aerocapture::data::{
    Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit, SimData,
    SphericalState, SuccessCriteria, TimePeriods,
};
use aerocapture::gnc::navigation::estimator::{NavigationBiases, NavigationOutput};

/// Build a `NavigationOutput` from key flight-state parameters.
///
/// * `altitude`    — geometric altitude above Mars surface (m)
/// * `velocity`    — relative velocity magnitude (m/s)
/// * `flight_path` — flight-path angle (rad, negative = descending)
/// * `density`     — estimated atmospheric density (kg/m³)
/// * `drag_accel`  — drag acceleration magnitude (m/s²)
/// * `lift_accel`  — lift acceleration (m/s², signed)
pub fn nav_from_state(
    altitude: f64,
    velocity: f64,
    flight_path: f64,
    density: f64,
    drag_accel: f64,
    lift_accel: f64,
) -> NavigationOutput {
    let r = 3_396_200.0 + altitude; // Mars mean equatorial radius + altitude
    NavigationOutput {
        position_estimated: [r, 0.0, 0.0],
        velocity_estimated: [velocity, flight_path, 0.6],
        acceleration_estimated: [drag_accel, lift_accel],
        aero_coefficients: [1.269, -0.205],
        density_guidance: density,
        density_exit: 1e-6,
        dynamic_pressure_estimated: 0.5 * density * velocity * velocity,
        energy_estimated: -1e6,
        ..Default::default()
    }
}

/// Return a `SimData` with Mars MSR defaults, suitable for most unit/integration tests.
///
/// Uses a three-point atmosphere stub (avoids real file I/O), a two-point
/// aero table at the equilibrium AoA, and `PilotType::Perfect` so pilot
/// dynamics don't interfere with guidance assertions.
pub fn minimal_sim_data() -> SimData {
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
        neural_net: None,
        dispersion_config: None,
    }
}

/// Return `NavigationBiases` with all fields set to zero.
///
/// Use this when the navigation error model should be transparent (i.e. the
/// measured state equals the true state).
pub fn zero_nav_biases() -> NavigationBiases {
    NavigationBiases::default()
}
