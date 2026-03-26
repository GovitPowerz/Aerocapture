//! Per-run initialization and Monte Carlo dispersion application.
//!
//! Domain-based system: generates draws at runtime from seeded RNG.

use crate::data::dispersions::DispersionDraw;
use crate::data::{EntryConditions, SimData};
use crate::gnc::control::pilot::PilotBiases;
use crate::gnc::navigation::estimator::NavigationBiases;

/// Per-simulation-run state after applying dispersions.
///
/// **Note:** `wind_scale` must be initialized to `1.0` (identity), not `0.0`.
/// Setting it to `0.0` silently zeroes out all wind. Use `init_run_from_draw()`
/// for production; for tests, set `wind_scale: 1.0` explicitly.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RunState {
    pub entry: EntryConditions,
    pub cx_bias: f64,             // drag coefficient bias (fractional)
    pub cz_bias: f64,             // lift coefficient bias (fractional)
    pub density_bias: f64,        // atmosphere density bias (fractional)
    pub mass_bias: f64,           // mass bias (fractional)
    pub incidence_bias: f64,      // incidence error (radians)
    pub ref_area_bias: f64,       // reference area bias (fractional)
    pub max_bank_rate_bias: f64,  // max bank rate bias (fractional)
    pub filter_gain_bias: f64,    // density filter gain bias (absolute delta)
    pub wind_scale: f64,          // wind speed multiplier (1.0 = nominal, NOT 0.0)
    pub wind_direction_bias: f64, // wind direction rotation (radians)
    pub nav_biases: NavigationBiases,
    pub pilot_biases: PilotBiases,
}

/// Initialize a simulation run by applying dispersion draws to entry conditions.
pub fn init_run_from_draw(sim_data: &SimData, draw: &DispersionDraw) -> RunState {
    let mut entry = sim_data.entry;
    entry.state.altitude += draw.altitude;
    entry.state.longitude += draw.longitude;
    entry.state.latitude += draw.latitude;
    entry.state.velocity += draw.velocity;
    entry.state.flight_path += draw.flight_path;
    entry.state.azimuth += draw.azimuth;

    RunState {
        entry,
        cx_bias: draw.drag_coeff,
        cz_bias: draw.lift_coeff,
        density_bias: draw.density,
        mass_bias: draw.mass,
        incidence_bias: draw.incidence,
        ref_area_bias: draw.ref_area,
        max_bank_rate_bias: draw.max_bank_rate,
        filter_gain_bias: draw.filter_gain,
        wind_scale: draw.wind_scale,
        wind_direction_bias: draw.wind_direction_bias,
        nav_biases: NavigationBiases {
            pos: [draw.nav_altitude, draw.nav_longitude, draw.nav_latitude],
            vel: [draw.nav_velocity, draw.nav_flight_path, draw.nav_azimuth],
            drag: draw.nav_drag_accel,
        },
        pilot_biases: PilotBiases {
            tau: draw.pilot_tau,
            damping: draw.pilot_damping,
            frequency: draw.pilot_frequency,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::dispersions::DispersionDraw;
    use crate::data::{
        Constraints, EntryConditions, FinalConditions, OrbitalTarget, ParkingOrbit, SphericalState,
        SuccessCriteria, TimePeriods, aerodynamics, atmosphere, capsule, guidance_params,
        incidence, pilot,
    };

    fn test_sim_data() -> SimData {
        SimData {
            capsule: capsule::Capsule {
                mass: 1089.0,
                reference_area: 14.7,
                cq: 0.0,
                max_bank_rate: 0.0,
                periods: TimePeriods::default(),
            },
            aero: aerodynamics::AeroTables::default(),
            atmosphere: atmosphere::AtmosphereModel::default(),
            entry: EntryConditions {
                state: SphericalState {
                    altitude: 130_000.0,
                    longitude: 0.1,
                    latitude: 0.2,
                    velocity: 5687.0,
                    flight_path: -0.189,
                    azimuth: 0.664,
                },
                initial_date: 0.0,
                initial_bank: 1.13,
                initial_aoa: -0.48,
            },
            constraints: Constraints::default(),
            final_conditions: FinalConditions::default(),
            target_orbit: OrbitalTarget::default(),
            parking_orbit: ParkingOrbit::default(),
            periods: TimePeriods::default(),
            guidance: guidance_params::GuidanceParams::default(),
            incidence: incidence::IncidenceProfile {
                n_points: 0,
                altitudes: vec![],
                incidences: vec![],
            },
            pilot: pilot::PilotModel {
                pilot_type: pilot::PilotType::Perfect,
                time_constant: 1.0,
                damping: 0.7,
                frequency: 0.072,
            },
            success: SuccessCriteria::default(),
            wind_enabled: false,
            wind_table: None,
            neural_net: None,
            dispersion_config: None,
            nav_mode: crate::data::NavMode::Bias,
            nav_config: None,
        }
    }

    #[test]
    fn test_zero_draw_preserves_entry() {
        let sim_data = test_sim_data();
        let draw = DispersionDraw::default();
        let run = init_run_from_draw(&sim_data, &draw);

        assert_eq!(run.entry.state.altitude, 130_000.0);
        assert_eq!(run.entry.state.velocity, 5687.0);
        assert_eq!(run.entry.state.longitude, 0.1);
        assert_eq!(run.cx_bias, 0.0);
        assert_eq!(run.cz_bias, 0.0);
        assert_eq!(run.density_bias, 0.0);
        assert_eq!(run.mass_bias, 0.0);
        assert_eq!(run.incidence_bias, 0.0);
        assert_eq!(run.ref_area_bias, 0.0);
        assert_eq!(run.max_bank_rate_bias, 0.0);
        assert_eq!(run.filter_gain_bias, 0.0);
        assert_eq!(run.pilot_biases.tau, 0.0);
        assert_eq!(run.pilot_biases.damping, 0.0);
        assert_eq!(run.pilot_biases.frequency, 0.0);
        assert_eq!(run.nav_biases.drag, 0.0);
    }

    #[test]
    fn test_nonzero_draw_shifts_entry() {
        let sim_data = test_sim_data();
        let draw = DispersionDraw {
            altitude: 500.0,
            velocity: -2.0,
            longitude: 0.001,
            latitude: -0.001,
            flight_path: 0.002,
            azimuth: -0.003,
            drag_coeff: 0.05,
            lift_coeff: -0.03,
            density: 0.15,
            mass: -0.01,
            incidence: 0.01,
            ref_area: 0.02,
            max_bank_rate: -0.05,
            pilot_tau: 0.08,
            pilot_damping: -0.03,
            pilot_frequency: 0.05,
            filter_gain: 0.07,
            ..Default::default()
        };
        let run = init_run_from_draw(&sim_data, &draw);

        assert_eq!(run.entry.state.altitude, 130_500.0);
        assert_eq!(run.entry.state.velocity, 5685.0);
        assert_eq!(run.cx_bias, 0.05);
        assert_eq!(run.cz_bias, -0.03);
        assert_eq!(run.density_bias, 0.15);
        assert_eq!(run.mass_bias, -0.01);
        assert_eq!(run.incidence_bias, 0.01);
        assert_eq!(run.ref_area_bias, 0.02);
        assert_eq!(run.max_bank_rate_bias, -0.05);
        assert_eq!(run.filter_gain_bias, 0.07);
        assert_eq!(run.pilot_biases.tau, 0.08);
        assert_eq!(run.pilot_biases.damping, -0.03);
        assert_eq!(run.pilot_biases.frequency, 0.05);
    }

    proptest::proptest! {
        /// For any reasonable dispersion draw, the dispersed altitude and velocity
        /// must remain positive — a negative altitude or velocity would cause
        /// the simulator to crash before a single integration step.
        #[test]
        fn proptest_dispersed_entry_positive(
            alt_draw in -5_000.0_f64..=5_000.0_f64,   // ±5 km
            vel_draw in -50.0_f64..=50.0_f64,          // ±50 m/s
            fp_draw  in -0.05_f64..=0.05_f64,          // ±~3 deg in rad
        ) {
            let sim_data = test_sim_data();
            let draw = DispersionDraw {
                altitude: alt_draw,
                velocity: vel_draw,
                flight_path: fp_draw,
                ..Default::default()
            };
            let run = init_run_from_draw(&sim_data, &draw);

            proptest::prop_assert!(
                run.entry.state.altitude > 0.0,
                "dispersed altitude must be positive, got {}",
                run.entry.state.altitude
            );
            proptest::prop_assert!(
                run.entry.state.velocity > 0.0,
                "dispersed velocity must be positive, got {}",
                run.entry.state.velocity
            );
        }
    }

    #[test]
    fn test_nav_bias_mapping() {
        let sim_data = test_sim_data();
        let draw = DispersionDraw {
            nav_altitude: 100.0,
            nav_longitude: 0.01,
            nav_latitude: 0.02,
            nav_velocity: 0.5,
            nav_flight_path: 0.03,
            nav_azimuth: 0.04,
            nav_drag_accel: 0.1,
            ..Default::default()
        };
        let run = init_run_from_draw(&sim_data, &draw);

        assert_eq!(run.nav_biases.pos[0], 100.0);
        assert_eq!(run.nav_biases.pos[1], 0.01);
        assert_eq!(run.nav_biases.pos[2], 0.02);
        assert_eq!(run.nav_biases.vel[0], 0.5);
        assert_eq!(run.nav_biases.vel[1], 0.03);
        assert_eq!(run.nav_biases.vel[2], 0.04);
        assert_eq!(run.nav_biases.drag, 0.1);
    }
}
