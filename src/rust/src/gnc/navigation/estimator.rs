//! Navigation state estimator.
//!
//! Matches Fortran naviga.f.
//! Adds navigation errors to the true state to produce measured state,
//! estimates atmospheric density, and manages guidance phase transitions.

use crate::config::Planet;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::orbit::elements;

/// Navigation error biases (constant during a run).
///
/// Matches Fortran common /pernav/ dispos(3), disvit(3), disdra.
#[derive(Debug, Clone, Copy, Default)]
pub struct NavigationBiases {
    pub pos: [f64; 3], // [altitude, longitude, latitude] bias
    pub vel: [f64; 3], // [velocity, flight_path, azimuth] bias
    pub drag: f64,     // drag acceleration measurement bias
}

/// Navigation filter state (persistent across steps).
#[derive(Debug, Clone, Copy)]
pub struct NavigationState {
    pub coefro: f64, // density estimation coefficient
    pub vitpre: f64, // previous radial velocity (m/s)
    pub ibounc: i32, // bounce indicator: 0=before, 1=after
    pub iphase: i32, // guidance phase: 1=capture, 2=exit, 3=emergency
    pub tcaptr: f64, // capture phase duration (s)
}

impl Default for NavigationState {
    fn default() -> Self {
        Self::new()
    }
}

impl NavigationState {
    pub fn new() -> Self {
        Self {
            coefro: 1.0,
            vitpre: 0.0,
            ibounc: 0,
            iphase: 1,
            tcaptr: 0.0,
        }
    }
}

/// Navigation output for guidance.
#[derive(Debug, Clone, Copy, Default)]
pub struct NavigationOutput {
    // Estimated state (with navigation errors added)
    pub positn: [f64; 3], // [r, lon, lat]
    pub vitesn: [f64; 3], // [V, gamma, psi]
    // Estimated aerodynamic quantities
    pub acceln: [f64; 2], // [drag accel, lift accel]
    pub coefan: [f64; 2], // [Cx, Cz]
    pub roguid: f64,      // estimated density for guidance
    pub roexit: f64,      // estimated exit density
    pub pdynan: f64,      // estimated dynamic pressure
    pub energn: f64,      // total energy
    // Orbital parameter errors
    pub ecartn: [f64; 4], // [Δa, Δe, Δi, ΔΩ]
    // Phase management
    pub ibounc: i32,
    pub iphase: i32,
    pub icrash: i32,
    pub indext: i32, // phase transition flag
    pub vitref: f64, // reference radial velocity
    pub tcaptr: f64, // capture duration
}

/// Run one navigation step.
///
/// Matches Fortran naviga.f.
#[allow(clippy::too_many_arguments)]
pub fn navigate(
    positr: &[f64; 3], // true position [r, lon, lat]
    vitesr: &[f64; 3], // true velocity [V, gamma, psi]
    alfcom: f64,       // commanded AoA
    temsim: f64,       // current time
    biases: &NavigationBiases,
    nav_state: &mut NavigationState,
    data: &SimData,
    planet: &Planet,
    run_density_bias: f64,
    run_cx_bias: f64,
    _run_cz_bias: f64,
    run_mass_bias: f64,
    run_incidence_bias: f64,
    run_ref_area_bias: f64,
    run_filter_gain_bias: f64,
) -> NavigationOutput {
    let mut out = NavigationOutput {
        indext: 0,
        icrash: 0,
        ..Default::default()
    };

    // Add navigation errors (bias constants)
    // Matches naviga.f lines 140-143
    out.positn[0] = positr[0] + biases.pos[0];
    out.positn[1] = positr[1] + biases.pos[1];
    out.positn[2] = positr[2] + biases.pos[2];
    out.vitesn[0] = vitesr[0] + biases.vel[0];
    out.vitesn[1] = vitesr[1] + biases.vel[1];
    out.vitesn[2] = vitesr[2] + biases.vel[2];

    let vitrel = out.vitesn[0];

    // Compute true drag acceleration (imodel=0)
    // Matches conphy with true state
    let (alt_true, _) = geodetic_from_spherical(positr[0], positr[1], positr[2], planet);
    let rho_true = data.atmosphere.density_at(alt_true);
    let rho_true = rho_true * (1.0 + run_density_bias);
    let cx_true = data.aero.interpolate_cx(alfcom + run_incidence_bias) * (1.0 + run_cx_bias);
    let mass_true = data.capsule.mass * (1.0 + run_mass_bias);
    let ref_area_true = data.capsule.reference_area * (1.0 + run_ref_area_bias);
    let acdrag_true =
        rho_true * ref_area_true * cx_true * vitesr[0] * vitesr[0] / (2.0 * mass_true);
    let acdram = acdrag_true + biases.drag;

    // Compute estimated aero coefficients (imodel=1)
    // Matches conphy with estimated state
    let (alt_est, _) = geodetic_from_spherical(out.positn[0], out.positn[1], out.positn[2], planet);
    let cx_est = data.aero.interpolate_cx(alfcom);
    let cz_est = data.aero.interpolate_cz(alfcom);
    out.coefan[0] = cx_est;
    out.coefan[1] = cz_est;

    // Density estimation via inverse dynamics
    // roesti = 2*|acdram|*mass / (Cx*S*V^2)
    let roesti = if cx_est.abs() > 1e-30 && vitrel.abs() > 1e-10 {
        2.0 * acdram.abs() * data.capsule.mass
            / (cx_est * data.capsule.reference_area * vitrel * vitrel)
    } else {
        0.0
    };

    // Model atmosphere density at estimated altitude
    let rho_model = data.atmosphere.density_at(alt_est);

    // Exponential filter for density correction
    // coefro = (1-λ)*coefro + λ*(roesti/rorefr)
    let lambda = (data.guidance.density_filter_gain + run_filter_gain_bias).clamp(0.01, 0.99);
    if rho_model.abs() > 1e-30 {
        nav_state.coefro = (1.0 - lambda) * nav_state.coefro + lambda * (roesti / rho_model);
    }
    if alt_est > 100e3 {
        nav_state.coefro = 1.0;
    }

    out.roguid = nav_state.coefro * rho_model;

    // Estimated drag and lift accelerations
    let mass_est = data.capsule.mass;
    let coefar = out.roguid * data.capsule.reference_area / (2.0 * mass_est);
    out.acceln[0] = coefar * cx_est * vitrel * vitrel;
    out.acceln[1] = coefar * cz_est * vitrel * vitrel;
    out.pdynan = 0.5 * out.roguid * vitrel * vitrel;

    // Exit density estimation
    let alt_exit = data.guidance.exit_altitude_threshold;
    let rho_exit_model = data.atmosphere.density_at(alt_exit);
    out.roexit = nav_state.coefro * rho_exit_model;

    // Total energy
    out.energn = total_energy(
        out.positn[0],
        out.positn[1],
        out.positn[2],
        out.vitesn[0],
        out.vitesn[1],
        out.vitesn[2],
        planet,
    );

    // Orbital elements
    let orbit = elements::from_spherical(
        out.positn[0],
        out.positn[1],
        out.positn[2],
        out.vitesn[0],
        out.vitesn[1],
        out.vitesn[2],
        planet,
    );
    out.ecartn[0] = orbit.semi_major_axis - data.target_orbit.semi_major_axis;
    out.ecartn[1] = orbit.eccentricity - data.target_orbit.eccentricity;
    out.ecartn[2] = orbit.inclination - data.target_orbit.inclination;
    out.ecartn[3] = orbit.raan - data.target_orbit.raan;

    // Bounce detection
    if nav_state.ibounc == 0 && out.vitesn[1].sin() > 0.0 {
        nav_state.ibounc = 1;
    }

    let vitrad = vitrel * out.vitesn[1].sin();

    // Phase management (matches naviga.f lines 256-299)
    if nav_state.ibounc == 0 {
        nav_state.iphase = 1;
    } else {
        let vphase = data.guidance.exit_velocity_threshold;
        if vitrel >= vphase && vitrad < 0.0 {
            nav_state.iphase = 1;
        }
        if vitrel <= vphase && nav_state.iphase == 1 {
            nav_state.iphase = 2;
            nav_state.tcaptr = temsim;
            out.indext = 1;
            out.vitref = vitrad;
        }
    }

    // Crash detection after bounce
    if nav_state.ibounc >= 1 {
        let dvitrd = vitrad - nav_state.vitpre;
        nav_state.vitpre = vitrad;
        if dvitrd < 0.0 {
            out.icrash = 1;
        }
    }

    if out.icrash == 1 {
        nav_state.iphase = 3;
    } else if vitrad >= 120.0 {
        nav_state.iphase = 2;
    }

    // Fortran has "iphase=1" hardcoded at line 301 (override)
    nav_state.iphase = 1;
    if nav_state.iphase == 1 {
        nav_state.tcaptr += data.periods.navigation;
    }

    out.ibounc = nav_state.ibounc;
    out.iphase = nav_state.iphase;
    out.tcaptr = nav_state.tcaptr;

    out
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

    /// Build a minimal SimData suitable for navigation tests.
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
                nominal_cx: 1.269,
                nominal_cz: -0.205,
                nominal_finesse: -0.205 / 1.269,
                ballistic_coeff: 0.0,
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

    /// Mars equatorial radius for converting altitude to geocentric radius.
    const MARS_REQ: f64 = 3.39394e6;

    fn zero_biases() -> NavigationBiases {
        NavigationBiases::default()
    }

    fn no_run_biases() -> [f64; 7] {
        [0.0; 7] // density, cx, cz, mass, incidence, ref_area, filter_gain
    }

    /// Helper: call navigate with a convenient tuple of run biases.
    fn call_navigate(
        positr: &[f64; 3],
        vitesr: &[f64; 3],
        biases: &NavigationBiases,
        nav_state: &mut NavigationState,
        data: &SimData,
        run_biases: &[f64; 7],
    ) -> NavigationOutput {
        navigate(
            positr,
            vitesr,
            data.entry.initial_aoa,
            0.0, // temsim
            biases,
            nav_state,
            data,
            &Planet::Mars,
            run_biases[0], // density
            run_biases[1], // cx
            run_biases[2], // cz
            run_biases[3], // mass
            run_biases[4], // incidence
            run_biases[5], // ref_area
            run_biases[6], // filter_gain
        )
    }

    // ── Test 1: biases_are_additive ──

    #[rstest]
    #[case::zero_bias(
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        "zero bias should not change the state"
    )]
    #[case::nonzero_bias(
        [100.0, 0.001, -0.002], [5.0, 0.01, -0.005],
        "nonzero bias should be added to true state"
    )]
    fn biases_are_additive(
        #[case] pos_bias: [f64; 3],
        #[case] vel_bias: [f64; 3],
        #[case] _label: &str,
    ) {
        let data = test_sim_data();
        let planet = Planet::Mars;
        // Use high altitude so density filter doesn't complicate things
        let r = planet.equatorial_radius() + 120_000.0;
        let positr = [r, 0.5, 0.3];
        let vitesr = [5500.0, -0.15, 1.2];

        let biases = NavigationBiases {
            pos: pos_bias,
            vel: vel_bias,
            drag: 0.0,
        };
        let mut nav_state = NavigationState::new();
        let out = call_navigate(
            &positr,
            &vitesr,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        for i in 0..3 {
            assert_relative_eq!(out.positn[i], positr[i] + pos_bias[i], max_relative = 1e-14);
            assert_relative_eq!(out.vitesn[i], vitesr[i] + vel_bias[i], max_relative = 1e-14);
        }
    }

    // ── Test 2: density_filter_convergence ──

    #[test]
    fn density_filter_convergence() {
        let data = test_sim_data();
        // Altitude ~40 km where there's meaningful atmosphere
        let r = MARS_REQ + 40_000.0;
        let positr = [r, 0.0, 0.0];
        let vitesr = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let mut coefro_values = Vec::new();
        for step in 0..50 {
            let _out = navigate(
                &positr,
                &vitesr,
                data.entry.initial_aoa,
                step as f64,
                &biases,
                &mut nav_state,
                &data,
                &Planet::Mars,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            );
            coefro_values.push(nav_state.coefro);
        }

        // After many steps with constant inputs, coefro should converge
        // (difference between successive values should shrink).
        let late_delta = (coefro_values[49] - coefro_values[48]).abs();
        let early_delta = (coefro_values[5] - coefro_values[4]).abs();

        // Late deltas should be smaller or equal to early deltas (convergence).
        // If coefro converges immediately (same input each step), both could be 0.
        assert!(
            late_delta <= early_delta + 1e-15,
            "density filter should converge: early_delta={early_delta:.6e}, late_delta={late_delta:.6e}"
        );

        // The final coefro should be finite and positive
        assert!(
            nav_state.coefro.is_finite() && nav_state.coefro > 0.0,
            "coefro should be finite and positive, got {}",
            nav_state.coefro
        );
    }

    // ── Test 3: high_altitude_resets_coefro ──

    #[test]
    fn high_altitude_resets_coefro() {
        let data = test_sim_data();
        // Altitude above 100 km
        let r = MARS_REQ + 110_000.0;
        let positr = [r, 0.0, 0.0];
        let vitesr = [5687.0, -0.15, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        // Perturb coefro away from 1.0
        nav_state.coefro = 2.5;

        let _out = call_navigate(
            &positr,
            &vitesr,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        assert_relative_eq!(nav_state.coefro, 1.0, max_relative = 1e-14,);
    }

    // ── Test 4: filter_gain_clamped ──

    #[rstest]
    #[case::extreme_negative(-10.0, "large negative bias should be clamped")]
    #[case::extreme_positive(10.0, "large positive bias should be clamped")]
    fn filter_gain_clamped(#[case] filter_gain_bias: f64, #[case] _label: &str) {
        let data = test_sim_data();
        // Use 40 km altitude so filter runs
        let r = MARS_REQ + 40_000.0;
        let positr = [r, 0.0, 0.0];
        let vitesr = [5000.0, -0.10, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let run_biases = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, filter_gain_bias];
        let out = call_navigate(
            &positr,
            &vitesr,
            &biases,
            &mut nav_state,
            &data,
            &run_biases,
        );

        // Function should not crash, and all outputs should be finite
        assert!(out.positn[0].is_finite(), "positn[0] should be finite");
        assert!(out.vitesn[0].is_finite(), "vitesn[0] should be finite");
        assert!(out.roguid.is_finite(), "roguid should be finite");
        assert!(
            nav_state.coefro.is_finite(),
            "coefro should be finite with filter_gain_bias={filter_gain_bias}"
        );
    }

    // ── Test 5: bounce_detection ──

    #[rstest]
    #[case::descending(-0.15, 0, "negative gamma (descending) => no bounce")]
    #[case::ascending(0.05, 1, "positive gamma (ascending) => bounce detected")]
    fn bounce_detection(#[case] gamma: f64, #[case] expected_ibounc: i32, #[case] _label: &str) {
        let data = test_sim_data();
        let r = MARS_REQ + 50_000.0;
        let positr = [r, 0.0, 0.0];
        let vitesr = [5000.0, gamma, 1.0];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let _out = call_navigate(
            &positr,
            &vitesr,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        assert_eq!(
            nav_state.ibounc, expected_ibounc,
            "with gamma={gamma}, expected ibounc={expected_ibounc}, got {}",
            nav_state.ibounc
        );
    }

    // ── Test 6: zero_biases_no_nav_errors ──

    #[test]
    fn zero_biases_no_nav_errors() {
        let data = test_sim_data();
        let r = MARS_REQ + 80_000.0;
        let positr = [r, 0.3, -0.1];
        let vitesr = [5200.0, -0.12, 0.8];
        let biases = zero_biases();
        let mut nav_state = NavigationState::new();

        let out = call_navigate(
            &positr,
            &vitesr,
            &biases,
            &mut nav_state,
            &data,
            &no_run_biases(),
        );

        // With zero biases, output position should exactly equal input
        for i in 0..3 {
            assert_eq!(
                out.positn[i], positr[i],
                "positn[{i}] should exactly match input with zero biases"
            );
            assert_eq!(
                out.vitesn[i], vitesr[i],
                "vitesn[{i}] should exactly match input with zero biases"
            );
        }
    }
}
