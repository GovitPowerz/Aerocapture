//! CPAG unit tests: solver smoke, eps identities, replan throttle, constraint
//! activation, convergence fallback. E2E behavior on the real mission config is
//! covered by the golden regression (configs/test/test_cpag_golden.toml).

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
use std::sync::Arc;

fn test_nav(velocity: f64, altitude: f64) -> NavigationOutput {
    let r = PlanetConfig::mars().equatorial_radius + altitude;
    NavigationOutput {
        position_estimated: [r, 0.0, 0.0],
        velocity_estimated: [velocity, -0.15, 0.6],
        density_guidance: 0.001,
        heat_load_fraction: 0.1,
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
            cx: vec![1.22, 1.22],
            cz: vec![0.404, 0.404],
            equilibrium_aoa: -0.48,
            ..Default::default()
        },
        // Mars-like analytic exponential (rho0 = 0.0134, H = 11.1 km) sampled
        // every 5 km: a REAL capture corridor exists in it, unlike the 3-point
        // legacy fixture whose linear interpolation makes mid-altitudes
        // absurdly dense (645 kW/m^2 peaks, no flyable corridor).
        atmosphere: Arc::new(AtmosphereModel {
            n_points: 27,
            altitudes: (0..27).map(|i| 5_000.0 * i as f64).collect(),
            densities: (0..27)
                .map(|i| 0.0134 * (-(5_000.0 * i as f64) / 11_100.0).exp())
                .collect(),
            ref_density: 1.0984316253131566e-7,
            scale_factor: 1.0 / 11_100.0,
            ref_altitude: 130_000.0,
            gas_constant: 1.3,
            density_profile: DensityProfile::default(),
        }),
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
        guidance: GuidanceParams::default(),
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
        final_conditions: FinalConditions {
            altitude: 130_988.0,
            ..Default::default()
        },
        parking_orbit: ParkingOrbit::default(),
        constraints: Constraints {
            max_heat_flux: 200e3,
            max_load_factor: 4.0 * 9.81,
            max_dynamic_pressure: 10_810.0,
            max_heat_load: 25_000e3,
        },
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

/// Reduced SCP settings so debug-build tests stay fast.
fn fast_params() -> CpagParams {
    CpagParams {
        seg_dt: 16.0,
        n_sub: 2,
        horizon_max: 800.0,
        max_iters: 6,
        max_iters_warm: 2,
        ..CpagParams::default()
    }
}

fn entry_state8(data: &SimData, planet: &PlanetConfig) -> State8 {
    [
        planet.equatorial_radius + data.entry.state.altitude,
        0.0,
        0.0,
        data.entry.state.velocity,
        data.entry.state.flight_path,
        0.6,
        data.entry.initial_bank,
        0.0,
    ]
}

// ── Solver + formulation identities ─────────────────────────────────────────

/// Clarabel API smoke: min 1/2 x^2 s.t. x >= 1 has x* = 1.
#[test]
fn clarabel_solves_known_qp() {
    let p_mat: CscMatrix<f64> = CscMatrix::new_from_triplets(1, 1, vec![0], vec![0], vec![1.0]);
    // -x <= -1 as b - Ax in Nonneg: A = [-1], b = [-1]
    let a_mat: CscMatrix<f64> = CscMatrix::new_from_triplets(1, 1, vec![0], vec![0], vec![-1.0]);
    let cones = [SupportedConeT::NonnegativeConeT(1)];
    let settings = DefaultSettings {
        verbose: false,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&p_mat, &[0.0], &a_mat, &[-1.0], &cones, settings).unwrap();
    solver.solve();
    assert!(matches!(solver.solution.status, SolverStatus::Solved));
    assert_relative_eq!(solver.solution.x[0], 1.0, epsilon = 1e-6);
}

/// The eps root coincides with apoapsis == target (Keplerian identity).
#[test]
fn eps_zero_iff_apoapsis_on_target() {
    let data = test_sim_data();
    let planet = PlanetConfig::mars();
    let m = Model::new(&data, &planet, 1.0);
    let mut x: State8 = [
        planet.equatorial_radius + 131e3,
        0.0,
        0.0,
        0.0,
        3.4_f64.to_radians(),
        60.0_f64.to_radians(),
        0.0,
        0.0,
    ];
    let (mut lo, mut hi) = (3000.0, 4600.0);
    for _ in 0..60 {
        x[IV] = 0.5 * (lo + hi);
        if eps_apoapsis(&x, &m) > 0.0 {
            hi = x[IV];
        } else {
            lo = x[IV];
        }
    }
    assert_relative_eq!(
        apoapsis_radius(&x, &planet),
        m.target_apo_radius,
        epsilon = 200.0
    );
}

/// eps stays finite and monotone in v across the parabolic boundary — the
/// paper's fix for the apoapsis-Jacobian singularity at escape velocity.
#[test]
fn eps_smooth_through_escape() {
    let data = test_sim_data();
    let planet = PlanetConfig::mars();
    let m = Model::new(&data, &planet, 1.0);
    let r = planet.equatorial_radius + 131e3;
    let v_esc = (2.0 * planet.mu / r).sqrt();
    let mut x: State8 = [r, 0.0, 0.0, 0.0, 0.05, 0.8, 0.0, 0.0];
    let mut prev = f64::NEG_INFINITY;
    for i in 0..41 {
        x[IV] = v_esc - 200.0 + 10.0 * i as f64;
        let e = eps_apoapsis(&x, &m);
        assert!(e.is_finite(), "eps not finite at v={}", x[IV]);
        assert!(e > prev, "eps not monotone at v={}", x[IV]);
        prev = e;
    }
}

// ── Guidance-loop behavior ───────────────────────────────────────────────────

/// Within `replan_period` the profile is played back without re-solving.
#[test]
fn replan_throttled_within_replan_period() {
    let mut data = test_sim_data();
    data.guidance.cpag = fast_params();
    let planet = PlanetConfig::mars();
    let mut state = CpagState::new(64.77_f64.to_radians());

    let nav = test_nav(5687.0, 60_000.0);
    let _ = cpag_bank(&nav, &mut state, &data, &planet, 0.0);
    assert!(state.initialized, "first in-atmosphere call must replan");
    assert_relative_eq!(state.last_replan_time, 0.0, epsilon = 1e-12);
    let profile_before = state.u_profile.clone();

    // 1 s later with a substantially different nav state: playback only.
    let nav_later = test_nav(4000.0, 55_000.0);
    let _ = cpag_bank(&nav_later, &mut state, &data, &planet, 1.0);
    assert_relative_eq!(state.last_replan_time, 0.0, epsilon = 1e-12);
    assert_eq!(
        state.u_profile, profile_before,
        "held profile must not change"
    );

    // Past the period: replans (timestamp advances).
    let _ = cpag_bank(&nav_later, &mut state, &data, &planet, 2.5);
    assert_relative_eq!(state.last_replan_time, 2.5, epsilon = 1e-12);
}

/// Outside the sensible atmosphere the plan is held (no replan) and the
/// command is the playback of the initial (constant-bank) profile.
#[test]
fn vacuum_holds_plan() {
    let mut data = test_sim_data();
    data.guidance.cpag = fast_params();
    let planet = PlanetConfig::mars();
    let initial_bank = 64.77_f64.to_radians();
    let mut state = CpagState::new(initial_bank);

    let nav = test_nav(5687.0, 250_000.0); // exponential tail ~2e-12 < 1e-10
    let bank = cpag_bank(&nav, &mut state, &data, &planet, 0.0);

    assert!(!state.initialized, "vacuum must not trigger a replan");
    assert_relative_eq!(bank, initial_bank, epsilon = 1e-12);
}

/// The commanded playback bank stays inside the sigma box at all times.
#[test]
fn playback_respects_sigma_box() {
    let mut data = test_sim_data();
    data.guidance.cpag = fast_params();
    let planet = PlanetConfig::mars();
    let mut state = CpagState::new(170.0_f64.to_radians());

    let nav = test_nav(5400.0, 50_000.0);
    let _ = cpag_bank(&nav, &mut state, &data, &planet, 0.0);
    let sigma_max = data.guidance.cpag.sigma_max_deg.to_radians();
    for i in 0..200 {
        let bank = state.sigma_at(i as f64, &data.guidance.cpag);
        assert!(
            bank.abs() <= sigma_max + 1e-12,
            "playback bank {bank} outside sigma box at t={i}"
        );
    }
}

/// Enforcing the heat-flux constraint must lower the planned peak flux when
/// the unconstrained optimum violates a limit that is physically satisfiable
/// (set between the full-lift-up floor and the unconstrained optimum's peak).
#[test]
fn heat_flux_constraint_activates() {
    let data = test_sim_data();
    let planet = PlanetConfig::mars();
    let m = Model::new(&data, &planet, 1.0);
    let mut params = fast_params();
    params.max_iters = 10;

    let x0 = entry_state8(&data, &planet);
    let n_seg = (params.horizon_max / params.seg_dt).ceil() as usize;
    let plan_peak = |result: &ReplanResult, mm: &Model, p: &CpagParams| -> f64 {
        shoot_profile(&x0, &result.u_profile, mm, p)
            .x_nodes
            .iter()
            .map(|x| path_quantities(x, mm).0)
            .fold(0.0, f64::max)
    };

    // Physical floor: full lift-up (max recovery authority) peak flux.
    let u_up = roll_to_bank_profile(x0[ISIGMA], 0.0, n_seg, &params, data.capsule.max_bank_rate);
    let peak_floor = shoot_profile(&x0, &u_up, &m, &params)
        .x_nodes
        .iter()
        .map(|x| path_quantities(x, &m).0)
        .fold(0.0, f64::max);

    params.enforce_heat_flux = false;
    let uncon = scp_replan(
        &x0,
        &m,
        &params,
        None,
        data.capsule.max_bank_rate,
        params.max_iters,
    );
    let peak_unconstrained = plan_peak(&uncon, &m, &params);
    assert!(
        peak_unconstrained > peak_floor * 1.02,
        "fixture precondition: unconstrained peak {peak_unconstrained:.0} must exceed the \
         lift-up floor {peak_floor:.0} for the limit to be satisfiable"
    );

    // Satisfiable limit between the floor and the unconstrained optimum.
    let mut data_tight = test_sim_data();
    data_tight.constraints.max_heat_flux = 0.5 * (peak_floor + peak_unconstrained);
    let m_tight = Model::new(&data_tight, &planet, 1.0);
    params.enforce_heat_flux = true;
    let con = scp_replan(
        &x0,
        &m_tight,
        &params,
        None,
        data_tight.capsule.max_bank_rate,
        params.max_iters,
    );
    let peak_constrained = plan_peak(&con, &m_tight, &params);

    assert!(
        peak_constrained < peak_unconstrained * 0.99,
        "enforced peak {peak_constrained:.0} not below unconstrained {peak_unconstrained:.0}"
    );
    assert!(
        peak_constrained <= data_tight.constraints.max_heat_flux * 1.05,
        "enforced peak {peak_constrained:.0} far above the limit {:.0}",
        data_tight.constraints.max_heat_flux
    );
}

/// Unrecoverable state (deep, steep, lift-down): the replan must settle on the
/// crash tier without panicking and the command must stay finite and bounded.
#[test]
fn convergence_fallback_on_unrecoverable_state() {
    let mut data = test_sim_data();
    data.guidance.cpag = fast_params();
    let planet = PlanetConfig::mars();
    let mut state = CpagState::new(160.0_f64.to_radians());

    let mut nav = test_nav(5200.0, 25_000.0);
    nav.velocity_estimated[1] = -0.26; // ~-15 deg, diving
    let bank = cpag_bank(&nav, &mut state, &data, &planet, 0.0);

    assert!(bank.is_finite(), "fallback bank not finite: {bank}");
    assert!(
        bank.abs() <= data.guidance.cpag.sigma_max_deg.to_radians() + 1e-12,
        "fallback bank {bank} outside sigma box"
    );
    assert!(state.initialized);
}

/// First replan from the entry interface improves the terminal eps residual
/// versus the initial constant-bank hold (the corrector corrects).
#[test]
fn entry_replan_improves_eps() {
    let data = test_sim_data();
    let planet = PlanetConfig::mars();
    let m = Model::new(&data, &planet, 1.0);
    let params = fast_params();

    let x0 = entry_state8(&data, &planet);
    let n_seg = (params.horizon_max / params.seg_dt).ceil() as usize;
    let hold = shoot_profile(&x0, &vec![0.0; n_seg], &m, &params);
    let eps_hold = (eps_apoapsis(hold.x_nodes.last().unwrap(), &m) / EPS_SCALE).abs();

    let result = scp_replan(&x0, &m, &params, None, data.capsule.max_bank_rate, 8);
    assert!(
        result.eps_mj.abs() < eps_hold,
        "replan eps {:.4} MJ/kg not below hold eps {:.4}",
        result.eps_mj.abs(),
        eps_hold
    );
    assert!(result.iters >= 1);
}

// ── Proptest ─────────────────────────────────────────────────────────────────

mod prop {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(12))]
        /// For valid atmospheric entry conditions, CPAG must always return a
        /// finite signed bank within the sigma box.
        #[test]
        fn output_always_finite_and_bounded(
            alt in 30_000.0..100_000.0_f64,
            vel in 3_500.0..6_000.0_f64,
            fpa in -0.15..0.05_f64,
            sigma0 in -3.0..3.0_f64,
        ) {
            let mut data = test_sim_data();
            data.guidance.cpag = fast_params();
            data.guidance.cpag.max_iters = 3;
            let planet = PlanetConfig::mars();
            let mut state = CpagState::new(sigma0);

            let mut nav = test_nav(vel, alt);
            nav.velocity_estimated[1] = fpa;
            let bank = cpag_bank(&nav, &mut state, &data, &planet, 0.0);

            prop_assert!(bank.is_finite(), "bank not finite: {}", bank);
            prop_assert!(
                bank.abs() <= data.guidance.cpag.sigma_max_deg.to_radians() + 1e-10,
                "bank outside sigma box: {}",
                bank
            );
        }
    }
}
