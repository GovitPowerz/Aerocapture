//! Tests that guidance schemes handle degenerate inputs gracefully.
//!
//! Not testing correctness — testing that nothing panics or produces NaN/Inf.
//! Each test drives one guidance function with a pathological input and asserts
//! the returned bank angle is finite.

mod common;

use aerocapture::config::PlanetConfig;
use aerocapture::gnc::guidance::energy_controller::{
    EnergyControllerState, energy_controller_bank,
};
use aerocapture::gnc::guidance::equilibrium_glide::equilibrium_glide_bank;
use aerocapture::gnc::guidance::fnpag::{FnpagState, fnpag_bank};
use aerocapture::gnc::guidance::predguid::{PredGuidState, predguid_bank};
use aerocapture::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};

use common::fixtures::{minimal_sim_data, nav_from_state};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn planet() -> PlanetConfig {
    PlanetConfig {
        name: "mars".into(),
        mu: 4.282829e13,
        equatorial_radius: 3393940.0,
        polar_radius: 3376780.0,
        omega: 7.088218e-5,
        j2: 1.958616e-3,
        j3: 3.145e-5,
        j4: -1.538e-5,
    }
}

// ─── Equilibrium Glide ────────────────────────────────────────────────────────

#[test]
fn eq_glide_zero_velocity() {
    let nav = nav_from_state(60_000.0, 0.0, -10.0_f64.to_radians(), 1e-4, 0.0, 0.0);
    let data = minimal_sim_data();
    let p = planet();
    let (alt, _) = geodetic_from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        &p,
    );
    let bank = equilibrium_glide_bank(&nav, &data, &p, alt);
    assert!(
        bank.is_finite(),
        "eq_glide zero velocity produced non-finite bank: {bank}"
    );
}

#[test]
fn eq_glide_zero_density() {
    let data = minimal_sim_data();
    // equilibrium_glide recomputes density from the atmosphere table itself,
    // so we use a very high altitude to get near-zero density from the table.
    let nav_high = nav_from_state(300_000.0, 5_000.0, -10.0_f64.to_radians(), 0.0, 0.0, 0.0);
    let p = planet();
    let (alt, _) = geodetic_from_spherical(
        nav_high.position_estimated[0],
        nav_high.position_estimated[1],
        nav_high.position_estimated[2],
        &p,
    );
    let bank = equilibrium_glide_bank(&nav_high, &data, &p, alt);
    assert!(
        bank.is_finite(),
        "eq_glide zero density produced non-finite bank: {bank}"
    );
}

#[test]
fn eq_glide_extreme_fpa_down() {
    // gamma = -π/2: straight down
    let nav = nav_from_state(
        60_000.0,
        5_000.0,
        -std::f64::consts::FRAC_PI_2,
        1e-4,
        10.0,
        -2.0,
    );
    let data = minimal_sim_data();
    let p = planet();
    let (alt, _) = geodetic_from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        &p,
    );
    let bank = equilibrium_glide_bank(&nav, &data, &p, alt);
    assert!(
        bank.is_finite(),
        "eq_glide straight-down produced non-finite bank: {bank}"
    );
}

#[test]
fn eq_glide_extreme_fpa_up() {
    // gamma = +π/2: straight up
    let nav = nav_from_state(
        60_000.0,
        5_000.0,
        std::f64::consts::FRAC_PI_2,
        1e-4,
        10.0,
        -2.0,
    );
    let data = minimal_sim_data();
    let p = planet();
    let (alt, _) = geodetic_from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        &p,
    );
    let bank = equilibrium_glide_bank(&nav, &data, &p, alt);
    assert!(
        bank.is_finite(),
        "eq_glide straight-up produced non-finite bank: {bank}"
    );
}

#[test]
fn eq_glide_very_high_altitude() {
    // 300 km — well above the stub atmosphere table
    let nav = nav_from_state(300_000.0, 5_000.0, -10.0_f64.to_radians(), 0.0, 0.0, 0.0);
    let data = minimal_sim_data();
    let p = planet();
    let (alt, _) = geodetic_from_spherical(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        &p,
    );
    let bank = equilibrium_glide_bank(&nav, &data, &p, alt);
    assert!(
        bank.is_finite(),
        "eq_glide 300 km altitude produced non-finite bank: {bank}"
    );
}

// ─── FNPAG ────────────────────────────────────────────────────────────────────

#[test]
fn fnpag_zero_velocity() {
    let nav = nav_from_state(60_000.0, 0.0, -10.0_f64.to_radians(), 1e-4, 0.0, 0.0);
    let data = minimal_sim_data();
    let mut state = FnpagState::new(60.0_f64.to_radians());
    let bank = fnpag_bank(&nav, &mut state, &data, &planet());
    assert!(
        bank.is_finite(),
        "fnpag zero velocity produced non-finite bank: {bank}"
    );
}

#[test]
fn fnpag_zero_density() {
    // 300 km → tabulated density ≈ 0, FNPAG should return previous bank
    let nav = nav_from_state(300_000.0, 5_000.0, -10.0_f64.to_radians(), 0.0, 0.0, 0.0);
    let data = minimal_sim_data();
    let init_bank = 60.0_f64.to_radians();
    let mut state = FnpagState::new(init_bank);
    let bank = fnpag_bank(&nav, &mut state, &data, &planet());
    assert!(
        bank.is_finite(),
        "fnpag zero density produced non-finite bank: {bank}"
    );
}

#[test]
fn fnpag_extreme_fpa_down() {
    let nav = nav_from_state(
        60_000.0,
        5_000.0,
        -std::f64::consts::FRAC_PI_2,
        1e-4,
        10.0,
        -2.0,
    );
    let data = minimal_sim_data();
    let mut state = FnpagState::new(60.0_f64.to_radians());
    let bank = fnpag_bank(&nav, &mut state, &data, &planet());
    assert!(
        bank.is_finite(),
        "fnpag straight-down produced non-finite bank: {bank}"
    );
}

#[test]
fn fnpag_extreme_fpa_up() {
    let nav = nav_from_state(
        60_000.0,
        5_000.0,
        std::f64::consts::FRAC_PI_2,
        1e-4,
        10.0,
        -2.0,
    );
    let data = minimal_sim_data();
    let mut state = FnpagState::new(60.0_f64.to_radians());
    let bank = fnpag_bank(&nav, &mut state, &data, &planet());
    assert!(
        bank.is_finite(),
        "fnpag straight-up produced non-finite bank: {bank}"
    );
}

#[test]
fn fnpag_very_high_altitude() {
    let nav = nav_from_state(300_000.0, 5_000.0, -10.0_f64.to_radians(), 0.0, 0.0, 0.0);
    let data = minimal_sim_data();
    let mut state = FnpagState::new(60.0_f64.to_radians());
    let bank = fnpag_bank(&nav, &mut state, &data, &planet());
    assert!(
        bank.is_finite(),
        "fnpag 300 km produced non-finite bank: {bank}"
    );
}

// ─── Energy Controller ────────────────────────────────────────────────────────
//
// NOTE: energy_controller_bank requires a reference trajectory in SimData
// (data.guidance.ref_trajectory.n_points > 0).  minimal_sim_data() leaves it
// empty (n_points == 0), so the function returns the 60° fallback.  These tests
// therefore exercise the fallback path — but that's intentional: the fallback
// must also be finite and not panic.

#[test]
fn energy_ctrl_zero_velocity() {
    let nav = nav_from_state(60_000.0, 0.0, -10.0_f64.to_radians(), 1e-4, 0.0, 0.0);
    let data = minimal_sim_data();
    let state = EnergyControllerState::new();
    let p = planet();
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &p,
    );
    let bank = energy_controller_bank(&nav, &state, &data, energy);
    assert!(
        bank.is_finite(),
        "energy_ctrl zero velocity produced non-finite bank: {bank}"
    );
}

#[test]
fn energy_ctrl_zero_density() {
    let nav = nav_from_state(300_000.0, 5_000.0, -10.0_f64.to_radians(), 0.0, 0.0, 0.0);
    let data = minimal_sim_data();
    let state = EnergyControllerState::new();
    let p = planet();
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &p,
    );
    let bank = energy_controller_bank(&nav, &state, &data, energy);
    assert!(
        bank.is_finite(),
        "energy_ctrl zero density produced non-finite bank: {bank}"
    );
}

#[test]
fn energy_ctrl_extreme_fpa_down() {
    let nav = nav_from_state(
        60_000.0,
        5_000.0,
        -std::f64::consts::FRAC_PI_2,
        1e-4,
        10.0,
        -2.0,
    );
    let data = minimal_sim_data();
    let state = EnergyControllerState::new();
    let p = planet();
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &p,
    );
    let bank = energy_controller_bank(&nav, &state, &data, energy);
    assert!(
        bank.is_finite(),
        "energy_ctrl straight-down produced non-finite bank: {bank}"
    );
}

#[test]
fn energy_ctrl_extreme_fpa_up() {
    let nav = nav_from_state(
        60_000.0,
        5_000.0,
        std::f64::consts::FRAC_PI_2,
        1e-4,
        10.0,
        -2.0,
    );
    let data = minimal_sim_data();
    let state = EnergyControllerState::new();
    let p = planet();
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &p,
    );
    let bank = energy_controller_bank(&nav, &state, &data, energy);
    assert!(
        bank.is_finite(),
        "energy_ctrl straight-up produced non-finite bank: {bank}"
    );
}

#[test]
fn energy_ctrl_very_high_altitude() {
    let nav = nav_from_state(300_000.0, 5_000.0, -10.0_f64.to_radians(), 0.0, 0.0, 0.0);
    let data = minimal_sim_data();
    let state = EnergyControllerState::new();
    let p = planet();
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &p,
    );
    let bank = energy_controller_bank(&nav, &state, &data, energy);
    assert!(
        bank.is_finite(),
        "energy_ctrl 300 km produced non-finite bank: {bank}"
    );
}

// ─── PredGuid ─────────────────────────────────────────────────────────────────
//
// Same note as energy_controller: minimal_sim_data() has no reference
// trajectory, so all tests hit the 60° fallback path.

#[test]
fn predguid_zero_velocity() {
    let nav = nav_from_state(60_000.0, 0.0, -10.0_f64.to_radians(), 1e-4, 0.0, 0.0);
    let data = minimal_sim_data();
    let state = PredGuidState::new();
    let p = planet();
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &p,
    );
    let bank = predguid_bank(&nav, &state, &data, energy);
    assert!(
        bank.is_finite(),
        "predguid zero velocity produced non-finite bank: {bank}"
    );
}

#[test]
fn predguid_zero_density() {
    let nav = nav_from_state(300_000.0, 5_000.0, -10.0_f64.to_radians(), 0.0, 0.0, 0.0);
    let data = minimal_sim_data();
    let state = PredGuidState::new();
    let p = planet();
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &p,
    );
    let bank = predguid_bank(&nav, &state, &data, energy);
    assert!(
        bank.is_finite(),
        "predguid zero density produced non-finite bank: {bank}"
    );
}

#[test]
fn predguid_extreme_fpa_down() {
    let nav = nav_from_state(
        60_000.0,
        5_000.0,
        -std::f64::consts::FRAC_PI_2,
        1e-4,
        10.0,
        -2.0,
    );
    let data = minimal_sim_data();
    let state = PredGuidState::new();
    let p = planet();
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &p,
    );
    let bank = predguid_bank(&nav, &state, &data, energy);
    assert!(
        bank.is_finite(),
        "predguid straight-down produced non-finite bank: {bank}"
    );
}

#[test]
fn predguid_extreme_fpa_up() {
    let nav = nav_from_state(
        60_000.0,
        5_000.0,
        std::f64::consts::FRAC_PI_2,
        1e-4,
        10.0,
        -2.0,
    );
    let data = minimal_sim_data();
    let state = PredGuidState::new();
    let p = planet();
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &p,
    );
    let bank = predguid_bank(&nav, &state, &data, energy);
    assert!(
        bank.is_finite(),
        "predguid straight-up produced non-finite bank: {bank}"
    );
}

#[test]
fn predguid_very_high_altitude() {
    let nav = nav_from_state(300_000.0, 5_000.0, -10.0_f64.to_radians(), 0.0, 0.0, 0.0);
    let data = minimal_sim_data();
    let state = PredGuidState::new();
    let p = planet();
    let energy = total_energy(
        nav.position_estimated[0],
        nav.position_estimated[1],
        nav.position_estimated[2],
        nav.velocity_estimated[0],
        nav.velocity_estimated[1],
        nav.velocity_estimated[2],
        &p,
    );
    let bank = predguid_bank(&nav, &state, &data, energy);
    assert!(
        bank.is_finite(),
        "predguid 300 km produced non-finite bank: {bank}"
    );
}
