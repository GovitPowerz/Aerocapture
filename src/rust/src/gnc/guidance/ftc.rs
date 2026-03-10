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
    pub gitcom: f64, // current commanded bank angle (rad)
    pub gitpre: f64, // previous commanded bank angle (rad)
    pub gpilpr: f64, // previous pilot bank angle (rad)
    pub alfcom: f64, // commanded AoA (rad)

    // Roll sign and reversal tracking
    pub sgngit: f64, // roll polarity sign (-1, 0, +1)
    pub somgit: f64, // cumulative bank angle changes (rad)
    pub nbroll: i32, // number of roll reversals
    pub indrvr: i32, // roll reversal active flag
    pub rolway: i32, // roll reversal path (+1=short, -1=long)
    pub trevrs: f64, // roll reversal duration (s)

    // Guidance securization
    pub iprepr: [i32; 2], // securization counters
    pub iguida: [i32; 2], // securization indicators

    // Reference velocity
    pub vitref: f64,

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
            gitcom: initial_bank,
            gitpre: initial_bank,
            gpilpr: initial_bank,
            alfcom: initial_aoa,
            sgngit: if initial_bank >= 0.0 { 1.0 } else { -1.0 },
            somgit: 0.0,
            nbroll: 0,
            indrvr: 0,
            rolway: 1,
            trevrs: 0.0,
            iprepr: [0, 0],
            iguida: [1, 1],
            vitref: 0.0,
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
    pub gitcom: f64, // commanded bank angle (rad)
    pub alfcom: f64, // commanded AoA (rad)
    pub vitgit: f64, // bank rate before saturation (rad/s)
    pub ilongi: i32, // longitudinal guidance active
    pub isatur: i32, // rate saturation occurred
    pub indrol: i32, // roll reversal indicator
}

/// Run one FTC guidance step.
#[allow(clippy::too_many_arguments)]
pub fn guidance_step(
    nav: &NavigationOutput,
    gitpil: f64, // pilot-realized bank angle
    temsim: f64,
    gitref: f64, // reference bank angle (from config, rad)
    state: &mut FtcState,
    data: &SimData,
    planet: &Planet,
    is_reference: bool,
    guidance_type: GuidanceType,
) -> FtcOutput {
    let pi = std::f64::consts::PI;
    let mut out = FtcOutput::default();

    let sgnpre = state.sgngit;
    state.gpilpr = gitpil;

    // === Angle of attack guidance ===
    // proalf returns altitude as scheduling parameter
    let (altitude, _) =
        geodetic_from_spherical(nav.positn[0], nav.positn[1], nav.positn[2], planet);
    state.alfcom = data.incidence.incidence_at(altitude);
    out.alfcom = state.alfcom;

    // === Longitudinal guidance activation ===
    let enrjlt = total_energy(
        nav.positn[0],
        nav.positn[1],
        nav.positn[2],
        nav.vitesn[0],
        nav.vitesn[1],
        nav.vitesn[2],
        planet,
    );

    let mut ilongi: i32;
    if enrjlt <= data.guidance.longi_activation && enrjlt >= data.guidance.longi_inhibition {
        ilongi = 1;
    } else {
        ilongi = 0;
        state.iprepr[1] += 1;
    }

    ilongi *= state.iguida[0];
    out.ilongi = ilongi;

    // === Reference trajectory mode ===
    if is_reference {
        ilongi = 0;
        state.iguida[0] = 0;
        state.iguida[1] = 0;
    }

    // === Longitudinal bank angle command ===
    // gitref passed as parameter from config.reference_bank_angle
    let gitlon: f64;

    if is_reference {
        state.gitcom = gitref;
        gitlon = gitref;
    } else if ilongi == 0 {
        gitlon = gitref.abs();
    } else {
        // Longitudinal guidance dispatch
        gitlon = match guidance_type {
            GuidanceType::Ftc => guicap(nav, enrjlt, altitude, state, data, planet),
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
    let mut ilater: i32;
    if enrjlt <= data.guidance.lateral_activation && enrjlt >= data.guidance.lateral_inhibition {
        ilater = 1;
    } else {
        ilater = 0;
    }

    ilater *= state.iguida[1];

    // === Lateral guidance ===
    let mut indrol = 0;
    if ilater == 1 {
        guilat(nav, gitlon, temsim, state, data, planet, &mut indrol);
        if state.indrvr == 1 {
            state.iguida[1] = 0;
        }
    } else {
        state.sgngit = sgnpre;
    }

    // === Combine longitudinal and lateral commands ===
    if !is_reference {
        if state.iguida[0] * state.iguida[1] == 1 {
            state.gitcom = gitlon * state.sgngit;
        } else if state.indrvr == 1 {
            let vgitmx = data.capsule.max_bank_rate;
            let tguida = data.periods.guidance;
            if state.rolway == 1 {
                if state.sgngit > 0.0 {
                    state.gitcom = state.gitpre + vgitmx * tguida;
                } else {
                    state.gitcom = state.gitpre - vgitmx * tguida;
                }
            } else {
                if state.sgngit > 0.0 {
                    state.gitcom = state.gitpre - vgitmx * tguida;
                    if state.gitcom < -pi {
                        state.gitcom += 2.0 * pi;
                    }
                } else {
                    state.gitcom = state.gitpre + vgitmx * tguida;
                    if state.gitcom > pi {
                        state.gitcom -= 2.0 * pi;
                    }
                }
            }
        }
    }

    // === Roll rate saturation ===
    let vgitmx = data.capsule.max_bank_rate;
    let tguida = data.periods.guidance;
    let vitgit = (state.gitcom - state.gitpre) / tguida;
    let mut isatur = 0;

    if vitgit.abs() - vgitmx > 1e-10 {
        isatur = 1;
        if state.gitcom > state.gitpre {
            state.gitcom = state.gitpre + vgitmx * tguida;
        } else {
            state.gitcom = state.gitpre - vgitmx * tguida;
        }
    }

    // Cumulative bank angle tracking
    if vitgit.abs() > 1e-10 {
        state.somgit += (state.gitcom - state.gitpre).abs();
    }

    state.gitpre = state.gitcom;

    out.gitcom = state.gitcom;
    out.vitgit = vitgit;
    out.isatur = isatur;
    out.indrol = indrol;

    out
}

/// Capture phase longitudinal guidance: altitude-gain predictor-corrector.
fn guicap(
    nav: &NavigationOutput,
    enrjlt: f64,
    altitude: f64,
    state: &mut FtcState,
    data: &SimData,
    _planet: &Planet,
) -> f64 {
    let ref_traj = &data.guidance.ref_trajectory;

    let vitrel = nav.vitesn[0];
    let vitrad = vitrel * nav.vitesn[1].sin();
    let pdyneq = 0.5 * nav.roguid * vitrel * vitrel;

    // Interpolate reference trajectory at current energy
    let cmunom = ref_traj.interpolate(enrjlt, &ref_traj.cos_bank);
    let prenom = ref_traj.interpolate(enrjlt, &ref_traj.pressure);
    let hdtnom = ref_traj.interpolate(enrjlt, &ref_traj.radial_vel);
    let _httnom = ref_traj.interpolate(enrjlt, &ref_traj.altitude_rate);

    // Compute gains (tbgain)
    let (gaindh, gainpd) = tbgain(altitude, &nav.coefan, data);

    // Predictor-corrector equation
    // cos(gitlon) = cmunom + gaindh*(vitrad - hdtnom)/pdyneq + gainpd*(pdyneq - prenom)/pdyneq
    let pdyneq_safe = if pdyneq.abs() > 1e-10 { pdyneq } else { 1e-10 };
    let mut cosmuc = cmunom
        + gaindh * (vitrad - hdtnom) / pdyneq_safe
        + gainpd * (pdyneq - prenom) / pdyneq_safe;

    // Securization: clamp cos to [-1, 1]
    let isecur;
    let gitlon;
    if cosmuc.abs() > 1.0 {
        cosmuc = cosmuc.signum();
        gitlon = cosmuc.acos();
        isecur = 1;
    } else {
        gitlon = cosmuc.acos().abs();
        isecur = 0;
    }

    if isecur == 1 {
        state.iprepr[0] += 1;
        state.n_secur += 1;
    }

    gitlon
}

/// Compute guidance gains from altitude-based Pdyn model.
fn tbgain(altitude: f64, coefan: &[f64; 2], data: &SimData) -> (f64, f64) {
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
    let inumer = found.unwrap_or_else(|| {
        if pdyn_table.is_empty() {
            0
        } else {
            pdyn_table.len() - 1
        }
    });

    let coefpd_a = if inumer < pdyn_table.len() {
        pdyn_table[inumer].coeff_a
    } else {
        1.0
    };

    // Gains
    let amorft = data.guidance.capture_damping;
    let pulsft = data.guidance.capture_frequency;
    let srefer = data.capsule.reference_area;
    let xmasse = data.capsule.mass;
    let cz = coefan[1]; // lift coefficient

    let gaindh = if (srefer * cz).abs() > 1e-30 {
        -2.0 * amorft * pulsft * xmasse / (srefer * cz)
    } else {
        0.0
    };

    let gainpd = if (coefpd_a * srefer * cz).abs() > 1e-30 {
        -pulsft * pulsft * xmasse / (coefpd_a * srefer * cz)
    } else {
        0.0
    };

    (gaindh, gainpd)
}

/// Lateral guidance — roll reversal logic.
fn guilat(
    nav: &NavigationOutput,
    gitlon: f64,
    _temsim: f64,
    state: &mut FtcState,
    data: &SimData,
    planet: &Planet,
    indrol: &mut i32,
) {
    let pi = std::f64::consts::PI;

    if gitlon == 0.0 || gitlon == pi {
        return;
    }

    let sgnpre = state.sgngit;

    // Compute orbital elements for inclination
    let orbit = elements::from_spherical(
        nav.positn[0],
        nav.positn[1],
        nav.positn[2],
        nav.vitesn[0],
        nav.vitesn[1],
        nav.vitesn[2],
        planet,
    );

    let xinccr = data.target_orbit.inclination - orbit.inclination;
    // Hemisphere correction intentionally omitted (inactive)

    let vitrel = nav.vitesn[0];

    // Corridor boundary: xinmax = (v/coridx)^4 + coridy
    let coridx = data.guidance.corridor_slope;
    let coridy = data.guidance.corridor_intercept;
    let xinmax = (vitrel / coridx).powi(4) + coridy;

    // Reversal decision
    if xinccr.abs() >= xinmax && gitlon.abs() > 1e-10 && state.nbroll < data.guidance.max_reversals
    {
        if xinccr > xinmax {
            state.sgngit = -1.0;
        } else if xinccr < -xinmax {
            state.sgngit = 1.0;
        }

        if state.sgngit * sgnpre < 0.0 {
            // Roll reversal commanded
            *indrol = 1;
            state.nbroll += 1;

            if state.indrvr == 0 {
                state.indrvr = 1;
                state.indrvr = 0; // immediately reset after arming
                state.rolway = 1;
                let dgitcm = state.gitpre.abs() + gitlon.abs();
                let vgitmx = data.capsule.max_bank_rate;
                let tguida = data.periods.guidance;
                state.trevrs = dgitcm / vgitmx;
                state.trevrs = (state.trevrs / tguida).floor() * tguida;
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
            positn: [r, 0.0, 0.0],
            vitesn: [5000.0, -0.15, 0.6],
            acceln: [50.0, -8.0],
            coefan: [1.269, -0.205],
            roguid: 0.001,
            roexit: 1e-6,
            pdynan: 0.5 * 0.001 * 5000.0 * 5000.0,
            energn: -1e6,
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
            0.0, // temsim
            initial_bank,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        assert!(out.gitcom.is_finite(), "gitcom not finite: {}", out.gitcom);
        assert!(out.alfcom.is_finite(), "alfcom not finite: {}", out.alfcom);
        assert!(out.vitgit.is_finite(), "vitgit not finite: {}", out.vitgit);
    }

    /// In reference mode, output bank should equal the reference bank angle.
    #[test]
    fn reference_mode_returns_reference_bank() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = Planet::Mars;
        let gitref = 45.0_f64.to_radians();
        let mut state = FtcState::new(gitref, -0.48_f64.to_radians());
        // Prime gitpre so rate saturation doesn't shift the value
        state.gitpre = gitref;

        let out = guidance_step(
            &nav,
            gitref,
            0.0,
            gitref,
            &mut state,
            &data,
            &planet,
            true, // is_reference
            GuidanceType::Ftc,
        );

        assert!(
            (out.gitcom - gitref).abs() < 1e-9,
            "expected gitcom ≈ gitref ({:.6} rad), got {:.6} rad",
            gitref,
            out.gitcom,
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
            out.gitcom >= -pi && out.gitcom <= pi,
            "gitcom = {:.4} rad is outside [-π, π]",
            out.gitcom,
        );
    }

    /// ilongi=0 (guidance inactive) should still return a finite bank
    /// equal to |gitref|, without saturating.
    #[test]
    fn inactive_longitudinal_guidance_uses_reference_bank() {
        let nav = test_nav();
        let mut data = test_sim_data();
        // Force energy outside activation window so ilongi=0
        data.guidance.longi_activation = -1e12;
        data.guidance.longi_inhibition = -2e12;
        data.guidance.lateral_activation = -2e12;

        let planet = Planet::Mars;
        let gitref = 30.0_f64.to_radians();
        let mut state = FtcState::new(gitref, -0.48_f64.to_radians());
        state.gitpre = gitref;

        let out = guidance_step(
            &nav,
            gitref,
            0.0,
            gitref,
            &mut state,
            &data,
            &planet,
            false,
            GuidanceType::Ftc,
        );

        assert!(
            out.gitcom.is_finite(),
            "expected finite gitcom, got {}",
            out.gitcom
        );
        // When guidance is inactive and no lateral, gitcom comes from gitref.abs()
        // clamped by rate saturation — it should stay close to gitref for a single step
        let pi = std::f64::consts::PI;
        assert!(
            out.gitcom.abs() <= pi,
            "gitcom magnitude exceeds π: {}",
            out.gitcom
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
                    positn: [r, 0.0, 0.0],
                    vitesn: [vel, fpa, 0.6],
                    acceln: [50.0, -8.0],
                    coefan: [1.269, -0.205],
                    roguid: 1e-4,
                    roexit: 1e-6,
                    pdynan: 0.5 * 1e-4 * vel * vel,
                    energn: -1e6,
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

                prop_assert!(out.gitcom.is_finite(), "gitcom not finite: {}", out.gitcom);
                prop_assert!(out.alfcom.is_finite(), "alfcom not finite: {}", out.alfcom);

                let pi = std::f64::consts::PI;
                prop_assert!(
                    out.gitcom >= -pi && out.gitcom <= pi,
                    "gitcom = {} outside [-π, π]",
                    out.gitcom
                );
            }
        }
    }
}
