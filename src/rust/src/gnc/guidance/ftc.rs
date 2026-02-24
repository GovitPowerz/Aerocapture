//! FTC (Full Trajectory Control) predictor-corrector guidance.
//!
//! Matches Fortran guidag.f, guicap.f, guilon.f, guilat.f, vigite.f, guialf.f.

use crate::config::{MissionType, Planet};
use crate::data::SimData;
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::gnc::navigation::estimator::NavigationOutput;
use crate::orbit::elements;

/// FTC guidance persistent state.
#[derive(Debug, Clone)]
pub struct FtcState {
    // Bank angle command
    pub gitcom: f64,     // current commanded bank angle (rad)
    pub gitpre: f64,     // previous commanded bank angle (rad)
    pub gpilpr: f64,     // previous pilot bank angle (rad)
    pub alfcom: f64,     // commanded AoA (rad)

    // Roll sign and reversal tracking
    pub sgngit: f64,     // roll polarity sign (-1, 0, +1)
    pub somgit: f64,     // cumulative bank angle changes (rad)
    pub nbroll: i32,     // number of roll reversals
    pub indrvr: i32,     // roll reversal active flag
    pub rolway: i32,     // roll reversal path (+1=short, -1=long)
    pub trevrs: f64,     // roll reversal duration (s)

    // Guidance securization
    pub iprepr: [i32; 2], // securization counters
    pub iguida: [i32; 2], // securization indicators

    // Reference velocity
    pub vitref: f64,

    // Counters
    pub n_secur: i32,    // number of securization events
    pub n_active: i32,   // number of active guidance calls
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
        }
    }
}

/// FTC guidance output.
#[derive(Debug, Clone, Copy, Default)]
pub struct FtcOutput {
    pub gitcom: f64,     // commanded bank angle (rad)
    pub alfcom: f64,     // commanded AoA (rad)
    pub vitgit: f64,     // bank rate before saturation (rad/s)
    pub ilongi: i32,     // longitudinal guidance active
    pub isatur: i32,     // rate saturation occurred
    pub indrol: i32,     // roll reversal indicator
}

/// Run one FTC guidance step.
///
/// Matches Fortran guidag.f.
pub fn guidance_step(
    nav: &NavigationOutput,
    gitpil: f64,         // pilot-realized bank angle
    temsim: f64,
    gitref: f64,         // reference bank angle (from config, rad)
    state: &mut FtcState,
    data: &SimData,
    planet: &Planet,
    mission_type: MissionType,
    is_reference: bool,
) -> FtcOutput {
    let pi = std::f64::consts::PI;
    let mut out = FtcOutput::default();

    let sgnpre = state.sgngit;
    state.gpilpr = gitpil;

    // === Angle of attack guidance (guialf) ===
    // proalf returns altitude as scheduling parameter
    let (altitude, _) = geodetic_from_spherical(
        nav.positn[0], nav.positn[1], nav.positn[2], planet,
    );
    state.alfcom = data.incidence.incidence_at(altitude);
    out.alfcom = state.alfcom;

    // === Longitudinal guidance activation ===
    let enrjlt = total_energy(
        nav.positn[0], nav.positn[1], nav.positn[2],
        nav.vitesn[0], nav.vitesn[1], nav.vitesn[2],
        planet,
    );

    let mut ilongi: i32;
    if mission_type == MissionType::Aerocapture {
        // Aerocapture: energy-based activation
        if enrjlt <= data.guidance.longi_activation && enrjlt >= data.guidance.longi_inhibition {
            ilongi = 1;
        } else {
            ilongi = 0;
            state.iprepr[1] += 1;
        }
    } else {
        // Aero-gravity assist: pressure-based activation
        if nav.ibounc == 0 {
            ilongi = if nav.pdynan < data.guidance.longi_activation { 0 } else { 1 };
        } else {
            ilongi = if nav.pdynan < data.guidance.longi_inhibition { 0 } else { 1 };
        }
        if ilongi == 0 {
            state.iprepr[1] += 1;
        }
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
    let mut gitlon: f64;

    if is_reference {
        state.gitcom = gitref;
        gitlon = gitref;
    } else if ilongi == 0 {
        gitlon = gitref.abs();
    } else {
        // Call guicap (capture phase guidance)
        gitlon = guicap(
            nav, enrjlt, altitude, state, data, planet,
        );
        state.n_active += 1;
    }

    // === Lateral guidance activation ===
    let mut ilater: i32;
    if mission_type == MissionType::Aerocapture {
        if enrjlt <= data.guidance.lateral_activation && enrjlt >= data.guidance.lateral_inhibition {
            ilater = 1;
        } else {
            ilater = 0;
        }
    } else {
        // AGA lateral logic
        if data.guidance.lateral_activation < 0.0 {
            if nav.ibounc == 1 {
                if nav.pdynan >= data.guidance.lateral_inhibition
                    && nav.pdynan <= -data.guidance.lateral_activation
                {
                    ilater = 1;
                } else {
                    ilater = 0;
                }
            } else {
                ilater = 0;
            }
        } else {
            if nav.ibounc == 0 {
                ilater = if nav.pdynan < data.guidance.lateral_activation { 0 } else { 1 };
            } else {
                ilater = if nav.pdynan < data.guidance.lateral_inhibition { 0 } else { 1 };
            }
        }
    }

    ilater *= state.iguida[1];

    // === Lateral guidance ===
    let mut indrol = 0;
    if ilater == 1 {
        guilat(
            nav, gitlon, temsim, state, data, planet,
            &mut indrol,
        );
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

    // === Roll rate saturation (vigite) ===
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

/// Capture phase longitudinal guidance.
///
/// Matches Fortran guicap.f (flag=0 path — tbgain-based predictor-corrector).
fn guicap(
    nav: &NavigationOutput,
    enrjlt: f64,
    altitude: f64,
    state: &mut FtcState,
    data: &SimData,
    planet: &Planet,
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
    let (gaindh, gainpd) = tbgain(
        altitude, &nav.coefan, data,
    );

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
///
/// Matches Fortran tbgain.f.
fn tbgain(
    altitude: f64,
    coefan: &[f64; 2],
    data: &SimData,
) -> (f64, f64) {
    let pdyn_table = &data.guidance.pdyn_table;
    let alt_km = altitude / 1e3;

    // Find altitude bracket — matches Fortran tbgain.f
    // Fortran uses 1-based indexing with inumer=0 as "not found" sentinel.
    // We use Option<usize> to avoid off-by-one with 0-based indexing.
    let mut found: Option<usize> = None;
    for i in 0..pdyn_table.len().saturating_sub(1) {
        if alt_km >= pdyn_table[i].altitude && alt_km < pdyn_table[i + 1].altitude && found.is_none() {
            found = Some(i);
        }
    }
    // Fortran: if inumer==0 then inumer=nzapd (last entry)
    let inumer = found.unwrap_or_else(|| {
        if pdyn_table.is_empty() { 0 } else { pdyn_table.len() - 1 }
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
///
/// Matches Fortran guilat.f.
fn guilat(
    nav: &NavigationOutput,
    gitlon: f64,
    temsim: f64,
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
        nav.positn[0], nav.positn[1], nav.positn[2],
        nav.vitesn[0], nav.vitesn[1], nav.vitesn[2],
        planet,
    );

    let mut xinccr = data.target_orbit.inclination - orbit.inclination;
    // Hemisphere correction (commented out in Fortran: if positn(3) < 0 then xinccr = -xinccr)

    let vitrel = nav.vitesn[0];

    // Corridor boundary: xinmax = (v/coridx)^4 + coridy
    let coridx = data.guidance.corridor_slope;
    let coridy = data.guidance.corridor_intercept;
    let xinmax = (vitrel / coridx).powi(4) + coridy;

    // Reversal decision
    if xinccr.abs() >= xinmax && gitlon.abs() > 1e-10 {
        if state.nbroll < data.guidance.max_reversals {
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
                    state.indrvr = 0; // Fortran: immediately reset (line 157)
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
}
