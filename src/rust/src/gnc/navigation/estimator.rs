//! Navigation state estimator.
//!
//! Matches Fortran naviga.f.
//! Adds navigation errors to the true state to produce measured state,
//! estimates atmospheric density, and manages guidance phase transitions.

use crate::config::Planet;
use crate::data::{SimData, OrbitalElements, OrbitalTarget};
use crate::gnc::navigation::coordinates::{geodetic_from_spherical, total_energy};
use crate::orbit::elements;

/// Navigation error biases (constant during a run).
///
/// Matches Fortran common /pernav/ dispos(3), disvit(3), disdra.
#[derive(Debug, Clone, Copy, Default)]
pub struct NavigationBiases {
    pub pos: [f64; 3],  // [altitude, longitude, latitude] bias
    pub vel: [f64; 3],  // [velocity, flight_path, azimuth] bias
    pub drag: f64,       // drag acceleration measurement bias
}

/// Navigation filter state (persistent across steps).
#[derive(Debug, Clone, Copy)]
pub struct NavigationState {
    pub coefro: f64,     // density estimation coefficient
    pub vitpre: f64,     // previous radial velocity (m/s)
    pub ibounc: i32,     // bounce indicator: 0=before, 1=after
    pub iphase: i32,     // guidance phase: 1=capture, 2=exit, 3=emergency
    pub tcaptr: f64,     // capture phase duration (s)
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
    pub positn: [f64; 3],  // [r, lon, lat]
    pub vitesn: [f64; 3],  // [V, gamma, psi]
    // Estimated aerodynamic quantities
    pub acceln: [f64; 2],  // [drag accel, lift accel]
    pub coefan: [f64; 2],  // [Cx, Cz]
    pub roguid: f64,       // estimated density for guidance
    pub roexit: f64,       // estimated exit density
    pub pdynan: f64,       // estimated dynamic pressure
    pub energn: f64,       // total energy
    // Orbital parameter errors
    pub ecartn: [f64; 4],  // [Δa, Δe, Δi, ΔΩ]
    // Phase management
    pub ibounc: i32,
    pub iphase: i32,
    pub icrash: i32,
    pub indext: i32,       // phase transition flag
    pub vitref: f64,       // reference radial velocity
    pub tcaptr: f64,       // capture duration
}

/// Run one navigation step.
///
/// Matches Fortran naviga.f.
pub fn navigate(
    positr: &[f64; 3],  // true position [r, lon, lat]
    vitesr: &[f64; 3],  // true velocity [V, gamma, psi]
    alfcom: f64,         // commanded AoA
    temsim: f64,         // current time
    biases: &NavigationBiases,
    nav_state: &mut NavigationState,
    data: &SimData,
    planet: &Planet,
    run_density_bias: f64,
    run_cx_bias: f64,
    run_cz_bias: f64,
    run_mass_bias: f64,
) -> NavigationOutput {
    let mut out = NavigationOutput::default();
    out.indext = 0;
    out.icrash = 0;

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
    let (alt_true, _) = geodetic_from_spherical(
        positr[0], positr[1], positr[2], planet,
    );
    let rho_true = data.atmosphere.density_at(alt_true);
    let rho_true = rho_true * (1.0 + run_density_bias);
    let cx_true = data.aero.interpolate_cx(alfcom) * (1.0 + run_cx_bias);
    let mass_true = data.capsule.mass * (1.0 + run_mass_bias);
    let acdrag_true = rho_true * data.capsule.reference_area * cx_true
        * vitesr[0] * vitesr[0] / (2.0 * mass_true);
    let acdram = acdrag_true + biases.drag;

    // Compute estimated aero coefficients (imodel=1)
    // Matches conphy with estimated state
    let (alt_est, _) = geodetic_from_spherical(
        out.positn[0], out.positn[1], out.positn[2], planet,
    );
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
    let lambda = data.guidance.density_filter_gain;
    if rho_model.abs() > 1e-30 {
        nav_state.coefro = (1.0 - lambda) * nav_state.coefro
            + lambda * (roesti / rho_model);
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
        out.positn[0], out.positn[1], out.positn[2],
        out.vitesn[0], out.vitesn[1], out.vitesn[2],
        planet,
    );

    // Orbital elements
    let orbit = elements::from_spherical(
        out.positn[0], out.positn[1], out.positn[2],
        out.vitesn[0], out.vitesn[1], out.vitesn[2],
        planet,
    );
    out.ecartn[0] = orbit.semi_major_axis - data.target_orbit.semi_major_axis;
    out.ecartn[1] = orbit.eccentricity - data.target_orbit.eccentricity;
    out.ecartn[2] = orbit.inclination - data.target_orbit.inclination;
    out.ecartn[3] = orbit.raan - data.target_orbit.raan;

    // Bounce detection
    if nav_state.ibounc == 0 {
        if out.vitesn[1].sin() > 0.0 {
            nav_state.ibounc = 1;
        }
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
