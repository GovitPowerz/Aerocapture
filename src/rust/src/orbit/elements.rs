//! Orbital element computation from state vectors.
//!
//! Matches Fortran orbito.f exactly.

use crate::config::Planet;
use crate::data::OrbitalElements;
use crate::gnc::navigation::coordinates::{cross, dot, norm, to_absolute_cartesian};

/// Compute orbital elements from spherical state.
///
/// Matches Fortran orbito.f.
/// Position: [r, lon, lat] geocentric spherical
/// Velocity: [V, gamma, psi] local spherical
pub fn from_spherical(
    radius: f64,
    longitude: f64,
    latitude: f64,
    velocity: f64,
    flight_path: f64,
    azimuth: f64,
    planet: &Planet,
) -> OrbitalElements {
    let mu = planet.mu();
    let req = planet.equatorial_radius();
    let enrmin = 1e-6; // Fortran: satorb common, small threshold

    // Get absolute position and velocity in Cartesian
    let (posita, vitesa) = to_absolute_cartesian(
        radius,
        longitude,
        latitude,
        velocity,
        flight_path,
        azimuth,
        planet,
    );

    // Angular momentum: L = r × v
    let xmocin = cross(&posita, &vitesa);

    let rayvec = norm(&posita);
    let vitabs = norm(&vitesa);
    let xcinet = norm(&xmocin);

    // Total energy (thresholded to avoid parabolic singularity)
    let enrorb_raw = vitabs * vitabs / 2.0 - mu / rayvec;
    let sigenr = enrorb_raw.signum();
    let enrorb = sigenr * enrorb_raw.abs().max(enrmin);

    // Semi-major axis
    let demiax = -mu / (2.0 * enrorb);

    // Eccentricity
    let parexc = xcinet * xcinet / (mu * demiax);
    let excent = if (parexc - 1.0).abs() < 1e-20 {
        0.0
    } else {
        (1.0 - parexc).abs().sqrt()
    };

    // Inclination
    let cosinc = xmocin[2] / xcinet;
    let sininc = (1.0 - cosinc * cosinc).max(0.0).sqrt();
    let xincli = sininc.atan2(cosinc);

    // RAAN (longitude of ascending node)
    let gomega = if sininc.abs() > 1e-10 {
        let sinomg = xmocin[0] / (xcinet * sininc);
        let cosomg = -xmocin[1] / (xcinet * sininc);
        sinomg.atan2(cosomg)
    } else {
        0.0
    };

    // True anomaly
    let posvit = dot(&posita, &vitesa);
    let v0 = if enrorb < 0.0 {
        // Elliptical orbit
        let sv0 =
            posvit * (1.0 - excent * excent).max(0.0).sqrt() / (excent * (mu * demiax).sqrt());
        let cv0 = (1.0 - rayvec / demiax) / excent - excent;
        sv0.atan2(cv0)
    } else {
        // Hyperbolic orbit
        let sv0 = posvit * (excent * excent - 1.0).max(0.0).sqrt()
            / (excent * (mu * demiax.abs()).sqrt());
        let cv0 = -((1.0 + rayvec / demiax.abs()) / excent - excent);
        sv0.atan2(cv0)
    };

    // Argument of periapsis
    let sinomg = gomega.sin();
    let cosomg = gomega.cos();
    let pomega = if xincli > 1e-3 {
        let arg1 = posita[2] / (xincli.sin() * rayvec);
        let arg2 = (posita[0] * cosomg + posita[1] * sinomg) / rayvec;
        let mut w = arg1.atan2(arg2) - v0;
        if w < 0.0 {
            w += 2.0 * std::f64::consts::PI;
        }
        w
    } else {
        let mut w = posita[1].atan2(posita[0]) - v0;
        if w < 0.0 {
            w += 2.0 * std::f64::consts::PI;
        }
        w
    };

    // Periapsis and apoapsis radii → altitudes
    let rayper = demiax * (1.0 - excent);
    let rayapo = demiax * (1.0 + excent);

    OrbitalElements {
        semi_major_axis: demiax,
        eccentricity: excent,
        inclination: xincli,
        raan: gomega,
        arg_periapsis: pomega,
        true_anomaly: v0,
        periapsis_alt: rayper - req,
        apoapsis_alt: rayapo - req,
    }
}
