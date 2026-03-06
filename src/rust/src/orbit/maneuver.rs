//! Orbit maneuver cost computation.
//!
//! Matches Fortran ergols.f.
//! Computes delta-V cost for orbit correction after aerocapture.

use crate::config::Planet;
use crate::data::{OrbitalElements, OrbitalTarget, ParkingOrbit};

/// Delta-V results for orbit correction maneuvers.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeltaV {
    /// Maneuver 1: periapsis correction at apoapsis (m/s)
    pub dv1: f64,
    /// Maneuver 2: apoapsis correction at periapsis (m/s)
    pub dv2: f64,
    /// Maneuver 3: inclination correction (m/s)
    pub dv3: f64,
    /// Total: |dv1| + |dv2| + |dv3|
    pub total: f64,
}

/// Compute delta-V cost for orbit correction.
///
/// Matches Fortran ergols.f exactly:
/// - If ifinal != 3 (not atmosphere exit), returns 1e30 for all values
/// - Maneuver 1 at apoapsis: correct periapsis to target
/// - Maneuver 2 at new periapsis: correct apoapsis (circularize)
/// - Maneuver 3: inclination plane change at ascending/descending node
pub fn compute_deltav(
    orbit: &OrbitalElements,
    ifinal: i32,
    target: &OrbitalTarget,
    parking: &ParkingOrbit,
    planet: &Planet,
) -> DeltaV {
    let mu = planet.mu();
    let req = planet.equatorial_radius();

    if ifinal != 3 {
        return DeltaV {
            dv1: 1e30,
            dv2: 1e30,
            dv3: 1e30,
            total: 1e30,
        };
    }

    let rapoge = req + orbit.apoapsis_alt;
    let rperig = req + orbit.periapsis_alt;
    let rapotf = req + parking.apoapsis;
    let rpertf = req + parking.periapsis;

    // Maneuver 1: at apoapsis, correct periapsis
    let vitfin1 = (2.0 * mu * rpertf / (rapoge * (rapoge + rpertf))).sqrt();
    let vitini1 = (2.0 * mu * rperig / (rapoge * (rapoge + rperig))).sqrt();
    let dv1 = vitfin1 - vitini1;

    // Maneuver 2: at new periapsis, correct apoapsis
    let vitfin2 = (2.0 * mu * rapotf / (rpertf * (rapotf + rpertf))).sqrt();
    let vitini2 = (2.0 * mu * rapoge / (rpertf * (rapoge + rpertf))).sqrt();
    let dv2 = vitfin2 - vitini2;

    // Maneuver 3: inclination correction at ascending/descending node
    // Uses target orbit parameters for node velocity computation
    let target_sma = target.semi_major_axis;
    let target_ecc = target.eccentricity;
    let pi = std::f64::consts::PI;

    let anoneu = [2.0 * pi - orbit.arg_periapsis, pi - orbit.arg_periapsis];
    let mut vitneu = [0.0_f64; 2];
    for i in 0..2 {
        let rayneu =
            target_sma * (1.0 - target_ecc * target_ecc) / (1.0 + target_ecc * anoneu[i].cos());
        vitneu[i] = (2.0 * mu * (1.0 / rayneu - 1.0 / (2.0 * target_sma))).sqrt();
    }
    let dincli = (target.inclination - orbit.inclination).abs();
    let dv3 = 2.0 * vitneu[0].min(vitneu[1]) * (dincli / 2.0).sin();

    let total = dv1.abs() + dv2.abs() + dv3.abs();

    DeltaV {
        dv1,
        dv2,
        dv3,
        total,
    }
}

/// Compute optimal delta-V (from target orbit to parking orbit).
///
/// Matches Fortran ergols.f "dvopti" section.
#[allow(dead_code)]
pub fn compute_deltav_optimal(
    target: &OrbitalTarget,
    parking: &ParkingOrbit,
    planet: &Planet,
) -> DeltaV {
    let mu = planet.mu();
    let req = planet.equatorial_radius();

    let rapoge = req + target.apoapsis;
    let rperig = req + target.periapsis;
    let rapotf = req + parking.apoapsis;
    let rpertf = req + parking.periapsis;

    let vitfin1 = (2.0 * mu * rpertf / (rapoge * (rapoge + rpertf))).sqrt();
    let vitini1 = (2.0 * mu * rperig / (rapoge * (rapoge + rperig))).sqrt();
    let dv1 = vitfin1 - vitini1;

    let vitfin2 = (2.0 * mu * rapotf / (rpertf * (rapotf + rpertf))).sqrt();
    let vitini2 = (2.0 * mu * rapoge / (rpertf * (rapoge + rpertf))).sqrt();
    let dv2 = vitfin2 - vitini2;

    let dv3 = 0.0_f64; // Fortran: dvopti(3) = 0
    let total = dv1.abs() + dv2.abs() + dv3.abs();

    DeltaV {
        dv1,
        dv2,
        dv3,
        total,
    }
}
