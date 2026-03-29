//! Orbit maneuver cost computation.
//!
//! Computes delta-V cost for orbit correction after aerocapture.

use crate::config::PlanetConfig;
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

/// Compute delta-V cost for orbit correction (confirmed captures only).
///
/// Only called when the trajectory has exited the atmosphere into a bound orbit.
/// The caller is responsible for routing non-capture cases (hyperbolic exit, crash,
/// pending crash, timeout) to virtual DV computation instead.
///
/// - Maneuver 1 at apoapsis: correct periapsis to target
/// - Maneuver 2 at new periapsis: correct apoapsis (circularize)
/// - Maneuver 3: inclination plane change at ascending/descending node
pub fn compute_deltav(
    orbit: &OrbitalElements,
    target: &OrbitalTarget,
    parking: &ParkingOrbit,
    planet: &PlanetConfig,
) -> DeltaV {
    let mu = planet.mu;
    let req = planet.equatorial_radius;

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
#[allow(dead_code)]
pub fn compute_deltav_optimal(
    target: &OrbitalTarget,
    parking: &ParkingOrbit,
    planet: &PlanetConfig,
) -> DeltaV {
    let mu = planet.mu;
    let req = planet.equatorial_radius;

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

    let dv3 = 0.0_f64; // inclination correction not computed for optimal case
    let total = dv1.abs() + dv2.abs() + dv3.abs();

    DeltaV {
        dv1,
        dv2,
        dv3,
        total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a realistic post-aerocapture orbit at Mars.
    fn mars_test_fixtures() -> (OrbitalElements, OrbitalTarget, ParkingOrbit, PlanetConfig) {
        let orbit = OrbitalElements {
            semi_major_axis: 4.0e6,
            eccentricity: 0.3,
            inclination: 0.45, // ~25.8 deg
            raan: 1.0,
            arg_periapsis: 0.5,
            true_anomaly: 0.0,
            periapsis_alt: 100_000.0, // 100 km
            apoapsis_alt: 500_000.0,  // 500 km
        };
        let target = OrbitalTarget {
            apoapsis: 500_000.0,
            periapsis: 250_000.0,
            semi_major_axis: 3.77e6,
            eccentricity: 0.03,
            inclination: 0.45, // same as orbit for some tests; overridden where needed
            raan: 1.0,
        };
        let parking = ParkingOrbit {
            apoapsis: 500_000.0,
            periapsis: 250_000.0,
        };
        (orbit, target, parking, PlanetConfig::mars())
    }

    #[test]
    fn exit_returns_finite_cost() {
        let (orbit, target, parking, planet) = mars_test_fixtures();
        let dv = compute_deltav(&orbit, &target, &parking, &planet);
        assert!(dv.total.is_finite(), "total should be finite");
        assert!(dv.total > 0.0, "total should be positive");
        assert!(
            dv.total < 5000.0,
            "total should be < 5000 m/s for reasonable orbit"
        );
    }

    #[test]
    fn total_is_sum_of_abs() {
        let (orbit, target, parking, planet) = mars_test_fixtures();
        let dv = compute_deltav(&orbit, &target, &parking, &planet);
        let expected = dv.dv1.abs() + dv.dv2.abs() + dv.dv3.abs();
        assert!(
            (dv.total - expected).abs() < 1e-10,
            "total ({}) should equal |dv1|+|dv2|+|dv3| ({})",
            dv.total,
            expected
        );
    }

    #[test]
    fn optimal_has_zero_dv3() {
        let (_, target, parking, planet) = mars_test_fixtures();
        let dv = compute_deltav_optimal(&target, &parking, &planet);
        assert_eq!(dv.dv3, 0.0, "optimal dv3 should be exactly zero");
        assert!(dv.total.is_finite(), "optimal total should be finite");
        let expected = dv.dv1.abs() + dv.dv2.abs();
        assert!(
            (dv.total - expected).abs() < 1e-10,
            "optimal total should equal |dv1|+|dv2|"
        );
    }

    #[test]
    fn zero_inclination_error_small_dv3() {
        let (mut orbit, target, parking, planet) = mars_test_fixtures();
        // Set orbit inclination exactly equal to target inclination
        orbit.inclination = target.inclination;
        let dv = compute_deltav(&orbit, &target, &parking, &planet);
        assert!(
            dv.dv3.abs() < 1e-6,
            "dv3 ({}) should be near-zero when inclinations match",
            dv.dv3
        );

        // Also test with a tiny offset — should still be very small
        orbit.inclination = target.inclination + 1e-4; // ~0.006 deg
        let dv2 = compute_deltav(&orbit, &target, &parking, &planet);
        assert!(
            dv2.dv3.abs() < 1.0,
            "dv3 ({}) should be < 1 m/s for ~0.006 deg inclination error",
            dv2.dv3
        );
    }
}
