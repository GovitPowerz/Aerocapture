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

/// Speed at `r_apsis` on a Hohmann-like transfer ellipse whose two apsides are
/// `r_apsis` and `r_other`.  Vis-viva evaluated at `r_apsis`:
///   v = sqrt(2*mu*r_other / (r_apsis*(r_apsis+r_other)))
/// Same arithmetic / same operand order as every original inline site.
#[inline]
fn hohmann_leg_speed(mu: f64, r_apsis: f64, r_other: f64) -> f64 {
    (2.0 * mu * r_other / (r_apsis * (r_apsis + r_other))).sqrt()
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
    let vitfin1 = hohmann_leg_speed(mu, rapoge, rpertf);
    let vitini1 = hohmann_leg_speed(mu, rapoge, rperig);
    let dv1 = vitfin1 - vitini1;

    // Maneuver 2: at new periapsis, correct apoapsis
    let vitfin2 = hohmann_leg_speed(mu, rpertf, rapotf);
    let vitini2 = hohmann_leg_speed(mu, rpertf, rapoge);
    let dv2 = vitfin2 - vitini2;

    // Maneuver 3: inclination correction at ascending/descending node
    // Uses target orbit parameters for node velocity computation.
    let target_sma = target.semi_major_axis;
    let target_ecc = target.eccentricity;
    let pi = std::f64::consts::PI;

    let anoneu = [2.0 * pi - orbit.arg_periapsis, pi - orbit.arg_periapsis];
    let mut vitneu = [0.0_f64; 2];
    for i in 0..2 {
        let rayneu =
            target_sma * (1.0 - target_ecc * target_ecc) / (1.0 + target_ecc * anoneu[i].cos());
        vitneu[i] = if rayneu > 0.0 {
            (2.0 * mu * (1.0 / rayneu - 1.0 / (2.0 * target_sma)))
                .max(0.0)
                .sqrt()
        } else {
            0.0
        };
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

/// NN-input correction-DV: signed components, defined + smooth across e=1.
/// Distinct from `compute_deltav` (the terminal maneuver plan).
/// - dv1: energy-closing burn at current periapsis (vis-viva) -> "Δv to close the orbit".
/// - dv2: periapsis-correction at apoapsis; 0 when hyperbolic (continuous limit).
/// - dv3: inclination plane change (same as compute_deltav).
pub fn predicted_dv_for_nn(
    orbit: &OrbitalElements,
    target: &OrbitalTarget,
    parking: &ParkingOrbit,
    planet: &PlanetConfig,
) -> [f64; 3] {
    let mu = planet.mu;
    let req = planet.equatorial_radius;
    let a = orbit.semi_major_axis;
    let rp = req + orbit.periapsis_alt;
    let ra_t = req + parking.apoapsis;
    let rp_t = req + parking.periapsis;

    // Guards below also absorb non-finite rp/apoapsis (NaN comparisons are false in IEEE-754),
    // so a degenerate/uninitialized orbit yields 0.0 rather than NaN -- keeps the contract "always-defined".
    let dv1 = if rp > 0.0 && a.abs() > 0.0 {
        let v_cur = (mu * (2.0 / rp - 1.0 / a)).max(0.0).sqrt();
        let a_t1 = (rp + ra_t) / 2.0;
        let v_tgt = (mu * (2.0 / rp - 1.0 / a_t1)).max(0.0).sqrt();
        v_cur - v_tgt
    } else {
        0.0
    };

    let rapoge = req + orbit.apoapsis_alt;
    let dv2 = if orbit.eccentricity < 1.0 && rapoge.is_finite() && rapoge > 0.0 {
        let vitfin1 = hohmann_leg_speed(mu, rapoge, rp_t);
        let vitini1 = hohmann_leg_speed(mu, rapoge, rp);
        vitfin1 - vitini1
    } else {
        0.0
    };

    let target_sma = target.semi_major_axis;
    let target_ecc = target.eccentricity;
    let pi = std::f64::consts::PI;
    let anoneu = [2.0 * pi - orbit.arg_periapsis, pi - orbit.arg_periapsis];
    let mut vitneu = [0.0_f64; 2];
    for i in 0..2 {
        let rayneu =
            target_sma * (1.0 - target_ecc * target_ecc) / (1.0 + target_ecc * anoneu[i].cos());
        // Guard + clamp on the vis-viva term (negative under extreme eccentricity).
        vitneu[i] = if rayneu > 0.0 {
            (2.0 * mu * (1.0 / rayneu - 1.0 / (2.0 * target_sma)))
                .max(0.0)
                .sqrt()
        } else {
            0.0
        };
    }
    let dincli = (target.inclination - orbit.inclination).abs();
    let dv3 = 2.0 * vitneu[0].min(vitneu[1]) * (dincli / 2.0).sin();

    [dv1, dv2, dv3]
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

        // Also test with a tiny offset -- should still be very small
        orbit.inclination = target.inclination + 1e-4; // ~0.006 deg
        let dv2 = compute_deltav(&orbit, &target, &parking, &planet);
        assert!(
            dv2.dv3.abs() < 1.0,
            "dv3 ({}) should be < 1 m/s for ~0.006 deg inclination error",
            dv2.dv3
        );
    }

    // --- predicted_dv_for_nn: smooth, always-defined NN-input correction DV ---

    fn mk_orbit(sma: f64, ecc: f64, incl: f64, planet: &PlanetConfig) -> OrbitalElements {
        let a = sma;
        let rp = a * (1.0 - ecc);
        let ra = a * (1.0 + ecc);
        OrbitalElements {
            semi_major_axis: a,
            eccentricity: ecc,
            inclination: incl,
            periapsis_alt: rp - planet.equatorial_radius,
            apoapsis_alt: ra - planet.equatorial_radius,
            arg_periapsis: 0.0,
            ..Default::default()
        }
    }

    /// Build an orbit from a FIXED periapsis radius `rp` and eccentricity `ecc`.
    /// Unlike `mk_orbit` (which derives rp from sma and so hits `inf * 0 = NaN`
    /// exactly at e=1 since `a = rp/(1-e) -> inf`), this keeps rp finite and
    /// well-defined across the parabolic boundary -- the right fixture for the
    /// continuity sweep.
    fn mk_orbit_from_rp(rp: f64, ecc: f64, incl: f64, planet: &PlanetConfig) -> OrbitalElements {
        let a = rp / (1.0 - ecc); // a>0 for e<1, a<0 for e>1, +-inf at e=1
        let ra = a * (1.0 + ecc); // negative (finite) for hyperbolic, matching elements.rs
        OrbitalElements {
            semi_major_axis: a,
            eccentricity: ecc,
            inclination: incl,
            periapsis_alt: rp - planet.equatorial_radius,
            apoapsis_alt: ra - planet.equatorial_radius,
            arg_periapsis: 0.0,
            ..Default::default()
        }
    }
    fn parking() -> ParkingOrbit {
        ParkingOrbit {
            apoapsis: 500_000.0,
            periapsis: 300_000.0,
        }
    }
    fn target() -> OrbitalTarget {
        OrbitalTarget {
            semi_major_axis: 3.796e6 + 400_000.0,
            eccentricity: 0.05,
            inclination: 0.9,
            ..Default::default()
        }
    }

    #[test]
    fn predicted_dv_finite_for_elliptical_and_hyperbolic() {
        let p = PlanetConfig::mars();
        for ecc in [0.2_f64, 0.8, 1.2, 2.0] {
            let sma = if ecc < 1.0 { 5.0e6 } else { -5.0e6 };
            let o = mk_orbit(sma, ecc, 0.8, &p);
            let dv = predicted_dv_for_nn(&o, &target(), &parking(), &p);
            assert!(
                dv[0].is_finite() && dv[1].is_finite() && dv[2].is_finite(),
                "ecc={ecc} -> {dv:?}"
            );
        }
    }
    #[test]
    fn predicted_dv2_is_zero_when_hyperbolic() {
        let p = PlanetConfig::mars();
        let o = mk_orbit(-5.0e6, 1.5, 0.8, &p);
        let dv = predicted_dv_for_nn(&o, &target(), &parking(), &p);
        assert_eq!(dv[1], 0.0, "dv2 must be 0 for hyperbolic, got {}", dv[1]);
    }
    #[test]
    fn predicted_dv1_continuous_across_e1() {
        let p = PlanetConfig::mars();
        let rp = 3.796e6 + 50_000.0;
        let mut prev: Option<f64> = None;
        for k in 0..=40 {
            let e = 0.98 + 0.001 * k as f64;
            // Fixed periapsis across the sweep: a = rp/(1-e) -> +-inf at e=1, but
            // rp stays finite so the orbit is well-defined through the parabolic
            // boundary (mk_orbit would produce NaN periapsis_alt here).
            let o = mk_orbit_from_rp(rp, e, 0.8, &p);
            let dv1 = predicted_dv_for_nn(&o, &target(), &parking(), &p)[0];
            assert!(dv1.is_finite(), "e={e} dv1 not finite");
            if let Some(pv) = prev {
                assert!((dv1 - pv).abs() < 50.0, "dv1 jump at e={e}: {pv} -> {dv1}");
            }
            prev = Some(dv1);
        }
    }
    #[test]
    fn predicted_dv3_finite_for_pathological_target() {
        let p = PlanetConfig::mars();
        let o = mk_orbit(5.0e6, 0.3, 0.8, &p);
        let mut t = target();
        t.eccentricity = 1.2; // hyperbolic target -> rayneu can go <= 0
        let dv = predicted_dv_for_nn(&o, &t, &parking(), &p);
        assert!(
            dv[2].is_finite(),
            "dv3 must stay finite for pathological target, got {}",
            dv[2]
        );
    }

    /// Pathological target: eccentricity > 1 makes rayneu <= 0 for some node angles,
    /// which previously produced NaN via sqrt of negative.
    /// The guard must clamp to 0.0 so dv3 (and total) is finite.
    #[test]
    fn compute_deltav_inclination_finite_on_pathological_target() {
        let p = PlanetConfig::mars();
        // Use arg_periapsis = 0.0 so anoneu[0] = 2pi, anoneu[1] = pi.
        // With target_ecc = 1.2 (hyperbolic target), the denominator
        // 1 + 1.2*cos(2pi) = 2.2 -> rayneu > 0, but
        // 1 + 1.2*cos(pi)  = 1 - 1.2 = -0.2 -> rayneu < 0.
        // Previously: vitneu[1] = sqrt(negative) = NaN -> dv3 = NaN.
        let orbit = OrbitalElements {
            semi_major_axis: 4.0e6,
            eccentricity: 0.3,
            inclination: 0.3,
            raan: 0.0,
            arg_periapsis: 0.0,
            true_anomaly: 0.0,
            periapsis_alt: 100_000.0,
            apoapsis_alt: 500_000.0,
        };
        let target = OrbitalTarget {
            apoapsis: 500_000.0,
            periapsis: 250_000.0,
            semi_major_axis: 3.77e6,
            eccentricity: 1.2, // hyperbolic -> rayneu <= 0 for node angle pi
            inclination: 0.8,  // non-zero dincli so dv3 would normally be non-trivial
            raan: 0.0,
        };
        let parking = ParkingOrbit {
            apoapsis: 500_000.0,
            periapsis: 250_000.0,
        };
        let dv = compute_deltav(&orbit, &target, &parking, &p);
        assert!(
            dv.dv3.is_finite(),
            "dv3 must be finite for pathological target (rayneu <= 0), got {}",
            dv.dv3
        );
        assert!(
            dv.total.is_finite(),
            "total must be finite for pathological target, got {}",
            dv.total
        );
        assert!(
            dv.dv3 >= 0.0,
            "dv3 must be non-negative (it is 2*v*sin(half-angle)), got {}",
            dv.dv3
        );
    }

    /// Exact characterization test: pins current f64 outputs of both functions
    /// across three representative inputs. All assertions use `assert_eq!` (bit-exact).
    /// Values were captured by running the unmodified code with --nocapture.
    /// This test MUST pass before and after the helper extraction refactor.
    #[test]
    fn characterization_exact_outputs() {
        let p = PlanetConfig::mars();

        // -- Input A: elliptical orbit, inclination matches target (dv3=0) --
        let orbit_a = OrbitalElements {
            semi_major_axis: 4.0e6,
            eccentricity: 0.3,
            inclination: 0.45,
            raan: 1.0,
            arg_periapsis: 0.5,
            true_anomaly: 0.0,
            periapsis_alt: 100_000.0,
            apoapsis_alt: 500_000.0,
        };
        let target_a = OrbitalTarget {
            apoapsis: 500_000.0,
            periapsis: 250_000.0,
            semi_major_axis: 3.77e6,
            eccentricity: 0.03,
            inclination: 0.45,
            raan: 1.0,
        };
        let parking_a = ParkingOrbit {
            apoapsis: 500_000.0,
            periapsis: 250_000.0,
        };

        let dv_a = compute_deltav(&orbit_a, &target_a, &parking_a, &p);
        assert_eq!(dv_a.dv1, 35.56978761992332_f64);
        assert_eq!(dv_a.dv2, 0.0_f64);
        assert_eq!(dv_a.dv3, 0.0_f64);
        assert_eq!(dv_a.total, 35.56978761992332_f64);

        let nn_a = predicted_dv_for_nn(&orbit_a, &target_a, &parking_a, &p);
        assert_eq!(nn_a[0], 121.34751952410579_f64);
        assert_eq!(nn_a[1], 35.56978761992332_f64);
        assert_eq!(nn_a[2], 0.0_f64);

        // -- Input B: near-circular bound orbit, inclination error present --
        let orbit_b = OrbitalElements {
            semi_major_axis: 3.77e6,
            eccentricity: 0.03,
            inclination: 0.43,
            raan: 0.5,
            arg_periapsis: 1.2,
            true_anomaly: 0.0,
            periapsis_alt: (3.77e6 * (1.0 - 0.03)) - p.equatorial_radius,
            apoapsis_alt: (3.77e6 * (1.0 + 0.03)) - p.equatorial_radius,
        };
        let target_b = OrbitalTarget {
            apoapsis: 500_000.0,
            periapsis: 250_000.0,
            semi_major_axis: 3.77e6,
            eccentricity: 0.03,
            inclination: 0.45,
            raan: 0.5,
        };
        let parking_b = ParkingOrbit {
            apoapsis: 500_000.0,
            periapsis: 250_000.0,
        };

        let dv_b = compute_deltav(&orbit_b, &target_b, &parking_b, &p);
        assert_eq!(dv_b.dv1, -2.991416330771699_f64);
        assert_eq!(dv_b.dv2, 2.3489198373895306_f64);
        assert_eq!(dv_b.dv3, 66.73281990163794_f64);
        assert_eq!(dv_b.total, 72.07315606979917_f64);

        let nn_b = predicted_dv_for_nn(&orbit_b, &target_b, &parking_b, &p);
        assert_eq!(nn_b[0], -2.3470332994697856_f64);
        assert_eq!(nn_b[1], -2.991416330771699_f64);
        assert_eq!(nn_b[2], 66.73281990163794_f64);

        // -- Input C: hyperbolic orbit (e=1.5); predicted_dv_for_nn must return dv2=0 --
        let orbit_c = OrbitalElements {
            semi_major_axis: -5.0e6,
            eccentricity: 1.5,
            inclination: 0.8,
            raan: 0.0,
            arg_periapsis: 0.0,
            true_anomaly: 0.0,
            periapsis_alt: (-5.0e6 * (1.0 - 1.5)) - p.equatorial_radius,
            apoapsis_alt: (-5.0e6 * (1.0 + 1.5)) - p.equatorial_radius,
        };
        let target_c = OrbitalTarget {
            apoapsis: 500_000.0,
            periapsis: 300_000.0,
            semi_major_axis: 3.796e6 + 400_000.0,
            eccentricity: 0.05,
            inclination: 0.9,
            raan: 0.0,
        };
        let parking_c = ParkingOrbit {
            apoapsis: 500_000.0,
            periapsis: 300_000.0,
        };

        let nn_c = predicted_dv_for_nn(&orbit_c, &target_c, &parking_c, &p);
        assert_eq!(nn_c[0], 1976.3893027170516_f64);
        assert_eq!(nn_c[1], 0.0_f64);
        assert_eq!(nn_c[2], 303.76225600253156_f64);
    }
}
