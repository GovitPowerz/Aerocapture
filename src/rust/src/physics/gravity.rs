//! Gravity model with J2/J3/J4 zonal harmonic corrections.

use crate::config::PlanetConfig;

/// Compute gravitational acceleration components in spherical coordinates.
///
/// Returns (gravtl, gravtr):
///   - gravtl: lateral (latitudinal) component (m/s^2), convention: -g_lat
///   - gravtr: radial component (m/s^2), convention: -g_r (positive inward)
///
/// Supports zonal harmonics J2, J3, J4. When J3=J4=0, reduces to J2-only model.
pub fn gravity(radius: f64, latitude: f64, planet: &PlanetConfig) -> (f64, f64) {
    let mu = planet.mu;
    let req = planet.equatorial_radius;
    let j2 = planet.j2;
    let j3 = planet.j3;
    let j4 = planet.j4;

    let r2 = radius * radius;
    let r4 = r2 * r2;
    let sin_lat = latitude.sin();
    let cos_lat = latitude.cos();
    let sin2 = sin_lat * sin_lat;
    let req2 = req * req;

    // Hoist shared intermediates for J3/J4 (computed unconditionally, the multiplications are cheap)
    let r5 = r4 * radius;
    let r6 = r4 * r2;
    let req3 = req2 * req;
    let req4 = req2 * req2;
    let sin4 = sin2 * sin2;

    // ── Radial component (positive inward): gravtr = -g_r ──
    // Zonal harmonic acceleration expansion (J2-J4), Vallado,
    // Fundamentals of Astrodynamics and Applications, ch. 8.
    // Keplerian + J2
    let mut gravtr = mu / r2 + 1.5 * mu * j2 * req2 * (1.0 - 3.0 * sin2) / r4;

    // J3: 2*mu*J3*R^3 * sin*(3 - 5*sin^2) / r^5
    if j3 != 0.0 {
        gravtr += 2.0 * mu * j3 * req3 * sin_lat * (3.0 - 5.0 * sin2) / r5;
    }

    // J4: -(5/8)*mu*J4*R^4 * (3 - 30*sin^2 + 35*sin^4) / r^6
    if j4 != 0.0 {
        gravtr -= 0.625 * mu * j4 * req4 * (3.0 - 30.0 * sin2 + 35.0 * sin4) / r6;
    }

    // ── Lateral component: gravtl = -g_lat ──
    // J2: 3*mu*J2*R^2 * sin*cos / r^4
    let mut gravtl = 3.0 * mu * j2 * req2 * sin_lat * cos_lat / r4;

    // J3: (3/2)*mu*J3*R^3 * cos*(5*sin^2 - 1) / r^5
    if j3 != 0.0 {
        gravtl += 1.5 * mu * j3 * req3 * cos_lat * (5.0 * sin2 - 1.0) / r5;
    }

    // J4: -(5/2)*mu*J4*R^4 * sin*cos*(3 - 7*sin^2) / r^6
    if j4 != 0.0 {
        gravtl -= 2.5 * mu * j4 * req4 * sin_lat * cos_lat * (3.0 - 7.0 * sin2) / r6;
    }

    (gravtl, gravtr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    #[test]
    fn spherical_gravity_at_surface() {
        // Moon has tiny J2 (4.458e-6), so gravity should be nearly spherical: mu/r^2
        let planet = PlanetConfig::moon();
        let r = planet.equatorial_radius;
        let (gravtl, gravtr) = gravity(r, 0.0, &planet);
        let expected = planet.mu / (r * r);
        assert_relative_eq!(gravtr, expected, max_relative = 1e-4);
        assert_relative_eq!(gravtl, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn lateral_at_equator_from_j3() {
        // J2 lateral is zero at equator (sin=0), but J3 contributes
        // cos(0)*(5*0-1) = -1, so lateral is nonzero
        let planet = PlanetConfig::mars();
        let (gravtl, _) = gravity(planet.equatorial_radius, 0.0, &planet);
        // Should be small but nonzero (J3 contribution only)
        assert!(gravtl.abs() > 1e-12);
        assert!(gravtl.abs() < 0.01); // Still a tiny correction
    }

    #[test]
    fn lateral_zero_at_pole() {
        // cos(pi/2) ≈ 0 → gravtl ≈ 0 (all J2/J3/J4 lateral terms contain cos(lat))
        let planet = PlanetConfig::mars();
        let (gravtl, _) = gravity(planet.equatorial_radius, FRAC_PI_2, &planet);
        assert_relative_eq!(gravtl, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn j2_radial_stronger_at_pole_on_surface() {
        // Oblate planet: surface gravity is stronger at the pole because the polar surface
        // is closer to the center. Use actual surface radii (equatorial vs polar).
        let planet = PlanetConfig::earth();
        let r_eq = planet.equatorial_radius;
        let r_pole = planet.polar_radius;
        let (_, gravtr_eq) = gravity(r_eq, 0.0, &planet);
        let (_, gravtr_pole) = gravity(r_pole, FRAC_PI_2, &planet);
        assert!(
            gravtr_pole > gravtr_eq,
            "polar surface gravtr ({gravtr_pole}) should exceed equatorial surface gravtr ({gravtr_eq})"
        );
    }

    #[test]
    fn gravity_decreases_with_altitude() {
        let planet = PlanetConfig::mars();
        let r_low = planet.equatorial_radius;
        let r_high = r_low + 200_000.0; // 200 km above surface
        let (_, gravtr_low) = gravity(r_low, 0.0, &planet);
        let (_, gravtr_high) = gravity(r_high, 0.0, &planet);
        assert!(
            gravtr_low > gravtr_high,
            "surface gravtr ({gravtr_low}) should exceed 200km gravtr ({gravtr_high})"
        );
    }

    #[test]
    fn mars_surface_gravity_ballpark() {
        // Mars equatorial surface gravity ≈ 3.72 m/s²
        let planet = PlanetConfig::mars();
        let (_, gravtr) = gravity(planet.equatorial_radius, 0.0, &planet);
        assert_relative_eq!(gravtr, 3.72, max_relative = 0.05);
    }

    #[test]
    fn earth_surface_gravity_ballpark() {
        // Earth equatorial surface gravity ≈ 9.81 m/s²
        let planet = PlanetConfig::earth();
        let (_, gravtr) = gravity(planet.equatorial_radius, 0.0, &planet);
        assert_relative_eq!(gravtr, 9.81, max_relative = 0.05);
    }

    #[test]
    fn lateral_approximate_antisymmetry() {
        // With J3 active, exact antisymmetry is broken, but it's still nearly antisymmetric
        // because J3/J2 ~ 1.6% for Mars
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius;
        let lat = 0.7;
        let (gravtl_pos, _) = gravity(r, lat, &planet);
        let (gravtl_neg, _) = gravity(r, -lat, &planet);
        // The sum should be small relative to the individual values (dominated by J3)
        let asymmetry = (gravtl_pos + gravtl_neg).abs();
        let magnitude = gravtl_pos.abs().max(gravtl_neg.abs());
        assert!(
            asymmetry / magnitude < 0.05,
            "asymmetry ratio {:.4} exceeds 5%",
            asymmetry / magnitude
        );
    }

    #[test]
    fn lateral_peak_near_45_deg() {
        // sin(2*lat) peaks at lat=pi/4, so |gravtl| should be maximal there
        // (J3/J4 shift the peak slightly but not enough to change this for Mars)
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius;
        let (gravtl_45, _) = gravity(r, FRAC_PI_4, &planet);
        let (gravtl_low, _) = gravity(r, 0.5, &planet); // 0.5 rad ≈ 28.6°
        let (gravtl_high, _) = gravity(r, 1.0, &planet); // 1.0 rad ≈ 57.3°
        assert!(
            gravtl_45.abs() > gravtl_low.abs(),
            "|gravtl(pi/4)| ({}) should exceed |gravtl(0.5)| ({})",
            gravtl_45.abs(),
            gravtl_low.abs()
        );
        assert!(
            gravtl_45.abs() > gravtl_high.abs(),
            "|gravtl(pi/4)| ({}) should exceed |gravtl(1.0)| ({})",
            gravtl_45.abs(),
            gravtl_high.abs()
        );
    }

    // ── New J3/J4 tests ─────────────────────────────────────────────────────

    #[test]
    fn j3_j4_zero_matches_j2_only() {
        // A PlanetConfig with J3=J4=0 must produce bit-identical results to the
        // J2-only formula for any (r, lat) pair.
        let planet = PlanetConfig::mars_j2_only();
        let r = planet.equatorial_radius + 50_000.0;
        for lat_deg in [-60.0_f64, -30.0, 0.0, 30.0, 60.0] {
            let lat = lat_deg.to_radians();
            let (gravtl, gravtr) = gravity(r, lat, &planet);
            // Manually compute J2-only values
            let mu = planet.mu;
            let req = planet.equatorial_radius;
            let j2 = planet.j2;
            let r2 = r * r;
            let r4 = r2 * r2;
            let sin_lat = lat.sin();
            let cos_lat = lat.cos();
            let sin2 = sin_lat * sin_lat;
            let req2 = req * req;
            let expected_tl = 3.0 * mu * j2 * req2 * sin_lat * cos_lat / r4;
            let expected_tr = mu / r2 + 1.5 * mu * j2 * req2 * (1.0 - 3.0 * sin2) / r4;
            assert_eq!(gravtl, expected_tl, "gravtl mismatch at lat={lat_deg}");
            assert_eq!(gravtr, expected_tr, "gravtr mismatch at lat={lat_deg}");
        }
    }

    #[test]
    fn j3_breaks_north_south_symmetry() {
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius + 50_000.0;
        let lat = 0.5_f64; // ~28.6 degrees
        let (gravtl_pos, _) = gravity(r, lat, &planet);
        let (gravtl_neg, _) = gravity(r, -lat, &planet);
        // J3 (odd harmonic) breaks exact antisymmetry
        assert!(
            (gravtl_pos + gravtl_neg).abs() > 1e-10,
            "J3 should break north-south symmetry: sum = {}",
            gravtl_pos + gravtl_neg
        );
    }

    #[test]
    fn j3_lateral_nonzero_at_equator() {
        // J3 contributes cos(0)*(5*0-1) = -1 at equator, unlike J2 which is zero there
        let planet = PlanetConfig::mars();
        let r = planet.equatorial_radius;
        let (gravtl, _) = gravity(r, 0.0, &planet);
        assert!(
            gravtl.abs() > 1e-12,
            "J3 lateral should be nonzero at equator: gravtl = {gravtl}"
        );
    }

    #[test]
    fn j4_preserves_lateral_antisymmetry() {
        // J4 (even harmonic) preserves antisymmetry in the lateral component
        // Test with a planet that has J3=0, only J4 active
        let planet = PlanetConfig {
            j3: 0.0,
            j4: -1.538e-5,
            ..PlanetConfig::mars_j2_only()
        };
        let r = planet.equatorial_radius + 50_000.0;
        let lat = 0.7;
        let (gravtl_pos, _) = gravity(r, lat, &planet);
        let (gravtl_neg, _) = gravity(r, -lat, &planet);
        assert_relative_eq!(gravtl_pos, -gravtl_neg, max_relative = 1e-14);
    }

    #[test]
    fn j3_j4_small_correction_at_mars_surface() {
        let planet = PlanetConfig::mars();
        let planet_j2 = PlanetConfig::mars_j2_only();
        let r = planet.equatorial_radius;
        let lat = 0.5_f64;
        let (tl_full, tr_full) = gravity(r, lat, &planet);
        let (tl_j2, tr_j2) = gravity(r, lat, &planet_j2);
        let rel_tl = ((tl_full - tl_j2) / tl_j2).abs();
        let rel_tr = ((tr_full - tr_j2) / tr_j2).abs();
        assert!(
            rel_tl < 0.05,
            "J3+J4 lateral correction is {:.4}%, expected < 5%",
            rel_tl * 100.0
        );
        assert!(
            rel_tr < 0.05,
            "J3+J4 radial correction is {:.4}%, expected < 5%",
            rel_tr * 100.0
        );
    }

    // ── Proptest ─────────────────────────────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn gravity_magnitude_finite(
            alt_km in 0.0_f64..10000.0,
            lat_deg in -90.0_f64..90.0,
        ) {
            let planet = PlanetConfig::mars();
            let r = planet.equatorial_radius + alt_km * 1000.0;
            let lat = lat_deg.to_radians();
            let (gravtl, gravtr) = gravity(r, lat, &planet);
            prop_assert!(gravtl.is_finite(), "gravtl is not finite at alt={alt_km} lat={lat_deg}");
            prop_assert!(gravtr.is_finite(), "gravtr is not finite at alt={alt_km} lat={lat_deg}");
            prop_assert!(gravtr > 0.0, "gravtr should be positive (inward pull)");
        }
    }

    #[test]
    fn gravity_matches_potential_gradient() {
        // Independent oracle: the analytic (gravtl, gravtr) expansion must equal the
        // numerical gradient of the geopotential U(r,phi) = (mu/r)[1 - sum Jn (Re/r)^n Pn(sin phi)].
        // Legendre: P2=(3x^2-1)/2, P3=(5x^3-3x)/2, P4=(35x^4-30x^2+3)/8.
        // gravtr = -dU/dr ; gravtl = -(1/r) dU/dlat.
        let mut planet = PlanetConfig::mars();
        planet.j2 = 1.96e-3;
        planet.j3 = 3.1e-5;
        planet.j4 = -1.5e-5; // force all three terms non-trivial
        let re = planet.equatorial_radius;
        let mu = planet.mu;
        let (j2, j3, j4) = (planet.j2, planet.j3, planet.j4);
        let potential = |r: f64, lat: f64| -> f64 {
            let x = lat.sin();
            let p2 = (3.0 * x * x - 1.0) / 2.0;
            let p3 = (5.0 * x * x * x - 3.0 * x) / 2.0;
            let p4 = (35.0 * x.powi(4) - 30.0 * x * x + 3.0) / 8.0;
            (mu / r)
                * (1.0 - j2 * (re / r).powi(2) * p2 - j3 * (re / r).powi(3) * p3 - j4 * (re / r).powi(4) * p4)
        };
        for &(r_mult, lat) in &[(1.05_f64, 0.3_f64), (1.2, -0.7), (1.5, 1.1)] {
            let r = re * r_mult;
            let (gravtl, gravtr) = gravity(r, lat, &planet);
            let hr = r * 1e-6;
            let hl = 1e-6;
            let dudr = (potential(r + hr, lat) - potential(r - hr, lat)) / (2.0 * hr);
            let dudlat = (potential(r, lat + hl) - potential(r, lat - hl)) / (2.0 * hl);
            assert_relative_eq!(gravtr, -dudr, max_relative = 1e-5);
            assert_relative_eq!(gravtl, -dudlat / r, max_relative = 1e-5);
        }
    }
}
