//! Gravity model with J2 oblate planet correction.

use crate::config::Planet;

/// Compute gravitational acceleration components in spherical coordinates.
///
/// Returns (gravtl, gravtr):
///   - gravtl: lateral (latitudinal) component from J2 (m/s^2)
///   - gravtr: radial component, positive outward (m/s^2)
pub fn gravity(radius: f64, latitude: f64, planet: &Planet) -> (f64, f64) {
    let mu = planet.mu();
    let req = planet.equatorial_radius();
    let j2 = planet.j2();

    let r2 = radius * radius;
    let r4 = r2 * r2;
    let sin_lat = latitude.sin();
    let cos_lat = latitude.cos();
    let sin2 = sin_lat * sin_lat;
    let req2 = req * req;

    // Lateral component (from J2)
    let gravtl = 3.0 * mu * j2 * req2 * sin_lat * cos_lat / r4;

    // Radial component (positive outward)
    let gravtr = mu / r2 + 3.0 * mu * j2 * req2 * (1.0 - 3.0 * sin2) / (2.0 * r4);

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
        let planet = Planet::Moon;
        let r = planet.equatorial_radius();
        let (gravtl, gravtr) = gravity(r, 0.0, &planet);
        let expected = planet.mu() / (r * r);
        assert_relative_eq!(gravtr, expected, max_relative = 1e-4);
        assert_relative_eq!(gravtl, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn j2_lateral_zero_at_equator() {
        // sin(0) = 0 → gravtl must be exactly 0
        let (gravtl, _) = gravity(Planet::Mars.equatorial_radius(), 0.0, &Planet::Mars);
        assert_eq!(gravtl, 0.0);
    }

    #[test]
    fn j2_lateral_zero_at_pole() {
        // cos(pi/2) ≈ 0 → gravtl ≈ 0
        let (gravtl, _) = gravity(Planet::Mars.equatorial_radius(), FRAC_PI_2, &Planet::Mars);
        assert_relative_eq!(gravtl, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn j2_radial_stronger_at_pole_on_surface() {
        // Oblate planet: surface gravity is stronger at the pole because the polar surface
        // is closer to the center. Use actual surface radii (equatorial vs polar).
        let r_eq = Planet::Earth.equatorial_radius();
        let r_pole = Planet::Earth.polar_radius();
        let (_, gravtr_eq) = gravity(r_eq, 0.0, &Planet::Earth);
        let (_, gravtr_pole) = gravity(r_pole, FRAC_PI_2, &Planet::Earth);
        assert!(
            gravtr_pole > gravtr_eq,
            "polar surface gravtr ({gravtr_pole}) should exceed equatorial surface gravtr ({gravtr_eq})"
        );
    }

    #[test]
    fn gravity_decreases_with_altitude() {
        let r_low = Planet::Mars.equatorial_radius();
        let r_high = r_low + 200_000.0; // 200 km above surface
        let (_, gravtr_low) = gravity(r_low, 0.0, &Planet::Mars);
        let (_, gravtr_high) = gravity(r_high, 0.0, &Planet::Mars);
        assert!(
            gravtr_low > gravtr_high,
            "surface gravtr ({gravtr_low}) should exceed 200km gravtr ({gravtr_high})"
        );
    }

    #[test]
    fn mars_surface_gravity_ballpark() {
        // Mars equatorial surface gravity ≈ 3.72 m/s²
        let (_, gravtr) = gravity(Planet::Mars.equatorial_radius(), 0.0, &Planet::Mars);
        assert_relative_eq!(gravtr, 3.72, max_relative = 0.05);
    }

    #[test]
    fn earth_surface_gravity_ballpark() {
        // Earth equatorial surface gravity ≈ 9.81 m/s²
        let (_, gravtr) = gravity(Planet::Earth.equatorial_radius(), 0.0, &Planet::Earth);
        assert_relative_eq!(gravtr, 9.81, max_relative = 0.05);
    }

    #[test]
    fn j2_lateral_symmetry() {
        // gravtl(lat) = -gravtl(-lat) because sin(lat)*cos(lat) is odd
        let r = Planet::Mars.equatorial_radius();
        let lat = 0.7; // arbitrary latitude
        let (gravtl_pos, _) = gravity(r, lat, &Planet::Mars);
        let (gravtl_neg, _) = gravity(r, -lat, &Planet::Mars);
        assert_relative_eq!(gravtl_pos, -gravtl_neg, max_relative = 1e-14);
    }

    #[test]
    fn j2_lateral_max_at_45_deg() {
        // sin(2*lat) peaks at lat=pi/4, so |gravtl| should be maximal there
        let r = Planet::Mars.equatorial_radius();
        let (gravtl_45, _) = gravity(r, FRAC_PI_4, &Planet::Mars);
        let (gravtl_low, _) = gravity(r, 0.5, &Planet::Mars); // 0.5 rad ≈ 28.6°
        let (gravtl_high, _) = gravity(r, 1.0, &Planet::Mars); // 1.0 rad ≈ 57.3°
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
}
