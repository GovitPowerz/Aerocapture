//! Orbital element computation from state vectors.

use crate::config::Planet;
use crate::data::OrbitalElements;
use crate::gnc::navigation::coordinates::{cross, dot, norm, to_absolute_cartesian};

/// Compute orbital elements from spherical state.
///
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
    let enrmin = 1e-6; // small threshold to avoid parabolic singularity

    // Get absolute position and velocity in Cartesian
    let (position_abs, velocity_abs) = to_absolute_cartesian(
        radius,
        longitude,
        latitude,
        velocity,
        flight_path,
        azimuth,
        planet,
    );

    // Angular momentum: L = r × v
    let angular_momentum = cross(&position_abs, &velocity_abs);

    let radius_val = norm(&position_abs);
    let speed_abs = norm(&velocity_abs);
    let angular_momentum_magnitude = norm(&angular_momentum);

    // Total energy (thresholded to avoid parabolic singularity)
    let energy_raw = speed_abs * speed_abs / 2.0 - mu / radius_val;
    let energy_sign = energy_raw.signum();
    let energy = energy_sign * energy_raw.abs().max(enrmin);

    // Semi-major axis
    let semi_major_axis_raw = -mu / (2.0 * energy);

    // Eccentricity
    let eccentricity_param = angular_momentum_magnitude * angular_momentum_magnitude / (mu * semi_major_axis_raw);
    let eccentricity_raw = if (eccentricity_param - 1.0).abs() < 1e-20 {
        0.0
    } else {
        (1.0 - eccentricity_param).abs().sqrt()
    };

    // Inclination
    let cos_inclination = angular_momentum[2] / angular_momentum_magnitude;
    let sin_inclination = (1.0 - cos_inclination * cos_inclination).max(0.0).sqrt();
    let inclination_raw = sin_inclination.atan2(cos_inclination);

    // RAAN (longitude of ascending node)
    let raan_raw = if sin_inclination.abs() > 1e-10 {
        let sinomg = angular_momentum[0] / (angular_momentum_magnitude * sin_inclination);
        let cosomg = -angular_momentum[1] / (angular_momentum_magnitude * sin_inclination);
        sinomg.atan2(cosomg)
    } else {
        0.0
    };

    // True anomaly
    let pos_dot_vel = dot(&position_abs, &velocity_abs);
    let v0 = if energy < 0.0 {
        // Elliptical orbit
        let sv0 =
            pos_dot_vel * (1.0 - eccentricity_raw * eccentricity_raw).max(0.0).sqrt() / (eccentricity_raw * (mu * semi_major_axis_raw).sqrt());
        let cv0 = (1.0 - radius_val / semi_major_axis_raw) / eccentricity_raw - eccentricity_raw;
        sv0.atan2(cv0)
    } else {
        // Hyperbolic orbit
        let sv0 = pos_dot_vel * (eccentricity_raw * eccentricity_raw - 1.0).max(0.0).sqrt()
            / (eccentricity_raw * (mu * semi_major_axis_raw.abs()).sqrt());
        let cv0 = -((1.0 + radius_val / semi_major_axis_raw.abs()) / eccentricity_raw - eccentricity_raw);
        sv0.atan2(cv0)
    };

    // Argument of periapsis
    let sinomg = raan_raw.sin();
    let cosomg = raan_raw.cos();
    let arg_periapsis_raw = if inclination_raw > 1e-3 {
        let arg1 = position_abs[2] / (inclination_raw.sin() * radius_val);
        let arg2 = (position_abs[0] * cosomg + position_abs[1] * sinomg) / radius_val;
        let mut w = arg1.atan2(arg2) - v0;
        if w < 0.0 {
            w += 2.0 * std::f64::consts::PI;
        }
        w
    } else {
        let mut w = position_abs[1].atan2(position_abs[0]) - v0;
        if w < 0.0 {
            w += 2.0 * std::f64::consts::PI;
        }
        w
    };

    // Periapsis and apoapsis radii → altitudes
    let rayper = semi_major_axis_raw * (1.0 - eccentricity_raw);
    let rayapo = semi_major_axis_raw * (1.0 + eccentricity_raw);

    OrbitalElements {
        semi_major_axis: semi_major_axis_raw,
        eccentricity: eccentricity_raw,
        inclination: inclination_raw,
        raan: raan_raw,
        arg_periapsis: arg_periapsis_raw,
        true_anomaly: v0,
        periapsis_alt: rayper - req,
        apoapsis_alt: rayapo - req,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn mars() -> Planet {
        Planet::Mars
    }

    /// Circular equatorial orbit at 300 km altitude.
    /// V_rel = V_inertial - omega*r, heading east (psi=PI/2), gamma=0, lat=0.
    /// Expect: SMA ≈ r, e ≈ 0, inclination ≈ 0.
    #[test]
    fn circular_equatorial_orbit() {
        let planet = mars();
        let mu = planet.mu();
        let req = planet.equatorial_radius();
        let omega = planet.omega();
        let alt = 300_000.0;
        let r = req + alt;

        // Inertial circular velocity, then subtract planet rotation for relative velocity
        let v_inertial = (mu / r).sqrt();
        let v_rel = v_inertial - omega * r;

        let oe = from_spherical(r, 0.0, 0.0, v_rel, 0.0, PI / 2.0, &planet);

        assert_relative_eq!(oe.semi_major_axis, r, max_relative = 1e-6);
        assert_relative_eq!(oe.eccentricity, 0.0, epsilon = 1e-3);
        assert!(
            oe.inclination.abs() < 0.01,
            "inclination should be near zero, got {}",
            oe.inclination
        );
    }

    /// Hyperbolic entry at Mars: V = 5687 m/s at 125 km alt.
    /// Expect: SMA < 0, e > 1.
    #[test]
    fn hyperbolic_orbit_has_negative_sma() {
        let planet = mars();
        let req = planet.equatorial_radius();
        let r = req + 125_000.0;
        let v = 5687.0;
        let gamma = -10.0_f64.to_radians(); // shallow entry
        let psi = PI / 2.0;

        let oe = from_spherical(r, 0.0, 0.0, v, gamma, psi, &planet);

        assert!(
            oe.semi_major_axis < 0.0,
            "SMA should be negative for hyperbolic orbit, got {}",
            oe.semi_major_axis
        );
        assert!(
            oe.eccentricity > 1.0,
            "eccentricity should be > 1, got {}",
            oe.eccentricity
        );
    }

    /// For an elliptical orbit (e < 1), periapsis_alt < apoapsis_alt.
    #[test]
    fn periapsis_below_apoapsis() {
        let planet = mars();
        let mu = planet.mu();
        let req = planet.equatorial_radius();
        let omega = planet.omega();
        let alt = 300_000.0;
        let r = req + alt;

        // Slightly super-circular relative velocity → elliptical orbit
        let v_inertial = (mu / r).sqrt();
        let v_rel = v_inertial * 1.05 - omega * r;

        let oe = from_spherical(r, 0.0, 0.0, v_rel, 0.0, PI / 2.0, &planet);

        assert!(
            oe.eccentricity < 1.0,
            "orbit should be elliptical, e = {}",
            oe.eccentricity
        );
        assert!(
            oe.periapsis_alt < oe.apoapsis_alt,
            "periapsis ({}) should be below apoapsis ({})",
            oe.periapsis_alt,
            oe.apoapsis_alt
        );
    }

    /// East heading at equator → low inclination; north heading → ~90° inclination.
    #[test]
    fn inclination_from_azimuth() {
        let planet = mars();
        let mu = planet.mu();
        let req = planet.equatorial_radius();
        let omega = planet.omega();
        let r = req + 300_000.0;
        let v_inertial = (mu / r).sqrt();
        let v_rel = v_inertial - omega * r;

        // Eastward at equator → near-zero inclination
        let oe_east = from_spherical(r, 0.0, 0.0, v_rel, 0.0, PI / 2.0, &planet);
        assert!(
            oe_east.inclination.abs() < 0.01,
            "east heading at equator should give low inclination, got {} deg",
            oe_east.inclination.to_degrees()
        );

        // Northward at equator → ~90° inclination
        let oe_north = from_spherical(r, 0.0, 0.0, v_rel, 0.0, 0.0, &planet);
        // Not exactly PI/2 because planet rotation adds an eastward component
        // to the inertial velocity, tilting the orbit plane slightly
        assert_relative_eq!(oe_north.inclination, PI / 2.0, max_relative = 0.06);
    }

    /// Verify SMA matches vis-viva: a = -mu / (2*E) using total_energy().
    #[test]
    fn sma_matches_vis_viva() {
        use crate::gnc::navigation::coordinates::total_energy;

        let planet = mars();
        let mu = planet.mu();
        let req = planet.equatorial_radius();
        let omega = planet.omega();
        let r = req + 400_000.0;
        let v_inertial = (mu / r).sqrt() * 1.1;
        let v_rel = v_inertial - omega * r;
        let lon = 0.5;
        let lat = 0.1;
        let gamma = 0.0;
        let psi = PI / 2.0;

        let oe = from_spherical(r, lon, lat, v_rel, gamma, psi, &planet);
        let energy = total_energy(r, lon, lat, v_rel, gamma, psi, &planet);
        let sma_from_energy = -mu / (2.0 * energy);

        assert_relative_eq!(oe.semi_major_axis, sma_from_energy, max_relative = 1e-6);
    }
}
