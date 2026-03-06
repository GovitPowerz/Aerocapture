//! Coordinate transformations.
//!
//! Matches Fortran frayon.f, geodes.f, cartes.f, reploc.f, xvabsl.f.

use crate::config::Planet;

/// Compute geodetic altitude and latitude from geocentric spherical position.
///
/// Matches Fortran frayon.f exactly.
///
/// Input: position = [radius, longitude, latitude] (geocentric spherical)
/// Output: (geodetic_altitude, geodetic_latitude)
pub fn geodetic_from_spherical(
    radius: f64,
    longitude: f64,
    latitude: f64,
    planet: &Planet,
) -> (f64, f64) {
    let req = planet.equatorial_radius();
    let rpol = planet.polar_radius();

    let cos_lat = latitude.cos();
    let sin_lat = latitude.sin();

    // Convert to Cartesian
    let pos_x = radius * cos_lat * longitude.cos();
    let pos_y = radius * cos_lat * longitude.sin();
    let pos_z = radius * sin_lat;

    let pos_p = (pos_x * pos_x + pos_y * pos_y).sqrt();
    let pos_r = (pos_z * pos_z + pos_p * pos_p).sqrt();

    let altitr = pos_r - req;

    if (req - rpol).abs() < 1e-10 {
        // Spherical planet
        let sin_lat_geo = pos_z / pos_r;
        let cos_lat_geo = pos_p / pos_r;
        let lat_geo = sin_lat_geo.atan2(cos_lat_geo);
        (altitr, lat_geo)
    } else {
        // Oblate planet — iterative computation
        let excent = ((req * req - rpol * rpol) / (req * req)).sqrt();
        let e2 = excent * excent;

        let mut rplant = req;
        let mut altitz = altitr - (req * rpol).sqrt();
        let mut altitude;
        let lat_geo;

        for _ in 0..10 {
            let tan_lat = (pos_z / pos_p) / (1.0 - e2 * rplant / (rplant + altitz));
            let sin_l = (tan_lat * tan_lat / (1.0 + tan_lat * tan_lat)).sqrt();
            let cos_l = (1.0 / (1.0 + tan_lat * tan_lat)).sqrt();
            altitude = pos_p / cos_l - rplant;
            let sin_l = if tan_lat < 0.0 { -sin_l } else { sin_l };

            if (altitude - altitz).abs() < 0.01 {
                lat_geo = sin_l.atan2(cos_l);
                return (altitude, lat_geo);
            }

            rplant = req / (1.0 - e2 * sin_l * sin_l).sqrt();
            altitz = altitude;
        }

        // Fallback after max iterations
        let tan_lat = (pos_z / pos_p) / (1.0 - e2 * rplant / (rplant + altitz));
        let sin_l = (tan_lat * tan_lat / (1.0 + tan_lat * tan_lat)).sqrt();
        let cos_l = (1.0 / (1.0 + tan_lat * tan_lat)).sqrt();
        altitude = pos_p / cos_l - rplant;
        let sin_l = if tan_lat < 0.0 { -sin_l } else { sin_l };
        lat_geo = sin_l.atan2(cos_l);
        (altitude, lat_geo)
    }
}

/// Convert geodetic to geocentric Cartesian position.
///
/// Matches Fortran geodes.f.
#[allow(dead_code)]
pub fn geodetic_to_cartesian(
    altitude: f64,
    latitude: f64,
    longitude: f64,
    planet: &Planet,
) -> [f64; 3] {
    let req = planet.equatorial_radius();
    let rpol = planet.polar_radius();
    let excent = ((req * req - rpol * rpol) / (req * req)).sqrt();
    let e2 = excent * excent;

    let sin_lat = latitude.sin();
    let cos_lat = latitude.cos();

    let n = req / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let r = n + altitude;

    [
        r * cos_lat * longitude.cos(),
        r * cos_lat * longitude.sin(),
        (n * (1.0 - e2) + altitude) * sin_lat,
    ]
}

/// Convert spherical position to Cartesian.
///
/// Matches Fortran cartes.f with iposvi=0.
/// Input: [r, longitude, latitude]
/// Output: [x, y, z] geocentric Cartesian
pub fn position_to_cartesian(r: f64, lon: f64, lat: f64) -> [f64; 3] {
    [
        r * lat.cos() * lon.cos(),
        r * lat.cos() * lon.sin(),
        r * lat.sin(),
    ]
}

/// Convert spherical velocity to local Cartesian.
///
/// Matches Fortran cartes.f with iposvi=1.
/// Input: [V, gamma, psi] (speed, flight path angle, azimuth)
/// Output: local Cartesian velocity [vx, vy, vz]
pub fn velocity_to_local_cartesian(v: f64, gamma: f64, psi: f64) -> [f64; 3] {
    let two_pi = 2.0 * std::f64::consts::PI;
    let anglxy = -psi + two_pi;
    let anglxz = gamma;
    [
        v * anglxz.cos() * anglxy.cos(),
        v * anglxz.cos() * anglxy.sin(),
        v * anglxz.sin(),
    ]
}

/// Build local-to-geocentric rotation matrix.
///
/// Matches Fortran reploc.f with indloc=0.
/// Input: position as [r, longitude, latitude]
/// Output: 3x3 rotation matrix (row-major)
pub fn local_to_geocentric_matrix(lon: f64, lat: f64) -> [[f64; 3]; 3] {
    let sinlat = lat.sin();
    let coslat = lat.cos();
    let sinlon = lon.sin();
    let coslon = lon.cos();

    [
        [-coslon * sinlat, sinlon, coslon * coslat],
        [-sinlon * sinlat, -coslon, sinlon * coslat],
        [coslat, 0.0, sinlat],
    ]
}

/// Matrix-vector product (3x3 matrix × 3-vector).
///
/// Matches Fortran matvec.f.
pub fn mat_vec_3(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Cross product of two 3-vectors.
///
/// Matches Fortran pvecto.f.
pub fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-vectors.
pub fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Norm of a 3-vector.
///
/// Matches Fortran pnorme.f.
pub fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Compute absolute (inertial) position and velocity from spherical state.
///
/// Matches Fortran xvabsl.f.
/// Takes geocentric spherical position [r, lon, lat] and local spherical velocity [V, gamma, psi].
/// Returns (position_cartesian, velocity_absolute_cartesian).
pub fn to_absolute_cartesian(
    r: f64,
    lon: f64,
    lat: f64,
    v: f64,
    gamma: f64,
    psi: f64,
    planet: &Planet,
) -> ([f64; 3], [f64; 3]) {
    // Position: spherical → Cartesian
    let posita = position_to_cartesian(r, lon, lat);

    // Velocity: spherical → local Cartesian
    let vitesl = velocity_to_local_cartesian(v, gamma, psi);

    // Local-to-geocentric rotation matrix
    let plocal = local_to_geocentric_matrix(lon, lat);

    // Velocity in geocentric frame = P * vitesl
    let vitesr = mat_vec_3(&plocal, &vitesl);

    // Entrainment velocity = omega × position
    let omega = planet.omega();
    let omega_vec = [0.0, 0.0, omega]; // Fortran: xomega = [0, 0, omega]
    let vitese = cross(&omega_vec, &posita);

    // Absolute velocity = entrainment + relative geocentric
    let vitesa = [
        vitese[0] + vitesr[0],
        vitese[1] + vitesr[1],
        vitese[2] + vitesr[2],
    ];

    (posita, vitesa)
}

/// Compute total orbital energy from spherical state.
///
/// Matches Fortran enrtot.f.
/// E = |v_abs|^2/2 - mu/|r|
pub fn total_energy(
    r: f64,
    lon: f64,
    lat: f64,
    v: f64,
    gamma: f64,
    psi: f64,
    planet: &Planet,
) -> f64 {
    let (posita, vitesa) = to_absolute_cartesian(r, lon, lat, v, gamma, psi, planet);
    let vitabs = norm(&vitesa);
    let rayvec = norm(&posita);
    vitabs * vitabs / 2.0 - planet.mu() / rayvec
}
