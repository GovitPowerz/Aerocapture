//! Equations of motion and the aerodynamic loads derived from them.
//!
//! `compute_derivatives` is the plant the integrators step (Gill RK4 and
//! DOPRI45 both take it as a closure); `effective_airspeed`, `heat_flux` and
//! `aero_loads` are the aero quantities every consumer (the EOM, peak tracking,
//! the photo rows, the guidance thermal fractions) must compute identically.
//! The dispersions enter through `AeroDispersions`, a `Copy` view of the run's
//! `RunState` (`simulation::init`), so this module depends on `data` and
//! `physics` only.
//!
//! Every expression here is bit-for-bit the one the simulator has always
//! evaluated (goldens, `run_grid` and the subprocess bit-identity gate pin it):
//! do not reassociate, factor or `mul_add` anything.

use crate::config::PlanetConfig;
use crate::data::SimData;
use crate::gnc::navigation::coordinates::geodetic_from_spherical;
use crate::physics::{atmosphere, gravity};

/// The per-run dispersion biases the aerodynamics read.
#[derive(Debug, Clone, Copy)]
pub struct AeroDispersions {
    pub density_bias: f64,         // atmosphere density bias (fractional)
    pub density_perturbation: f64, // time-varying GM perturbation (fractional)
    pub cx_bias: f64,              // drag coefficient bias (fractional)
    pub cz_bias: f64,              // lift coefficient bias (fractional)
    pub mass_bias: f64,            // mass bias (fractional)
    pub incidence_bias: f64,       // incidence error (radians)
    pub ref_area_bias: f64,        // reference area bias (fractional)
    pub wind_scale: f64,           // wind speed multiplier (1.0 = nominal)
    pub wind_direction_bias: f64,  // wind direction rotation (radians)
}

/// Compute effective airspeed accounting for wind.
///
/// The state velocity `v` is relative to the planet-fixed atmosphere.
/// Wind adds a velocity perturbation: we subtract wind from the vehicle's
/// ground-relative velocity components to get the airspeed used for aero forces.
/// Returns the original `v` when wind is disabled or no wind table is loaded.
pub fn effective_airspeed(
    v: f64,
    gamma: f64,
    psi: f64,
    lat: f64,
    altitude: f64,
    data: &SimData,
    aero: &AeroDispersions,
) -> f64 {
    if !data.wind_enabled {
        return v;
    }
    if let Some(ref wt) = data.wind_table {
        let w = wt.wind_at(altitude, lat);
        let scale = aero.wind_scale;
        let rot = aero.wind_direction_bias;
        // Apply dispersions: scale and rotate wind vector
        let we = scale * (w.east * rot.cos() - w.north * rot.sin());
        let wn = scale * (w.east * rot.sin() + w.north * rot.cos());
        // Project into trajectory frame and compute effective speed
        let cos_g = gamma.cos();
        let v_east = v * cos_g * psi.sin() - we;
        let v_north = v * cos_g * psi.cos() - wn;
        let v_vert = v * gamma.sin();
        (v_east * v_east + v_north * v_north + v_vert * v_vert).sqrt()
    } else {
        v
    }
}

/// Convective heat flux (W/m^2): `cq * sqrt(rho) * v_eff^3.05`, the one law the
/// EOM flux integral, the peak tracker, the photo rows and the guidance thermal
/// fraction all evaluate.
pub fn heat_flux(cq: f64, rho: f64, v_eff: f64) -> f64 {
    cq * rho.sqrt() * v_eff.powf(3.05)
}

/// Instantaneous aerodynamic loads at one state.
#[derive(Debug, Clone, Copy)]
pub struct AeroLoads {
    pub heat_flux: f64,   // W/m^2
    pub pdyn: f64,        // Pa
    pub load_factor: f64, // m/s^2 (aerodynamic acceleration magnitude)
}

/// Heat flux, dynamic pressure and load factor from the dispersed density, the
/// wind-corrected airspeed and the commanded angle of attack (radians, undispersed).
pub fn aero_loads(
    rho: f64,
    v_eff: f64,
    aoa: f64,
    data: &SimData,
    aero: &AeroDispersions,
) -> AeroLoads {
    let heat_flux = heat_flux(data.capsule.cq, rho, v_eff);
    let pdyn = 0.5 * rho * v_eff * v_eff;

    let aoa_dispersed = aoa + aero.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + aero.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + aero.cz_bias);
    let mass = data.capsule.mass * (1.0 + aero.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + aero.ref_area_bias);
    let aero_accel = rho * ref_area * v_eff * v_eff / (2.0 * mass);
    let load_factor = aero_accel * (cx * cx + cz * cz).sqrt();

    AeroLoads {
        heat_flux,
        pdyn,
        load_factor,
    }
}

/// Compute state derivatives (equations of motion).
///
/// State = [r, lon, lat, V, gamma, psi, flux, time]
pub fn compute_derivatives(
    state: &[f64; 8],
    bank_angle: f64,
    aoa: f64,
    planet: &PlanetConfig,
    data: &SimData,
    aero: &AeroDispersions,
) -> [f64; 8] {
    let r = state[0];
    let _lon = state[1];
    let lat = state[2];
    let v = state[3];
    let gamma = state[4];
    let psi = state[5];

    let (gravtl, gravtr) = gravity::gravity(r, lat, planet);
    let (altitude, _lat_geo) = geodetic_from_spherical(r, state[1], lat, planet);
    let rho = atmosphere::density(
        &data.atmosphere,
        altitude,
        aero.density_bias,
        aero.density_perturbation,
    );

    let aoa_dispersed = aoa + aero.incidence_bias;
    let cx = data.aero.interpolate_cx(aoa_dispersed) * (1.0 + aero.cx_bias);
    let cz = data.aero.interpolate_cz(aoa_dispersed) * (1.0 + aero.cz_bias);

    let mass = data.capsule.mass * (1.0 + aero.mass_bias);
    let ref_area = data.capsule.reference_area * (1.0 + aero.ref_area_bias);

    // Wind-corrected velocity for aero forces and heat flux.
    // Note: aero force *magnitude* uses v_eff (airspeed) but is applied along the
    // planet-relative velocity direction. This is a first-order approximation valid
    // when wind << vehicle speed. At Mars entry (100 m/s wind vs 5700 m/s), the
    // direction error is O(wind/V)² ≈ 0.03%.
    let v_eff = effective_airspeed(v, gamma, psi, lat, altitude, data, aero);

    let aero_factor = rho * ref_area / (2.0 * mass);
    let acdrag = aero_factor * cx * v_eff * v_eff;
    let aclift = aero_factor * cz * v_eff * v_eff;

    let cos_bank = bank_angle.cos();
    let sin_bank = bank_angle.sin();
    let cos_gamma = gamma.cos();
    let sin_gamma = gamma.sin();
    let cos_psi = psi.cos();
    let sin_psi = psi.sin();
    let cos_lat = lat.cos();
    let sin_lat = lat.sin();
    let tan_gamma = sin_gamma / cos_gamma;
    let tan_lat = sin_lat / cos_lat;

    let omega = planet.omega;

    // Kinematic derivatives use original v (planet-relative)
    let dr = v * sin_gamma;
    let dlon = v * cos_gamma * sin_psi / (r * cos_lat);
    let dlat = v * cos_gamma * cos_psi / r;

    let dv = -acdrag - gravtr * sin_gamma - gravtl * cos_gamma * cos_psi
        + omega * omega * r * cos_lat * (cos_lat * sin_gamma - sin_lat * cos_gamma * cos_psi);

    let dgamma = (aclift * cos_bank / v) + (v * cos_gamma / r)
        - ((gravtr * cos_gamma - gravtl * sin_gamma * cos_psi) / v)
        + (2.0 * omega * sin_psi * cos_lat)
        + (omega * omega * r * cos_lat * (sin_lat * sin_gamma * cos_psi + cos_lat * cos_gamma) / v);

    let dpsi = (aclift * sin_bank / (v * cos_gamma))
        + (v * cos_gamma * sin_psi * tan_lat / r)
        + (2.0 * omega * (sin_lat - cos_psi * cos_lat * tan_gamma))
        + (gravtl * sin_psi / (v * cos_gamma))
        + (omega * omega * r * cos_lat * sin_lat * sin_psi / (v * cos_gamma));

    // Heat flux uses wind-corrected velocity
    let dflux = heat_flux(data.capsule.cq, rho, v_eff);
    let dtime = 1.0;

    [dr, dlon, dlat, dv, dgamma, dpsi, dflux, dtime]
}
