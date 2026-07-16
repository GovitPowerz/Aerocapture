"""Prediction dynamics for the CPAG C0 prototype.

CPAG (Rataczak/McMahon/Boyd, JGCD 2025) plans a SIGNED bank profile with the
bank angle as a state and the bank RATE as the control -- no separate lateral
logic. The prediction model here is the repo's own dynamics (`runner.rs::
compute_derivatives` / `fnpag.rs::pred_derivatives`): 3-DOF point mass over a
rotating oblate planet (J2/J3/J4 -- richer than the paper's J2-only), the
mission atmosphere table scaled by the nav-estimated density factor (the repo's
FNPAG lesson, playing the role of the paper's fading-memory filters), constant
stability-axis aero coefficients at the entry AoA, no winds. Unlike FNPAG's
predictor the lateral lift term is INCLUDED (sin sigma) -- CPAG knows the roll
sign because it plans it.

State vector x = [r, lon, lat, v, gamma, psi, sigma, Q]:
    r      radius (m, geocentric spherical)
    lon    longitude (rad)
    lat    geocentric latitude (rad)
    v      planet-relative velocity (m/s)
    gamma  relative flight path angle (rad)
    psi    relative heading, from North positive East (rad)
    sigma  signed bank angle (rad) -- augmented state, d(sigma)/dt = u
    Q      accumulated heat load (kJ/m^2) -- augmented state for the path bound

Conventions mirrored from the Rust sim:
    density   queried at GEODETIC altitude (iterative oblate conversion)
    exit      checked on SPHERICAL altitude r - Req (FNPAG predictor convention)
    aero      body ca/cn -> stability cx/cz at load (data/mod.rs), cz > 0 = lift up
    heat flux cq * sqrt(rho) * v^3.05  (W/m^2)
    g-load    (rho Sref v^2 / 2m) sqrt(cx^2+cz^2) / 9.81
    apoapsis  osculating, from INERTIAL velocity (relative + omega x r)
All functions are vectorized over a leading batch axis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

G0 = 9.81  # matches Rust sim_types::G0
HEAT_FLUX_VEL_EXPONENT = 3.05
UNBOUND_APOAPSIS_RADIUS_M = 1e9  # matches fnpag.rs sentinel

# State indices
IR, ILON, ILAT, IV, IGAMMA, IPSI, ISIGMA, IQ = range(8)
N_STATES = 8


@dataclass(frozen=True)
class Planet:
    mu: float
    req: float
    rpol: float
    omega: float
    j2: float
    j3: float
    j4: float


@dataclass(frozen=True)
class Vehicle:
    mass: float
    sref: float
    cq: float
    cx: float  # stability-axis drag coefficient at entry AoA
    cz: float  # stability-axis lift coefficient at entry AoA (positive = lift up)
    max_bank_rate: float  # rad/s


@dataclass(frozen=True)
class PathLimits:
    """SI units (converted from TOML kW/kPa/kJ/g like the Rust config parser)."""

    max_heat_flux: float  # W/m^2
    max_g_load: float  # g
    max_pdyn: float  # Pa
    max_heat_load: float  # kJ/m^2 (kept in kJ -- the Q state unit)


@dataclass(frozen=True)
class Atmosphere:
    altitudes: FloatArray  # m, non-decreasing
    densities: FloatArray  # kg/m^3
    ref_density: float  # exponential tail rho0
    scale_factor: float  # exponential tail 1/H (1/m)
    ref_altitude: float  # exponential tail z0 (m)

    def density(self, altitude: FloatArray | float) -> FloatArray:
        """Table interpolation + exponential tail, matching Rust `density_at`."""
        alt = np.asarray(altitude, dtype=np.float64)
        tail = self.ref_density * np.exp(-self.scale_factor * (alt - self.ref_altitude))
        interp = np.interp(alt, self.altitudes, self.densities)
        rho = np.where(alt >= self.altitudes[-1], tail, interp)
        return np.where(alt <= self.altitudes[0], self.densities[0], rho)


@dataclass(frozen=True)
class CpagModel:
    planet: Planet
    vehicle: Vehicle
    limits: PathLimits
    atmosphere: Atmosphere
    exit_alt: float  # m, spherical-altitude exit threshold (final_conditions.altitude)
    target_apoapsis_radius: float  # m
    target_inclination: float  # rad
    density_factor: float = 1.0  # nav-estimated dispersion factor on the onboard table

    def with_density_factor(self, factor: float) -> CpagModel:
        return replace(self, density_factor=factor)


def load_atmosphere_table(path: Path) -> Atmosphere:
    """Parse the legacy atmosphere .dat (Fortran D-notation, 3 header lines)."""
    rows: list[list[float]] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append([float(tok.replace("D", "E").replace("d", "e")) for tok in ln.split()])
        except ValueError:
            continue  # header/comment lines
    n_points = int(rows[0][0])
    table = np.array(rows[1 : 1 + n_points], dtype=np.float64)
    # After the table: -1 marker, 4 profile altitudes, 4 dispersions, then
    # rozmod / facech / zromod / cstgam single-value lines.
    tail_scalars = [r[0] for r in rows[1 + n_points :]]
    marker_offset = 1 if tail_scalars and tail_scalars[0] == -1.0 else 0
    exp_params = tail_scalars[marker_offset + 8 : marker_offset + 11]
    ref_density, scale_factor, ref_altitude = (exp_params + [0.0, 0.0, 0.0])[:3]
    return Atmosphere(
        altitudes=np.ascontiguousarray(table[:, 0]),
        densities=np.ascontiguousarray(table[:, 1]),
        ref_density=ref_density,
        scale_factor=scale_factor,
        ref_altitude=ref_altitude,
    )


def load_model(toml_path: str | Path, density_factor: float = 1.0) -> CpagModel:
    """Build a CpagModel from a resolved config (PyO3 loader, base inheritance)."""
    import aerocapture_rs  # noqa: PLC0415

    cfg = aerocapture_rs.load_config(str(toml_path))
    planet_cfg = cfg["planet"]
    planet = Planet(
        mu=float(planet_cfg["mu"]),
        req=float(planet_cfg["equatorial_radius"]),
        rpol=float(planet_cfg["polar_radius"]),
        omega=float(planet_cfg["omega"]),
        j2=float(planet_cfg["j2"]),
        j3=float(planet_cfg.get("j3", 0.0)),
        j4=float(planet_cfg.get("j4", 0.0)),
    )
    # Body ca/cn -> stability cx/cz per point (data/mod.rs), then interpolate at
    # the entry AoA (the FNPAG predictor's constant-coefficient convention).
    aoa = np.radians(float(cfg["entry"]["initial_aoa"]))
    points = cfg["aerodynamics"]["points"]
    alphas = np.radians(np.array([p["aoa"] for p in points], dtype=np.float64))
    ca = np.array([p["ca"] for p in points], dtype=np.float64)
    cn = np.array([p["cn"] for p in points], dtype=np.float64)
    cx_tab = ca * np.cos(alphas) + cn * np.sin(alphas)
    cz_tab = -ca * np.sin(alphas) + cn * np.cos(alphas)
    order = np.argsort(alphas)
    vehicle_cfg = cfg["vehicle"]
    vehicle = Vehicle(
        mass=float(vehicle_cfg["mass"]),
        sref=float(vehicle_cfg["reference_area"]),
        cq=float(vehicle_cfg["cq"]),
        cx=float(np.interp(aoa, alphas[order], cx_tab[order])),
        cz=float(np.interp(aoa, alphas[order], cz_tab[order])),
        max_bank_rate=np.radians(float(vehicle_cfg["max_bank_rate"])),
    )
    constraints = cfg["flight"]["constraints"]
    limits = PathLimits(
        max_heat_flux=float(constraints["max_heat_flux"]) * 1e3,
        max_g_load=float(constraints["max_load_factor"]),
        max_pdyn=float(constraints["max_dynamic_pressure"]) * 1e3,
        max_heat_load=float(constraints["max_heat_load"]),
    )
    repo_root = _find_repo_root(Path(toml_path))
    atmosphere = load_atmosphere_table(repo_root / cfg["data"]["atmosphere"])
    orbit = cfg["flight"]["target_orbit"]
    return CpagModel(
        planet=planet,
        vehicle=vehicle,
        limits=limits,
        atmosphere=atmosphere,
        exit_alt=float(cfg["flight"]["final_conditions"]["altitude"]) * 1e3,
        target_apoapsis_radius=planet.req + float(orbit["apoapsis"]) * 1e3,
        target_inclination=np.radians(float(orbit["inclination"])),
        density_factor=density_factor,
    )


def entry_state(toml_path: str | Path) -> FloatArray:
    """Undispersed entry state x0 from the config's [entry] section."""
    import aerocapture_rs  # noqa: PLC0415

    cfg = aerocapture_rs.load_config(str(toml_path))
    e = cfg["entry"]
    req = float(cfg["planet"]["equatorial_radius"])
    return np.array(
        [
            req + float(e["altitude"]) * 1e3,
            np.radians(float(e["longitude"])),
            np.radians(float(e["latitude"])),
            float(e["velocity"]),
            np.radians(float(e["flight_path_angle"])),
            np.radians(float(e["azimuth"])),
            np.radians(float(e["initial_bank_angle"])),
            0.0,
        ],
        dtype=np.float64,
    )


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    if p.is_file():
        p = p.parent
    while not (p / "pyproject.toml").exists():
        if p.parent == p:
            raise FileNotFoundError(f"repo root not found above {start}")
        p = p.parent
    return p


def gravity(r: FloatArray, lat: FloatArray, planet: Planet) -> tuple[FloatArray, FloatArray]:
    """(gravtl, gravtr): lateral and radial components, port of physics/gravity.rs."""
    mu, req = planet.mu, planet.req
    r2 = r * r
    r4 = r2 * r2
    r5 = r4 * r
    r6 = r4 * r2
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin2 = sin_lat * sin_lat
    sin4 = sin2 * sin2
    req2 = req * req
    req3 = req2 * req
    req4 = req2 * req2

    gravtr = mu / r2 + 1.5 * mu * planet.j2 * req2 * (1.0 - 3.0 * sin2) / r4
    gravtr = gravtr + 2.0 * mu * planet.j3 * req3 * sin_lat * (3.0 - 5.0 * sin2) / r5
    gravtr = gravtr - 0.625 * mu * planet.j4 * req4 * (3.0 - 30.0 * sin2 + 35.0 * sin4) / r6

    gravtl = 3.0 * mu * planet.j2 * req2 * sin_lat * cos_lat / r4
    gravtl = gravtl + 1.5 * mu * planet.j3 * req3 * cos_lat * (5.0 * sin2 - 1.0) / r5
    gravtl = gravtl - 2.5 * mu * planet.j4 * req4 * sin_lat * cos_lat * (3.0 - 7.0 * sin2) / r6
    return gravtl, gravtr


def geodetic_altitude(r: FloatArray, lat: FloatArray, planet: Planet) -> FloatArray:
    """Geodetic (ellipsoid) altitude, port of coordinates.rs::geodetic_from_spherical.

    Fixed-count vectorized iteration (the Rust loop early-exits at |dalt| < 0.01 m;
    ten iterations converge well past that for atmospheric-entry geometries).
    """
    req, rpol = planet.req, planet.rpol
    if abs(req - rpol) < 1e-10:
        return np.asarray(r - req, dtype=np.float64)
    pos_p = r * np.cos(lat)
    pos_z = r * np.sin(lat)
    e2 = (req * req - rpol * rpol) / (req * req)

    rplant = np.full_like(np.asarray(r, dtype=np.float64), req)
    altitude_z = (r - req) - np.sqrt(req * rpol)
    altitude = np.zeros_like(rplant)
    for _ in range(10):
        tan_lat = (pos_z / pos_p) / (1.0 - e2 * rplant / (rplant + altitude_z))
        t2 = tan_lat * tan_lat
        sin_l = np.sqrt(t2 / (1.0 + t2))
        cos_l = np.sqrt(1.0 / (1.0 + t2))
        altitude = pos_p / cos_l - rplant
        sin_l = np.where(tan_lat < 0.0, -sin_l, sin_l)
        rplant = req / np.sqrt(1.0 - e2 * sin_l * sin_l)
        altitude_z = altitude
    return altitude


def density_at_state(x: FloatArray, model: CpagModel) -> FloatArray:
    """Dispersion-scaled onboard density at the state's geodetic altitude."""
    alt = geodetic_altitude(x[..., IR], x[..., ILAT], model.planet)
    return model.density_factor * model.atmosphere.density(alt)


def eom(x: FloatArray, u: FloatArray, model: CpagModel) -> FloatArray:
    """State derivatives; u = bank rate sigma_dot (rad/s). Vectorized.

    Port of runner.rs::compute_derivatives with the CPAG bank-as-state
    augmentation: signed lift terms (cos sigma in gamma, sin sigma in psi).
    """
    r = x[..., IR]
    lat = x[..., ILAT]
    v = x[..., IV]
    gamma = x[..., IGAMMA]
    psi = x[..., IPSI]
    sigma = x[..., ISIGMA]

    rho = density_at_state(x, model)
    veh = model.vehicle
    aero_factor = rho * veh.sref / (2.0 * veh.mass)
    drag = aero_factor * veh.cx * v * v
    lift = aero_factor * veh.cz * v * v

    gravtl, gravtr = gravity(r, lat, model.planet)

    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    tan_gamma = sin_gamma / cos_gamma
    tan_lat = sin_lat / cos_lat
    omega = model.planet.omega

    dr = v * sin_gamma
    dlon = v * cos_gamma * sin_psi / (r * cos_lat)
    dlat = v * cos_gamma * cos_psi / r

    dv = -drag - gravtr * sin_gamma - gravtl * cos_gamma * cos_psi + omega * omega * r * cos_lat * (cos_lat * sin_gamma - sin_lat * cos_gamma * cos_psi)

    v_safe = np.where(np.abs(v) > 1.0, v, 1.0)
    dgamma = np.where(
        np.abs(v) > 1.0,
        (lift * np.cos(sigma) / v_safe)
        + (v_safe * cos_gamma / r)
        - ((gravtr * cos_gamma - gravtl * sin_gamma * cos_psi) / v_safe)
        + (2.0 * omega * sin_psi * cos_lat)
        + (omega * omega * r * cos_lat * (sin_lat * sin_gamma * cos_psi + cos_lat * cos_gamma) / v_safe),
        0.0,
    )
    dpsi = np.where(
        (np.abs(v) > 1.0) & (np.abs(cos_gamma) > 1e-10),
        (lift * np.sin(sigma) / (v_safe * cos_gamma))
        + (v_safe * cos_gamma * sin_psi * tan_lat / r)
        + (2.0 * omega * (sin_lat - cos_psi * cos_lat * tan_gamma))
        + (gravtl * sin_psi / (v_safe * cos_gamma))
        + (omega * omega * r * cos_lat * sin_lat * sin_psi / (v_safe * cos_gamma)),
        0.0,
    )
    dsigma = np.broadcast_to(np.asarray(u, dtype=np.float64), dr.shape)
    dq = 1e-3 * veh.cq * np.sqrt(rho) * v**HEAT_FLUX_VEL_EXPONENT  # kW/m^2 == kJ/m^2/s

    return np.stack([dr, dlon, dlat, dv, dgamma, dpsi, dsigma, dq], axis=-1)


def rk4_step(x: FloatArray, u: FloatArray, dt: float | FloatArray, model: CpagModel) -> FloatArray:
    """Classic RK4 with the bank rate held across stages.

    `dt` broadcasts against the batch axis (scalar or per-row array).
    """
    h = np.asarray(dt, dtype=np.float64)[..., np.newaxis] if isinstance(dt, np.ndarray) else dt
    k1 = eom(x, u, model)
    k2 = eom(x + 0.5 * h * k1, u, model)
    k3 = eom(x + 0.5 * h * k2, u, model)
    k4 = eom(x + h * k3, u, model)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def path_quantities(x: FloatArray, model: CpagModel) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(heat_flux W/m^2, g_load g, pdyn Pa) at the state, sim conventions."""
    v = x[..., IV]
    rho = density_at_state(x, model)
    veh = model.vehicle
    heat_flux = veh.cq * np.sqrt(rho) * v**HEAT_FLUX_VEL_EXPONENT
    aero_accel = rho * veh.sref * v * v / (2.0 * veh.mass)
    g_load = aero_accel * np.sqrt(veh.cx**2 + veh.cz**2) / G0
    pdyn = 0.5 * rho * v * v
    return heat_flux, g_load, pdyn


def _inertial_velocity_pieces(x: FloatArray, planet: Planet) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """(v_abs^2, v_radial, v_east_abs, v_north) from the relative spherical state."""
    r = x[..., IR]
    lat = x[..., ILAT]
    v = x[..., IV]
    gamma = x[..., IGAMMA]
    psi = x[..., IPSI]
    we = planet.omega * r * np.cos(lat)
    v_east = v * np.cos(gamma) * np.sin(psi) + we
    v_north = v * np.cos(gamma) * np.cos(psi)
    v_radial = v * np.sin(gamma)
    v_abs2 = v_east * v_east + v_north * v_north + v_radial * v_radial
    return v_abs2, v_radial, v_east, v_north


def inertial_energy(x: FloatArray, planet: Planet) -> FloatArray:
    """Specific orbital energy from inertial velocity: E = v_abs^2/2 - mu/r."""
    v_abs2, _, _, _ = _inertial_velocity_pieces(x, planet)
    return 0.5 * v_abs2 - planet.mu / x[..., IR]


def angular_momentum_sq(x: FloatArray, planet: Planet) -> FloatArray:
    """|r x v_abs|^2 (horizontal inertial velocity times radius, squared)."""
    v_abs2, v_radial, _, _ = _inertial_velocity_pieces(x, planet)
    r = x[..., IR]
    return r * r * np.maximum(v_abs2 - v_radial * v_radial, 0.0)


def apoapsis_radius(x: FloatArray, planet: Planet) -> FloatArray:
    """Osculating apoapsis radius from INERTIAL velocity, frame-free.

    Unbound orbits get the FNPAG sentinel (used for reporting/feasibility only;
    the SCP terminal signal is the smooth `eps_apoapsis` below).
    """
    mu = planet.mu
    energy = inertial_energy(x, planet)
    h2 = angular_momentum_sq(x, planet)
    sma = -mu / (2.0 * energy)
    ecc = np.sqrt(np.abs(1.0 - h2 / (mu * sma)))
    r_apo = sma * (1.0 + ecc)
    unbound = (energy >= 0.0) | (ecc >= 1.0) | ~np.isfinite(r_apo)
    return np.where(unbound, UNBOUND_APOAPSIS_RADIUS_M, r_apo)


def eps_apoapsis(x: FloatArray, model: CpagModel) -> FloatArray:
    """CPAG's energy-based apoapsis targeting residual (J/kg), Keplerian form.

    eps = E(x) - [ -mu/ra_t + h(x)^2 / (2 ra_t^2) ]: zero exactly when the
    osculating orbit's apoapsis radius equals ra_t; smooth and well-defined for
    ALL orbits including hyperbolic (the paper's fix for the apoapsis-Jacobian
    singularity at escape velocity). Positive = too much energy for the target.
    """
    ra_t = model.target_apoapsis_radius
    return inertial_energy(x, model.planet) + model.planet.mu / ra_t - angular_momentum_sq(x, model.planet) / (2.0 * ra_t * ra_t)


def cos_inclination(x: FloatArray, planet: Planet) -> FloatArray:
    """cos(i) of the osculating orbit: h_z / |h| (frame-free, inertial)."""
    _, _, v_east, v_north = _inertial_velocity_pieces(x, planet)
    lat = x[..., ILAT]
    h_horiz2 = v_east * v_east + v_north * v_north
    denom = np.sqrt(np.maximum(h_horiz2, 1e-30))
    return np.cos(lat) * v_east / denom
