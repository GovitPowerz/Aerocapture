"""AMAT oracle for the external physics cross-check (issue #112, docs/validation.md).

AMAT is a one-off oracle, never a dependency: this script runs only in a scratch venv, and its
outputs are frozen in amat_frozen.json next to it. From the repo root:

    uv venv .venv-amat --python 3.12
    VIRTUAL_ENV=.venv-amat uv pip install AMAT==2.4.0
    git clone https://github.com/athulpg007/AMAT /tmp/AMAT-src   # atmdata/ ships only in the repo
    .venv-amat/bin/python experiments/external_validation/amat_oracle.py --amat-src /tmp/AMAT-src

Two blocks:
- matched: configs/validation/amat_matched.toml flown by AMAT with our planet constants and our
  density table (linear interpolation, like the Rust lookup) at three constant banks, plus the
  lift-modulation corridor at the entry state. AMAT's own heating law and g-load are kept
  alongside, since both differ from ours by construction (Sutton-Graves; a g-load bug).
- published: the Girija 2023 Mars smallsat drag-modulation corridor, re-derived by AMAT with the
  notebook's settings (its planet, its Mars-GRAM mean table) and with ours (our constants, our
  table), which splits the published-vs-ours gap into tool and atmosphere contributions.

The inputs AMAT flew are written next to its outputs; the slow gate asserts the Rust-resolved
configs still equal them.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import types
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from AMAT.planet import Planet
from AMAT.vehicle import Vehicle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/python"))  # the repo's own `base` resolver (stdlib only), not a third copy
from aerocapture.training.toml_utils import load_toml_with_bases  # noqa: E402

MATCHED_TOML = "configs/validation/amat_matched.toml"
DRM_TOML = "configs/validation/mars_smallsat_drm.toml"
OUT = Path(__file__).with_name("amat_frozen.json")

MATCHED_BANKS_DEG = {"lift_up": 0.0, "mid_bank": 60.0, "lift_down": 180.0}
MATCHED_TOL = 1e-10  # solve_ivp rtol = atol on AMAT's non-dimensional state
T_MAX_S = 3000.0  # = the Rust default [simulation] max_time
DT_OUT_S = 0.1
EFPA_TOL_DEG = 1e-4
G0 = 9.81  # the Rust G0 (sim_types.rs); AMAT divides by 9.80665
SUTTON_GRAVES_MARS = 1.8980e-8  # AMAT qStagConvective, W/cm2 with rho in kg/m3, v in m/s, RN in m

DRM: dict[str, Any] = {
    "citation": (
        "A. P. Girija, Aerocapture Design Reference Missions for Solar System Exploration: from Venus to Neptune, "
        "arXiv:2308.10384 (2023), Table 1, Mars smallsat row"
    ),
    "published": {"undershoot_efpa_deg": -9.86, "overshoot_efpa_deg": -8.78, "width_deg": 1.09},
    "beta1_kg_m2": 20.0,
    "beta_ratio": 7.5,
    "notebook": "docs/source/mdpi-aerospace-notebooks/smallsat-mission-concepts/section-3-4-mars-smallsat-nominal-aerocapture-trajectory.ipynb",
}
# The notebook's own call: findUnderShootLimitD2(2400.0, 0.1, -20.0, -5.0, 1E-10, 2000.0) at solver tol 1e-6.
NOTEBOOK_TOL = 1e-6
NOTEBOOK_T_SEC = 2400.0
NOTEBOOK_EFPA_BRACKET_DEG = (-20.0, -5.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_inputs(rel: str) -> dict:
    """The physical inputs AMAT flies, as the config states them (compared field by field by the gate)."""
    cfg = load_toml_with_bases(REPO / rel)
    pt = cfg["aerodynamics"]["points"][0]
    alpha = np.radians(cfg["incidence"]["angles"][0])
    # data/aerodynamics.rs: Cx = Ca cos(a) + Cn sin(a), Cz = -Ca sin(a) + Cn cos(a)
    cx = pt["ca"] * np.cos(alpha) + pt["cn"] * np.sin(alpha)
    cz = -pt["ca"] * np.sin(alpha) + pt["cn"] * np.cos(alpha)
    atm = REPO / cfg["data"]["atmosphere"]
    return {
        "config": rel,
        "planet": {k: cfg["planet"][k] for k in ("mu", "equatorial_radius", "polar_radius", "omega", "j2", "j3", "j4")},
        "entry": {k: cfg["entry"][k] for k in ("altitude", "longitude", "latitude", "velocity", "flight_path_angle", "azimuth")},
        "vehicle": {k: cfg["vehicle"][k] for k in ("mass", "reference_area", "cq")},
        "aerodynamics": {"aoa_deg": cfg["incidence"]["angles"][0], "ca": pt["ca"], "cn": pt["cn"], "cx": float(cx), "cz": float(cz)},
        "exit_altitude_km": cfg["flight"]["final_conditions"]["altitude"],
        "target_apoapsis_km": cfg["flight"]["target_orbit"]["apoapsis"],
        "atmosphere": {"path": cfg["data"]["atmosphere"], "sha256": sha256(atm)},
        "wind": cfg["flight"]["wind"],
    }


def read_density_table(path: Path) -> np.ndarray:
    """(h_m, rho) rows of a Rust atmosphere file: first numeric line = row count, then the rows."""
    rows = []
    for line in path.read_text().splitlines():
        try:
            vals = [float(t) for t in line.replace("D", "E").split()[:2]]
        except ValueError:
            continue
        if vals:
            rows.append(vals)
    n = int(rows[0][0])
    return np.array(rows[1 : 1 + n])


def export_for_amat(table: np.ndarray, dst: Path) -> float:
    """AMAT's loadAtmosphereModel wants h, T, p, rho columns. T and p only feed Mach and stagnation
    pressure, neither compared here, so they are placeholders (p = rho * R_CO2 * T keeps AMAT's sonic
    speed finite). Rows with rho = 0 (the Rust table's top two, above any compared flight) are dropped.
    Returns the top altitude kept, which becomes AMAT's h_thres."""
    keep = table[table[:, 1] > 0.0]
    t = np.full(len(keep), 200.0)
    np.savetxt(dst, np.column_stack([keep[:, 0], t, keep[:, 1] * 188.9 * t, keep[:, 1]]))
    return float(keep[-1, 0])


def matched_planet(inp: dict, atm_file: Path, h_thres: float) -> Planet:
    p = inp["planet"]
    planet = Planet("MARS")
    planet.RP = p["equatorial_radius"]
    planet.GM = p["mu"]
    planet.OMEGA = p["omega"]
    planet.J2 = p["j2"]
    planet.J3 = p["j3"]
    # Planet.__init__ derives these from the constants just replaced.
    planet.Vref = np.sqrt(planet.GM / planet.RP)
    planet.tau = planet.RP / planet.Vref
    planet.OMEGAbar = planet.OMEGA * planet.tau
    planet.loadAtmosphereModel(str(atm_file), 0, 1, 2, 3, intType="linear")
    planet.h_thres = h_thres
    planet.h_skip = inp["exit_altitude_km"] * 1e3  # classifyTrajectory: exit = first sample above
    planet.h_trap = 0.0  # crash = first sample below the surface, as the Rust altitude <= 0 check
    return planet


def make_vehicle(name: str, planet: Planet, mass: float, area: float, cx: float, cz: float, rn: float, entry: dict, tol: float) -> Vehicle:
    vehicle = Vehicle(name, mass, mass / (area * cx), cz / cx, area, 0.0, rn, planet)
    # AMAT's heading is measured from east toward north; the Rust azimuth from north toward east.
    vehicle.setInitialState(
        entry["altitude"], entry["longitude"], entry["latitude"], entry["velocity"] / 1e3, 90.0 - entry["azimuth"], entry["flight_path_angle"], 0.0, 0.0
    )
    vehicle.setSolverParams(tol)
    return vehicle


def crossing_time(t: np.ndarray, h: np.ndarray, i: int, level: float) -> float:
    """Linear interpolation of the time h crosses `level` between samples i-1 and i."""
    return float(t[i - 1] + (level - h[i - 1]) * (t[i] - t[i - 1]) / (h[i] - h[i - 1]))


def fly_constant_bank(vehicle: Vehicle, bank_deg: float, cq: float) -> dict:
    """propogateEntry2's integration (solve_ivp + AMAT's gamma = -88 deg event), keeping the untruncated
    samples so the exit/impact crossing can be interpolated; every physical quantity is AMAT's.

    `bank_deg` is ours. AMAT's heading turns counter-clockwise, so its positive bank turns left and ours
    right: AMAT flies delta = -bank."""
    planet = vehicle.planetObj
    d2r = np.pi / 180.0
    r0 = planet.computeR(vehicle.h0_km * 1e3)
    nondim = planet.nonDimState(
        r0, vehicle.theta0_deg * d2r, vehicle.phi0_deg * d2r, vehicle.v0_kms * 1e3, vehicle.psi0_deg * d2r, vehicle.gamma0_deg * d2r, 0.0
    )
    delta = -bank_deg * d2r
    sol = vehicle.solveTrajectory2(*nondim, T_MAX_S, DT_OUT_S, delta)
    t, r, theta, phi, v, psi, gamma, _ = planet.dimensionalize(*sol)
    h = r - planet.RP
    index, exitflag = vehicle.classifyTrajectory(r)
    if exitflag == 1.0:
        outcome, t_end = "exit", crossing_time(t, h, index, planet.h_skip)
    elif exitflag == -1.0:
        outcome, t_end = "crash", crossing_time(t, h, index, planet.h_trap)
    else:
        outcome, t_end = "timeout", float(t[-1])
    s = slice(0, index)
    rho = planet.rhovectorized(r[s])
    q_ours = cq * np.sqrt(rho) * v[s] ** 3.05  # dynamics.rs heat_flux, W/m2
    a_s = vehicle.a_svectorized(r[s], theta[s], phi[s], v[s], delta)
    a_n = vehicle.a_nvectorized(r[s], theta[s], phi[s], v[s], delta)
    a_w = vehicle.a_wvectorized(r[s], theta[s], phi[s], v[s], delta)
    q_native = vehicle.qStagTotal(r[s], v[s])  # W/cm2
    g_reported = float(vehicle.computeAccelerationLoad(t[s], r[s], theta[s], phi[s], v[s], index, delta).max())
    out = {
        "bank_deg": bank_deg,
        "amat_delta_deg": -bank_deg,
        "outcome": outcome,
        "flight_time_s": t_end,
        "min_alt_km": float(h[s].min() / 1e3),
        "peak_heat_flux_kw_m2": float(q_ours.max() / 1e3),
        "heat_load_mj_m2": float(np.trapezoid(q_ours, t[s]) / 1e6),
        "peak_g": float(np.sqrt(a_s**2 + a_n**2 + a_w**2).max() / G0),
        "amat_native": {
            "peak_heat_flux_kw_m2": float(q_native.max() * 10.0),
            "heat_load_mj_m2": float(np.trapezoid(q_native, t[s]) / 100.0),
            # computeAccelerationLoad as shipped: sqrt(a_s^2 + a_n^2 + a_w*2) / 9.80665, NaN once a_w < 0
            "peak_g_reported": None if np.isnan(g_reported) else g_reported,
        },
    }
    if outcome == "exit":
        k = index - 1  # last sample below h_skip, AMAT's terminal state
        rk, vk, gk, pk, sk = r[k], v[k], gamma[k], phi[k], psi[k]
        v_east = vk * np.cos(gk) * np.cos(sk) + planet.OMEGA * rk * np.cos(pk)
        v_in2 = (vk * np.sin(gk)) ** 2 + v_east**2 + (vk * np.cos(gk) * np.sin(sk)) ** 2
        out["exit_energy_mj_kg"] = float((0.5 * v_in2 - planet.GM / rk) / 1e6)
        out["apoapsis_alt_km"] = float(vehicle.compute_ApoapsisAltitudeKm(rk, vk, gk, theta[k], pk, sk))
    return out


def exact_heading_eom2(self: Vehicle, t: float, y: list[float], delta: float) -> list[float]:
    """Vehicle.EOM2 with the heading equation's three 1 / (cos(gamma) + 1e-2) factors made exact
    (the lateral-force term, cfpsibar, copsibar); every other term is AMAT's own. Diagnostic only."""
    rbar, theta, phi, vbar, psi, gamma, _ = y
    w = self.planetObj.OMEGAbar
    cg, sg = np.cos(gamma), np.sin(gamma)
    return [
        vbar * sg,
        vbar * cg * np.cos(psi) / (rbar * np.cos(phi)),
        vbar * cg * np.sin(psi) / rbar,
        self.a_sbar(rbar, theta, phi, vbar, delta) + self.gsbar(rbar, phi, gamma, psi) + self.cfvbar(rbar, phi, vbar, psi, gamma),
        (self.a_wbar(rbar, theta, phi, vbar, delta) + self.gwbar(rbar, phi, gamma, psi)) / (vbar * cg)
        - (vbar / rbar) * cg * np.cos(psi) * np.tan(phi)
        - w**2 * rbar / (vbar * cg) * np.sin(phi) * np.cos(phi) * np.cos(psi)
        + 2.0 * w * (sg / cg * np.cos(phi) * np.sin(psi) - np.sin(phi)),
        (self.a_nbar(rbar, theta, phi, vbar, delta) + self.gnbar(rbar, phi, gamma, psi)) / vbar
        + (vbar / rbar) * cg
        + self.cfgammabar(rbar, phi, vbar, psi, gamma)
        + self.cogammabar(rbar, phi, vbar, psi, gamma),
        vbar * cg,
    ]


def matched_block(workdir: Path) -> dict:
    inp = config_inputs(MATCHED_TOML)
    atm_file = workdir / "matched_atm.dat"
    h_thres = export_for_amat(read_density_table(REPO / inp["atmosphere"]["path"]), atm_file)
    planet = matched_planet(inp, atm_file, h_thres)
    a, e = inp["aerodynamics"], inp["entry"]
    # The nose radius at which AMAT's Mars Sutton-Graves law equals ours at the entry speed:
    # k sqrt(rho / RN) v^3 [W/cm2] = cq sqrt(rho) v^3.05 [W/m2]. Only the AMAT-native heating uses it.
    rn = (SUTTON_GRAVES_MARS * 1e4 / (inp["vehicle"]["cq"] * e["velocity"] ** 0.05)) ** 2
    veh = make_vehicle("MSR", planet, inp["vehicle"]["mass"], inp["vehicle"]["reference_area"], a["cx"], a["cz"], rn, e, MATCHED_TOL)
    cases = {name: fly_constant_bank(veh, bank, inp["vehicle"]["cq"]) for name, bank in MATCHED_BANKS_DEG.items()}
    exact = make_vehicle("MSR", planet, inp["vehicle"]["mass"], inp["vehicle"]["reference_area"], a["cx"], a["cz"], rn, e, MATCHED_TOL)
    exact.EOM2 = types.MethodType(exact_heading_eom2, exact)
    # AMAT's native lift-modulation corridor: overshoot = full lift-down, undershoot = full lift-up.
    target = inp["target_apoapsis_km"]
    os_efpa, os_flag = veh.findOverShootLimit2(T_MAX_S, DT_OUT_S, -14.0, -8.0, EFPA_TOL_DEG, target)
    us_efpa, us_flag = veh.findUnderShootLimit2(T_MAX_S, DT_OUT_S, -14.0, -8.0, EFPA_TOL_DEG, target)
    assert os_flag == us_flag == 1.0, "corridor bound outside the [-14, -8] deg bracket"
    return {
        "inputs": inp,
        "amat_setup": {
            "solver": "Vehicle.solveTrajectory2 (scipy solve_ivp RK45, gamma = -88 deg terminal event)",
            "tol": MATCHED_TOL,
            "dt_out_s": DT_OUT_S,
            "t_max_s": T_MAX_S,
            "density_interpolation": "linear",
            "h_thres_m": h_thres,
            "nose_radius_m": rn,
            "beta_kg_m2": inp["vehicle"]["mass"] / (inp["vehicle"]["reference_area"] * a["cx"]),
            "lift_to_drag": a["cz"] / a["cx"],
        },
        "cases": cases,
        "diagnostics": {"mid_bank_exact_heading": fly_constant_bank(exact, MATCHED_BANKS_DEG["mid_bank"], inp["vehicle"]["cq"])},
        "corridor": {
            "target_apoapsis_km": target,
            "efpa_tol_deg": EFPA_TOL_DEG,
            "overshoot_efpa_deg": os_efpa,
            "undershoot_efpa_deg": us_efpa,
            "width_deg": os_efpa - us_efpa,
        },
    }


def drm_corridor(planet: Planet, inp: dict, tol: float, t_sec: float) -> dict:
    """The notebook's findUnderShootLimitD2 / findOverShootLimitD2 calls on a given planet."""
    e, vcl = inp["entry"], inp["vehicle"]
    veh = Vehicle("MarsSmallSat1", vcl["mass"], DRM["beta1_kg_m2"], 0.0, vcl["reference_area"], 0.0, 0.35, planet)
    veh.setInitialState(e["altitude"], e["longitude"], e["latitude"], e["velocity"] / 1e3, 90.0 - e["azimuth"], -5.0, 0.0, 0.0)
    veh.setSolverParams(tol)
    veh.setDragModulationVehicleParams(DRM["beta1_kg_m2"], DRM["beta_ratio"])
    lo, hi = NOTEBOOK_EFPA_BRACKET_DEG
    us, us_flag = veh.findUnderShootLimitD2(t_sec, DT_OUT_S, lo, hi, 1e-10, inp["target_apoapsis_km"])
    os_, os_flag = veh.findOverShootLimitD2(t_sec, DT_OUT_S, lo, hi, 1e-10, inp["target_apoapsis_km"])
    assert us_flag == os_flag == 1.0
    return {"undershoot_efpa_deg": us, "overshoot_efpa_deg": os_, "width_deg": os_ - us}


def published_block(workdir: Path, amat_src: Path) -> dict:
    inp = config_inputs(DRM_TOML)
    # (a) the notebook as published: AMAT's Mars constants and Mars-GRAM mean table.
    gram = amat_src / "atmdata/Mars/mars-gram-avg.dat"
    own = Planet("MARS")
    own.loadAtmosphereModel(str(gram), 0, 1, 2, 3)
    own.h_skip = inp["exit_altitude_km"] * 1e3
    as_published = drm_corridor(own, inp, NOTEBOOK_TOL, NOTEBOOK_T_SEC)
    # (b) our Mars: our constants (spherical, J2/J3; AMAT has no oblateness or J4) and our table.
    atm_file = workdir / "drm_atm.dat"
    h_thres = export_for_amat(read_density_table(REPO / inp["atmosphere"]["path"]), atm_file)
    ours = matched_planet(inp, atm_file, h_thres)
    with_our_mars = drm_corridor(ours, inp, NOTEBOOK_TOL, NOTEBOOK_T_SEC)
    return {
        **DRM,
        "inputs": inp,
        "amat_gram_mean_table": {"atmosphere": {"path": "atmdata/Mars/mars-gram-avg.dat", "sha256": sha256(gram)}, **as_published},
        "amat_our_mars": with_our_mars,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--amat-src", type=Path, required=True, help="AMAT git checkout (for atmdata/ and the commit id)")
    args = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(args.amat_src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        frozen = {
            "generator": "experiments/external_validation/amat_oracle.py",
            "generator_sha256": sha256(Path(__file__)),  # the gate fails when the script moves past its outputs
            "amat": {
                "version": version("AMAT"),
                "source_commit": commit,
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "scipy": version("scipy"),
            },
            "g0_m_s2": G0,
            "matched": matched_block(Path(tmp)),
            "published": published_block(Path(tmp), args.amat_src),
        }
    OUT.write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
