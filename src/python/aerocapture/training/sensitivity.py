"""SALib sensitivity analysis support.

Builds a SALib problem dict from a [monte_carlo] config section, mirroring the
per-dimension transforms in Rust dispersions.rs build_dim_transforms().

Also provides run_morris(), run_sobol(), run_full_analysis(), and a CLI entry point.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

# Column order matches DispersionDraw::to_array() in dispersions.rs
DISPERSION_COLUMNS: list[str] = [
    "altitude",  # 0  initial state (Gaussian, meters)
    "longitude",  # 1  initial state (Gaussian, radians)
    "latitude",  # 2  initial state (Gaussian, radians)
    "velocity",  # 3  initial state (Gaussian, m/s)
    "flight_path",  # 4  initial state (Gaussian, radians)
    "azimuth",  # 5  initial state (Gaussian, radians)
    "density",  # 6  atmosphere (Uniform, fractional)
    "drag_coeff",  # 7  aerodynamics (Uniform, fractional)
    "lift_coeff",  # 8  aerodynamics (Uniform, fractional)
    "incidence",  # 9  aerodynamics (Uniform, radians)
    "nav_altitude",  # 10 navigation (Gaussian, meters)
    "nav_longitude",  # 11 navigation (Gaussian, radians)
    "nav_latitude",  # 12 navigation (Gaussian, radians)
    "nav_velocity",  # 13 navigation (Gaussian, m/s)
    "nav_flight_path",  # 14 navigation (Gaussian, radians)
    "nav_azimuth",  # 15 navigation (Gaussian, radians)
    "nav_drag_accel",  # 16 navigation (Gaussian, m/s²)
    "mass",  # 17 mass (Uniform, fractional)
    "ref_area",  # 18 vehicle (Uniform, fractional)
    "max_bank_rate",  # 19 vehicle (Uniform, fractional)
    "pilot_tau",  # 20 pilot (Uniform, fractional)
    "pilot_damping",  # 21 pilot (Uniform, fractional)
    "pilot_frequency",  # 22 pilot (Uniform, fractional)
    "filter_gain",  # 23 nav_filter (Gaussian, absolute delta)
    "wind_scale",  # 24 wind (Uniform range [min, max])
    "wind_direction_bias",  # 25 wind (Uniform, radians)
]

_DEG2RAD = math.pi / 180.0

# ── Sigma presets (mirror Rust from_level() for each struct) ─────────────────

_INITIAL_STATE_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"altitude": 0.0, "longitude": 0.0, "latitude": 0.0, "velocity": 0.0, "flight_path": 0.0, "azimuth": 0.0},
    "low": {"altitude": 0.0, "longitude": 0.01, "latitude": 0.01, "velocity": 0.13, "flight_path": 0.043, "azimuth": 0.043},
    "medium": {"altitude": 0.1, "longitude": 0.1, "latitude": 0.05, "velocity": 1.0, "flight_path": 0.1, "azimuth": 0.05},
    "high": {"altitude": 0.5, "longitude": 0.5, "latitude": 0.1, "velocity": 2.0, "flight_path": 0.2, "azimuth": 0.1},
}
_INITIAL_STATE_SIGMAS["custom"] = _INITIAL_STATE_SIGMAS["medium"]

_ATMOSPHERE_SIGMAS: dict[str, float] = {
    "off": 0.0,
    "low": 20.0,
    "medium": 50.0,
    "high": 100.0,
}
_ATMOSPHERE_SIGMAS["custom"] = _ATMOSPHERE_SIGMAS["medium"]

_AERODYNAMICS_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"drag": 0.0, "lift": 0.0, "incidence": 0.0},
    "low": {"drag": 3.0, "lift": 5.0, "incidence": 0.5},
    "medium": {"drag": 5.0, "lift": 10.0, "incidence": 1.0},
    "high": {"drag": 10.0, "lift": 15.0, "incidence": 2.0},
}
_AERODYNAMICS_SIGMAS["custom"] = _AERODYNAMICS_SIGMAS["medium"]

_NAVIGATION_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"altitude": 0.0, "longitude": 0.0, "latitude": 0.0, "velocity": 0.0, "flight_path": 0.0, "azimuth": 0.0, "drag_accel": 0.0},
    "low": {"altitude": 0.3, "longitude": 0.01, "latitude": 0.01, "velocity": 0.2, "flight_path": 0.02, "azimuth": 0.02, "drag_accel": 0.05},
    "medium": {"altitude": 0.667, "longitude": 0.05, "latitude": 0.05, "velocity": 0.4, "flight_path": 0.03, "azimuth": 0.03, "drag_accel": 0.1},
    "high": {"altitude": 1.0, "longitude": 0.1, "latitude": 0.1, "velocity": 1.0, "flight_path": 0.05, "azimuth": 0.05, "drag_accel": 0.2},
}
_NAVIGATION_SIGMAS["custom"] = _NAVIGATION_SIGMAS["medium"]

_MASS_SIGMAS: dict[str, float] = {
    "off": 0.0,
    "low": 0.5,
    "medium": 1.0,
    "high": 2.0,
}
_MASS_SIGMAS["custom"] = _MASS_SIGMAS["medium"]

_VEHICLE_SIGMAS: dict[str, dict[str, float]] = {
    "off": {"ref_area": 0.0, "max_bank_rate": 0.0},
    "low": {"ref_area": 1.0, "max_bank_rate": 5.0},
    "medium": {"ref_area": 2.0, "max_bank_rate": 10.0},
    "high": {"ref_area": 5.0, "max_bank_rate": 20.0},
}
_VEHICLE_SIGMAS["custom"] = _VEHICLE_SIGMAS["medium"]

_PILOT_SIGMAS: dict[str, float] = {
    "off": 0.0,
    "low": 5.0,
    "medium": 10.0,
    "high": 20.0,
}
_PILOT_SIGMAS["custom"] = _PILOT_SIGMAS["medium"]

_NAV_FILTER_SIGMAS: dict[str, float] = {
    "off": 0.0,
    "low": 0.05,
    "medium": 0.10,
    "high": 0.15,
}
_NAV_FILTER_SIGMAS["custom"] = _NAV_FILTER_SIGMAS["medium"]

_WIND_LEVELS: dict[str, dict[str, float]] = {
    "off": {"scale_min": 1.0, "scale_max": 1.0, "direction_bias_deg": 0.0},
    "low": {"scale_min": 0.7, "scale_max": 1.3, "direction_bias_deg": 5.0},
    "medium": {"scale_min": 0.5, "scale_max": 1.5, "direction_bias_deg": 10.0},
    "high": {"scale_min": 0.2, "scale_max": 2.0, "direction_bias_deg": 20.0},
}
_WIND_LEVELS["custom"] = _WIND_LEVELS["medium"]


def _get_level(cfg: dict[str, object], domain: str) -> str:
    """Extract level string from mc_config for a domain. Defaults to 'off'."""
    domain_cfg = cfg.get(domain)
    if not isinstance(domain_cfg, dict):
        return "off"
    level = domain_cfg.get("level", "off")
    return str(level)


def build_problem(mc_config: dict[str, object]) -> dict[str, object]:
    """Build a SALib problem dict from a [monte_carlo] config section.

    Mirrors build_dim_transforms() in dispersions.rs. Units are SI throughout:
    - angles in radians, altitudes in meters, velocities in m/s.

    Gaussian dims use SALib dists='norm' with bounds=[mean=0, std=sigma].
    Uniform dims use SALib dists='unif' with bounds=[-hw, hw].
    Wind scale uses dists='unif' with bounds=[scale_min, scale_max].
    """
    bounds: list[list[float]] = []
    dists: list[str] = []

    # ── Initial state (dims 0-5, Gaussian) ───────────────────────────────────
    is_level = _get_level(mc_config, "initial_state")
    is_s = _INITIAL_STATE_SIGMAS[is_level]
    bounds.append([0.0, is_s["altitude"] * 1e3])  # 0: altitude (m)
    bounds.append([0.0, is_s["longitude"] * _DEG2RAD])  # 1: longitude (rad)
    bounds.append([0.0, is_s["latitude"] * _DEG2RAD])  # 2: latitude (rad)
    bounds.append([0.0, is_s["velocity"]])  # 3: velocity (m/s)
    bounds.append([0.0, is_s["flight_path"] * _DEG2RAD])  # 4: flight_path (rad)
    bounds.append([0.0, is_s["azimuth"] * _DEG2RAD])  # 5: azimuth (rad)
    dists.extend(["norm"] * 6)

    # ── Atmosphere (dim 6, Uniform) ──────────────────────────────────────────
    atm_level = _get_level(mc_config, "atmosphere")
    atm_hw = _ATMOSPHERE_SIGMAS[atm_level] / 100.0
    bounds.append([-atm_hw, atm_hw])  # 6: density (fractional)
    dists.append("unif")

    # ── Aerodynamics (dims 7-9, Uniform) ─────────────────────────────────────
    aero_level = _get_level(mc_config, "aerodynamics")
    aero_s = _AERODYNAMICS_SIGMAS[aero_level]
    drag_hw = aero_s["drag"] / 100.0
    lift_hw = aero_s["lift"] / 100.0
    inc_hw = aero_s["incidence"] * _DEG2RAD
    bounds.append([-drag_hw, drag_hw])  # 7: drag_coeff
    bounds.append([-lift_hw, lift_hw])  # 8: lift_coeff
    bounds.append([-inc_hw, inc_hw])  # 9: incidence (rad)
    dists.extend(["unif"] * 3)

    # ── Navigation (dims 10-16, Gaussian) ────────────────────────────────────
    nav_level = _get_level(mc_config, "navigation")
    nav_s = _NAVIGATION_SIGMAS[nav_level]
    bounds.append([0.0, nav_s["altitude"] * 1e3])  # 10: nav_altitude (m)
    bounds.append([0.0, nav_s["longitude"] * _DEG2RAD])  # 11: nav_longitude (rad)
    bounds.append([0.0, nav_s["latitude"] * _DEG2RAD])  # 12: nav_latitude (rad)
    bounds.append([0.0, nav_s["velocity"]])  # 13: nav_velocity (m/s)
    bounds.append([0.0, nav_s["flight_path"] * _DEG2RAD])  # 14: nav_flight_path (rad)
    bounds.append([0.0, nav_s["azimuth"] * _DEG2RAD])  # 15: nav_azimuth (rad)
    bounds.append([0.0, nav_s["drag_accel"]])  # 16: nav_drag_accel (m/s²)
    dists.extend(["norm"] * 7)

    # ── Mass (dim 17, Uniform) ───────────────────────────────────────────────
    mass_level = _get_level(mc_config, "mass")
    mass_hw = _MASS_SIGMAS[mass_level] / 100.0
    bounds.append([-mass_hw, mass_hw])  # 17: mass (fractional)
    dists.append("unif")

    # ── Vehicle (dims 18-19, Uniform) ────────────────────────────────────────
    veh_level = _get_level(mc_config, "vehicle")
    veh_s = _VEHICLE_SIGMAS[veh_level]
    area_hw = veh_s["ref_area"] / 100.0
    bank_rate_hw = veh_s["max_bank_rate"] / 100.0
    bounds.append([-area_hw, area_hw])  # 18: ref_area (fractional)
    bounds.append([-bank_rate_hw, bank_rate_hw])  # 19: max_bank_rate (fractional)
    dists.extend(["unif"] * 2)

    # ── Pilot (dims 20-22, Uniform) ──────────────────────────────────────────
    pilot_level = _get_level(mc_config, "pilot")
    pilot_hw = _PILOT_SIGMAS[pilot_level] / 100.0
    bounds.append([-pilot_hw, pilot_hw])  # 20: pilot_tau (fractional)
    bounds.append([-pilot_hw, pilot_hw])  # 21: pilot_damping (fractional)
    bounds.append([-pilot_hw, pilot_hw])  # 22: pilot_frequency (fractional)
    dists.extend(["unif"] * 3)

    # ── Nav filter (dim 23, Gaussian) ────────────────────────────────────────
    nf_level = _get_level(mc_config, "nav_filter")
    nf_sigma = _NAV_FILTER_SIGMAS[nf_level]
    bounds.append([0.0, nf_sigma])  # 23: filter_gain (absolute)
    dists.append("norm")

    # ── Wind (dims 24-25, Uniform) ───────────────────────────────────────────
    wind_cfg = mc_config.get("wind")
    if isinstance(wind_cfg, dict):
        wind_level = str(wind_cfg.get("level", "medium"))
        w = _WIND_LEVELS[wind_level]
        scale_min = float(wind_cfg.get("scale_min", w["scale_min"]))
        scale_max = float(wind_cfg.get("scale_max", w["scale_max"]))
        dir_hw = float(wind_cfg.get("direction_bias_deg", w["direction_bias_deg"])) * _DEG2RAD
    else:
        # Wind absent: zero-width uniform (no dispersion)
        scale_min = 0.0
        scale_max = 0.0
        dir_hw = 0.0
    bounds.append([scale_min, scale_max])  # 24: wind_scale
    bounds.append([-dir_hw, dir_hw])  # 25: wind_direction_bias (rad)
    dists.extend(["unif"] * 2)

    return {
        "num_vars": len(DISPERSION_COLUMNS),
        "names": DISPERSION_COLUMNS,
        "bounds": bounds,
        "dists": dists,
    }


# ── Default mc_config used when a TOML has no [monte_carlo] section ──────────
_DEFAULT_MC_CONFIG: dict[str, Any] = {
    "seed": 42,
    "initial_state": {"level": "medium"},
    "atmosphere": {"level": "medium"},
    "aerodynamics": {"level": "medium"},
    "navigation": {"level": "medium"},
    "mass": {"level": "medium"},
    "vehicle": {"level": "medium"},
    "pilot": {"level": "medium"},
    "nav_filter": {"level": "medium"},
    "wind": {"level": "medium"},
}


def _load_mc_config(toml_path: str) -> dict[str, Any]:
    """Load [monte_carlo] section from a TOML with base inheritance."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    cfg = load_toml_with_bases(Path(toml_path))
    mc = cfg.get("monte_carlo")
    if isinstance(mc, dict):
        return mc  # type: ignore[return-value]
    return _DEFAULT_MC_CONFIG


def _evaluate_draws(
    toml_path: str,
    draws: npt.NDArray[np.float64],
    overrides: dict[str, Any] | None = None,
    sim_timeout_secs: float | None = None,
) -> npt.NDArray[np.float64]:
    """Run simulations for the given draw matrix, return DV total (column 41) for each."""
    import aerocapture_rs  # type: ignore[import-untyped]

    result = aerocapture_rs.run_with_draws(toml_path, draws, overrides=overrides, sim_timeout_secs=sim_timeout_secs)
    return result.final_records[:, 41].copy()  # type: ignore[no-any-return]


def run_morris(
    toml_path: str,
    n: int = 1000,
    *,
    overrides: dict[str, Any] | None = None,
    mc_config: dict[str, Any] | None = None,
    sim_timeout_secs: float | None = None,
) -> dict[str, Any]:
    """Run Morris elementary effects sensitivity analysis.

    Returns dict with mu_star, sigma, mu_star_conf, names (all as lists).
    """
    from SALib.analyze.morris import analyze as morris_analyze  # type: ignore[import-untyped]
    from SALib.sample.morris import sample as morris_sample  # type: ignore[import-untyped]

    if mc_config is None:
        mc_config = _load_mc_config(toml_path)

    problem = build_problem(mc_config)
    X = morris_sample(problem, N=n, num_levels=4, seed=42)
    Y = _evaluate_draws(toml_path, X, overrides=overrides, sim_timeout_secs=sim_timeout_secs)
    Si = morris_analyze(problem, X, Y, num_levels=4)

    return {
        "mu_star": Si["mu_star"].tolist(),
        "sigma": Si["sigma"].tolist(),
        "mu_star_conf": Si["mu_star_conf"].tolist(),
        "names": list(cast(list[str], problem["names"])),
    }


def run_sobol(
    toml_path: str,
    n: int = 1024,
    *,
    param_indices: list[int] | None = None,
    overrides: dict[str, Any] | None = None,
    mc_config: dict[str, Any] | None = None,
    calc_second_order: bool = True,
    sim_timeout_secs: float | None = None,
) -> dict[str, Any]:
    """Run Sobol variance-based sensitivity analysis.

    Returns dict with S1, ST, S1_conf, ST_conf, names, param_indices,
    and optionally S2/S2_conf (all as lists).
    """
    from SALib.analyze.sobol import analyze as sobol_analyze  # type: ignore[import-untyped]
    from SALib.sample.sobol import sample as sobol_sample  # type: ignore[import-untyped]

    if mc_config is None:
        mc_config = _load_mc_config(toml_path)

    problem = build_problem(mc_config)
    all_names: list[str] = list(cast(list[str], problem["names"]))
    all_bounds: list[list[float]] = list(cast(list[list[float]], problem["bounds"]))
    all_dists: list[str] = list(cast(list[str], problem["dists"]))

    if param_indices is None:
        param_indices = list(range(26))

    # Build reduced problem for the selected parameters only
    sub_names = [all_names[i] for i in param_indices]
    sub_bounds = [all_bounds[i] for i in param_indices]
    sub_dists = [all_dists[i] for i in param_indices]
    sub_problem: dict[str, Any] = {
        "num_vars": len(param_indices),
        "names": sub_names,
        "bounds": sub_bounds,
        "dists": sub_dists,
    }

    X_sub = sobol_sample(sub_problem, N=n, calc_second_order=calc_second_order, scramble=True, seed=42)

    # Expand to full 26-dim draw matrix; non-selected dims fixed at 0.0
    # except wind_scale (dim 24) which defaults to 1.0 (neutral)
    n_rows = X_sub.shape[0]
    X_full = np.zeros((n_rows, 26), dtype=np.float64)
    X_full[:, 24] = 1.0  # wind_scale neutral default
    for col, idx in enumerate(param_indices):
        X_full[:, idx] = X_sub[:, col]

    Y = _evaluate_draws(toml_path, X_full, overrides=overrides, sim_timeout_secs=sim_timeout_secs)
    Si = sobol_analyze(sub_problem, Y, calc_second_order=calc_second_order)

    result: dict[str, Any] = {
        "S1": Si["S1"].tolist(),
        "ST": Si["ST"].tolist(),
        "S1_conf": Si["S1_conf"].tolist(),
        "ST_conf": Si["ST_conf"].tolist(),
        "names": sub_names,
        "param_indices": param_indices,
    }
    if calc_second_order and "S2" in Si:
        result["S2"] = Si["S2"].tolist()
        result["S2_conf"] = Si["S2_conf"].tolist()

    return result


def run_full_analysis(
    toml_path: str,
    *,
    morris_n: int = 1000,
    sobol_n: int = 1024,
    top_k: int = 10,
    morris_only: bool = False,
    sobol_only: bool = False,
    output_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
    sim_timeout_secs: float | None = None,
) -> dict[str, Any]:
    """Run Morris + Sobol sensitivity analysis and save results to JSON.

    Workflow:
    - Morris first: ranks parameters by mu_star, identifies top_k influencers.
    - Sobol on top_k (or all 26 if sobol_only): higher-fidelity variance decomposition.
    - Results saved to output_dir/sensitivity_results.json.
    """
    mc_config = _load_mc_config(toml_path)

    # Determine output dir from guidance type if not given
    if output_dir is None:
        from aerocapture.training.toml_utils import load_toml_with_bases

        cfg = load_toml_with_bases(Path(toml_path))
        guidance_type = str(cfg.get("guidance", {}).get("type", "unknown"))  # type: ignore[union-attr]
        output_dir = Path("training_output") / guidance_type / "sensitivity"

    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    top_k_indices: list[int] | None = None

    if not sobol_only:
        print(f"Running Morris (N={morris_n})...")
        morris_result = run_morris(
            toml_path,
            n=morris_n,
            overrides=overrides,
            mc_config=mc_config,
            sim_timeout_secs=sim_timeout_secs,
        )
        results["morris"] = morris_result

        # Rank by mu_star, pick top-k indices
        mu_star = morris_result["mu_star"]
        ranked = sorted(range(len(mu_star)), key=lambda i: mu_star[i], reverse=True)
        top_k_indices = ranked[:top_k]

        print(f"\nTop-{top_k} parameters by mu_star:")
        names = morris_result["names"]
        for rank, idx in enumerate(top_k_indices, 1):
            print(f"  {rank:2d}. {names[idx]:<25s}  mu*={mu_star[idx]:.4f}")

    if not morris_only:
        sobol_indices = top_k_indices if top_k_indices is not None else None
        k_label = len(sobol_indices) if sobol_indices is not None else 26
        print(f"\nRunning Sobol (N={sobol_n}, k={k_label})...")
        sobol_result = run_sobol(
            toml_path,
            n=sobol_n,
            param_indices=sobol_indices,
            overrides=overrides,
            mc_config=mc_config,
            sim_timeout_secs=sim_timeout_secs,
        )
        results["sobol"] = sobol_result

    out_path = output_dir / "sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Variance-based sensitivity analysis for MC dispersions")
    parser.add_argument("toml", help="Path to training TOML config")
    parser.add_argument("--morris-n", type=int, default=1000)
    parser.add_argument("--sobol-n", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--morris-only", action="store_true")
    parser.add_argument("--sobol-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sim-timeout", type=float, default=None)
    args = parser.parse_args()

    run_full_analysis(
        args.toml,
        morris_n=args.morris_n,
        sobol_n=args.sobol_n,
        top_k=args.top_k,
        morris_only=args.morris_only,
        sobol_only=args.sobol_only,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        sim_timeout_secs=args.sim_timeout,
    )


if __name__ == "__main__":
    main()
