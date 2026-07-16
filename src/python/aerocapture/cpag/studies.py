"""C0 convergence studies: replan states across the corridor + dispersed batch.

Replan states are harvested from Rust-simulated trajectories (the truth plant,
not the prototype's own model): an undispersed constant-bank sweep spanning the
corridor sentinel range (corridor.py convention, 0-180 deg) and a dispersed
FTC-guided MC batch flying the deployed GA optimum. Each state gets one cold
SCP replan (hold-current-bank initialization) and is classified against the
physically achievable apoapsis bracket so "SCP failed" is never conflated with
"target unreachable from this state".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aerocapture.cpag.model import (
    IQ,
    ISIGMA,
    CpagModel,
    FloatArray,
    apoapsis_radius,
    entry_state,
    load_model,
)
from aerocapture.cpag.scp import ScpConfig, ScpResult, roll_to_bank_profile, scp_replan, shoot_profile

# Trajectory columns (BatchResults contract)
_COL_ALT_KM = 0
_COL_LON_DEG = 1
_COL_LAT_DEG = 2
_COL_VEL = 3
_COL_FPA_DEG = 4
_COL_HEAD_DEG = 5
_COL_TIME_S = 7
_COL_BANK_DEG = 10
_COL_NAV_DENSITY_RATIO = 13
_COL_HEAT_LOAD_KJ = 15


@dataclass
class ReplanCase:
    """One harvested replan state."""

    source: str  # e.g. "bank_072" or "mc_seed_17"
    time_s: float
    x0: FloatArray
    density_factor: float


def geodetic_to_geocentric(alt_m: float, lat_gd_rad: float, req: float, rpol: float) -> tuple[float, float]:
    """(r, geocentric lat) from geodetic altitude + latitude (standard ellipsoid)."""
    e2 = (req * req - rpol * rpol) / (req * req)
    sin_l = np.sin(lat_gd_rad)
    n_curv = req / np.sqrt(1.0 - e2 * sin_l * sin_l)
    p = (n_curv + alt_m) * np.cos(lat_gd_rad)
    z = (n_curv * (1.0 - e2) + alt_m) * sin_l
    return float(np.hypot(p, z)), float(np.arctan2(z, p))


def state_from_trajectory_row(row: FloatArray, model: CpagModel) -> FloatArray:
    """Reconstruct the 8-state vector from one (17,) trajectory row."""
    r, lat_gc = geodetic_to_geocentric(float(row[_COL_ALT_KM]) * 1e3, float(np.radians(row[_COL_LAT_DEG])), model.planet.req, model.planet.rpol)
    return np.array(
        [
            r,
            np.radians(float(row[_COL_LON_DEG])),
            lat_gc,
            float(row[_COL_VEL]),
            np.radians(float(row[_COL_FPA_DEG])),
            np.radians(float(row[_COL_HEAD_DEG])),
            np.radians(float(row[_COL_BANK_DEG])),
            float(row[_COL_HEAT_LOAD_KJ]),
        ],
        dtype=np.float64,
    )


def _sample_trajectory(traj: FloatArray, model: CpagModel, source: str, dt_sample: float, dispersed: bool) -> list[ReplanCase]:
    """Replan states every ~dt_sample seconds while inside the atmosphere."""
    cases: list[ReplanCase] = []
    if traj.ndim != 2 or traj.shape[0] < 3:
        return cases
    t = traj[:, _COL_TIME_S]
    next_t = 30.0  # skip the shaper's initial bank transient
    for i in range(traj.shape[0] - 2):
        if t[i] < next_t:
            continue
        alt_km = float(traj[i, _COL_ALT_KM])
        if alt_km >= model.exit_alt / 1e3 or alt_km <= 5.0:
            continue
        factor = float(traj[i, _COL_NAV_DENSITY_RATIO]) if dispersed else 1.0
        if not (np.isfinite(factor) and 0.05 < factor < 20.0):
            factor = 1.0
        cases.append(ReplanCase(source=source, time_s=float(t[i]), x0=state_from_trajectory_row(traj[i], model), density_factor=factor))
        next_t = t[i] + dt_sample
    return cases


def harvest_constant_bank_states(toml_path: str, model: CpagModel, banks_deg: list[float], dt_sample: float = 60.0) -> list[ReplanCase]:
    """Undispersed constant-bank sweep (corridor sentinel range) via the Rust sim."""
    import aerocapture_rs  # noqa: PLC0415

    from aerocapture.training.reference import nominal_flight_overrides  # noqa: PLC0415
    from aerocapture.training.toml_utils import load_toml_with_bases  # noqa: PLC0415

    mc_config = load_toml_with_bases(Path(toml_path)).get("monte_carlo", {})
    overrides_list: list[dict[str, object]] = []
    for bank in banks_deg:
        ov = nominal_flight_overrides({}, "piecewise_constant", mc_config)
        ov["guidance.piecewise_constant.n_segments"] = 1
        ov["guidance.piecewise_constant.bank_angle_0"] = float(bank)
        ov["monte_carlo.seed"] = 42  # the off-level overrides materialize [monte_carlo]; seed is required
        overrides_list.append(ov)
    batch = aerocapture_rs.run_batch(toml_path=toml_path, overrides_list=overrides_list, include_trajectories=True, sim_timeout_secs=120.0)
    cases: list[ReplanCase] = []
    for i, bank in enumerate(banks_deg):
        traj = np.asarray(batch.trajectories[i])
        cases.extend(_sample_trajectory(traj, model, f"bank_{bank:05.1f}", dt_sample, dispersed=False))
    return cases


def harvest_dispersed_states(toml_path: str, model: CpagModel, n_sims: int, dt_sample: float = 60.0, seed: int = 4242) -> list[ReplanCase]:
    """Dispersed FTC-guided MC batch flying the deployed GA optimum."""
    import aerocapture_rs  # noqa: PLC0415

    from aerocapture.training.param_spaces import route_scaffolding_param  # noqa: PLC0415

    overrides: dict[str, object] = {"simulation.n_sims": n_sims, "monte_carlo.seed": seed}
    best_params = Path("training_output/ftc/best_params.json")
    if best_params.exists():
        for key, value in json.loads(best_params.read_text()).items():
            path, coerced = route_scaffolding_param(key, value, "ftc")
            overrides[path] = coerced
    batch = aerocapture_rs.run_mc(toml_path=toml_path, overrides=overrides, include_trajectories=True, sim_timeout_secs=120.0)
    cases: list[ReplanCase] = []
    for i in range(len(batch.trajectories)):
        traj = np.asarray(batch.trajectories[i])
        cases.extend(_sample_trajectory(traj, model, f"mc_{i:03d}", dt_sample, dispersed=True))
    return cases


def classify_reachability(x0: FloatArray, model: CpagModel, cfg: ScpConfig) -> tuple[str, float, float]:
    """Ground-truth apoapsis bracket from the two max-authority shoots.

    Full lift-up (sigma -> 0) gives the achievable maximum; full lift-down
    (sigma -> +-180 shortest way) the minimum. Returns (class, apo_max_err_m,
    apo_min_err_m): 'unrecoverable' (even full lift-up crashes), 'under_reach'
    (max below target - tol), 'over_reach' (min above target + tol — post-bounce
    energy excess the thin upper atmosphere cannot shed), or 'reachable'.
    """
    rec = shoot_profile(x0, roll_to_bank_profile(x0, 0.0, cfg), model, cfg)
    if rec.event == "crash":
        return "unrecoverable", float("-inf"), float("-inf")
    apo_max_err = float(apoapsis_radius(rec.x_nodes[-1], model.planet)) - model.target_apoapsis_radius
    if apo_max_err < -cfg.tol_apo_m:
        return "under_reach", apo_max_err, apo_max_err

    sig_down = float(np.copysign(np.pi, x0[ISIGMA]) if x0[ISIGMA] != 0.0 else np.pi)
    low = shoot_profile(x0, roll_to_bank_profile(x0, sig_down, cfg), model, cfg)
    apo_min = 0.0 if low.event == "crash" else float(apoapsis_radius(low.x_nodes[-1], model.planet))
    apo_min_err = apo_min - model.target_apoapsis_radius
    if apo_min_err > cfg.tol_apo_m:
        return "over_reach", apo_max_err, apo_min_err
    return "reachable", apo_max_err, apo_min_err


def run_replan_study(cases: list[ReplanCase], model: CpagModel, cfg: ScpConfig) -> list[dict[str, Any]]:
    """One cold SCP replan per case; returns JSON-ready per-case records."""
    records: list[dict[str, Any]] = []
    for case in cases:
        m = model.with_density_factor(case.density_factor)
        reach, apo_max_err, apo_min_err = classify_reachability(case.x0, m, cfg)
        r: ScpResult = scp_replan(case.x0, m, cfg)
        records.append(
            {
                "source": case.source,
                "time_s": case.time_s,
                "alt_km": (float(case.x0[0]) - model.planet.req) / 1e3,
                "sigma0_deg": float(np.degrees(case.x0[ISIGMA])),
                "heat_load0_kj": float(case.x0[IQ]),
                "density_factor": case.density_factor,
                "reachability": reach,
                "apo_max_err_km": apo_max_err / 1e3 if np.isfinite(apo_max_err) else None,
                "apo_min_err_km": apo_min_err / 1e3 if np.isfinite(apo_min_err) else None,
                "converged": r.converged,
                "feasible": r.feasible,
                "n_iters": r.n_iters,
                "n_solves": len(r.solves),
                "apo_error_km": r.apo_error_m / 1e3 if np.isfinite(r.apo_error_m) else None,
                "eps_mj": r.eps_mj,
                "inc_error_deg": r.inc_error_deg,
                "event": r.event,
                "path_peaks": r.path_peaks,
                "heat_load_frac": r.heat_load_frac,
                "wall_time_s": r.wall_time,
                "solver_iters": [s.iterations for s in r.solves],
                "solver_time_s": [s.solve_time for s in r.solves],
                "qp_vars": r.solves[-1].n_vars if r.solves else 0,
                "qp_rows": r.solves[-1].n_rows if r.solves else 0,
            }
        )
        rec = records[-1]
        gap = None
        if rec["apo_error_km"] is not None:
            if reach == "under_reach" and rec["apo_max_err_km"] is not None:
                gap = abs(rec["apo_error_km"]) - abs(rec["apo_max_err_km"])
            elif reach == "over_reach" and rec["apo_min_err_km"] is not None:
                gap = rec["apo_error_km"] - rec["apo_min_err_km"]
        rec["least_bad_gap_km"] = gap
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate convergence stats, split by reachability class."""

    def _agg(recs: list[dict[str, Any]]) -> dict[str, Any]:
        if not recs:
            return {"n": 0}
        conv = [r for r in recs if r["converged"]]
        feas = [r for r in recs if r["feasible"]]
        apo = np.array([abs(r["apo_error_km"]) for r in recs if r["apo_error_km"] is not None])
        iters = np.array([r["n_iters"] for r in recs], dtype=np.float64)
        solver_t = np.concatenate([np.asarray(r["solver_time_s"], dtype=np.float64) for r in recs if r["solver_time_s"]])
        gaps = np.array([r["least_bad_gap_km"] for r in recs if r.get("least_bad_gap_km") is not None])
        return {
            "n": len(recs),
            "converged_rate": len(conv) / len(recs),
            "feasible_rate": len(feas) / len(recs),
            "iters_p50": float(np.percentile(iters, 50)),
            "iters_p95": float(np.percentile(iters, 95)),
            "apo_err_km_p50": float(np.percentile(apo, 50)) if apo.size else None,
            "apo_err_km_p95": float(np.percentile(apo, 95)) if apo.size else None,
            "least_bad_gap_km_p50": float(np.percentile(gaps, 50)) if gaps.size else None,
            "least_bad_gap_km_p95": float(np.percentile(gaps, 95)) if gaps.size else None,
            "solver_time_ms_p50": float(np.percentile(solver_t, 50) * 1e3) if solver_t.size else None,
            "solver_time_ms_p95": float(np.percentile(solver_t, 95) * 1e3) if solver_t.size else None,
        }

    by_class: dict[str, Any] = {}
    for cls in ("reachable", "under_reach", "over_reach", "unrecoverable"):
        by_class[cls] = _agg([r for r in records if r["reachability"] == cls])
    return {"all": _agg(records), "by_reachability": by_class}


def run_c0_studies(
    toml_path: str = "configs/nominal/msr_aller_ftc_nominal.toml",
    dispersed_toml: str = "configs/training/msr_aller_ftc_train.toml",
    out_dir: str | Path = "training_output/cpag_c0",
    n_dispersed: int = 40,
    cfg: ScpConfig | None = None,
) -> dict[str, Any]:
    """Full Stage C0 study battery. Returns the summary dict (also written to disk)."""
    cfg = cfg or ScpConfig(max_iters=30, horizon_max=1200.0)
    model = load_model(toml_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x0 = entry_state(toml_path)
    nominal = run_replan_study([ReplanCase(source="entry", time_s=0.0, x0=x0, density_factor=1.0)], model, cfg)

    banks = [float(b) for b in np.arange(0.0, 181.0, 18.0)]
    sweep_cases = harvest_constant_bank_states(toml_path, model, banks)
    sweep = run_replan_study(sweep_cases, model, cfg)

    disp_cases = harvest_dispersed_states(dispersed_toml, model, n_sims=n_dispersed)
    disp = run_replan_study(disp_cases, model, cfg)

    result = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "nominal": nominal,
        "sweep_summary": summarize(sweep),
        "dispersed_summary": summarize(disp),
        "n_sweep_cases": len(sweep_cases),
        "n_dispersed_cases": len(disp_cases),
    }
    (out / "study_records.json").write_text(json.dumps({"sweep": sweep, "dispersed": disp, "nominal": nominal}, indent=1))
    (out / "study_summary.json").write_text(json.dumps(result, indent=1))
    return result
