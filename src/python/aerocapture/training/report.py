"""PDF report orchestrator — generates Typst-compiled PDF reports from training data.

Loads JSONL training logs, optionally runs final MC evaluation via PyO3,
generates all SVG charts, writes metadata/summary JSON, and invokes
``typst compile`` to produce a PDF.

Usage:
    uv run python -m aerocapture.training.report training_output/equilibrium_glide/
    uv run python -m aerocapture.training.report training_output/equilibrium_glide/ --toml configs/training/msr_aller_eqglide_train.toml
    uv run python -m aerocapture.training.report --compare training_output/
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from aerocapture.training import charts
from aerocapture.training.metrics import convergence_speed, stagnation_count

# ---------------------------------------------------------------------------
# Typst template directory — src/typst/ relative to this file
# report.py lives at src/python/aerocapture/training/report.py
# typst dir lives at src/typst/
# ---------------------------------------------------------------------------
_TYPST_DIR = Path(__file__).resolve().parent.parent.parent.parent / "typst"

# Percentiles for summary table
_PERCENTILES = [5, 25, 50, 75, 95]


# ---------------------------------------------------------------------------
# Typst availability check
# ---------------------------------------------------------------------------
def _load_nn_scaffolding_overrides(scheme_dir: Path, optimized_toml: Path) -> dict[str, object]:
    """Build dot-path overrides from `best_params.json` (NN+optimize_scaffolding deploys).

    Non-NN schemes bake params into `optimized_<scheme>.toml`, so when that file
    exists we return empty (the optimized TOML already carries everything).
    NN schemes do NOT write an optimized TOML — they write `best_model.json`
    plus `best_params.json` (scaffolding). When `optimize_scaffolding=false`
    is the only knob, `best_params.json` is absent and we return empty.
    """
    if optimized_toml.exists():
        return {}
    scaff_path = scheme_dir / "best_params.json"
    if not scaff_path.exists():
        return {}
    scaff_params: dict[str, object] = json.loads(scaff_path.read_text())
    overrides: dict[str, object] = {}
    for key, value in scaff_params.items():
        if key.startswith("lateral."):
            bare = key.removeprefix("lateral.")
            if bare == "max_reversals":
                value = int(round(float(value)))  # type: ignore[arg-type]
            overrides[f"guidance.lateral.{bare}"] = value
        elif key.startswith("exit."):
            overrides[f"guidance.ftc.{key.removeprefix('exit.')}"] = value
        elif key.startswith("nav."):
            overrides[f"navigation.{key.removeprefix('nav.')}"] = value
        elif key.startswith("thermal."):
            overrides[f"guidance.thermal_limiter.{key.removeprefix('thermal.')}"] = value
        elif key.startswith("shaping."):
            overrides[f"guidance.command_shaping.{key.removeprefix('shaping.')}"] = value
            overrides["guidance.command_shaping.enabled"] = True
    return overrides


def _check_typst() -> bool:
    """Return True if the ``typst`` CLI is available on PATH."""
    return shutil.which("typst") is not None


# ---------------------------------------------------------------------------
# JSONL loading (preserved from original report.py)
# ---------------------------------------------------------------------------
def load_run_data(scheme_dir: Path) -> tuple[list[dict], list[int]]:
    """Load all JSONL records from a scheme directory, sorted by generation.

    Returns:
        Tuple of (records, resume_generations) where resume_generations
        contains the first generation number from each JSONL file after
        the first (i.e., where training was resumed).
    """
    file_records: list[list[dict]] = []
    for jsonl_file in sorted(scheme_dir.glob("*.jsonl")):
        file_recs: list[dict] = []
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    file_recs.append(json.loads(line))
        if file_recs:
            file_records.append(file_recs)

    records: list[dict] = []
    for file_recs in file_records:
        records.extend(file_recs)
    records.sort(key=lambda r: r["generation"])

    # Deduplicate: last-writer-wins for same generation (safety net for legacy logs)
    seen: dict[int, int] = {}
    deduped: list[dict] = []
    for r in records:
        gen = r["generation"]
        if gen in seen:
            deduped[seen[gen]] = r
        else:
            seen[gen] = len(deduped)
            deduped.append(r)

    # Detect resume points: first generation of each file after the first
    resume_gens: list[int] = []
    for file_recs in file_records[1:]:
        if file_recs:
            first_gen = min(r["generation"] for r in file_recs)
            if first_gen not in resume_gens:
                resume_gens.append(first_gen)
    resume_gens.sort()

    return deduped, resume_gens


# ---------------------------------------------------------------------------
# Final MC evaluation via PyO3
# ---------------------------------------------------------------------------
def run_final_evaluation(
    toml_path: Path,
    scheme_dir: Path,
    n_sims: int = 1000,
    sim_timeout_secs: float | None = None,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None:
    """Run final MC evaluation using reserved seeds that never overlap with training or validation.

    Uses optimized TOML if it exists (``scheme_dir / f"optimized_{scheme_dir.name}.toml"``),
    otherwise the provided *toml_path*.  Seeds are generated deterministically from the
    TOML's ``[monte_carlo].seed`` via :func:`make_reserved_seeds` with
    ``FINAL_EVAL_SEED_OFFSET``, guaranteeing disjointness from training and validation.

    Returns ``(final_records, trajectories, dispersions)`` or None on failure.
    """
    try:
        import aerocapture_rs  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        print("PyO3 bindings not available -- skipping final evaluation")
        return None

    from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.toml_utils import load_toml_with_bases

    optimized = scheme_dir / f"optimized_{scheme_dir.name}.toml"
    eval_toml = optimized if optimized.exists() else toml_path

    toml_data = load_toml_with_bases(eval_toml)
    base_mc_seed = toml_data.get("monte_carlo", {}).get("seed", 42)
    reserved_seeds = make_reserved_seeds(base_mc_seed, FINAL_EVAL_SEED_OFFSET, n_sims)

    scaffolding_overrides = _load_nn_scaffolding_overrides(scheme_dir, optimized)
    if scaffolding_overrides:
        print(f"  Using optimized NN scaffolding from {scheme_dir / 'best_params.json'}")

    try:
        base_overrides: dict[str, object] = {"simulation.n_sims": 1, **scaffolding_overrides}
        overrides_list = [{**base_overrides, "monte_carlo.seed": s} for s in reserved_seeds]
        results = aerocapture_rs.run_batch(
            toml_path=str(eval_toml.resolve()),
            overrides_list=overrides_list,
            include_trajectories=True,
            sim_timeout_secs=sim_timeout_secs,
        )
        return (results.final_records, results.trajectories, results.dispersions)
    except Exception:
        import traceback

        traceback.print_exc()
        return None


def print_eval_summary(final_records: npt.NDArray[np.float64], n_sims: int, cost_kwargs: dict[str, Any] | None = None) -> None:
    """Print a human-readable summary of the final MC evaluation to stdout."""
    from aerocapture.training.evaluate import compute_cost

    ecc = final_records[:, charts._FR_ECC]
    ifinal = final_records[:, charts._FR_IFINAL]
    captured = (ifinal == 3) & (ecc < 1.0)  # only AtmosphereExit on bound orbit
    n_captured = int(np.sum(captured))
    cap = final_records[captured]

    # Objective cost (over all sims)
    per_sim_costs = np.array([compute_cost(final_records[i : i + 1], **(cost_kwargs or {})) for i in range(len(final_records))])
    rms_cost = float(np.sqrt(np.mean(per_sim_costs**2)))

    print(f"\n  Final evaluation ({n_sims} sims):")
    print(f"    Objective cost:     p50={np.median(per_sim_costs):.1f}  p95={np.percentile(per_sim_costs, 95):.1f}  RMS={rms_cost:.1f}")
    print(f"    Capture rate:       {n_captured}/{n_sims} ({100 * n_captured / n_sims:.1f}%)")

    if n_captured > 0:
        dv = np.clip(cap[:, charts._FR_DV_TOTAL], charts.DV_FLOOR, charts.DV_CAP)
        apo = cap[:, charts._FR_APO_ERR]
        peri = cap[:, charts._FR_PERI_ERR]
        incl = cap[:, charts._FR_INCL_ERR]
        print(f"    Delta-V (m/s):      p50={np.median(dv):.1f}  p95={np.percentile(dv, 95):.1f}  mean={np.mean(dv):.1f}")
        print(f"    Apoapsis err (km):  p50={np.median(apo):.1f}  p95={np.percentile(apo, 95):.1f}  mean={np.mean(apo):.1f}")
        print(f"    Periapsis err (km): p50={np.median(peri):.1f}  p95={np.percentile(peri, 95):.1f}  mean={np.mean(peri):.1f}")
        print(f"    Inclin. err (deg):  p50={np.median(incl):.2f}  p95={np.percentile(incl, 95):.2f}  mean={np.mean(incl):.2f}")

    # Constraint violation stats (over ALL sims)
    all_q = final_records[:, charts._FR_MAX_HEAT_FLUX]
    all_g = final_records[:, charts._FR_MAX_G_LOAD]
    all_hl = final_records[:, charts._FR_INTEGRATED_FLUX] * 1e3  # MJ/m2 -> kJ/m2
    _q = cost_kwargs.get("heat_flux_limit") if cost_kwargs else None
    _g = cost_kwargs.get("g_load_limit") if cost_kwargs else None
    _hl = cost_kwargs.get("heat_load_limit") if cost_kwargs else None
    q_limit = float(_q) if isinstance(_q, (int, float)) else None
    g_limit = float(_g) if isinstance(_g, (int, float)) else None
    hl_limit = float(_hl) if isinstance(_hl, (int, float)) else None
    q_viol = f"  {np.mean(all_q > q_limit) * 100:.1f}% > {q_limit:.0f}" if q_limit else ""
    g_viol = f"  {np.mean(all_g > g_limit) * 100:.1f}% > {g_limit:.1f}" if g_limit else ""
    hl_viol = f"  {np.mean(all_hl > hl_limit) * 100:.1f}% > {hl_limit:.0f}" if hl_limit else ""
    print(f"    Heat flux (kW/m2):  p50={np.median(all_q):.1f}  p95={np.percentile(all_q, 95):.1f}  max={np.max(all_q):.1f}{q_viol}")
    print(f"    G-load (g):         p50={np.median(all_g):.2f}  p95={np.percentile(all_g, 95):.2f}  max={np.max(all_g):.2f}{g_viol}")
    print(f"    Heat load (kJ/m2):  p50={np.median(all_hl):.0f}  p95={np.percentile(all_hl, 95):.0f}  max={np.max(all_hl):.0f}{hl_viol}")


# ---------------------------------------------------------------------------
# TOML metadata reader
# ---------------------------------------------------------------------------
def _read_mission_name(toml_path: Path) -> str:
    """Read planet name and mission type from TOML, returning a human label."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    data = load_toml_with_bases(toml_path)
    planet: str = data.get("planet", {}).get("name", "Unknown")
    mission_type: str = data.get("mission", {}).get("type", "")
    if mission_type:
        return f"{planet} — {mission_type}"
    return planet


def _read_constraint_limits(toml_path: Path) -> tuple[float | None, float | None]:
    """Read heat flux and g-load limits from TOML [flight.constraints] section."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    data = load_toml_with_bases(toml_path)
    constraints = data.get("flight", {}).get("constraints", {})
    heat_flux: float | None = constraints.get("max_heat_flux")
    g_load: float | None = constraints.get("max_load_factor")
    return heat_flux, g_load


def read_cost_kwargs(toml_path: Path) -> dict[str, Any]:
    """Read cost function parameters from TOML for objective cost computation."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    data = load_toml_with_bases(toml_path)
    cost_cfg = data.get("cost_function", {})
    constraints = data.get("flight", {}).get("constraints", {})
    return {
        "dv_threshold": float(cost_cfg.get("dv_threshold", 1000.0)),
        "g_load_limit": float(constraints.get("max_load_factor", 15.0)),
        "heat_flux_limit": float(constraints.get("max_heat_flux", 200.0)),
        "heat_load_limit": float(constraints.get("max_heat_load", 25000.0)),
        "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
        "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
        "heat_load_weight": float(cost_cfg.get("heat_load_weight", 1000.0)),
        "cost_transform": str(cost_cfg.get("cost_transform", "linear")),
    }


# ---------------------------------------------------------------------------
# Metadata builder (for cover page)
# ---------------------------------------------------------------------------
def _build_metadata(
    records: list[dict],
    scheme_dir: Path,
    n_sims: int,
    has_trajectories: bool,
    has_final_eval: bool,
    toml_path: Path | None,
    has_cost_distribution: bool,
) -> dict:
    """Build metadata dict for the Typst cover page."""
    scheme = records[0].get("scheme", scheme_dir.name) if records else scheme_dir.name
    best_cost = records[-1]["best_cost"] if records else 0.0
    capture_rate = records[-1].get("capture_rate", 0.0) if records else 0.0
    config_hash = records[0].get("config_hash", "N/A") if records else "N/A"

    cost_history = [r["best_cost"] for r in records]
    conv_speed = convergence_speed(cost_history) if cost_history else 0
    stag = stagnation_count(cost_history) if cost_history else 0

    mission = ""
    if toml_path is not None and toml_path.exists():
        try:
            mission = _read_mission_name(toml_path)
        except Exception:
            mission = ""

    return {
        "scheme": scheme,
        "mission": mission or "N/A",
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "best_cost": f"{best_cost:.4e}",
        "capture_rate": f"{capture_rate * 100:.0f}%",
        "total_generations": str(len(records)),
        "convergence_speed": str(conv_speed),
        "stagnation": str(stag),
        "n_sims": str(n_sims),
        "config_hash": config_hash,
        "has_trajectories": has_trajectories,
        "has_final_eval": has_final_eval,
        "has_cost_distribution": has_cost_distribution,
    }


# ---------------------------------------------------------------------------
# Summary table builder
# ---------------------------------------------------------------------------
def _build_summary_table(
    final_records: npt.NDArray[np.float64],
    heat_flux_limit: float | None = None,
    g_load_limit: float | None = None,
    cost_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Build the performance summary table dict for Typst.

    Returns dict with ``rows`` key — each row is
    [name, mean, std, min, p5, p25, p50, p75, p95, max].
    Only captured trajectories (eccentricity < 1.0) are included.
    Adds constraint violation rates when limits are provided.
    """
    n_total = len(final_records)
    ecc = final_records[:, charts._FR_ECC]
    ifinal = final_records[:, charts._FR_IFINAL]
    captured = (ifinal == 3) & (ecc < 1.0)  # only AtmosphereExit on bound orbit
    cap_data = final_records[captured]

    if len(cap_data) == 0:
        return {"rows": []}

    def _row(name: str, values: npt.NDArray[np.float64]) -> list[str]:
        pcts = np.percentile(values, _PERCENTILES)
        return [
            name,
            f"{np.mean(values):.2f}",
            f"{np.std(values):.2f}",
            f"{np.min(values):.2f}",
            *[f"{p:.2f}" for p in pcts],
            f"{np.max(values):.2f}",
        ]

    dv_total = np.clip(cap_data[:, charts._FR_DV_TOTAL], charts.DV_FLOOR, charts.DV_CAP)

    dv1_abs = np.abs(cap_data[:, charts._FR_DV1])
    dv2_abs = np.abs(cap_data[:, charts._FR_DV2])
    dv3_abs = np.abs(cap_data[:, charts._FR_DV3])

    rows = [
        _row("Max G-load (g)", cap_data[:, charts._FR_MAX_G_LOAD]),
        _row("Max heat flux (kW/m2)", cap_data[:, charts._FR_MAX_HEAT_FLUX]),
        _row("Bank consumption (deg)", cap_data[:, charts._FR_BANK_CONSUMPTION]),
        _row("Periapsis error (km)", cap_data[:, charts._FR_PERI_ERR]),
        _row("Apoapsis error (km)", cap_data[:, charts._FR_APO_ERR]),
        _row("Inclination error (deg)", cap_data[:, charts._FR_INCL_ERR]),
        _row("|DV1| periapsis (m/s)", dv1_abs),
        _row("|DV2| apoapsis (m/s)", dv2_abs),
        _row("|DV3| inclination (m/s)", dv3_abs),
        _row("Total DV (m/s)", dv_total),
    ]

    # Objective cost (over ALL sims)
    from aerocapture.training.evaluate import compute_cost

    per_sim_costs = np.array([compute_cost(final_records[i : i + 1], **(cost_kwargs or {})) for i in range(n_total)])
    rms_cost = float(np.sqrt(np.mean(per_sim_costs**2)))

    # Constraint statistics (over ALL sims, not just captured)
    all_g = final_records[:, charts._FR_MAX_G_LOAD]
    all_q = final_records[:, charts._FR_MAX_HEAT_FLUX]

    n_captured = int(np.sum(captured))
    capture_pct = 100 * n_captured / n_total if n_total > 0 else 0.0

    violation_rows: list[list[str]] = []
    violation_rows.append(_row(f"Objective cost, all sims — RMS={rms_cost:.1f}", per_sim_costs))
    if g_load_limit is not None:
        g_exceed = float(np.mean(all_g > g_load_limit) * 100)
        violation_rows.append(_row(f"G-load, all sims (g) — {g_exceed:.1f}% > {g_load_limit:.1f}", all_g))
    if heat_flux_limit is not None:
        q_exceed = float(np.mean(all_q > heat_flux_limit) * 100)
        violation_rows.append(_row(f"Heat flux, all sims (kW/m2) — {q_exceed:.1f}% > {heat_flux_limit:.0f}", all_q))
    # Capture rate: single value, fill remaining columns with empty strings
    cr_label = f"Capture rate: {capture_pct:.1f}% ({n_captured}/{n_total})"
    violation_rows.append([cr_label, "", "", "", "", "", "", "", "", ""])

    return {"rows": rows, "violation_rows": violation_rows}


# ---------------------------------------------------------------------------
# Chart generation helpers
# ---------------------------------------------------------------------------
def _generate_training_charts(
    records: list[dict],
    resume_gens: list[int],
    out_dir: Path,
) -> bool:
    """Generate Part 1 (training convergence) SVG charts. Returns has_cost_distribution."""
    charts.chart_convergence(records, out_dir / "convergence.svg", resume_gens=resume_gens)
    charts.chart_diversity_cost(records, out_dir / "diversity_cost.svg", resume_gens=resume_gens)
    has_cost_distribution = charts.chart_cost_distribution(records, out_dir / "cost_distribution.svg")
    charts.chart_parameter_evolution(records, out_dir / "parameter_evolution.svg", resume_gens=resume_gens)

    return has_cost_distribution


def _load_corridor_data(scheme_dir: Path) -> dict[str, Any] | None:
    """Load corridor boundaries .npz from the mission-level training output directory."""
    from aerocapture.training.corridor import load_corridor

    # corridor_boundaries.npz lives one level up (mission directory, e.g. training_output/)
    # or in the piecewise_constant sibling directory
    candidates = [
        scheme_dir.parent / "corridor_boundaries.npz",
        scheme_dir.parent / "piecewise_constant" / "corridor_boundaries.npz",
    ]
    for path in candidates:
        data = load_corridor(path)
        if data is not None:
            return data
    return None


def _run_undispersed_nominal(toml_path: Path, scheme_dir: Path, sim_timeout_secs: float | None = None) -> npt.NDArray[np.float64] | None:
    """Run a single undispersed simulation to get the nominal trajectory."""
    try:
        import aerocapture_rs  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        return None

    optimized = scheme_dir / f"optimized_{scheme_dir.name}.toml"
    eval_toml = optimized if optimized.exists() else toml_path

    # NN+optimize_scaffolding schemes write best_params.json sibling to best_model.json;
    # without loading it here, the nominal overlay would use TOML-default scaffolding
    # while the dispersed MC corridor uses the GA-tuned values — visually inconsistent.
    scaffolding_overrides = _load_nn_scaffolding_overrides(scheme_dir, optimized)

    overrides: dict[str, object] = {
        "simulation.n_sims": 1,
        "monte_carlo.initial_state.level": "off",
        "monte_carlo.atmosphere.level": "off",
        "monte_carlo.aerodynamics.level": "off",
        "monte_carlo.navigation.level": "off",
        "monte_carlo.mass.level": "off",
        **scaffolding_overrides,
    }

    try:
        results = aerocapture_rs.run_mc(
            toml_path=str(eval_toml.resolve()),
            overrides=overrides,
            include_trajectories=True,
            sim_timeout_secs=sim_timeout_secs,
        )
        if results.trajectories:
            traj: npt.NDArray[np.float64] = results.trajectories[0]
            return traj
    except Exception as exc:
        print(f"Warning: undispersed nominal run failed: {exc}")
    return None


def _find_best_trajectory(
    final_records: npt.NDArray[np.float64],
    trajectories: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64] | None:
    """Find the trajectory with the lowest total DV among captured cases."""
    ecc = final_records[:, charts._FR_ECC]
    ifinal = final_records[:, charts._FR_IFINAL]
    captured_indices = np.where((ifinal == 3) & (ecc < 1.0))[0]
    if len(captured_indices) == 0:
        return None
    dv = final_records[captured_indices, charts._FR_DV_TOTAL]
    best_idx = captured_indices[int(np.argmin(dv))]
    result: npt.NDArray[np.float64] = trajectories[best_idx]
    return result


def _generate_sensitivity_charts(sensitivity_dir: Path, out_dir: Path) -> dict[str, bool]:
    """Generate Part 3 (sensitivity) SVG charts from pre-computed results."""
    results_path = sensitivity_dir / "sensitivity_results.json"
    if not results_path.exists():
        return {"has_sensitivity": False, "has_morris": False, "has_sobol": False, "has_sobol_heatmap": False}

    results = json.loads(results_path.read_text())
    flags: dict[str, bool] = {"has_sensitivity": True, "has_morris": False, "has_sobol": False, "has_sobol_heatmap": False}

    if "morris" in results:
        morris = results["morris"]
        charts.chart_morris_scatter(morris, out_dir / "morris_scatter.svg")
        flags["has_morris"] = True

        # Write ranked table for Typst
        names = morris["names"]
        mu_star = morris["mu_star"]
        sigma = morris["sigma"]
        mu_star_conf = morris.get("mu_star_conf", [0.0] * len(names))
        order = sorted(range(len(names)), key=lambda i: mu_star[i], reverse=True)
        rows = []
        for rank, i in enumerate(order, 1):
            rows.append([str(rank), names[i], f"{mu_star[i]:.1f}", f"{sigma[i]:.1f}", f"{mu_star_conf[i]:.1f}"])
        (out_dir / "morris_table.json").write_text(json.dumps({"rows": rows}, indent=2))

    if "sobol" in results:
        charts.chart_sobol_bars(results["sobol"], out_dir / "sobol_bars.svg")
        flags["has_sobol"] = True
        if "S2" in results["sobol"]:
            charts.chart_sobol_heatmap(results["sobol"], out_dir / "sobol_heatmap.svg")
            flags["has_sobol_heatmap"] = True

    return flags


def _generate_trajectory_charts(
    final_records: npt.NDArray[np.float64],
    trajectories: list[npt.NDArray[np.float64]],
    dispersions: npt.NDArray[np.float64],
    out_dir: Path,
    scheme_dir: Path | None = None,
    toml_path: Path | None = None,
    sim_timeout_secs: float | None = None,
    cost_kwargs: dict[str, Any] | None = None,
) -> None:
    """Generate Part 2 (mission performance) SVG charts from final eval data."""
    # Load constraint limits and classify trajectories
    heat_flux_limit, g_load_limit = _read_constraint_limits(toml_path) if toml_path is not None else (None, None)
    traj_class = charts.classify_trajectories(final_records, heat_flux_limit=heat_flux_limit, g_load_limit=g_load_limit)

    # Load corridor boundaries and nominal trajectories
    corridor_data = _load_corridor_data(scheme_dir) if scheme_dir is not None else None
    undispersed = (
        _run_undispersed_nominal(toml_path, scheme_dir, sim_timeout_secs=sim_timeout_secs) if toml_path is not None and scheme_dir is not None else None
    )
    best_traj = _find_best_trajectory(final_records, trajectories)

    # Corridor panels
    nominal_kwargs: dict[str, Any] = {"undispersed_nominal": undispersed, "best_nominal": best_traj}
    charts.chart_corridor_pdyn(
        trajectories,
        traj_class,
        out_dir / "corridor_pdyn.svg",
        corridor_data=corridor_data,
        **nominal_kwargs,
    )
    charts.chart_corridor_inclination(trajectories, traj_class, out_dir / "corridor_inclination.svg", **nominal_kwargs)
    charts.chart_corridor_bank(trajectories, traj_class, out_dir / "corridor_bank.svg", **nominal_kwargs)

    # Time-domain panels
    charts.chart_altitude_time(trajectories, traj_class, out_dir / "altitude_time.svg", **nominal_kwargs)
    charts.chart_heat_flux_time(trajectories, traj_class, out_dir / "heat_flux_time.svg", limit_kw_m2=heat_flux_limit, **nominal_kwargs)
    charts.chart_gload_time(trajectories, traj_class, out_dir / "gload_time.svg", limit_g=g_load_limit, **nominal_kwargs)
    charts.chart_bank_angle_time(trajectories, traj_class, out_dir / "bank_angle_time.svg", **nominal_kwargs)
    charts.chart_nav_density_ratio(trajectories, traj_class, out_dir / "nav_density_ratio.svg", **nominal_kwargs)

    # Distribution panels
    charts.chart_cost_objective(final_records, out_dir / "cost_objective.svg", **(cost_kwargs or {}))
    charts.chart_dv_distribution(final_records, out_dir / "dv_distribution.svg")
    charts.chart_dv_individual_burns(final_records, out_dir / "dv_individual_burns.svg")

    # Entry/exit conditions
    charts.chart_entry_conditions(trajectories, traj_class, out_dir / "entry_conditions.svg")
    charts.chart_exit_conditions(final_records, out_dir / "exit_conditions.svg")

    # Dispersion grid
    charts.chart_dispersion_grid(final_records, dispersions, out_dir / "dispersion_grid.svg", traj_class=traj_class)


# ---------------------------------------------------------------------------
# Main entry point: single-scheme report
# ---------------------------------------------------------------------------
def generate_report(
    scheme_dir: Path,
    toml_path: Path | None = None,
    skip_final_eval: bool = False,
    keep_artifacts: bool = False,
    n_sims_override: int | None = None,
    sim_timeout_secs: float | None = None,
    sensitivity: bool = False,
) -> Path | None:
    """Generate a PDF training report for a single guidance scheme.

    Loads JSONL training data, optionally runs final MC evaluation,
    generates SVG charts, writes JSON metadata, and compiles PDF via Typst.

    Returns the path to the generated PDF, or None if no data / Typst unavailable.
    """
    records, resume_gens = load_run_data(scheme_dir)
    if not records:
        print(f"No JSONL data found in {scheme_dir}")
        return None

    n_sims = n_sims_override if n_sims_override is not None else 1000

    # Create temp directory for artifacts
    tmp_dir = Path(tempfile.mkdtemp(prefix="aerocapture_report_"))

    try:
        # Part 1: training convergence charts
        has_cost_distribution = _generate_training_charts(records, resume_gens, tmp_dir)

        # Read cost function config from TOML (needed for cost stats)
        cost_kwargs = read_cost_kwargs(toml_path) if toml_path is not None else None

        # Part 2: final evaluation (optional)
        has_trajectories = False
        final_records = None
        if not skip_final_eval and toml_path is not None:
            print(f"\nRunning {n_sims}-sim final evaluation...")
            eval_result = run_final_evaluation(toml_path, scheme_dir, n_sims=n_sims, sim_timeout_secs=sim_timeout_secs)
            if eval_result is not None:
                final_records_arr, trajectories, dispersions = eval_result
                has_trajectories = True
                print_eval_summary(final_records_arr, n_sims, cost_kwargs=cost_kwargs)
                _generate_trajectory_charts(
                    final_records_arr,
                    trajectories,
                    dispersions,
                    tmp_dir,
                    scheme_dir=scheme_dir,
                    toml_path=toml_path,
                    sim_timeout_secs=sim_timeout_secs,
                    cost_kwargs=cost_kwargs,
                )
                final_records = final_records_arr

                # Write Parquet output for analysis
                try:
                    from aerocapture.training.parquet_output import write_parquet
                    from aerocapture.training.toml_utils import load_toml_with_bases

                    resolved_config = load_toml_with_bases(toml_path)
                    parquet_path = scheme_dir / "final_eval.parquet"
                    write_parquet(parquet_path, final_records_arr, dispersions, resolved_config, toml_path=str(toml_path))
                    print(f"Parquet output: {parquet_path}")
                except ImportError:
                    pass  # pyarrow not installed
                except Exception as exc:  # noqa: BLE001
                    print(f"Warning: Parquet write failed: {exc}")

        # Part 3: Sensitivity Analysis
        sensitivity_flags: dict[str, bool] = {"has_sensitivity": False, "has_morris": False, "has_sobol": False, "has_sobol_heatmap": False}
        if sensitivity:
            sensitivity_dir = scheme_dir / "sensitivity"
            sensitivity_flags = _generate_sensitivity_charts(sensitivity_dir, tmp_dir)
            if not sensitivity_flags["has_sensitivity"]:
                print(f"No sensitivity data found in {sensitivity_dir} -- skipping Part 3")

        # Write metadata.json
        metadata = _build_metadata(
            records,
            scheme_dir,
            n_sims=n_sims,
            has_trajectories=has_trajectories,
            has_final_eval=final_records is not None,
            toml_path=toml_path,
            has_cost_distribution=has_cost_distribution,
        )
        metadata.update(sensitivity_flags)
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Write summary_table.json
        heat_flux_limit, g_load_limit = _read_constraint_limits(toml_path) if toml_path is not None else (None, None)
        summary = (
            _build_summary_table(final_records, heat_flux_limit=heat_flux_limit, g_load_limit=g_load_limit, cost_kwargs=cost_kwargs)
            if final_records is not None
            else {"rows": [], "violation_rows": []}
        )
        (tmp_dir / "summary_table.json").write_text(json.dumps(summary, indent=2))

        # Compile PDF via Typst
        if not _check_typst():
            print("Typst CLI not found — skipping PDF compilation")
            if keep_artifacts:
                print(f"Chart artifacts available at: {tmp_dir}")
            return None

        output_pdf = scheme_dir / "report.pdf"
        template = _TYPST_DIR / "report.typ"

        result = subprocess.run(
            [
                "typst",
                "compile",
                str(template),
                "--root",
                "/",
                "--input",
                f"dir={tmp_dir}",
                str(output_pdf),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Typst compilation failed:\n{result.stderr}")
            return None

        print(f"\nReport saved to {output_pdf}")
        return output_pdf

    finally:
        if not keep_artifacts:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# RL reuse helpers — thin wrappers so report_rl.py avoids code duplication
# ---------------------------------------------------------------------------
def render_mission_performance_charts(
    final_records: npt.NDArray[np.float64],
    trajectories: list[npt.NDArray[np.float64]],
    dispersions: npt.NDArray[np.float64],
    tmp_dir: Path,
    toml_path: Path | None = None,
    scheme_dir: Path | None = None,
    cost_kwargs: dict[str, Any] | None = None,
    sim_timeout_secs: float | None = None,
) -> tuple[bool, dict]:
    """Generate Part 2 SVG charts and summary_table.json into *tmp_dir*.

    Returns (has_trajectories, summary_table_dict).
    """
    heat_flux_limit, g_load_limit = _read_constraint_limits(toml_path) if toml_path is not None else (None, None)
    _generate_trajectory_charts(
        final_records,
        trajectories,
        dispersions,
        tmp_dir,
        scheme_dir=scheme_dir,
        toml_path=toml_path,
        sim_timeout_secs=sim_timeout_secs,
        cost_kwargs=cost_kwargs,
    )
    summary = _build_summary_table(final_records, heat_flux_limit=heat_flux_limit, g_load_limit=g_load_limit, cost_kwargs=cost_kwargs)
    (tmp_dir / "summary_table.json").write_text(json.dumps(summary, indent=2))
    return True, summary


def maybe_render_sensitivity_charts(scheme_dir: Path, tmp_dir: Path) -> dict[str, bool]:
    """Generate Part 3 SVG charts if sensitivity results exist. Returns sensitivity flags."""
    sensitivity_dir = scheme_dir / "sensitivity"
    return _generate_sensitivity_charts(sensitivity_dir, tmp_dir)


# ---------------------------------------------------------------------------
# Cross-scheme comparison report
# ---------------------------------------------------------------------------
def generate_comparison_report(
    training_output_dir: Path,
    schemes: list[str] | None = None,
    keep_artifacts: bool = False,
) -> Path | None:
    """Generate a cross-scheme comparison PDF report.

    Scans subdirectories of *training_output_dir* for JSONL data, generates
    a comparison convergence chart and metrics table, and compiles PDF.

    Returns the path to the generated PDF, or None if no data / Typst unavailable.
    """
    scheme_dirs = sorted(d for d in training_output_dir.iterdir() if d.is_dir() and list(d.glob("*.jsonl")))

    if schemes:
        scheme_dirs = [d for d in scheme_dirs if d.name in schemes]

    if not scheme_dirs:
        print(f"No JSONL data found in subdirectories of {training_output_dir}")
        return None

    # Collect data per scheme
    all_data: dict[str, list[dict]] = {}
    summary_rows: list[list[str]] = []

    for scheme_dir in scheme_dirs:
        scheme_name = scheme_dir.name
        data, _resume_gens = load_run_data(scheme_dir)
        if not data:
            continue
        all_data[scheme_name] = data

        cost_history = [r["best_cost"] for r in data]
        conv = convergence_speed(cost_history)
        cap = data[-1].get("capture_rate", 0) * 100

        summary_rows.append([scheme_name, f"{cost_history[-1]:.2e}", str(len(data)), f"{cap:.0f}%", str(conv)])

    if not all_data:
        print("No valid scheme data found")
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="aerocapture_comparison_"))

    try:
        # Generate comparison chart
        charts.chart_comparison_convergence(all_data, tmp_dir / "comparison_convergence.svg")

        # Write metadata
        metadata = {
            "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
            "schemes": list(all_data.keys()),
        }
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Write comparison table
        comparison_table = {
            "headers": ["Scheme", "Best Cost", "Generations", "Capture %", "Conv. Speed"],
            "rows": summary_rows,
        }
        (tmp_dir / "comparison_table.json").write_text(json.dumps(comparison_table, indent=2))

        # Compile PDF
        if not _check_typst():
            print("Typst CLI not found — skipping PDF compilation")
            return None

        output_pdf = training_output_dir / "comparison_report.pdf"
        template = _TYPST_DIR / "comparison.typ"

        result = subprocess.run(
            [
                "typst",
                "compile",
                str(template),
                "--root",
                "/",
                "--input",
                f"dir={tmp_dir}",
                str(output_pdf),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Typst compilation failed:\n{result.stderr}")
            return None

        print(f"Comparison report saved to {output_pdf}")
        return output_pdf

    finally:
        if not keep_artifacts:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: generate training reports from JSONL logs."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDF training reports from JSONL logs")
    parser.add_argument("path", type=str, help="Path to scheme directory (single) or training_output/ (comparison)")
    parser.add_argument("--toml", type=str, default=None, help="Path to training TOML config (enables final MC evaluation)")
    parser.add_argument("--compare", action="store_true", help="Generate cross-scheme comparison report")
    parser.add_argument("--schemes", nargs="*", help="Filter by scheme names (comparison mode)")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep temporary SVG/JSON artifacts after PDF generation")
    parser.add_argument("--sim-timeout", type=float, default=None, help="Wall-clock timeout per simulation in seconds")
    parser.add_argument("--sensitivity", action="store_true", help="Include Part 3: Sensitivity Analysis (requires pre-computed data)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: Path not found: {path}")
        sys.exit(1)

    if args.compare:
        generate_comparison_report(path, schemes=args.schemes, keep_artifacts=args.keep_artifacts)
    else:
        toml_path = Path(args.toml) if args.toml else None
        generate_report(path, toml_path=toml_path, keep_artifacts=args.keep_artifacts, sim_timeout_secs=args.sim_timeout, sensitivity=args.sensitivity)


if __name__ == "__main__":
    main()
