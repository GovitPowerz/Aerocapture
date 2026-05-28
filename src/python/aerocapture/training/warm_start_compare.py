"""Supervisor-vs-NN trajectory comparison for the warm-start report.

Runs four MC batches at the end of warm-start (after `build_warm_start_chromosome`):

  1. Supervisor (primary scheme from `[warm_start] supervisor_schemes[0]`) on the
     TRAINING pool (`make_reserved_seeds(base, WARM_START_SEED_OFFSET, n_warm_seeds)`).
  2. NN (warm-started chromosome) on the same TRAINING pool.
  3. Supervisor on the VALIDATION pool
     (`make_reserved_seeds(base, VALIDATION_SEED_OFFSET, validation_n_sims)`).
  4. NN on the same VALIDATION pool.

For each batch, renders the 5 mission-performance panels from `charts.py`:
corridor pdyn / inclination / bank + altitude-vs-time + heat-flux-vs-time.
20 SVGs total, written to `<save_dir>/warm_start_report/compare_{pool}_{side}_*.svg`
and consumed by the Typst template in `warm_start_report.py`.

Trajectories on the same (pool, seed) ARE run twice -- once for supervisor, once
for NN -- because we want like-for-like dispersion draws between the two sides.
Reusing `aerocapture_rs.run_batch` with `monte_carlo.seed` overrides gives us
deterministic seeds matching the rest of the training pipeline.

Memory: trajectories are processed one (pool, side) at a time and released
before moving to the next; peak ~600 MB for 5000 sims at ~120 KB per trajectory.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from aerocapture.training.evaluate import (
    FINAL_EVAL_SEED_OFFSET,
    VALIDATION_SEED_OFFSET,
    WARM_START_SEED_OFFSET,
    make_reserved_seeds,
    write_nn_json,
)
from aerocapture.training.warm_start import _build_overrides_for_source

if TYPE_CHECKING:
    from aerocapture.training.config import TrainingConfig

try:
    import aerocapture_rs as _aero_rs
except ImportError as e:
    raise ImportError("warm_start_compare requires the aerocapture_rs PyO3 module") from e


_FALLBACK_SIDE_LABELS = {"supervisor": "Supervisor", "nn": "NN (warm-started)"}


def _run_one_pool_one_side(
    toml_path: str,
    seeds: list[int],
    side: str,
    primary_scheme: str,
    supervisor_params: dict[str, float],
    nn_json_path: Path,
    sim_timeout_secs: float | None,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    """Run a single pool through either the supervisor or the NN. Returns
    (final_records, trajectories). The supervisor branch uses
    `_build_overrides_for_source` so all scaffolding (lateral / exit / thermal /
    nav / shaping) values from the supervisor's `best_params.json` are honored,
    matching what `build_warm_start_chromosome` fed into `collect_supervised`.
    The NN branch swaps `data.neural_network` to the temp JSON written from the
    freshly-encoded warm-start chromosome.
    """
    if side == "supervisor":
        overrides_template: dict[str, object] = _build_overrides_for_source(supervisor_params, primary_scheme)
        overrides_template["guidance.type"] = primary_scheme
    elif side == "nn":
        overrides_template = {
            "guidance.type": "neural_network",
            "data.neural_network": str(nn_json_path),
        }
    else:
        raise ValueError(f"unknown side {side!r}; expected 'supervisor' or 'nn'")

    overrides_list = [{**overrides_template, "monte_carlo.seed": int(s), "simulation.n_sims": 1} for s in seeds]
    results = _aero_rs.run_batch(
        toml_path=str(toml_path),
        overrides_list=overrides_list,
        include_trajectories=True,
        sim_timeout_secs=sim_timeout_secs,
    )
    return results.final_records, list(results.trajectories)


def _render_pool_panels(
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    out_dir: Path,
    prefix: str,
    heat_flux_limit: float | None,
) -> None:
    """Write 5 SVG panels named `{prefix}_{panel}.svg` into `out_dir`.

    Reuses the same chart functions the final report uses so the visual style
    matches. No nominal overlays (no `best_traj` / `undispersed_nominal`) --
    the supervisor-vs-NN delta is the point; nominal references would crowd
    the plots.
    """
    from aerocapture.training import charts

    charts.chart_corridor_pdyn(trajectories, traj_class, out_dir / f"{prefix}_corridor_pdyn.svg")
    charts.chart_corridor_inclination(trajectories, traj_class, out_dir / f"{prefix}_corridor_inclination.svg")
    charts.chart_corridor_bank(trajectories, traj_class, out_dir / f"{prefix}_corridor_bank.svg")
    charts.chart_altitude_time(trajectories, traj_class, out_dir / f"{prefix}_altitude_time.svg")
    charts.chart_heat_flux_time(trajectories, traj_class, out_dir / f"{prefix}_heat_flux_time.svg", limit_kw_m2=heat_flux_limit)


def _read_constraint_limits(toml_path: Path) -> tuple[float | None, float | None]:
    """Mirror report._read_constraint_limits without taking the runtime import dep."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    try:
        doc = load_toml_with_bases(toml_path)
        constraints = doc.get("flight", {}).get("constraints", {})
        return (
            float(constraints["max_heat_flux"]) if "max_heat_flux" in constraints else None,
            float(constraints["max_load_factor"]) if "max_load_factor" in constraints else None,
        )
    except Exception:
        return None, None


def _decode_warm_start_weights(
    warm_chromo: npt.NDArray[np.float64],
    weight_specs: list,
) -> npt.NDArray[np.float64]:
    """Decode the NN-weight slab of a warm-start chromosome into physical weights.

    The warm-start chromosome is `[NN-weights | scaffolding-tail (optional)]`,
    laid out as normalized [0, 1] values per ParamSpec. The scaffolding tail
    must be excluded -- only the NN weights go into the JSON. Caller is
    responsible for passing only the NN-weight slab (and matching ParamSpec
    slab) here.
    """
    if len(warm_chromo) != len(weight_specs):
        raise ValueError(
            f"chromosome length ({len(warm_chromo)}) must match weight_specs length ({len(weight_specs)}); did you forget to drop the scaffolding tail?"
        )
    weights = np.empty(len(weight_specs), dtype=np.float64)
    for j, s in enumerate(weight_specs):
        weights[j] = s.p_min + float(warm_chromo[j]) * (s.p_max - s.p_min)
    return weights


def render_trajectory_comparison(
    cfg: TrainingConfig,
    base_mc_seed: int,
    warm_chromo: npt.NDArray[np.float64],
    nn_weight_specs: list,
) -> dict[str, Any]:
    """Orchestrate the four (pool, side) batches and render 20 comparison panels.

    Returns a `manifest` dict with the relative SVG filenames per (pool, side, panel)
    so `warm_start_report.py` can wire them into the Typst template without
    hard-coding the layout.

    Best-effort: any sub-step that raises is logged and the manifest's
    corresponding slot is `None`. The caller (train.py) catches the outer
    exception and continues without blocking training.
    """
    save_dir = Path(cfg.save_dir)
    report_dir = save_dir / "warm_start_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    toml_config = cfg.sim.toml_config
    if toml_config is None:
        raise ValueError("render_trajectory_comparison requires cfg.sim.toml_config to be set")

    # Resolve primary supervisor (first scheme in the list -- typically FTC).
    ws = cfg.warm_start
    primary_scheme = ws.supervisor_schemes[0]
    primary_params_path = ws.params_paths.get(primary_scheme) or f"training_output/{primary_scheme}/best_params.json"
    primary_params_path = str(primary_params_path)
    if not Path(primary_params_path).exists():
        raise FileNotFoundError(
            f"primary supervisor '{primary_scheme}' params not found at '{primary_params_path}'. "
            f"Train {primary_scheme} first or set [warm_start.params_paths].{primary_scheme}."
        )
    with open(primary_params_path) as f:
        supervisor_params = json.load(f)

    # Seed pools (matching offsets used elsewhere; deliberately disjoint from
    # FINAL_EVAL_SEED_OFFSET so the comparison plots don't leak final-eval seeds).
    assert WARM_START_SEED_OFFSET != VALIDATION_SEED_OFFSET != FINAL_EVAL_SEED_OFFSET
    train_seeds = make_reserved_seeds(base_mc_seed, WARM_START_SEED_OFFSET, ws.n_warm_seeds)
    val_n = int(cfg.optimizer.validation_n_sims)
    val_seeds = make_reserved_seeds(base_mc_seed, VALIDATION_SEED_OFFSET, val_n)

    # Write the warm-start NN to a temp JSON so the runtime can load it.
    # Cleanup happens unconditionally in the finally block.
    heat_flux_limit, _ = _read_constraint_limits(Path(toml_config))
    panels = ["corridor_pdyn", "corridor_inclination", "corridor_bank", "altitude_time", "heat_flux_time"]

    manifest: dict[str, Any] = {
        "primary_supervisor": primary_scheme,
        "pools": {},
        "panels": panels,
        "side_labels": {**_FALLBACK_SIDE_LABELS, "supervisor": f"Supervisor ({primary_scheme})"},
    }

    import os

    fd, nn_tmp_str = tempfile.mkstemp(suffix=".json", prefix="warm_start_nn_")
    os.close(fd)
    nn_tmp_path = Path(nn_tmp_str)
    try:
        # Decode the NN-weight slab and write the temp JSON. The chromosome may
        # have a scaffolding tail; nn_weight_specs covers only the NN weights.
        weights = _decode_warm_start_weights(warm_chromo[: len(nn_weight_specs)], nn_weight_specs)
        write_nn_json(
            weights,
            cfg.network,
            nn_tmp_path,
            input_mask=cfg.network.input_mask,
            output_param=cfg.network.output_parameterization,
        )

        pools: list[tuple[str, list[int], int]] = [
            ("train", train_seeds, ws.n_warm_seeds),
            ("val", val_seeds, val_n),
        ]
        sides = ["supervisor", "nn"]

        from aerocapture.training import charts as _charts

        for pool_name, seeds, n_seeds in pools:
            pool_entry: dict[str, Any] = {"n_sims": int(n_seeds), "sides": {}}
            for side in sides:
                t0 = time.monotonic()
                print(f"  [warm_start_compare] {side} on {pool_name} pool ({n_seeds} sims)...")
                try:
                    final_records, trajectories = _run_one_pool_one_side(
                        toml_path=toml_config,
                        seeds=seeds,
                        side=side,
                        primary_scheme=primary_scheme,
                        supervisor_params=supervisor_params,
                        nn_json_path=nn_tmp_path,
                        sim_timeout_secs=cfg.sim.sim_timeout_secs,
                    )
                    traj_class = _charts.classify_trajectories(
                        final_records,
                        heat_flux_limit=heat_flux_limit,
                        g_load_limit=None,  # g-load violation already counted via heat-flux path
                    )
                    n_captured = int(((final_records[:, _charts._FR_IFINAL] == 3) & (final_records[:, _charts._FR_ECC] < 1.0)).sum())
                    prefix = f"compare_{pool_name}_{side}"
                    _render_pool_panels(
                        trajectories=trajectories,
                        traj_class=traj_class,
                        out_dir=report_dir,
                        prefix=prefix,
                        heat_flux_limit=heat_flux_limit,
                    )
                    pool_entry["sides"][side] = {
                        "n_captured": n_captured,
                        "capture_rate": n_captured / max(n_seeds, 1),
                        "prefix": prefix,
                        "panels": {p: f"{prefix}_{p}.svg" for p in panels},
                    }
                    print(f"  [warm_start_compare]   {side}/{pool_name}: {n_captured}/{n_seeds} captured  ({time.monotonic() - t0:.1f}s)")
                    # Release memory before next pool/side -- 5000 trajectories
                    # at ~120 KB each is ~600 MB.
                    del trajectories, final_records
                except Exception as e:
                    pool_entry["sides"][side] = {"error": f"{type(e).__name__}: {e}"}
                    print(f"  [warm_start_compare] WARNING: {side}/{pool_name} failed: {type(e).__name__}: {e}")
            manifest["pools"][pool_name] = pool_entry
    finally:
        nn_tmp_path.unlink(missing_ok=True)

    # Persist manifest so warm_start_report.py and any CLI re-run can pick it up.
    (report_dir / "compare_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


__all__ = ["render_trajectory_comparison"]
