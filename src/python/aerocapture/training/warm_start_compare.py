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
and consumed by `src/typst/warm_start_report.typ` via `warm_start_report.py`.

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

from aerocapture.training.artifacts import write_best_artifacts
from aerocapture.training.cell_eval import evaluate_cell
from aerocapture.training.deploy_overrides import overrides_from_params
from aerocapture.training.param_spaces import ParamSpec
from aerocapture.training.report import _read_constraint_limits
from aerocapture.training.seeds import FINAL_EVAL_SEED_OFFSET, VALIDATION_SEED_OFFSET, WARM_START_SEED_OFFSET, make_reserved_seeds

if TYPE_CHECKING:
    from aerocapture.training.config import TrainingConfig

try:
    import aerocapture_rs  # noqa: F401
except ImportError as e:
    raise ImportError("warm_start_compare requires the aerocapture_rs PyO3 module") from e


_FALLBACK_SIDE_LABELS = {"supervisor": "Supervisor", "nn": "NN (warm-started)"}


def _run_one_pool_one_side(
    toml_path: str,
    seeds: list[int],
    side: str,
    primary_scheme: str,
    supervisor_params: dict[str, float],
    nn_cell_dir: Path,
    sim_timeout_secs: float | None,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    """Run a single pool through either the supervisor or the NN. Returns
    (final_records, trajectories). The supervisor branch uses
    `overrides_from_params` so all scaffolding (lateral / exit / thermal /
    nav / shaping) values from the supervisor's `best_params.json` are honored,
    matching what `build_warm_start_chromosome` fed into `collect_supervised`.
    The NN branch flies the scratch cell written from the warm-start chromosome:
    its `best_model.json` and its scaffolding tail (`best_params.json`).
    """
    if side == "supervisor":
        overrides_template: dict[str, object] = overrides_from_params(supervisor_params, primary_scheme)
        overrides_template["guidance.type"] = primary_scheme
    elif side == "nn":
        overrides_template = {"guidance.type": "neural_network"}
    else:
        raise ValueError(f"unknown side {side!r}; expected 'supervisor' or 'nn'")

    cell_dir = nn_cell_dir if side == "nn" else None
    model = nn_cell_dir / "best_model.json" if side == "nn" else None
    results = evaluate_cell(
        cell_dir, Path(toml_path), seeds, model=model, extra_overrides=overrides_template, include_trajectories=True, sim_timeout_secs=sim_timeout_secs
    )
    assert results.trajectories is not None
    return results.final_records, results.trajectories


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


def render_trajectory_comparison(
    cfg: TrainingConfig,
    base_mc_seed: int,
    warm_chromo: npt.NDArray[np.float64],
    param_specs: list[ParamSpec],
) -> dict[str, Any]:
    """Orchestrate the four (pool, side) batches and render 20 comparison panels.

    Returns a `manifest` dict with the relative SVG filenames per (pool, side, panel)
    so `warm_start_report.py` can hand them to `src/typst/warm_start_report.typ` without
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

    heat_flux_limit, _, _ = _read_constraint_limits(Path(toml_config))
    panels = ["corridor_pdyn", "corridor_inclination", "corridor_bank", "altitude_time", "heat_flux_time"]

    manifest: dict[str, Any] = {
        "primary_supervisor": primary_scheme,
        "pools": {},
        "panels": panels,
        "side_labels": {**_FALLBACK_SIDE_LABELS, "supervisor": f"Supervisor ({primary_scheme})"},
    }

    # The warm-started NN flies from a scratch cell written by the trainer's own
    # artifact writer (the chromosome's layout and scaffolding tail, the config's
    # normalization table), never the TOML's deploy path.
    with tempfile.TemporaryDirectory(prefix="warm_start_nn_") as scratch:
        nn_cell_dir = Path(scratch)
        write_best_artifacts(warm_chromo, cfg, param_specs, nn_cell_dir, cwd=None)

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
                        nn_cell_dir=nn_cell_dir,
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

    # Persist manifest so warm_start_report.py and any CLI re-run can pick it up.
    (report_dir / "compare_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


__all__ = ["render_trajectory_comparison"]
