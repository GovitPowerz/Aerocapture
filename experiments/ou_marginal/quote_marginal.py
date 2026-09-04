"""Quote the OU-marginal campaign: every retrained NN cell + the original
frozen-trained cells + classical baselines, scored on BOTH noise regimes
(frozen = legacy pipeline conditioning; marginal = varied noise realization
per scenario) over one shared paired seed pool.

The frozen/marginal protocol matches the 2026-08-27 investigation
(the OU investigation summarized in RESULTS.md): n=1000 shared seeds, marginal additionally sets
simulation.random_seed = 1000 + 7*i, identical across cells.

Usage: uv run python experiments/ou_marginal/quote_marginal.py [--n-sims 1000]
Writes experiments/ou_marginal/quote_results.json and prints the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import aerocapture_rs
import numpy as np
from aerocapture.training.deploy_overrides import load_scaffolding_overrides
from aerocapture.training.report import _read_constraint_limits

REPO = Path(__file__).resolve().parents[2]


def discover_ou_cells() -> list[tuple[str, str, str | None]]:
    """Every deployed run under training_output/ou_marginal/ (scratch s1, ft_*,
    *_s2/_s3 repeats), plus the original pilot. Labels = 'ou_' + dir name; the
    scoring TOML is the family config (its [data] path is irrelevant here --
    the model is pinned via override)."""
    rows: list[tuple[str, str, str | None]] = []
    for d in sorted((REPO / "training_output/ou_marginal").glob("*/best_model.json")):
        name = d.parent.name
        cell = name.removeprefix("ft_")
        for suf in ("_s2", "_s3"):
            cell = cell.removesuffix(suf)
        rows.append((f"ou_{name}", f"configs/training/ou_marginal/{cell}.toml", str(d.parent.relative_to(REPO))))
    pilot = REPO / "training_output/ou_pilot/mamba_p962/best_model.json"
    if pilot.exists():
        rows.append(("ou_pilot_mamba", "configs/training/sweep/mamba_p962.toml", "training_output/ou_pilot/mamba_p962"))
    return rows


# (label, toml, model_dir | None for classicals-via-optimized-toml)
CELLS: list[tuple[str, str, str | None]] = [
    # Frozen-trained originals (reference rows):
    ("mamba_p962", "configs/training/sweep/mamba_p962.toml", "training_output/mamba_p962_long"),
    ("lstm_p1082", "configs/training/sweep/lstm_p1082.toml", "training_output/lstm_p1082_long"),
    ("gru_p1014", "configs/training/sweep/gru_p1014.toml", "training_output/gru_p1014_long"),
    ("dense_p972", "configs/training/sweep/dense_p972.toml", "training_output/dense_p972_ga_paper_best"),
    ("dense_p515", "configs/training/paper/dense_p515_ga.toml", "training_output/dense_p515_ga_paper_best"),
    # Classical baselines (frozen-tuned; robust across regimes per the investigation):
    ("ftc", "training_output/ftc/optimized_ftc.toml", None),
    ("fnpag", "training_output/fnpag/optimized_fnpag.toml", None),
    ("pred_guid", "training_output/pred_guid/optimized_pred_guid.toml", None),
    ("energy_controller", "training_output/energy_controller/optimized_energy_controller.toml", None),
    ("equilibrium_glide", "training_output/equilibrium_glide/optimized_equilibrium_glide.toml", None),
    ("piecewise_constant", "training_output/piecewise_constant/optimized_piecewise_constant.toml", None),
]


def score(toml: str, model_dir: str | None, seeds: np.ndarray, regime: str) -> dict:
    idx = aerocapture_rs.final_record_indices()
    _, _, heat_load_limit_kj = _read_constraint_limits(REPO / toml)  # [flight.constraints] is authoritative
    assert heat_load_limit_kj is not None
    base: dict[str, object] = {"simulation.n_sims": 1}
    if model_dir is not None:
        d = REPO / model_dir
        base["data.neural_network"] = str(d / "best_model.json")
        base.update(load_scaffolding_overrides(d))
        # The ou_marginal configs bake per_draw into the TOML; pin the regime
        # explicitly so BOTH regimes are scored for every cell regardless of
        # which TOML it trained under.
    base["monte_carlo.noise_seeding"] = "legacy"
    ovr = []
    for i, s in enumerate(seeds):
        o = {**base, "monte_carlo.seed": int(s)}
        if regime == "marginal":
            o["simulation.random_seed"] = float(1000 + 7 * i)
        ovr.append(o)
    fr = np.asarray(aerocapture_rs.run_batch(str(REPO / toml), overrides_list=ovr, sim_timeout_secs=30.0).final_records)
    cap = (fr[:, idx["ifinal"]] == 3) & (fr[:, idx["ecc"]] < 1.0)
    dv = fr[cap, idx["dv_total_ms"]]
    hl = fr[:, idx["heat_load_mjm2"]]
    p50, p95, p99 = np.percentile(dv, [50, 95, 99])
    return {
        "capture_pct": round(100.0 * cap.mean(), 2),
        "dv_p50": round(float(p50), 2),
        "dv_p95": round(float(p95), 2),
        "dv_p99": round(float(p99), 2),
        "dv_cvar95": round(float(dv[dv >= p95].mean()), 2),
        "heat_load_viol_pct": round(100.0 * float((hl * 1e3 > heat_load_limit_kj).mean()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args()
    seeds = np.random.default_rng(987654321).integers(0, 2**31, size=args.n_sims)

    out: dict[str, dict] = {}
    for label, toml, model_dir in discover_ou_cells() + CELLS:
        if model_dir is not None and not (REPO / model_dir / "best_model.json").exists():
            print(f"{label:<20} SKIPPED (no best_model.json yet)")
            continue
        for regime in ("frozen", "marginal"):
            m = score(toml, model_dir, seeds, regime)
            out[f"{label}/{regime}"] = m
            print(
                f"{label:<20} {regime:<8} capture {m['capture_pct']:6.1f}%  p50 {m['dv_p50']:7.1f}  "
                f"p95 {m['dv_p95']:7.1f}  cvar95 {m['dv_cvar95']:7.1f}  hl_viol {m['heat_load_viol_pct']:.1f}%"
            )
    result_path = Path(__file__).resolve().parent / "quote_results.json"
    result_path.write_text(json.dumps({"n_sims": args.n_sims, "cells": out}, indent=1))
    print(f"\nWritten {result_path}")


if __name__ == "__main__":
    main()
