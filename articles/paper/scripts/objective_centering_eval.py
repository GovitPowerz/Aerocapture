"""objective-centering eval: score the five Phase-1 cells on the 9M stress pool
and extract the transform-independent validation capture-rate convergence series.

Same machinery / regime / pool as robustness_retrain_eval.py, so the deployed
off-nominal numbers are directly comparable. Convergence is read on
validation.capture_rate (NOT rms_cost, which is in each cell's transform space).

Usage:
    uv run python articles/paper/scripts/objective_centering_eval.py [--n-sims 1000]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.paper_stats import run_stats  # noqa: E402

# (label, run_dir under training_output/, training TOML). n_sims is the training
# budget used (for the convergence x-axis), set by the runner.
CELLS = [
    ("stacked", "paper/objective_centering/dense_stacked", "configs/training/paper/objective_centering/dense_stacked_high.toml", 2),
    ("plus_sims", "paper/objective_centering/dense_plus_sims", "configs/training/paper/objective_centering/dense_plus_sims_high.toml", 16),
    ("plus_bucket", "paper/objective_centering/dense_plus_bucket", "configs/training/paper/objective_centering/dense_plus_bucket_high.toml", 2),
    ("plus_transform", "paper/objective_centering/dense_plus_transform", "configs/training/paper/objective_centering/dense_plus_transform_high.toml", 2),
    ("centered", "paper/objective_centering/dense_centered", "configs/training/paper/objective_centering/dense_centered_high.toml", 16),
    ("mamba_centered", "paper/objective_centering/mamba_centered", "configs/training/paper/objective_centering/mamba_centered_high.toml", 16),
]
N_POP = 256
STRESS_OVERRIDES = {
    "monte_carlo.atmosphere.level": "high",
    "monte_carlo.density_perturbation.level": "high",
    "monte_carlo.navigation.level": "high",
    "monte_carlo.nav_filter.level": "high",
}
OUT = REPO / "articles/paper/data/objective_centering.json"


def _derive_n_pop(jsonl_path: str, fallback: int) -> int:
    """Population size = length of a generation's all_costs array; fallback if absent."""
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ac = r.get("all_costs")
            if isinstance(ac, list) and ac:
                return len(ac)
    return fallback


def extract_convergence(jsonl_path: str, n_pop: int, n_sims: int) -> list[list]:
    """Per-validation [cumulative_training_sims, capture_rate]. Transform-independent."""
    series: list[list] = []
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("validation") or {}
            cap = v.get("capture_rate")
            gen = r.get("generation")
            if cap is None or gen is None:
                continue
            series.append([int(gen) * n_pop * n_sims, float(cap)])
    series.sort(key=lambda p: p[0])
    return series


def _eval_one(label: str, run_dir: str, toml: str, n_sims_train: int, n_eval: int) -> dict:
    import aerocapture_rs
    from aerocapture.training.evaluate import STRESS_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES
    from aerocapture.training.report import _resolve_eval_toml
    from aerocapture.training.toml_utils import load_toml_with_bases

    scheme_dir = REPO / "training_output" / run_dir
    eval_toml, scaffolding = _resolve_eval_toml(Path(toml), scheme_dir)
    base_mc_seed = load_toml_with_bases(eval_toml).get("monte_carlo", {}).get("seed", 42)
    seeds = make_reserved_seeds(base_mc_seed, STRESS_EVAL_SEED_OFFSET, n_eval)
    base: dict = {"simulation.n_sims": 1, **STRESS_OVERRIDES, **scaffolding}
    local_model = scheme_dir / "best_model.json"
    if local_model.exists():
        base["data.neural_network"] = str(local_model.resolve())
    overrides = [{**base, "monte_carlo.seed": s} for s in seeds]
    results = aerocapture_rs.run_batch(toml_path=str(eval_toml.resolve()), overrides_list=overrides, sim_timeout_secs=5.0)
    recs = np.asarray(results.final_records)
    col = {name: recs[:, idx] for name, idx in zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True)}
    stats = {"label": label, **run_stats(col["ifinal"], col["eccentricity"], col["dv_total_m_s"], n_boot=2000)}
    jsonls = sorted(glob.glob(str(scheme_dir / "run_*.jsonl")))
    if jsonls:
        n_pop = _derive_n_pop(jsonls[-1], N_POP)
        stats["convergence"] = extract_convergence(jsonls[-1], n_pop, n_sims_train)
    else:
        stats["convergence"] = []
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args(argv)
    cells_out, convergence = [], {}
    for label, run_dir, toml, n_sims_train in CELLS:
        if not (REPO / "training_output" / run_dir / "final_eval.parquet").exists():
            print(f"  skip {label} ({run_dir} not deployed yet)")
            continue
        s = _eval_one(label, run_dir, toml, n_sims_train, args.n_sims)
        convergence[label] = s.pop("convergence")
        cells_out.append(s)
        print(f"  {label:16s} stress: capture {s['capture_pct']:5.1f}% | mean {s['dv_mean']:7.1f} | CVaR95 {s.get('dv_cvar95'):7.1f} | conv pts {len(convergence[label])}")
    if cells_out:
        OUT.write_text(json.dumps({"stress_overrides": STRESS_OVERRIDES, "n_sims_eval": args.n_sims, "pool": "STRESS_EVAL 9M", "n_pop": N_POP, "cells": cells_out, "convergence": convergence}, indent=2))
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
