"""objective-centering eval: score the five Phase-1 cells on the 9M stress pool
and extract the transform-independent validation capture-rate convergence series.

Same machinery / regime / pool as robustness_retrain_eval.py, so the deployed
off-nominal numbers are directly comparable. Convergence is read on
validation.capture_rate (NOT rms_cost, which is in each cell's transform space).

`--v4` (issue #177) scores the five dense lever cells retrained under per-scenario noise
(experiments/ou_marginal/jobs_centering_dense.txt) under that regime, into
objective_centering_v4.json; the arxiv-v3 objective_centering.json stays. Each v4 cell also carries
its `violation_pct` per constraint (heat_flux, g_load, heat_load) over every draw, against its TOML's
[flight.constraints] limits, as in centered_depth_eval.py --v4.

Usage:
    uv run python articles/paper/scripts/objective_centering_eval.py [--v4] [--n-sims 1000]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.deploy_overrides import LEGACY_NOISE_REGIME  # noqa: E402
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
# The v4 lever cells (#177): the five dense cells above retrained under per-scenario noise.
V4_CELLS = [(label, f"ou_marginal/centered/dense_{label}", f"configs/training/ou_marginal/centered/dense_{label}.toml", n) for label, _, _, n in CELLS[:5]]
V4_OUT = REPO / "articles/paper/data/objective_centering_v4.json"


def _read_run_log(jsonl_paths: list[str]) -> list[dict]:
    """The run's generation records across its run_*.jsonl fragments, oldest first: a resumed run
    writes one per launch and a later fragment supersedes the generations it re-logs
    (collect_runs._gzip_run_log bundles them the same way)."""
    records: list[dict] = []
    for path in jsonl_paths:
        new = []
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("generation") is not None:
                    new.append(r)
        if new:
            records = [r for r in records if r["generation"] < new[0]["generation"]] + new
    return records


def _derive_n_pop(records: list[dict], fallback: int) -> int:
    """Population size = length of a generation's all_costs array; fallback if absent."""
    for r in records:
        ac = r.get("all_costs")
        if isinstance(ac, list) and ac:
            return len(ac)
    return fallback


def extract_convergence(records: list[dict], n_pop: int, n_sims: int) -> list[list]:
    """Per-validation [cumulative_training_sims, capture_rate]. Transform-independent."""
    series: list[list] = []
    for r in records:
        cap = (r.get("validation") or {}).get("capture_rate")
        if cap is not None:
            series.append([int(r["generation"]) * n_pop * n_sims, float(cap)])
    series.sort(key=lambda p: p[0])
    return series


def _eval_one(label: str, run_dir: str, toml: str, n_sims_train: int, n_eval: int, regime: dict[str, str]) -> dict:
    from aerocapture.training.cell_eval import evaluate_cell
    from aerocapture.training.evaluate import constraint_violation_rates
    from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES
    from aerocapture.training.report import read_cost_kwargs
    from aerocapture.training.seeds import STRESS_EVAL_SEED_OFFSET

    scheme_dir = REPO / "training_output" / run_dir
    cost_kwargs = read_cost_kwargs(Path(toml))
    res = evaluate_cell(
        scheme_dir,
        Path(toml),
        pool=(STRESS_EVAL_SEED_OFFSET, n_eval),
        extra_overrides={**regime, **STRESS_OVERRIDES},
        sim_timeout_secs=5.0,
    )
    col = {name: res.final_records[:, idx] for name, idx in zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True)}
    stats = {"label": label, **run_stats(col["ifinal"], col["eccentricity"], col["dv_total_m_s"], n_boot=2000)}
    # Every draw against the ADR-0005 gate's own limits (the TOML's [flight.constraints]); None when none is set.
    stats["violation_rates"] = constraint_violation_rates(res.final_records, cost_kwargs)
    records = _read_run_log(sorted(glob.glob(str(scheme_dir / "run_*.jsonl"))))  # run_000_<UTC stamp>: name order is launch order
    stats["convergence"] = extract_convergence(records, _derive_n_pop(records, N_POP), n_sims_train)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--v4", action="store_true", help="the #177 per-scenario lever cells, per_draw, into objective_centering_v4.json")
    args = parser.parse_args(argv)
    cells, regime, out = (V4_CELLS, {"monte_carlo.noise_seeding": "per_draw"}, V4_OUT) if args.v4 else (CELLS, LEGACY_NOISE_REGIME, OUT)
    cells_out, convergence = [], {}
    for label, run_dir, toml, n_sims_train in cells:
        if not (REPO / "training_output" / run_dir / "final_eval.parquet").exists():
            if args.v4:
                sys.exit(f"{run_dir} is not deployed: every v4 lever cell is required (no thinner file)")
            print(f"  skip {label} ({run_dir} not deployed yet)")
            continue
        s = _eval_one(label, run_dir, toml, n_sims_train, args.n_sims, regime)
        convergence[label] = s.pop("convergence")
        rates = s.pop("violation_rates")
        if args.v4:  # the arxiv-v3 file's shape stays as committed
            if rates is None:
                sys.exit(f"{toml} configures no [flight.constraints] limit: the violation rates cannot be scored")
            s["violation_pct"] = {k: round(100 * v, 2) for k, v in rates.items()}
        cells_out.append(s)
        print(
            f"  {label:16s} stress: capture {s['capture_pct']:5.1f}% | mean {s['dv_mean']:7.1f}"
            f" | CVaR95 {s.get('dv_cvar95'):7.1f} | conv pts {len(convergence[label])}" + (f" | violations % {s['violation_pct']}" if args.v4 else "")
        )
    if cells_out:
        record = {
            "stress_overrides": STRESS_OVERRIDES,
            "n_sims_eval": args.n_sims,
            "pool": "STRESS_EVAL 9M",
            "n_pop": N_POP,
            "cells": cells_out,
            "convergence": convergence,
        }
        if args.v4:  # the arxiv-v3 file's bytes stay as committed
            record = {"noise_seeding": "per_draw", **record}
        out.write_text(json.dumps(record, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
