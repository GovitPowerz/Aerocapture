"""Fresh-pool MC re-quote of the deployed headline model (the abstract number).

Every sweep cell in the campaign is scored on the shared 2M-offset final-eval
pool, and the headline configuration is CHOSEN by those sweeps -- quoting the
same pool is selection-on-test (winner's curse, ~1-3 m/s optimism). This script
re-runs the deployed model once on the untouched 8M-offset pool and prints the
capture rate + DV mean/p50/p95/p99/CVaR95 to quote in the abstract.

Usage (after experiments/paper/02 deploys the headline cell):
    uv run python articles/paper/scripts/fresh_pool_requote.py \
        training_output/paper/optimizer_budget/ga_300 \
        --toml configs/training/paper/dense_p3998_ga.toml [--n-sims 1000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="campaign cell dir holding best_model.json (+ best_params.json)")
    parser.add_argument("--toml", required=True, help="the cell's training TOML")
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--sim-timeout", type=float, default=5.0)
    args = parser.parse_args(argv)

    import aerocapture_rs
    from aerocapture.training.evaluate import HEADLINE_REQUOTE_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.report import _load_nn_scaffolding_overrides
    from aerocapture.training.toml_utils import load_toml_with_bases

    run_dir = Path(args.run_dir)
    model = run_dir / "best_model.json"
    if not model.exists():
        sys.exit(f"{model} not found -- train/deploy the cell first")

    toml_data = load_toml_with_bases(Path(args.toml))
    base_mc_seed = toml_data.get("monte_carlo", {}).get("seed", 42)
    seeds = make_reserved_seeds(base_mc_seed, HEADLINE_REQUOTE_SEED_OFFSET, args.n_sims)

    scaffolding = _load_nn_scaffolding_overrides(run_dir, run_dir / f"optimized_{run_dir.name}.toml")
    base = {"simulation.n_sims": 1, "data.neural_network": str(model.resolve()), **scaffolding}
    results = aerocapture_rs.run_batch(
        toml_path=str(Path(args.toml).resolve()),
        overrides_list=[{**base, "monte_carlo.seed": s} for s in seeds],
        sim_timeout_secs=args.sim_timeout,
    )

    from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES

    records = np.asarray(results.final_records)
    col = {name: records[:, idx] for name, idx in zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True)}
    cap = (col["ifinal"] == 3) & (col["eccentricity"] < 1.0)
    dvc = np.sort(col["dv_total_m_s"][cap])
    out = {
        "pool": "fresh (offset 8M)",
        "n": int(len(records)),
        "capture_pct": round(100 * float(cap.mean()), 2),
        "dv_mean": round(float(dvc.mean()), 2),
        "dv_p50": round(float(np.percentile(dvc, 50)), 2),
        "dv_p95": round(float(np.percentile(dvc, 95)), 2),
        "dv_p99": round(float(np.percentile(dvc, 99)), 2),
        "dv_cvar95": round(float(dvc[-max(1, len(dvc) // 20) :].mean()), 2),
        "dv_max_descriptive": round(float(dvc.max()), 2),
        "scaffolding_applied": sorted(scaffolding),
    }
    out_path = run_dir / "fresh_pool_requote.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
