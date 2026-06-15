"""Single-core guidance compute benchmark for the 3-way (NN / FTC / FNPAG).

Measures wall-clock per trajectory on ONE core for each deployed scheme over a
shared set of representative scenarios (the 2M final-eval pool, first --n-sims).
The cross-scheme ratio of ms/sim is a CONSERVATIVE proxy for relative onboard
guidance cost: the integration/physics cost is shared across schemes and sits
in the denominator of both, so the ms/sim ratio UNDER-states the pure
guidance-cost ratio (FNPAG runs a full forward predictor -- many sub-
integrations -- per guidance call, FTC/NN are a single closed-form / feedforward
evaluation). The guidance cadence is identical across schemes, so ms/sim ratio
== ms-per-guidance-call ratio.

This is the "compute of the fastest" half of the NN deployability claim.
Deployed-state aware: reuses report._resolve_eval_toml + scaffolding so each
scheme runs at its deployed operating point. Gated on 01 (classical) + 02 (NN).

Usage:
    uv run python articles/paper/scripts/compute_benchmark.py [--n-sims 200]
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

# (label, run_dir relative to training_output, training TOML) -- the 3-way.
SCHEMES = [
    ("NN", "paper/optimizer_budget/ga_300", "configs/training/paper/dense_p3998_ga.toml"),
    ("FTC", "ftc", "configs/training/msr_aller_ftc_train.toml"),
    ("FNPAG", "fnpag", "configs/training/msr_aller_fnpag_train.toml"),
]
OUT = REPO / "articles/paper/data/compute_benchmark.json"


def _bench_one(label: str, run_dir: str, toml: str, n_sims: int) -> dict:
    import aerocapture_rs
    from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.report import _resolve_eval_toml
    from aerocapture.training.toml_utils import load_toml_with_bases

    scheme_dir = REPO / "training_output" / run_dir
    eval_toml, scaffolding = _resolve_eval_toml(Path(toml), scheme_dir)
    base_mc_seed = load_toml_with_bases(eval_toml).get("monte_carlo", {}).get("seed", 42)
    seeds = make_reserved_seeds(base_mc_seed, FINAL_EVAL_SEED_OFFSET, n_sims)

    base: dict = {"simulation.n_sims": 1, **scaffolding}
    local_model = scheme_dir / "best_model.json"
    if local_model.exists():
        base["data.neural_network"] = str(local_model.resolve())
    overrides = [{**base, "monte_carlo.seed": s} for s in seeds]

    def run() -> None:
        aerocapture_rs.run_batch(toml_path=str(eval_toml.resolve()), overrides_list=overrides, n_threads=1)

    run()  # warmup (discard: caches, page-ins)
    t0 = time.perf_counter()
    run()
    dt = time.perf_counter() - t0
    return {"label": label, "n_sims": n_sims, "wall_s": round(dt, 4), "ms_per_sim": round(1000 * dt / n_sims, 4), "sims_per_s": round(n_sims / dt, 1)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=200, help="representative scenarios per scheme (single-threaded)")
    args = parser.parse_args(argv)

    results = []
    for label, run_dir, toml in SCHEMES:
        if not (REPO / "training_output" / run_dir / "final_eval.parquet").exists():
            print(f"  skip {label} ({run_dir} not deployed yet)")
            continue
        r = _bench_one(label, run_dir, toml, args.n_sims)
        results.append(r)
        print(f"  {r['label']:6s} {r['ms_per_sim']:8.3f} ms/sim   {r['sims_per_s']:8.1f} sims/s (1 core)")

    if results:
        fastest = min(r["ms_per_sim"] for r in results)
        for r in results:
            r["slowdown_vs_fastest"] = round(r["ms_per_sim"] / fastest, 1)
        OUT.write_text(json.dumps({"single_core": True, "schemes": results}, indent=2))
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
