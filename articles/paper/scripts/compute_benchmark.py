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
# NN = the Mamba_962 sizing headline (stateful SSM runtime -- a deliberately
# pessimistic per-sim cost vs the dense efficiency reference; report both if asked).
SCHEMES = [
    ("NN-mamba", "mamba_p962_long", "configs/training/sweep/mamba_p962.toml"),
    ("NN-dense", "dense_p515_ga_paper_best", "configs/training/sweep/dense_p515.toml"),
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
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        run()
        reps.append(1000 * (time.perf_counter() - t0) / n_sims)
    reps.sort()
    ms = reps[len(reps) // 2]  # median of 5 timed repeats

    # mean flown duration -> guidance updates per sim (cadence 1 s)
    import pyarrow.parquet as pq

    sim_time = float(pq.read_table(scheme_dir / "final_eval.parquet", columns=["sim_time_s"]).to_pandas()["sim_time_s"].mean())
    return {
        "label": label,
        "n_sims": n_sims,
        "ms_per_sim": round(ms, 4),
        "ms_per_sim_repeats": [round(r, 4) for r in reps],
        "sims_per_s": round(1000 / ms, 1),
        "mean_sim_time_s": round(sim_time, 1),
        "n_guidance_updates": round(sim_time / 1.0),
        "us_per_update_incl_sim": round(1000 * ms / (sim_time / 1.0), 2),
    }


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
        # FNPAG per-REPLAN share: total minus the shared physics/GNC baseline
        # (~= FTC's whole-sim cost), split over its 2 s replan cycles. The
        # per-SIM 86 ms is NOT a per-replan cost -- conflating them overstates
        # flight-processor pressure by ~2 orders of magnitude.
        by = {r["label"]: r for r in results}
        if "FNPAG" in by and "FTC" in by:
            n_replans = by["FNPAG"]["mean_sim_time_s"] / 2.0
            by["FNPAG"]["n_replans"] = round(n_replans)
            by["FNPAG"]["ms_per_replan_derived"] = round((by["FNPAG"]["ms_per_sim"] - by["FTC"]["ms_per_sim"]) / n_replans, 4)
        import platform
        import subprocess

        meta = {
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "rustc": subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(),
            "profile": "release (lto), f64 throughout, n_threads=1, idle box",
            "timing": "median of 5 batch repeats after 1 warmup; per-scheme repeat spread in ms_per_sim_repeats",
            "guidance_cadence_s": 1.0,
        }
        OUT.write_text(json.dumps({"single_core": True, "meta": meta, "schemes": results}, indent=2))
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
