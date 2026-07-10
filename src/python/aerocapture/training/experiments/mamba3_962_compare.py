"""Head-to-head tail comparison of the 4 Mamba-3 962-cell arms on one shared pool.

SINGLE run per arm (no seed-repeats), so this reports POINT ESTIMATES of the tail
metrics -- there is NO sigma_run and therefore NO significance test. Treat gaps as
suggestive, not proven; the honest error bar needs seed-repeats. Each model is
scored WITH its co-trained scaffolding (best_params.json nav/shaping), on the
reserved 10M pool (disjoint from every arm's train / validation / final-eval
stream), so all arms see identical held-out scenarios at their true operating point.

    python -m aerocapture.training.experiments.mamba3_962_compare --n-sims 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ARMS = ["baseline", "trapz", "complex", "both"]
OUT_DIR = Path("training_output/mamba3_962")
CONFIG_DIR = Path("configs/training/mamba3_962")


def _score(arm: str, seeds: list[int], sim_timeout: float | None) -> dict[str, float]:
    import aerocapture_rs

    from aerocapture.training import charts
    from aerocapture.training.experiments.probe_common import cvar95 as _cvar95
    from aerocapture.training.report import _load_nn_scaffolding_overrides, compute_eval_summary, read_cost_kwargs

    d = OUT_DIR / arm
    config = CONFIG_DIR / f"{arm}.toml"
    model = d / "best_model.json"
    scaff = _load_nn_scaffolding_overrides(d, d / "__no_optimized_toml__.toml")  # forces best_params.json read
    overrides = [{"simulation.n_sims": 1, "data.neural_network": str(model), **scaff, "monte_carlo.seed": int(s)} for s in seeds]
    batch = aerocapture_rs.run_batch(str(config), overrides, n_threads=None, include_trajectories=False, sim_timeout_secs=sim_timeout)
    final = np.array(batch.final_records, dtype=np.float64)
    summary = compute_eval_summary(final, n_sims=len(seeds), cost_kwargs=read_cost_kwargs(config))
    captured = charts.is_captured(final)
    dv = np.clip(final[captured, charts._FR_DV_TOTAL], charts.DV_FLOOR, charts.DV_CAP)
    cap = summary["captured"]
    return {
        "capture_rate": float(summary["capture_rate"]),
        "dv_p50": float(cap["dv"]["p50"]) if cap else float("nan"),
        "dv_p95": float(cap["dv"]["p95"]) if cap else float("nan"),
        "dv_s3sigma": float(cap["dv"]["s3sigma"]) if cap else float("nan"),
        "cvar95": _cvar95(dv),
    }


def main() -> None:
    from aerocapture.training.evaluate import MAMBA3_EVAL_SEED_OFFSET, make_reserved_seeds

    p = argparse.ArgumentParser(description="Mamba-3 962-cell arm comparison (single-run, tail metrics)")
    p.add_argument("--n-sims", type=int, default=2000, help="shared reserved pool size (>=5000 for a meaningful 3-sigma)")
    p.add_argument("--sim-timeout", type=float, default=None)
    args = p.parse_args()

    seeds = make_reserved_seeds(0, MAMBA3_EVAL_SEED_OFFSET, args.n_sims)
    rows: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        if not (OUT_DIR / arm / "best_model.json").exists():
            print(f"skip {arm}: no best_model.json")
            continue
        rows[arm] = _score(arm, seeds, args.sim_timeout)

    print("\n" + "=" * 74)
    print(f"Mamba-3 962-cell comparison -- SINGLE run/arm, {args.n_sims} shared eval sims")
    print("Point estimates, NO sigma_run: gaps are suggestive, not significance-tested.")
    print("Lead with dv_p95 / CVaR95 (the sizing tail), not p50.")
    print("=" * 74)
    print(f"{'arm':10s} {'params':>6s} {'cap%':>6s} {'dvP50':>8s} {'dvP95':>8s} {'CVaR95':>8s} {'3sig*':>8s}")
    param_counts = {"baseline": 962, "trapz": 978, "complex": 1154, "both": 1170}
    for arm in ARMS:
        r = rows.get(arm)
        if r is None:
            continue
        print(
            f"{arm:10s} {param_counts[arm]:6d} {r['capture_rate'] * 100:6.2f} {r['dv_p50']:8.1f} {r['dv_p95']:8.1f} {r['cvar95']:8.1f} {r['dv_s3sigma']:8.1f}"
        )
    print(f"* 3-sigma (p99.87) is only meaningful at n >= ~5000; at n={args.n_sims} it approximates the max.")

    base = rows.get("baseline")
    if base is None:
        print("\nNo baseline -- cannot compute gaps.")
        return
    print("\nGap vs baseline (negative = better; NOT significance-tested, single run):")
    for metric, label in (("dv_p95", "dvP95"), ("cvar95", "CVaR95")):
        print(f"  [{label}] baseline = {base[metric]:.1f}")
        for arm in ("trapz", "complex", "both"):
            r = rows.get(arm)
            if r is None:
                continue
            gap = r[metric] - base[metric]
            print(f"    {arm:10s} {gap:+8.1f} ({'better' if gap < 0 else 'worse'})")


if __name__ == "__main__":
    main()
