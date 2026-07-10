"""Large-pool far-tail evaluation for mission-sizing decisions.

Propellant (ergols) sizing uses a FAR-tail design case -- 3-sigma (~p99.87),
p99.9, or the campaign worst case -- not p95. At n=1000 those quantiles are
estimated from ~1-10 samples (huge variance), so the optimization-process
choice (cost_transform, optimizer) cannot be decided on them. This re-evaluates
the DEPLOYED policy of each requested cell on the full reserved final-eval pool
(2M offset, n=10000 -- training-disjoint) so p99 / p99.9 / CVaR99 / CVaR99.9
become well-estimated (worst 100 / 10), with bootstrap CIs.

Eval-only (no retraining). Pins each cell's run-local best_model.json and its
co-trained scaffolding. Writes articles/paper/data/far_tail_eval.json.

Usage:
    uv run python articles/paper/scripts/far_tail_eval.py \
        --cells cost_transform/linear:configs/training/paper/dense_p3998_ga_transform_linear.toml \
                cost_transform/sqrt:configs/training/paper/dense_p3998_ga_transform_sqrt.toml \
                cost_transform/log:configs/training/paper/dense_p3998_ga_transform_log.toml \
                cost_transform/squared:configs/training/paper/dense_p3998_ga_transform_squared.toml \
                optimizer_budget/ga_300:configs/training/paper/dense_p3998_ga.toml \
        [--n-sims 10000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.paper_stats import bootstrap_ci, cvar  # noqa: E402

OUT = REPO / "articles/paper/data/far_tail_eval.json"


def _eval_one(label: str, toml: str, n_sims: int, bundle_key: str | None = None) -> dict:
    import aerocapture_rs
    from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES
    from aerocapture.training.report import _read_constraint_limits, _resolve_eval_toml
    from aerocapture.training.toml_utils import load_toml_with_bases

    scheme_dir = (
        REPO / "training_output" / "paper" / label if "/" in label and not (REPO / "training_output" / label).exists() else REPO / "training_output" / label
    )
    eval_toml, scaffolding = _resolve_eval_toml(Path(toml), scheme_dir)
    base_mc_seed = load_toml_with_bases(eval_toml).get("monte_carlo", {}).get("seed", 42)
    seeds = make_reserved_seeds(base_mc_seed, FINAL_EVAL_SEED_OFFSET, n_sims)

    base: dict = {"simulation.n_sims": 1, **scaffolding}
    # Pin the committed bundle's frozen weights when a bundle key is given --
    # training_output can drift from the bundle on a later resume (dense_p515
    # did: its local model far-tails at 140.3 vs the bundle's 128.1). Same
    # rationale as collect_appendix.py.
    bundle_model = REPO / "articles/paper/data/runs" / bundle_key / "best_model.json" if bundle_key else None
    local_model = scheme_dir / "best_model.json"
    model = bundle_model if bundle_model is not None and bundle_model.exists() else local_model
    if model.exists():
        base["data.neural_network"] = str(model.resolve())
    overrides = [{**base, "monte_carlo.seed": s} for s in seeds]
    res = aerocapture_rs.run_batch(toml_path=str(eval_toml.resolve()), overrides_list=overrides, sim_timeout_secs=5.0)
    recs = np.asarray(res.final_records)
    col = {name: recs[:, idx] for name, idx in zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True)}
    cap = (col["ifinal"] == 3) & (col["eccentricity"] < 1.0)
    x = np.sort(col["dv_total_m_s"][cap])
    # constraint feasibility on the same pool (the sizing tail must be flown INSIDE
    # the envelope -- a policy that buys its tail with heat-load violations is not
    # a clean competitor; see the LSTM disclosure in the paper's section 6.2)
    hfl, gll, hll = _read_constraint_limits(eval_toml)
    v_hf = col["max_heat_flux_kw_m2"] > hfl
    v_g = col["max_load_factor_g"] > gll
    v_hl = col["integrated_flux_mj_m2"] * 1e3 > hll
    r2 = lambda v: round(float(v), 2)  # noqa: E731
    return {
        "label": label,
        "n": int(len(recs)),
        "n_captured": int(cap.sum()),
        "capture_pct": r2(100 * cap.mean()),
        "p95": r2(np.percentile(x, 95)),
        "cvar95": r2(cvar(x, 0.95)),
        "cvar95_ci": [r2(v) for v in bootstrap_ci(x, lambda a: cvar(a, 0.95))],
        "p99": r2(np.percentile(x, 99)),
        "p99_ci": [r2(v) for v in bootstrap_ci(x, lambda a: float(np.percentile(a, 99)))],
        "cvar99": r2(cvar(x, 0.99)),
        "cvar99_ci": [r2(v) for v in bootstrap_ci(x, lambda a: cvar(a, 0.99))],
        "p999": r2(np.percentile(x, 99.9)),
        "p999_ci": [r2(v) for v in bootstrap_ci(x, lambda a: float(np.percentile(a, 99.9)))],
        "cvar999": r2(cvar(x, 0.999)),
        "cvar999_ci": [r2(v) for v in bootstrap_ci(x, lambda a: cvar(a, 0.999))],
        "max": r2(x.max()),  # descriptive bound, no CI (sample max)
        "viol_pct": r2(100 * (v_hf | v_g | v_hl).mean()),
        "heat_flux_viol_pct": r2(100 * v_hf.mean()),
        "g_load_viol_pct": r2(100 * v_g.mean()),
        "heat_load_viol_pct": r2(100 * v_hl.mean()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", required=True, help="label:toml[:bundle_key] (label = dir under training_output[/paper]; bundle_key pins articles/paper/data/runs/<key>/best_model.json)")
    parser.add_argument("--n-sims", type=int, default=10000, help="full reserved pool (training-disjoint up to 10000)")
    args = parser.parse_args(argv)

    # Accumulate by label across runs (don't clobber previously-evaluated cells).
    by_label: dict[str, dict] = {}
    if OUT.exists():
        for c in json.loads(OUT.read_text()).get("cells", []):
            by_label[c["label"]] = c
    for spec in args.cells:
        parts = spec.split(":")
        label, toml = parts[0], parts[1]
        bundle_key = parts[2] if len(parts) > 2 else None
        s = _eval_one(label, toml, args.n_sims, bundle_key=bundle_key)
        by_label[label] = s
        print(f"  {label:28s} n={s['n']} cap={s['capture_pct']:.1f}% | p99 {s['p99']} CVaR99 {s['cvar99']}")
        print(f"  {'':28s} p99.9 {s['p999']} CVaR99.9 {s['cvar999']} max {s['max']}")

    cells = [by_label[k] for k in sorted(by_label)]
    OUT.write_text(json.dumps({"n_sims": args.n_sims, "pool": "FINAL_EVAL 2M", "cells": cells}, indent=2))
    print(f"\nwrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
