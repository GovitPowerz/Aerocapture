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

from aerocapture.training.sizing_tail import bootstrap_ci, cvar, score_cell  # noqa: E402

OUT = REPO / "articles/paper/data/far_tail_eval.json"
# This file's spelling of the record (results.typ / figlib read `cvar999`, `p999`, `max`, ...).
_KEYS = {
    "p95": "dv_p95",
    "cvar95": "dv_cvar95",
    "p99": "dv_p99",
    "cvar99": "dv_cvar99",
    "p999": "dv_p999",
    "cvar999": "dv_cvar999",
    "max": "dv_max_descriptive",
}


def _parse_extra_overrides(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        k, _, v = item.partition("=")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _eval_one(label: str, toml: str, n_sims: int, bundle_key: str | None = None, extra: dict | None = None, scaffolding_from: str | None = None) -> dict:
    from aerocapture.training.cell_eval import evaluate_cell
    from aerocapture.training.deploy_overrides import LEGACY_NOISE_REGIME
    from aerocapture.training.report import read_cost_kwargs
    from aerocapture.training.seeds import FINAL_EVAL_SEED_OFFSET

    src = scaffolding_from or label
    scheme_dir = REPO / "training_output" / "paper" / src if "/" in src and not (REPO / "training_output" / src).exists() else REPO / "training_output" / src
    # Pin the committed bundle's frozen weights when a bundle key is given --
    # training_output can drift from the bundle on a later resume (dense_p515
    # did: its local model far-tails at 140.3 vs the bundle's 128.1). Same
    # rationale as collect_appendix.py. Without one, evaluate_cell pins the cell's own model.
    bundle_model = REPO / "articles/paper/data/runs" / bundle_key / "best_model.json" if bundle_key else None
    res = evaluate_cell(
        scheme_dir,
        Path(toml),
        pool=(FINAL_EVAL_SEED_OFFSET, n_sims),
        model=bundle_model if bundle_model is not None and bundle_model.exists() else None,
        extra_overrides={**LEGACY_NOISE_REGIME, **(extra or {})},
        sim_timeout_secs=5.0,
    )
    # The sizing-tail record with its CIs (constraint feasibility on the same pool: the
    # sizing tail must be flown INSIDE the envelope -- a policy that buys its tail with
    # heat-load violations is not a clean competitor; see the LSTM disclosure in the
    # paper's section 6.2), plus the far-tail CIs this n = 10000 pool is for.
    rec = score_cell(res, read_cost_kwargs(res.toml_path), pool="FINAL_EVAL 2M", n_boot=10_000)
    x = res.dv  # seed order, the basis of the record's CIs (the same resamples index the same rows)
    r2 = lambda v: round(float(v), 2)  # noqa: E731
    out = {"label": label, **{k: rec[k] for k in ("n", "n_captured", "capture_pct")}}
    out.update({k: rec[src] for k, src in _KEYS.items()})
    out.update({f"{k}_ci": rec[f"{src}_ci"] for k, src in _KEYS.items() if f"{src}_ci" in rec})
    out["p999_ci"] = [r2(v) for v in bootstrap_ci(x, lambda a: float(np.percentile(a, 99.9)))]
    out["cvar999_ci"] = [r2(v) for v in bootstrap_ci(x, lambda a: cvar(a, 0.999))]
    out.update({k: rec[k] for k in rec if k.endswith("_viol_pct") or k == "viol_pct"})
    out.update({k: rec[k] for k in ("estimator", "noise_seeding", "pool")})
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        nargs="+",
        required=True,
        help="label:toml[:bundle_key] (label = dir under training_output[/paper]; bundle_key pins articles/paper/data/runs/<key>/best_model.json)",
    )
    parser.add_argument("--n-sims", type=int, default=10000, help="full reserved pool (training-disjoint up to 10000)")
    parser.add_argument(
        "--extra-override", action="append", default=[], help="k=v applied to every sim (e.g. guidance.neural_network.reset_state_every_tick=true)"
    )
    parser.add_argument(
        "--scaffolding-from",
        default=None,
        help="resolve best_params.json scaffolding from this training_output dir instead of the label's (for ablation cells sharing a source run)",
    )
    args = parser.parse_args(argv)
    extra = _parse_extra_overrides(args.extra_override)

    # Accumulate by label across runs (don't clobber previously-evaluated cells).
    by_label: dict[str, dict] = {}
    if OUT.exists():
        for c in json.loads(OUT.read_text()).get("cells", []):
            by_label[c["label"]] = c
    for spec in args.cells:
        parts = spec.split(":")
        label, toml = parts[0], parts[1]
        bundle_key = parts[2] if len(parts) > 2 else None
        s = _eval_one(label, toml, args.n_sims, bundle_key=bundle_key, extra=extra, scaffolding_from=args.scaffolding_from)
        by_label[label] = s
        print(f"  {label:28s} n={s['n']} cap={s['capture_pct']:.1f}% | p99 {s['p99']} CVaR99 {s['cvar99']}")
        print(f"  {'':28s} p99.9 {s['p999']} CVaR99.9 {s['cvar999']} max {s['max']}")

    cells = [by_label[k] for k in sorted(by_label)]
    OUT.write_text(json.dumps({"n_sims": args.n_sims, "pool": "FINAL_EVAL 2M", "cells": cells}, indent=2))
    print(f"\nwrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
