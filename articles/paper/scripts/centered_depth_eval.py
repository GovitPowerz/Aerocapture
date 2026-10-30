"""Centered high-regime cells at sizing depth (issue #156): the three centered-Mamba trainer
seeds of Section 7.3 and the two joint-FTC baselines they are compared to, scored on the
reserved 9M stress pool at n = 10 000 under BOTH noise regimes, with bootstrap confidence
intervals and paired seed-versus-baseline deltas.

Same pool, high-regime overrides and scoring machinery as objective_centering_eval.py and
robustness_retrain_eval.py (their n = 1000 numbers are the first 1000 seeds of this pool
under the legacy regime); the per_draw regime (ADR-0006) gives every scenario its own
density-noise path and is the one the paper leads with. One file, both regimes, each
labelled (never mixed in one table).

`--v4` (issue #177) scores the same study retrained under per-scenario noise: the three centered
seeds of experiments/ou_marginal/jobs_centering.txt against the #172 medium joint-FTC retune and
the high-regime joint-FTC retuned the same way (classical/ftc_joint_high), per_draw ONLY, into
centered_depth_v4.json; the arxiv-v3 centered_depth.json stays. Each v4 cell also carries its
`violation_pct` per constraint (heat_flux, g_load, heat_load) over every draw of the pool, against
its TOML's [flight.constraints] limits (the ADR-0005 gate's), so a tail win is read next to the
constraint exceedance it may be bought with.

Usage:
    uv run python articles/paper/scripts/centered_depth_eval.py [--v4] [--n-sims 10000]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.cell_eval import FR_DV_TOTAL  # noqa: E402
from aerocapture.training.deploy_overrides import LEGACY_NOISE_REGIME  # noqa: E402
from aerocapture.training.sizing_tail import CONSTRAINTS, paired_tail, score_cell  # noqa: E402

MC_TOML = "configs/training/paper/objective_centering/mamba_centered_high.toml"
# (label, run_dir under training_output/, training TOML). The s2 / s3 repeats were trained by
# 16_sigma_extras.sh at the s1 allocation; the baselines are robustness_retrain_eval.py's.
SEEDS = [
    ("mamba_centered_s1", "paper/objective_centering/mamba_centered", MC_TOML),
    ("mamba_centered_s2", "paper/sigma_extras/mamba_centered_s2", MC_TOML),
    ("mamba_centered_s3", "paper/sigma_extras/mamba_centered_s3", MC_TOML),
]
BASELINES = [
    ("jointFTC-medium", "paper/joint_reference/ftc", "configs/training/msr_aller_ftc_joint_ref_train.toml"),
    ("jointFTC-high", "paper/robustness_retrain/ftc_joint", "configs/training/paper/robustness_retrain/ftc_joint_high.toml"),
]
REGIMES = {"per_draw": {"monte_carlo.noise_seeding": "per_draw"}, "legacy": LEGACY_NOISE_REGIME}
N_BOOT = 2000
STRESS_OVERRIDES = {
    "monte_carlo.atmosphere.level": "high",
    "monte_carlo.density_perturbation.level": "high",
    "monte_carlo.navigation.level": "high",
    "monte_carlo.nav_filter.level": "high",
}
OUT = REPO / "articles/paper/data/centered_depth.json"
# The v4 study (#177): every cell trained under per-scenario noise, scored under it alone.
V4_SEEDS = [
    (f"mamba_centered_s{s}", f"ou_marginal/centered/mamba_centered_s{s}", f"configs/training/ou_marginal/centered/mamba_centered_s{s}.toml") for s in (1, 2, 3)
]
V4_BASELINES = [
    ("jointFTC-medium", "ou_marginal/classical/ftc_joint", "configs/training/ou_marginal/classical/ftc_joint.toml"),
    ("jointFTC-high", "ou_marginal/classical/ftc_joint_high", "configs/training/ou_marginal/classical/ftc_joint_high.toml"),
]
V4_REGIMES = {"per_draw": REGIMES["per_draw"]}
V4_OUT = REPO / "articles/paper/data/centered_depth_v4.json"


def _fly(run_dir: str, toml: str, regime: str, n_sims: int) -> tuple[list[int], np.ndarray, np.ndarray, dict]:
    """(seeds, captured mask, correction DV per scenario, the sizing-tail record with bootstrap CIs) of
    one cell under one regime on the 9M stress pool; the record's violation rates are against the
    ADR-0005 gate's own limits (the TOML's [flight.constraints]; build_cost_kwargs defaults any
    missing one, so all three are always scored)."""
    from aerocapture.training.cell_eval import evaluate_cell
    from aerocapture.training.report import read_cost_kwargs
    from aerocapture.training.seeds import STRESS_EVAL_SEED_OFFSET

    res = evaluate_cell(
        REPO / "training_output" / run_dir,
        Path(toml),
        pool=(STRESS_EVAL_SEED_OFFSET, n_sims),
        extra_overrides={**REGIMES[regime], **STRESS_OVERRIDES},
        sim_timeout_secs=5.0,
    )
    rec = score_cell(res, read_cost_kwargs(Path(toml)), pool="STRESS_EVAL 9M", n_boot=N_BOOT)
    return res.seeds, res.captured, res.final_records[:, FR_DV_TOTAL], rec


def _paired(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> dict:
    """a versus b on the shared pool (`sizing_tail.paired_tail`): the both-captured mean delta, and the
    scenario-resampled bootstrap of the capture-rate and conditional-CVaR95 deltas (negative = a
    better on the tail, positive = a captures more)."""
    (cap_a, dv_a), (cap_b, dv_b) = a, b
    return paired_tail(dv_a, cap_a, dv_b, cap_b, n_boot=N_BOOT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--v4", action="store_true", help="the #177 per-scenario cells, per_draw only, into centered_depth_v4.json")
    args = parser.parse_args(argv)
    seed_cells, baseline_cells, regimes, out = (V4_SEEDS, V4_BASELINES, V4_REGIMES, V4_OUT) if args.v4 else (SEEDS, BASELINES, REGIMES, OUT)
    for _, run_dir, _ in seed_cells + baseline_cells:
        if not (REPO / "training_output" / run_dir / "best_params.json").exists():
            sys.exit(f"{run_dir} is not deployed: every cell is required (no thinner file)")
    # The seeds share one training TOML whose [data] neural_network deploy path is seed 1's run
    # directory: the seed 2 / 3 repeats once overwrote seed 1's model there (issue #156).
    models = [REPO / "training_output" / run_dir / "best_model.json" for _, run_dir, _ in seed_cells]
    if len({hashlib.sha256(m.read_bytes()).hexdigest() for m in models}) != len(models):
        sys.exit("two centered seeds share one best_model.json: a repeat overwrote another seed's model through the shared deploy path")

    cells: dict[str, list[dict]] = {}
    paired: dict[str, list[dict]] = {}
    for regime in regimes:
        flown: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pool: list[int] | None = None
        for label, run_dir, toml in seed_cells + baseline_cells:
            seeds, cap, dv, rec = _fly(run_dir, toml, regime, args.n_sims)
            if pool is not None and seeds != pool:
                sys.exit(f"{label} flew a different seed pool: the comparisons are paired")
            pool = seeds
            flown[label] = (cap, dv)
            s = {"label": label, **rec}
            if args.v4:  # the per-constraint block the v4 file has carried since #177 (the record's rates, as a dict)
                s["violation_pct"] = {c: rec[f"{c}_viol_pct"] for c in CONSTRAINTS}
            cells.setdefault(regime, []).append(s)
            print(
                f"  {regime:8s} {label:18s} capture {s['capture_pct']:5.1f}% [{s['capture_pct_ci'][0]:5.1f}, {s['capture_pct_ci'][1]:5.1f}]"
                f" | mean {s['dv_mean']:6.1f} | CVaR95 {s['dv_cvar95']:6.1f} [{s['dv_cvar95_ci'][0]:6.1f}, {s['dv_cvar95_ci'][1]:6.1f}]"
                + (f" | violations % {s['violation_pct']}" if args.v4 else "")
            )
        for a, _, _ in seed_cells:
            for b, _, _ in baseline_cells:
                p = {"a": a, "b": b, **_paired(flown[a], flown[b])}
                paired.setdefault(regime, []).append(p)
                print(
                    f"  {regime:8s} {a} - {b}: capture {p['delta_capture_pts']:+.2f} pts {p['delta_capture_pts_ci']}"
                    f" | CVaR95 {p['delta_cvar95']:+.1f} {p['delta_cvar95_ci']}"
                )

    out.write_text(
        json.dumps(
            {
                "stress_overrides": STRESS_OVERRIDES,
                "regimes": regimes,
                "n_sims": args.n_sims,
                "n_boot": N_BOOT,
                "pool": "STRESS_EVAL 9M",
                "cells": cells,
                "paired": paired,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
