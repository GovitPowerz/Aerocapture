"""Centered high-regime cells at sizing depth (issue #156): the three centered-Mamba trainer
seeds of Section 7.3 and the two joint-FTC references they are compared to, scored on the
reserved 9M stress pool at n = 10 000 under BOTH noise regimes, with bootstrap confidence
intervals and paired seed-versus-reference deltas.

Same pool, high-regime overrides and scoring machinery as objective_centering_eval.py and
robustness_retrain_eval.py (their n = 1000 numbers are the first 1000 seeds of this pool
under the legacy regime); the per_draw regime (ADR-0006) gives every scenario its own
density-noise path and is the one the paper leads with. One file, both regimes, each
labelled (never mixed in one table).

Usage:
    uv run python articles/paper/scripts/centered_depth_eval.py [--n-sims 10000] [--n-boot 2000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.paper_stats import bootstrap_ci, capture_mask, cvar, paired_comparison, run_stats  # noqa: E402

MC_TOML = "configs/training/paper/objective_centering/mamba_centered_high.toml"
# (label, run_dir under training_output/, training TOML). The s2 / s3 repeats were trained by
# 16_sigma_extras.sh at the s1 allocation; the references are robustness_retrain_eval.py's.
SEEDS = [
    ("mamba_centered_s1", "paper/objective_centering/mamba_centered", MC_TOML),
    ("mamba_centered_s2", "paper/sigma_extras/mamba_centered_s2", MC_TOML),
    ("mamba_centered_s3", "paper/sigma_extras/mamba_centered_s3", MC_TOML),
]
REFERENCES = [
    ("jointFTC-medium", "paper/joint_reference/ftc", "configs/training/msr_aller_ftc_joint_ref_train.toml"),
    ("jointFTC-high", "paper/robustness_retrain/ftc_joint", "configs/training/paper/robustness_retrain/ftc_joint_high.toml"),
]
REGIMES = {"per_draw": {"monte_carlo.noise_seeding": "per_draw"}, "legacy": {"monte_carlo.noise_seeding": "legacy"}}
STRESS_OVERRIDES = {
    "monte_carlo.atmosphere.level": "high",
    "monte_carlo.density_perturbation.level": "high",
    "monte_carlo.navigation.level": "high",
    "monte_carlo.nav_filter.level": "high",
}
OUT = REPO / "articles/paper/data/centered_depth.json"


def _fly(run_dir: str, toml: str, regime: str, n_sims: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    """(seeds, captured mask, correction DV per scenario) of one cell under one regime."""
    from aerocapture.training.cell_eval import evaluate_cell
    from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES
    from aerocapture.training.seeds import STRESS_EVAL_SEED_OFFSET

    res = evaluate_cell(
        REPO / "training_output" / run_dir,
        Path(toml),
        pool=(STRESS_EVAL_SEED_OFFSET, n_sims),
        extra_overrides={**REGIMES[regime], **STRESS_OVERRIDES},
        sim_timeout_secs=5.0,
    )
    col = {name: res.final_records[:, idx] for name, idx in zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True)}
    return res.seeds, capture_mask(col["ifinal"], col["eccentricity"]), col["dv_total_m_s"]


def _paired(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], n_boot: int) -> dict:
    """a versus b on the shared pool: the both-captured mean delta (paper_stats), plus a paired
    bootstrap over scenarios of the capture-rate and conditional-CVaR95 deltas (negative = a better
    on the tail, positive = a captures more)."""
    (cap_a, dv_a), (cap_b, dv_b) = a, b
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(cap_a), size=(n_boot, len(cap_a)))
    d_cap = np.array([100 * (cap_a[i].mean() - cap_b[i].mean()) for i in idx])
    d_tail = np.array([cvar(dv_a[i][cap_a[i]], 0.95) - cvar(dv_b[i][cap_b[i]], 0.95) for i in idx])
    r2 = lambda v: round(float(v), 2)  # noqa: E731
    return {
        **paired_comparison(dv_a, cap_a, dv_b, cap_b, n_boot=n_boot),
        "delta_capture_pts": r2(100 * (cap_a.mean() - cap_b.mean())),
        "delta_capture_pts_ci": [r2(np.percentile(d_cap, 2.5)), r2(np.percentile(d_cap, 97.5))],
        "delta_cvar95": r2(cvar(dv_a[cap_a], 0.95) - cvar(dv_b[cap_b], 0.95)),
        "delta_cvar95_ci": [r2(np.percentile(d_tail, 2.5)), r2(np.percentile(d_tail, 97.5))],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    for _, run_dir, _ in SEEDS + REFERENCES:
        if not (REPO / "training_output" / run_dir / "best_params.json").exists():
            sys.exit(f"{run_dir} is not deployed: every cell is required (no thinner file)")

    cells: dict[str, list[dict]] = {}
    paired: dict[str, list[dict]] = {}
    for regime in REGIMES:
        flown: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pool: list[int] | None = None
        for label, run_dir, toml in SEEDS + REFERENCES:
            seeds, cap, dv = _fly(run_dir, toml, regime, args.n_sims)
            assert pool is None or seeds == pool, f"{label} flew a different seed pool: the comparisons are paired"
            pool = seeds
            flown[label] = (cap, dv)
            ifinal = np.where(cap, 3.0, 0.0)  # run_stats re-derives capture from (ifinal, ecc)
            s = {"label": label, **run_stats(ifinal, np.zeros_like(dv), dv, n_boot=args.n_boot)}
            s["capture_pct_ci"] = [round(100 * v, 2) for v in bootstrap_ci(cap.astype(np.float64), np.mean, args.n_boot, 0)]
            cells.setdefault(regime, []).append(s)
            print(
                f"  {regime:8s} {label:18s} capture {s['capture_pct']:5.1f}% [{s['capture_pct_ci'][0]:5.1f}, {s['capture_pct_ci'][1]:5.1f}]"
                f" | mean {s['dv_mean']:6.1f} | CVaR95 {s['dv_cvar95']:6.1f} [{s['dv_cvar95_ci'][0]:6.1f}, {s['dv_cvar95_ci'][1]:6.1f}]"
            )
        for a, _, _ in SEEDS:
            for b, _, _ in REFERENCES:
                p = {"a": a, "b": b, **_paired(flown[a], flown[b], args.n_boot)}
                paired.setdefault(regime, []).append(p)
                print(
                    f"  {regime:8s} {a} - {b}: capture {p['delta_capture_pts']:+.2f} pts {p['delta_capture_pts_ci']}"
                    f" | CVaR95 {p['delta_cvar95']:+.1f} {p['delta_cvar95_ci']}"
                )

    OUT.write_text(
        json.dumps(
            {
                "stress_overrides": STRESS_OVERRIDES,
                "regimes": REGIMES,
                "n_sims": args.n_sims,
                "n_boot": args.n_boot,
                "pool": "STRESS_EVAL 9M",
                "cells": cells,
                "paired": paired,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
