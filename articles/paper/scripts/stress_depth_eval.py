"""Off-nominal stress at sizing depth on the v4 champion (issue #176): the champion and the four
classical references retuned under per-scenario noise (#172), scored on the reserved 9M stress
pool at n = 10 000 under per-scenario noise ONLY, with bootstrap confidence intervals and paired
champion-versus-classical deltas on capture, mean and conditional CVaR95.

Same pool, high-regime overrides and scoring machinery as centered_depth_eval.py (imported from
it); robustness_stress.py's n = 1000 file on the shared-path cells stays as the arxiv-v3 bundle's.
The regime is stated in the output and never mixed with the shared-path one.

Usage:
    uv run python articles/paper/scripts/stress_depth_eval.py [--n-sims 10000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aerocapture.training.paper_stats import _r2, bootstrap_ci, capture_mask, run_stats  # noqa: E402
from centered_depth_eval import STRESS_OVERRIDES, _fly, _paired  # noqa: E402

TRAINING = REPO / "training_output"
REGIME = "per_draw"
N_BOOT = 2000
# (label, run_dir under training_output/, training TOML): the champion (#173's seed 1) and the
# #172 retunes the v4 performance table quotes; the labels are robustness_stress.py's.
CELLS = [
    ("NN", "ou_marginal/hl_mamba_p962", "configs/training/ou_marginal/hl_mamba_p962.toml"),
    ("joint-FTC", "ou_marginal/classical/ftc_joint", "configs/training/ou_marginal/classical/ftc_joint.toml"),
    ("FTC-fixed", "ou_marginal/classical/ftc", "configs/training/ou_marginal/classical/ftc.toml"),
    ("PredGuid", "ou_marginal/classical/pred_guid", "configs/training/ou_marginal/classical/pred_guid.toml"),
    ("FNPAG", "ou_marginal/classical/fnpag", "configs/training/ou_marginal/classical/fnpag.toml"),
]
OUT = REPO / "articles/paper/data/stress_depth.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=10000)
    args = parser.parse_args(argv)
    for _, run_dir, _ in CELLS:
        if not (TRAINING / run_dir / "best_params.json").exists():
            sys.exit(f"{run_dir} is not deployed: every cell is required (the comparisons are paired)")

    champion = CELLS[0][0]
    flown: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pool: list[int] | None = None
    cells: list[dict] = []
    for label, run_dir, toml in CELLS:
        seeds, ifinal, ecc, dv = _fly(run_dir, toml, REGIME, args.n_sims)
        if pool is not None and seeds != pool:
            sys.exit(f"{label} flew a different seed pool: the comparisons are paired")
        pool = seeds
        cap = capture_mask(ifinal, ecc)
        flown[label] = (cap, dv)
        s = {"label": label, "run_dir": run_dir, **run_stats(ifinal, ecc, dv, n_boot=N_BOOT)}
        s["capture_pct_ci"] = [_r2(100 * v) for v in bootstrap_ci(cap.astype(np.float64), np.mean, N_BOOT)]
        cells.append(s)
        print(
            f"  {REGIME:8s} {label:10s} capture {s['capture_pct']:6.2f}% [{s['capture_pct_ci'][0]:6.2f}, {s['capture_pct_ci'][1]:6.2f}]"
            f" | mean {s['dv_mean']:6.1f} | CVaR95 {s['dv_cvar95']:6.1f} [{s['dv_cvar95_ci'][0]:6.1f}, {s['dv_cvar95_ci'][1]:6.1f}]",
            flush=True,
        )
    paired = []
    for label, _, _ in CELLS[1:]:
        p = {"a": champion, "b": label, **_paired(flown[champion], flown[label])}
        paired.append(p)
        print(
            f"  {champion} - {label}: capture {p['delta_capture_pts']:+.2f} pts {p['delta_capture_pts_ci']}"
            f" | CVaR95 {p['delta_cvar95']:+.1f} {p['delta_cvar95_ci']}"
        )

    OUT.write_text(
        json.dumps(
            {
                "regime": REGIME,
                "noise_seeding": REGIME,
                "stress_overrides": STRESS_OVERRIDES,
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
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
