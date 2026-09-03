"""Far-tail confirmatory pools under PER-SCENARIO noise (the Appendix E re-run).

Reuses the paper's confirmatory machinery (`_eval_cell`, `make_confirmatory_pools`
-- same 10 x 100k pre-registered pools, same estimators) with one extra override,
`monte_carlo.noise_seeding = per_draw`, and writes to its OWN results file so the
frozen-regime `confirmatory_eval.json` is never mixed with marginal rows.

Cells: the two fine-tuned champions, the frozen-trained headline champion
(before/after contrast), and FNPAG (best classical).

Usage: uv run python experiments/ou_marginal/confirmatory_marginal.py [--n 100000]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "articles/paper/scripts"))
OUT = Path(__file__).resolve().parent / "confirmatory_marginal.json"

CELLS = [
    ("ou_marginal/ft_dense_p515", "configs/training/ou_marginal/dense_p515.toml"),
    ("ou_marginal/ft_mamba_p962", "configs/training/ou_marginal/mamba_p962.toml"),
    ("ou_marginal/ft_mamba_p962_s2", "configs/training/ou_marginal/mamba_p962.toml"),
    ("ou_marginal/ft_mamba_p962_s3", "configs/training/ou_marginal/mamba_p962.toml"),
    ("mamba_p962_long", "configs/training/sweep/mamba_p962.toml"),
    ("fnpag", "configs/training/msr_aller_fnpag_train.toml"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--replicates", type=int, default=10)
    args = parser.parse_args()

    import confirmatory_eval as ce  # type: ignore[import-not-found]  # articles/paper/scripts via sys.path
    from aerocapture.training.evaluate import make_confirmatory_pools
    from aerocapture.training.toml_utils import load_toml_with_bases

    # Same pools as the paper's confirmatory: every cell must share the base MC seed
    # or the paired replicate deltas would silently break.
    seeds = {label: load_toml_with_bases(REPO / toml).get("monte_carlo", {}).get("seed", 42) for label, toml in CELLS}
    assert len(set(seeds.values())) == 1, f"base_mc_seed differs across cells: {seeds}"
    pools = make_confirmatory_pools(next(iter(seeds.values())), args.replicates, args.n)
    extra = {"monte_carlo.noise_seeding": "per_draw"}

    existing: dict = json.loads(OUT.read_text()) if OUT.exists() else {}
    by_label: dict = {c["label"]: c for c in existing.get("cells", [])}
    if existing:
        assert existing.get("n_replicates") == args.replicates and existing.get("n_per_replicate") == args.n, (
            f"pool shape mismatch vs existing {OUT.name} ({existing.get('n_replicates')}x{existing.get('n_per_replicate')})"
        )
    freeze_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    for label, toml in CELLS:
        if label in by_label:
            print(f"{label}: already done, skipping")
            continue
        print(f"== {label} ({toml})", flush=True)
        by_label[label] = ce._eval_cell(label, toml, pools, None, extra)
        OUT.write_text(
            json.dumps(
                {
                    "regime": "per_draw (marginal noise)",
                    "freeze_commit": existing.get("freeze_commit", freeze_commit),
                    "n_replicates": args.replicates,
                    "n_per_replicate": args.n,
                    "cells": [by_label[k] for k in sorted(by_label)],
                },
                indent=1,
            )
        )
        print(f"  written {OUT.name}", flush=True)
    for label in sorted(by_label):
        p = by_label[label]["pooled"]
        print(f"{label:<28} cap {by_label[label]['replicate_stats']['capture_pct']['mean']:6.2f}%  cvar95 {p['cvar95']:7.2f}  cvar999 {p['cvar999']:7.2f}")


if __name__ == "__main__":
    main()
