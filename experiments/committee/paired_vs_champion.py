"""Paired per-scenario delta of a committee cell against the deployed seed (#232).

Flies both cells on the same confirmatory pools (`make_confirmatory_pools(seed,
replicates, n)`: at the defaults the 10 x 100,000 scenarios of confirmatory_marginal.py;
a smaller --replicates / --n is a fresh draw from the same range, not a prefix of
those; per_draw noise, the scorer's 5 s wall clock) and reports the paired tail
deltas committee minus champion (negative = committee better): CVaR95 / p95 / capture from `sizing_tail.paired_tail`, plus the
far-tail CVaR99.9 delta with the same scenario-resampled bootstrap. Output:
experiments/committee/paired_vs_champion.json, one entry per committee label.

Usage (repo root):
  uv run python -u experiments/committee/paired_vs_champion.py \
      --cell committee/hl_mamba3_circular_mean [--replicates 10] [--n 100000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/python"))
sys.path.insert(0, str(REPO / "experiments/ou_marginal"))

from aerocapture.training.cell_eval import FR_DV_TOTAL, evaluate_cell  # noqa: E402
from aerocapture.training.seeds import make_confirmatory_pools  # noqa: E402
from aerocapture.training.sizing_tail import cvar, paired_tail  # noqa: E402
from aerocapture.training.toml_utils import load_toml_with_bases  # noqa: E402
from confirmatory_marginal import read_manifest  # type: ignore[import-not-found]  # noqa: E402

OUT = Path(__file__).resolve().parent / "paired_vs_champion.json"
MANIFEST = Path(__file__).resolve().parent / "committee_cells.txt"
CHAMPION = ("ou_marginal/hl_mamba_p962", "configs/training/ou_marginal/hl_mamba_p962.toml", "training_output/ou_marginal/hl_mamba_p962")
PER_DRAW = {"monte_carlo.noise_seeding": "per_draw"}


def fly(toml: str, model_dir: str, pools: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    dv, cap = [], []
    for k, seeds in enumerate(pools):
        t0 = time.time()
        res = evaluate_cell(REPO / model_dir, REPO / toml, seeds, extra_overrides=PER_DRAW, sim_timeout_secs=5.0)
        dv.append(res.final_records[:, FR_DV_TOTAL])
        cap.append(res.captured)
        print(f"    replicate {k}: {len(seeds)} sims in {time.time() - t0:.0f} s, captured {int(res.captured.sum())}", flush=True)
    return np.concatenate(dv), np.concatenate(cap)


def delta_cvar999(dv_a: np.ndarray, cap_a: np.ndarray, dv_b: np.ndarray, cap_b: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n = len(dv_a)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = cvar(dv_a[idx][cap_a[idx]], 0.999) - cvar(dv_b[idx][cap_b[idx]], 0.999)
    point = cvar(dv_a[cap_a], 0.999) - cvar(dv_b[cap_b], 0.999)
    return {
        "delta_cvar999": round(float(point), 2),
        "delta_cvar999_ci": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True, help="committee label from committee_cells.txt")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--n", type=int, default=100_000)
    args = parser.parse_args(argv)
    cells = {label: (toml, model_dir or f"training_output/{label}") for label, toml, model_dir in read_manifest(MANIFEST)}
    toml, model_dir = cells[args.cell]
    # The driver's rule: both sides share the base MC seed or the pools (and the pairing) differ.
    seeds = {load_toml_with_bases(REPO / t).get("monte_carlo", {}).get("seed", 42) for t in (toml, CHAMPION[1])}
    assert len(seeds) == 1, f"base_mc_seed differs between {args.cell} and the champion: {seeds}"
    pools = make_confirmatory_pools(seeds.pop(), args.replicates, args.n)

    print(f"== champion {CHAMPION[0]}", flush=True)
    dv_b, cap_b = fly(CHAMPION[1], CHAMPION[2], pools)
    print(f"== committee {args.cell}", flush=True)
    dv_a, cap_a = fly(toml, model_dir, pools)

    stats = {**paired_tail(dv_a, cap_a, dv_b, cap_b), **delta_cvar999(dv_a, cap_a, dv_b, cap_b)}
    entry = {
        "champion": CHAMPION[0],
        "pool": f"CONFIRMATORY {args.replicates} x {args.n}",
        "noise_seeding": "per_draw",
        "committee": {"n_captured": int(cap_a.sum()), "cvar95": round(float(cvar(dv_a[cap_a], 0.95)), 2), "cvar999": round(float(cvar(dv_a[cap_a], 0.999)), 2)},
        "champion_stats": {
            "n_captured": int(cap_b.sum()),
            "cvar95": round(float(cvar(dv_b[cap_b], 0.95)), 2),
            "cvar999": round(float(cvar(dv_b[cap_b], 0.999)), 2),
        },
        "paired": stats,
    }
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing[args.cell] = entry
    OUT.write_text(json.dumps(existing, indent=1) + "\n")
    print(json.dumps(entry, indent=1))


if __name__ == "__main__":
    main()
