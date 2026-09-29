"""Quote the OU-marginal campaign: every retrained NN cell + the original
frozen-trained cells + classical baselines, scored on BOTH noise regimes
(frozen = legacy pipeline conditioning; marginal = varied noise realization
per scenario) over one shared paired seed pool.

The frozen/marginal protocol matches the 2026-08-27 investigation
(the OU investigation summarized in RESULTS.md): n=1000 shared seeds, marginal additionally sets
simulation.random_seed = 1000 + 7*i, identical across cells.

Usage: uv run python experiments/ou_marginal/quote_marginal.py [--n-sims 1000] [--manifest FILE] [--only LABEL ...] [--force]
Writes experiments/ou_marginal/quote_results.json (the protocol record REGIMES /
SEED_POOL alongside the cells, so the paper's extract can copy it rather than
restate it) and prints the table. The cells are every deployed run under
training_output/ou_marginal/ plus CELLS below, or with `--manifest` the file's
rows instead (one `label|toml[|model_dir]` per line, `#` comments; no model_dir
= a classical cell flown from its optimized TOML, unlike confirmatory_marginal.py).
A row whose best_model.json or optimized TOML does not exist yet is skipped.
A `label/regime` already in the file is kept, not re-scored: `--only` re-scores the
named cells (both regimes) and keeps the rest, `--force` re-quotes the listed cells
into a fresh file (required after a protocol change: n_sims, regimes, seed pool).
The committed file holds the default cells and the classical_cells.txt manifest's:
re-quote with `--force`, then `--manifest classical_cells.txt` without `--force`.
A cell whose model changed since its quote (a resumed training,
audit_deployed_models.py --repair) keeps its old numbers until `--only` re-scores it.

Stop and resume: Ctrl-C at any point, then rerun the command without `--force` /
`--only`. The file is rewritten after every scored `label/regime`, so only the
one in flight is flown again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from aerocapture.training.cell_eval import evaluate_cell
from aerocapture.training.deploy_overrides import LEGACY_NOISE_REGIME
from aerocapture.training.evaluate import constraint_violation_rates
from aerocapture.training.report import read_cost_kwargs
from confirmatory_marginal import read_manifest

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "quote_results.json"

# The protocol, written into quote_results.json so downstream extracts copy it instead of
# restating it: both regimes pin the legacy (shared-path) seeding; the marginal one gives
# scenario i its own density-noise path through a per-seed simulation.random_seed override.
# The shared seed pool is one numpy stream over [0, 2^31).
MARGINAL_SEED_BASE, MARGINAL_SEED_STEP = 1000, 7
REGIMES: dict[str, dict[str, str | None]] = {
    "frozen": {"noise_seeding": LEGACY_NOISE_REGIME["monte_carlo.noise_seeding"], "per_seed_override": None},
    "marginal": {
        "noise_seeding": LEGACY_NOISE_REGIME["monte_carlo.noise_seeding"],
        "per_seed_override": f"simulation.random_seed = {MARGINAL_SEED_BASE} + {MARGINAL_SEED_STEP} i",
    },
}
SEED_POOL_RNG_SEED, SEED_POOL_HIGH = 987654321, 2**31
SEED_POOL = {"rng": "numpy.random.default_rng", "seed": SEED_POOL_RNG_SEED, "range": [0, SEED_POOL_HIGH]}


def discover_ou_cells() -> list[tuple[str, str, str | None]]:
    """Every deployed run under training_output/ou_marginal/ (scratch s1, ft_*,
    *_s2/_s3 repeats), plus the original pilot. Labels = 'ou_' + dir name; the
    scoring TOML is the family config (its [data] path is irrelevant here --
    the model is pinned via override)."""
    rows: list[tuple[str, str, str | None]] = []
    for d in sorted((REPO / "training_output/ou_marginal").glob("*/best_model.json")):
        name = d.parent.name
        cell = name.removeprefix("ft_")
        for suf in ("_s2", "_s3"):
            cell = cell.removesuffix(suf)
        rows.append((f"ou_{name}", f"configs/training/ou_marginal/{cell}.toml", str(d.parent.relative_to(REPO))))
    pilot = REPO / "training_output/ou_pilot/mamba_p962/best_model.json"
    if pilot.exists():
        rows.append(("ou_pilot_mamba", "configs/training/sweep/mamba_p962.toml", "training_output/ou_pilot/mamba_p962"))
    return rows


# (label, toml, model_dir | None for classicals-via-optimized-toml)
CELLS: list[tuple[str, str, str | None]] = [
    # Frozen-trained originals (reference rows):
    ("mamba_p962", "configs/training/sweep/mamba_p962.toml", "training_output/mamba_p962_long"),
    ("lstm_p1082", "configs/training/sweep/lstm_p1082.toml", "training_output/lstm_p1082_long"),
    ("gru_p1014", "configs/training/sweep/gru_p1014.toml", "training_output/gru_p1014_long"),
    ("dense_p972", "configs/training/sweep/dense_p972.toml", "training_output/dense_p972_ga_paper_best"),
    ("dense_p515", "configs/training/paper/dense_p515_ga.toml", "training_output/dense_p515_ga_paper_best"),
    # Classical baselines (frozen-tuned; robust across regimes per the investigation):
    ("ftc", "training_output/ftc/optimized_ftc.toml", None),
    ("fnpag", "training_output/fnpag/optimized_fnpag.toml", None),
    ("pred_guid", "training_output/pred_guid/optimized_pred_guid.toml", None),
    ("energy_controller", "training_output/energy_controller/optimized_energy_controller.toml", None),
    ("equilibrium_glide", "training_output/equilibrium_glide/optimized_equilibrium_glide.toml", None),
    ("piecewise_constant", "training_output/piecewise_constant/optimized_piecewise_constant.toml", None),
]


def score(toml: str, model_dir: str | None, seeds: np.ndarray, regime: str) -> dict:
    # The ADR-0005 gate's own limits ([flight.constraints] through read_cost_kwargs), read before the flights.
    cost_kwargs = read_cost_kwargs(REPO / toml)
    # The ou_marginal configs bake per_draw into the TOML; pin the regime
    # explicitly so BOTH regimes are scored for every cell regardless of
    # which TOML it trained under. The marginal regime re-draws the OU noise
    # path per seed (simulation.random_seed); legacy shares it (env_idx=0).
    res = evaluate_cell(
        REPO / model_dir if model_dir is not None else None,
        REPO / toml,
        seeds,
        model=REPO / model_dir / "best_model.json" if model_dir is not None else None,
        extra_overrides=LEGACY_NOISE_REGIME,
        per_seed_overrides=[{"simulation.random_seed": float(MARGINAL_SEED_BASE + MARGINAL_SEED_STEP * i)} for i in range(len(seeds))]
        if REGIMES[regime]["per_seed_override"] is not None
        else None,
        sim_timeout_secs=30.0,
    )
    rates = constraint_violation_rates(res.final_records, cost_kwargs)
    assert rates is not None
    cap = res.captured
    dv = res.dv
    p50, p95, p99 = np.percentile(dv, [50, 95, 99])
    return {
        "capture_pct": round(100.0 * cap.mean(), 2),
        "dv_p50": round(float(p50), 2),
        "dv_p95": round(float(p95), 2),
        "dv_p99": round(float(p99), 2),
        "dv_cvar95": round(float(dv[dv >= p95].mean()), 2),
        "heat_load_viol_pct": round(100.0 * rates["heat_load"], 2),
        "heat_flux_viol_pct": round(100.0 * rates["heat_flux"], 2),
        "g_load_viol_pct": round(100.0 * rates["g_load"], 2),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--manifest", type=Path, help="score this file's 'label|toml[|model_dir]' rows instead of the discovered + built-in cells")
    parser.add_argument("--only", nargs="+", metavar="LABEL", help="re-score only these cells and merge them into the existing quote_results.json")
    parser.add_argument("--force", action="store_true", help="re-quote the listed cells into a fresh quote_results.json (required after a protocol change)")
    args = parser.parse_args(argv)
    seeds = np.random.default_rng(SEED_POOL_RNG_SEED).integers(0, SEED_POOL_HIGH, size=args.n_sims)

    cells = read_manifest(args.manifest) if args.manifest else discover_ou_cells() + CELLS
    prior = json.loads(OUT.read_text()) if OUT.exists() else None
    same_protocol = prior is not None and (prior["n_sims"], prior["regimes"], prior["seed_pool"]) == (args.n_sims, REGIMES, SEED_POOL)
    if prior is not None and not same_protocol and (args.only or not args.force):
        raise SystemExit(f"{OUT.name} was quoted under another n_sims / protocol: re-quote every cell instead (--force)")
    out: dict[str, dict] = prior["cells"] if prior is not None and (args.only or not args.force) else {}
    if args.only:
        unknown = set(args.only) - {label for label, _, _ in cells}
        if unknown:
            raise SystemExit(f"unknown cell label(s): {', '.join(sorted(unknown))}")
        cells = [c for c in cells if c[0] in args.only]
        missing = [label for label, _, model_dir in cells if model_dir is not None and not (REPO / model_dir / "best_model.json").exists()]
        if missing:
            raise SystemExit(f"no best_model.json for {', '.join(missing)}: nothing to re-quote")
    for label, toml, model_dir in cells:
        if model_dir is not None and not (REPO / model_dir / "best_model.json").exists():
            print(f"{label:<20} SKIPPED (no best_model.json yet)")
            continue
        if model_dir is None and not (REPO / toml).exists():
            print(f"{label:<20} SKIPPED (no {toml} yet)")
            continue
        for regime in REGIMES:
            if f"{label}/{regime}" in out and not args.only:
                print(f"{label:<20} {regime:<8} already quoted, skipping (--only {label} re-scores it)")
                continue
            m = score(toml, model_dir, seeds, regime)
            out[f"{label}/{regime}"] = m
            tmp = OUT.with_name(f".tmp_{OUT.name}")
            tmp.write_text(json.dumps({"n_sims": args.n_sims, "regimes": REGIMES, "seed_pool": SEED_POOL, "cells": out}, indent=1))
            tmp.replace(OUT)
            print(
                f"{label:<20} {regime:<8} capture {m['capture_pct']:6.1f}%  p50 {m['dv_p50']:7.1f}  "
                f"p95 {m['dv_p95']:7.1f}  cvar95 {m['dv_cvar95']:7.1f}  viol hl/flux/g {m['heat_load_viol_pct']:.1f}/"
                f"{m['heat_flux_viol_pct']:.1f}/{m['g_load_viol_pct']:.1f}%"
            )
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
