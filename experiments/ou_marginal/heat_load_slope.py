"""The DV-vs-heat-load slope of the headline cells (#192).

The six `hs_<cell>_q<NN>` legs (`campaign.sh experiments/ou_marginal/jobs_heat_load.txt`) are
the #173 seed-1 champions `hl_mamba_p962` / `hl_dense_p515` fine-tuned 2000 gens under
`[flight.constraints] max_heat_load` 25000 (the v4 limit, the control) / 27500 / 30000 kJ/m2.
This script flies every deployed leg on quote_marginal.py's paired n = 1000 marginal pool
twice: at its own ceiling (the training TOML as report.py and quote_marginal.py fly it), and
re-flown under the v4 limit through the explicit `CEILING_KEY` override below, which also sets
the limit the violation column is scored against. The six `hl_<cell>{,_s2,_s3}` sources fly once
at their own 25000 ceiling on the same pool: the standard deviation of their CVaR95 is the seed
spread the slope (CVaR95 at q30 minus q25, per MJ/m2) is read against.

Usage: uv run python experiments/ou_marginal/heat_load_slope.py [--n-sims 1000]
Writes experiments/ou_marginal/heat_load_slope.json and prints the RESULTS.md table; a leg or
source not yet deployed is skipped. Two wiring checks stop the script: a q25 leg must score
identically with and without the override (the override path is the TOML path), and no two
legs of a cell may end on byte-identical checkpoints (a replay: the ceiling never reached the
trainer, since the legs share the copied checkpoint's RNG and population). Two legs may deploy
the same model when neither beat the copied champion; the script prints a note.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
from aerocapture.training import charts
from aerocapture.training.cell_eval import evaluate_cell
from aerocapture.training.deploy_overrides import LEGACY_NOISE_REGIME
from aerocapture.training.evaluate import constraint_violation_rates
from aerocapture.training.report import read_cost_kwargs
from quote_marginal import MARGINAL_SEED_BASE, MARGINAL_SEED_STEP, REGIMES, SEED_POOL, SEED_POOL_HIGH, SEED_POOL_RNG_SEED

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs/training/ou_marginal"
RUNS = REPO / "training_output/ou_marginal"
OUT = Path(__file__).resolve().parent / "heat_load_slope.json"

# The v4 limit: configs/missions/mars.toml [flight.constraints] max_heat_load, kJ/m2. A relaxed
# leg's training TOML carries its own ceiling, so flying it at the v4 limit is this override.
V4_MAX_HEAT_LOAD = 25000.0
CEILING_KEY = "flight.constraints.max_heat_load"
CELLS = ("mamba_p962", "dense_p515")
LEGS = {"q25": 25000.0, "q27": 27500.0, "q30": 30000.0}
SOURCE_SEEDS = ("", "_s2", "_s3")  # hl_<cell>, hl_<cell>_s2, hl_<cell>_s3: the #173 seed spread


def fly(toml: Path, run: Path, seeds: npt.NDArray[np.integer], *, ceiling: float | None) -> dict[str, float]:
    """One row: `run`'s model on the paired marginal pool, at `toml`'s own ceiling (None) or re-flown under `ceiling`."""
    cost_kwargs = read_cost_kwargs(toml)
    overrides: dict[str, object] = dict(LEGACY_NOISE_REGIME)
    if ceiling is not None:
        overrides[CEILING_KEY] = ceiling
        cost_kwargs["heat_load_limit"] = ceiling
    res = evaluate_cell(
        run,
        toml,
        seeds,
        model=run / "best_model.json",
        extra_overrides=overrides,
        per_seed_overrides=[{"simulation.random_seed": float(MARGINAL_SEED_BASE + MARGINAL_SEED_STEP * i)} for i in range(len(seeds))],
        sim_timeout_secs=30.0,
    )
    rates = constraint_violation_rates(res.final_records, cost_kwargs)
    assert rates is not None
    heat_load = res.final_records[:, charts._FR_INTEGRATED_FLUX]
    dv = res.dv
    p50, p95 = np.percentile(dv, [50, 95])
    return {
        "ceiling_kj_m2": float(cost_kwargs["heat_load_limit"]),
        "capture_pct": round(100.0 * float(res.captured.mean()), 2),
        "dv_p50": round(float(p50), 2),
        "dv_p95": round(float(p95), 2),
        "dv_cvar95": round(float(dv[dv >= p95].mean()), 2),
        "dv_max": round(float(dv.max()), 2),
        "heat_load_p95_mj_m2": round(float(np.percentile(heat_load, 95)), 2),
        "heat_load_max_mj_m2": round(float(heat_load.max()), 2),
        "heat_load_viol_pct": round(100.0 * rates["heat_load"], 2),
        "heat_flux_viol_pct": round(100.0 * rates["heat_flux"], 2),
        "g_load_viol_pct": round(100.0 * rates["g_load"], 2),
    }


def summarize(rows: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, object]]:
    """Per cell with both end legs scored: the q30 - q25 CVaR95 delta, its slope per MJ/m2, the source seed spread (sd, ddof = 1)."""
    out: dict[str, dict[str, object]] = {}
    for cell in CELLS:
        q25, q30 = rows.get(f"hs_{cell}_q25/own"), rows.get(f"hs_{cell}_q30/own")
        if q25 is None or q30 is None:
            continue
        delta = q30["dv_cvar95"] - q25["dv_cvar95"]
        seed_cvar = [rows[key]["dv_cvar95"] for s in SOURCE_SEEDS if (key := f"hl_{cell}{s}/own") in rows]
        sd = float(np.std(seed_cvar, ddof=1)) if len(seed_cvar) > 1 else None
        out[cell] = {
            "cvar95_q30_minus_q25": round(delta, 2),
            "slope_m_s_per_mj_m2": round(delta / ((LEGS["q30"] - LEGS["q25"]) / 1000.0), 3),
            "source_seed_cvar95": seed_cvar,
            "source_seed_sd": round(sd, 2) if sd is not None else None,
            "outside_seed_spread": abs(delta) > sd if sd is not None else None,
        }
    return out


def _print_table(rows: Mapping[str, Mapping[str, float]], summary: Mapping[str, Mapping[str, object]]) -> None:
    print("\n| run | flown at | capture | p50 | CVaR95 | max | heat load p95 / max (MJ/m2) | heat load > ceiling | flux > 200 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key, r in rows.items():
        name, pass_ = key.split("/")
        at = f"{r['ceiling_kj_m2'] / 1000.0:.1f}" + (" (v4)" if pass_ == "v4" else "")
        print(
            f"| {name} | {at} | {r['capture_pct']:.1f}% | {r['dv_p50']:.1f} | {r['dv_cvar95']:.1f} | {r['dv_max']:.1f} "
            f"| {r['heat_load_p95_mj_m2']:.2f} / {r['heat_load_max_mj_m2']:.2f} | {r['heat_load_viol_pct']:.1f}% | {r['heat_flux_viol_pct']:.1f}% |"
        )
    for cell, s in summary.items():
        seed_cvar = s["source_seed_cvar95"]
        assert isinstance(seed_cvar, list)
        seeds = " / ".join(f"{v:.1f}" for v in seed_cvar)
        outside = s["outside_seed_spread"]
        verdict = "no seed spread (fewer than two source seeds)" if outside is None else f"{'OUTSIDE' if outside else 'inside'} the seed spread"
        print(
            f"\n{cell}: CVaR95 q30 - q25 = {s['cvar95_q30_minus_q25']} m/s ({s['slope_m_s_per_mj_m2']} m/s per MJ/m2); "
            f"source seeds {seeds}, sd {s['source_seed_sd']}: {verdict}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args(argv)
    seeds = np.random.default_rng(SEED_POOL_RNG_SEED).integers(0, SEED_POOL_HIGH, size=args.n_sims)

    rows: dict[str, dict[str, float]] = {}
    md5: dict[str, str] = {}
    for cell in CELLS:
        for suffix in SOURCE_SEEDS:
            name = f"hl_{cell}{suffix}"
            if not (RUNS / name / "best_model.json").exists():
                print(f"{name:<20} SKIPPED (no best_model.json)")
                continue
            rows[f"{name}/own"] = fly(CONFIG_DIR / f"{name}.toml", RUNS / name, seeds, ceiling=None)
            print(f"{name:<20} own  CVaR95 {rows[f'{name}/own']['dv_cvar95']:7.1f}")
        final_ckpt: dict[str, bytes] = {}
        for leg, ceiling in LEGS.items():
            name = f"hs_{cell}_{leg}"
            toml, run = CONFIG_DIR / f"{name}.toml", RUNS / name
            if not (run / "best_model.json").exists():
                print(f"{name:<20} SKIPPED (no best_model.json yet)")
                continue
            ckpt = max(run.glob("checkpoint_g*.json")).read_bytes()
            model_md5 = hashlib.md5((run / "best_model.json").read_bytes()).hexdigest()
            for prev, prev_ckpt in final_ckpt.items():
                if prev_ckpt == ckpt:
                    raise SystemExit(f"{prev} and {name} end on byte-identical checkpoints: a replay, the ceiling never reached the trainer")
                if md5[prev] == model_md5:
                    print(f"{name:<20} deploys {prev}'s model (neither beat the copied champion): their flights differ through NN input 7 only")
            final_ckpt[name], md5[name] = ckpt, model_md5
            own = fly(toml, run, seeds, ceiling=None)
            if own["ceiling_kj_m2"] != ceiling:
                raise SystemExit(f"{name}: {toml} resolves max_heat_load = {own['ceiling_kj_m2']}, the leg says {ceiling}")
            v4 = fly(toml, run, seeds, ceiling=V4_MAX_HEAT_LOAD)
            if leg == "q25" and v4 != own:
                raise SystemExit(f"{name}: the explicit {CEILING_KEY} = {V4_MAX_HEAT_LOAD} override scores differently from the TOML path:\n{own}\n{v4}")
            rows[f"{name}/own"], rows[f"{name}/v4"] = own, v4
            print(
                f"{name:<20} own  CVaR95 {own['dv_cvar95']:7.1f}  viol {own['heat_load_viol_pct']:.1f}%"
                f"   at v4  CVaR95 {v4['dv_cvar95']:7.1f}  viol {v4['heat_load_viol_pct']:.1f}%"
            )

    summary = summarize(rows)
    OUT.write_text(
        json.dumps(
            {
                "n_sims": args.n_sims,
                "regime": REGIMES["marginal"],
                "seed_pool": SEED_POOL,
                "v4_max_heat_load": V4_MAX_HEAT_LOAD,
                "override": CEILING_KEY,
                "rows": rows,
                "summary": summary,
                "model_md5": md5,
            },
            indent=1,
        )
    )
    _print_table(rows, summary)
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
