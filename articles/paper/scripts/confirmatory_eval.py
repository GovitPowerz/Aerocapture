"""Frozen confirmatory sizing evaluation -- the R4/R5 revision's headline pool.

Selection-disjoint by construction: seeds are drawn from [2^31, 2^32) (every
historical pool, training draw, and curation probe lives in [0, 2^31)), the
pool was generated AFTER all methodology / architecture / checkpoint choices
were frozen (freeze_commit recorded), and each cell is evaluated on it exactly
once. R replicate pools of n scenarios each: per-replicate tail statistics give
design-based dispersion (t-based SE over replicates, no bootstrap-on-design
caveat), and replicates share seeds across cells so per-replicate stat
differences are paired.

Estimator: cvar(x, a) = mean of the worst max(1, round((1-a) * n_captured))
captured-DV observations, no interpolation (paper_stats.cvar).

Usage:
    uv run python articles/paper/scripts/confirmatory_eval.py \
        --cells <label:toml[:bundle_key]> ... \
        [--replicates 10] [--n 100000] [--extra-override k=v ...]

Accumulates cells across invocations into articles/paper/data/confirmatory_eval.json
(so the slow FNPAG cell can run separately). Paired deltas are recomputed on
every save from the per-replicate stats of whichever pairs are present.

Stop and resume: Ctrl-C (or a crash, or a shutdown) at any point, then rerun the
same command. Every finished replicate is on disk under <out>.partial/<label>/
(r<NN>.npy, then r<NN>.json) and is loaded, not flown again; the cell is
assembled once all its replicates exist, written into the results file, and its
partial store deleted. The store refuses a replicate flown under another merged
TOML, override set, sim timeout, model or pool, compared by content (delete it to
re-fly). A cell already in the file is skipped; --force re-flies it. make
mc-confirmatory passes --force for every cell of the file, so rerun after an
interruption it reuses the interrupted cell's finished replicates but re-flies the
cells it had completed: to resume by hand, pass --force --cells with the cells not
yet re-flown. Invocations sharing one --out may run concurrently (one cell each):
every save merges the cells the file holds by then.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.paper_stats import cvar  # noqa: E402

OUT = REPO / "articles/paper/data/confirmatory_eval.json"

# Paired comparisons recomputed on save (A, B, name) -- replicates share seeds.
PAIRS = [
    ("mamba_p962_long", "joint_reference/ftc", "mamba_vs_jointftc"),
    ("mamba_p962_long", "fnpag", "mamba_vs_fnpag"),
    ("mamba_p962_long", "dense_p515_ga_paper_best", "mamba_vs_dense515"),
    ("mamba_p962_long", "lstm_p1082_long", "mamba_vs_lstm"),
    ("joint_reference/ftc", "fnpag", "jointftc_vs_fnpag"),
    ("mamba_p962_long", "state_reset/mamba_s1", "mamba_vs_state_reset"),
]

T95_DF9 = 2.262  # t(0.975, df=9) for 10 replicates

REP_KEYS = ("capture_pct", "p50", "p95", "cvar95", "p99", "cvar99", "p999", "p9987", "cvar999", "max")
# Per-constraint violation rates ride along per replicate (ADR-0005: quote feasibility next to the tail),
# with their exact counts: a 2-decimal rate reads fewer than 5 violations per 100,000 as 0.00.
VIOL_KEYS = ("viol_pct", "heat_flux_viol_pct", "g_load_viol_pct", "heat_load_viol_pct")
VIOL_COUNT_KEYS = ("viol_n", "heat_flux_viol_n", "g_load_viol_n", "heat_load_viol_n")


def _r2(v: float) -> float:
    return round(float(v), 2)


def _parse_extra_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
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


def _replicate_stats(x: np.ndarray, n_total: int, n_captured: int, viol: dict[str, float]) -> dict:
    return {
        "n": n_total,
        "n_captured": n_captured,
        "capture_pct": _r2(100.0 * n_captured / n_total),
        "p50": _r2(np.percentile(x, 50)),
        "p95": _r2(np.percentile(x, 95)),
        "cvar95": _r2(cvar(x, 0.95)),
        "p99": _r2(np.percentile(x, 99)),
        "cvar99": _r2(cvar(x, 0.99)),
        "p999": _r2(np.percentile(x, 99.9)),
        "p9987": _r2(np.percentile(x, 99.87)),
        "cvar999": _r2(cvar(x, 0.999)),
        "max": _r2(x.max()),
        **{k: _r2(v) for k, v in viol.items()},
    }


def _agg(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return {
        "mean": _r2(v.mean()),
        "se": _r2(se),
        "ci95": [_r2(v.mean() - T95_DF9 * se), _r2(v.mean() + T95_DF9 * se)],
        "min": _r2(v.min()),
        "max": _r2(v.max()),
    }


def _partial_store(out: Path, label: str) -> Path:
    return out.parent / f"{out.name}.partial" / label


def _drop_partial(out: Path, label: str) -> None:
    """Delete a saved cell's replicate store, then the store's parents it leaves empty (labels nest)."""
    store = _partial_store(out, label)
    shutil.rmtree(store)
    for d in store.parents[: len(Path(label).parts)]:
        if any(d.iterdir()):
            break
        d.rmdir()


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".tmp_{path.name}")
    tmp.write_text(text)
    tmp.replace(path)


def _eval_cell(
    label: str,
    toml: str,
    pools: list[list[int]],
    bundle_key: str | None,
    extra: dict[str, Any],
    *,
    cell_dir: Path,
    sim_timeout: float,
    noise_seeding: str,
    store: Path,
) -> dict:
    """One cell over every pool. Each finished replicate lands in `store` (r<NN>.npy, then r<NN>.json
    as the completion marker), and a replicate found there is loaded instead of flown, so an
    interrupted cell resumes where it stopped. The caller deletes the store once the cell is saved."""
    from aerocapture.training.cell_eval import _resolve_cell, evaluate_cell
    from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES
    from aerocapture.training.report import _read_constraint_limits
    from aerocapture.training.toml_utils import load_toml_with_bases

    hfl, gll, hll = (None, None, None)

    bundle_model = REPO / "articles/paper/data/runs" / bundle_key / "best_model.json" if bundle_key else None
    if bundle_key and (bundle_model is None or not bundle_model.exists()):
        raise SystemExit(f"{label}: bundle key {bundle_key} given but {bundle_model} missing (refusing local fallback)")
    overrides = {"monte_carlo.noise_seeding": noise_seeding, **extra}
    # What a stored replicate must have been flown under to be reused, by content: the merged TOML
    # evaluate_cell will fly (a classical cell's optimized_<scheme>.toml, bases included), every
    # override, the sim timeout (a timed-out sim scores as a failure), and the flown model's bytes
    # (audit_deployed_models.py --repair rewrites one in place) rather than its checkout-specific path.
    eval_toml, cell_overrides = _resolve_cell(cell_dir, Path(toml), bundle_model)
    flown = {**cell_overrides, **overrides}
    model_path = flown.pop("data.neural_network", None)
    flight = {
        "toml": toml,
        "bundle_key": bundle_key,
        "sim_timeout": sim_timeout,
        "overrides": flown,
        "config_sha256": hashlib.sha256(json.dumps(load_toml_with_bases(eval_toml), sort_keys=True).encode()).hexdigest(),
        "model_sha256": hashlib.sha256(Path(str(model_path)).read_bytes()).hexdigest() if model_path else None,
    }

    reps: list[dict] = []
    pooled_parts: list[np.ndarray] = []
    model_used = None
    for r, seeds in enumerate(pools):
        key = {**flight, "seeds_sha256": hashlib.sha256(np.asarray(seeds, dtype=np.int64).tobytes()).hexdigest()}
        done = store / f"r{r:02d}.json"
        if done.exists():
            saved = json.loads(done.read_text())
            if saved["key"] != key:
                raise SystemExit(
                    f"{label}: {done.parent}/r{r:02d} was flown under another config or pool ({saved['key']} != {key}): delete the store to re-fly it"
                )
            rep, model_used, x = saved["replicate"], saved["model"], np.load(done.with_suffix(".npy"))
            reps.append(rep)
            pooled_parts.append(x)
            print(f"  {label} r{r}: loaded from {done.parent}", flush=True)
            continue
        res = evaluate_cell(cell_dir, Path(toml), seeds, model=bundle_model, extra_overrides=overrides, sim_timeout_secs=sim_timeout)
        if hfl is None:
            hfl, gll, hll = _read_constraint_limits(res.toml_path)
        model_used = res.overrides.get("data.neural_network")
        recs = res.final_records
        col = {name: recs[:, idx] for name, idx in zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True)}
        cap = res.captured
        x = np.sort(res.dv)
        over_flux = col["max_heat_flux_kw_m2"] > hfl
        over_g = col["max_load_factor_g"] > gll
        over_hl = col["integrated_flux_mj_m2"] * 1e3 > hll
        over = {"viol": over_flux | over_g | over_hl, "heat_flux_viol": over_flux, "g_load_viol": over_g, "heat_load_viol": over_hl}
        viol = {f"{k}_pct": 100 * float(m.mean()) for k, m in over.items()}
        rep = {"replicate": r, **_replicate_stats(x, len(recs), int(cap.sum()), viol), **{f"{k}_n": int(m.sum()) for k, m in over.items()}}
        failed = [int(s) for s, ok in zip(seeds, cap, strict=True) if not ok][:50]
        if failed:
            rep["failed_seeds"] = failed  # for post-hoc classification (timeout vs physical)
        store.mkdir(parents=True, exist_ok=True)
        npy_tmp = store / f".tmp_r{r:02d}.npy"
        np.save(npy_tmp, x)
        npy_tmp.replace(done.with_suffix(".npy"))
        _write_atomic(done, json.dumps({"key": key, "model": model_used, "replicate": rep}))
        reps.append(rep)
        pooled_parts.append(x)
        print(f"  {label} r{r}: cap={reps[-1]['capture_pct']}% cvar999={reps[-1]['cvar999']} max={reps[-1]['max']}", flush=True)

    pooled_x = np.sort(np.concatenate(pooled_parts))
    n_all = sum(rp["n"] for rp in reps)
    n_cap = sum(rp["n_captured"] for rp in reps)
    step = max(1, len(pooled_x) // 10_000)
    return {
        "label": label,
        "toml": toml,
        "bundle_key": bundle_key,
        "model": model_used,
        "extra_overrides": extra or None,
        "replicates": reps,
        "pooled": {
            "n": n_all,
            "n_captured": n_cap,
            "capture_pct": _r2(100.0 * n_cap / n_all),
            **{k: _r2(np.percentile(pooled_x, p)) for k, p in (("p95", 95), ("p99", 99), ("p999", 99.9), ("p9987", 99.87))},
            **{f"cvar{t}": _r2(cvar(pooled_x, lv)) for t, lv in (("95", 0.95), ("99", 0.99), ("999", 0.999))},
            "max": _r2(pooled_x.max()),
            "n_tail_obs_cvar999": max(1, int(round(0.001 * n_cap))),
            **{k: _r2(float(np.mean([rp[k] for rp in reps]))) for k in VIOL_KEYS},
            **{k: sum(rp[k] for rp in reps) for k in VIOL_COUNT_KEYS},
        },
        "replicate_stats": {k: _agg([rp[k] for rp in reps]) for k in REP_KEYS + VIOL_KEYS},
        "survival_sample": [_r2(v) for v in pooled_x[::step]],
    }


def _paired(by_label: dict[str, dict]) -> dict:
    out: dict[str, dict] = {}
    for a, b, name in PAIRS:
        if a not in by_label or b not in by_label:
            continue
        ra, rb = by_label[a]["replicates"], by_label[b]["replicates"]
        if len(ra) != len(rb):
            continue
        deltas = {f"delta_{k}": [_r2(x[k] - y[k]) for x, y in zip(ra, rb, strict=True)] for k in ("cvar95", "cvar999", "p999", "max")}
        out[name] = {"a": a, "b": b, **{k: _agg(v) for k, v in deltas.items()}}
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", required=True, help="label:toml[:bundle_key]")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument(
        "--extra-override", action="append", default=[], help="k=v applied to every sim (e.g. guidance.neural_network.reset_state_every_tick=true)"
    )
    parser.add_argument(
        "--scaffolding-from",
        default=None,
        help="resolve best_params.json scaffolding from this training_output dir instead of the label's (for ablation cells sharing a source run)",
    )
    parser.add_argument(
        "--sim-timeout",
        type=float,
        default=5.0,
        help="wall-clock timeout per sim (s); raise for slow schemes, a cap below their wall time reports every sim as a failure",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT, help="output JSON (use a separate file for shallow-pool campaigns - pairing requires all cells on the SAME pools)"
    )
    parser.add_argument("--force", action="store_true", help="re-fly the --cells already in --out instead of skipping them")
    parser.add_argument(
        "--noise-seeding",
        choices=("legacy", "per_draw"),
        default="legacy",
        help="noise regime of the pools (ADR-0003 / ADR-0006): the committed confirmatory_eval.json is legacy; per_draw pools live in a separate file",
    )
    args = parser.parse_args(argv)

    from aerocapture.training.seeds import make_confirmatory_pools
    from aerocapture.training.toml_utils import load_toml_with_bases

    extra = _parse_extra_overrides(args.extra_override)

    out_path: Path = args.out
    existing: dict = json.loads(out_path.read_text()) if out_path.exists() else {}
    by_label: dict[str, dict] = {c["label"]: c for c in existing.get("cells", [])}
    if existing:
        assert existing.get("n_replicates") == args.replicates and existing.get("n_per_replicate") == args.n, (
            f"pool shape mismatch vs existing {out_path.name} ({existing.get('n_replicates')}x{existing.get('n_per_replicate')})"
        )
        existing_regime = existing.get("noise_seeding", "legacy")  # files written before the key existed are legacy
        assert existing_regime == args.noise_seeding, (
            f"regime mismatch vs existing {out_path.name}: {existing_regime} != {args.noise_seeding} (one regime per file)"
        )

    specs = []
    base_seed: int | None = None
    for spec in args.cells:
        parts = spec.split(":")
        label, toml = parts[0], parts[1]
        bundle_key = parts[2] if len(parts) > 2 else None
        cell_seed = load_toml_with_bases(REPO / toml).get("monte_carlo", {}).get("seed", 42)
        if base_seed is None:
            base_seed = cell_seed
        assert cell_seed == base_seed, f"{label}: base_mc_seed {cell_seed} != {base_seed} -- pairing across cells would break"
        specs.append((label, toml, bundle_key))

    assert base_seed is not None
    pools = make_confirmatory_pools(base_seed, args.replicates, args.n)

    for label, toml, bundle_key in specs:
        if label in by_label and not args.force:
            print(f"{label}: already in {out_path.name}, skipping (--force re-flies it)", flush=True)
            continue
        src = args.scaffolding_from or label
        cell = _eval_cell(
            label,
            toml,
            pools,
            bundle_key,
            extra,
            # a study cell may be named without its training_output/paper/ prefix
            cell_dir=REPO / "training_output" / "paper" / src
            if "/" in src and not (REPO / "training_output" / src).exists()
            else REPO / "training_output" / src,
            sim_timeout=args.sim_timeout,
            noise_seeding=args.noise_seeding,
            store=_partial_store(out_path, label),
        )
        # Another invocation on the same file (the slow FNPAG cell runs separately) may have saved
        # a cell since startup: merge from disk, or this save would drop it.
        if out_path.exists():
            by_label.update({c["label"]: c for c in json.loads(out_path.read_text()).get("cells", [])})
        by_label[label] = cell
        cells = [by_label[k] for k in sorted(by_label)]
        _write_atomic(
            out_path,
            json.dumps(
                {
                    "freeze_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip(),
                    "pool": "CONFIRMATORY [2^31, 2^32)",
                    "noise_seeding": args.noise_seeding,
                    "base_mc_seed": base_seed,
                    "n_replicates": args.replicates,
                    "n_per_replicate": args.n,
                    "estimator": "cvar(x,a) = mean of worst max(1, round((1-a)*n_captured)) captured-DV obs; agg = t-CI over replicates (df=9)",
                    "cells": cells,
                    "paired": _paired(by_label),
                },
                indent=1,
            ),
        )
        _drop_partial(out_path, label)
        p = by_label[label]["pooled"]
        print(f"{label}: pooled cap={p['capture_pct']}% cvar999={p['cvar999']} (n_tail={p['n_tail_obs_cvar999']}) max={p['max']} -> saved", flush=True)

    print(f"\nwrote {out_path} ({len(by_label)} cells)")


if __name__ == "__main__":
    main()
