"""PTQ scale-jitter sweep (#178): how much of a b-bit PTQ verdict is the rounding realization.

Every cell fake-quantizes the champion's policy tensors at `--bits` with each absmax step
enlarged by (1 + eps), eps on a uniform grid from 0 (the PTQ sweep's own cell) to --eps-max,
for both granularities, and scores it on the PTQ sweep's pool with its scaffolding, cost and
noise regime (quantize.run_quant_sweep's defaults), so the cells differ only in which weights
round up or down. The mean relative weight error is recorded per cell: a larger step does not
grow it over this range. Writes quant_jitter.json next to this file, stamped with the commit
and the model hash, and prints the per-granularity spread.

Usage: uv run python experiments/quant_jitter/quant_jitter.py [--champion-dir DIR] [--toml TOML] [--noise-seeding per_draw] [--n-eps 21] [--eps-max 0.1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from aerocapture.training import quantize as qz

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "quant_jitter.json"
GRANULARITIES = ("per_tensor", "per_channel")


def _rel_err(model: dict, quantized: dict, keys: list[str]) -> float:
    """Mean over the quantized tensors of ||q - w|| / ||w||."""
    errs = []
    for key in keys:
        layer, field = key.split(".")
        w = np.asarray(model["weights"][layer][field], dtype=np.float64)
        errs.append(float(np.linalg.norm(np.asarray(quantized["weights"][layer][field]) - w) / np.linalg.norm(w)))
    return float(np.mean(errs))


def _spread(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cvar = np.array([r["dv_cvar95"] for r in rows], dtype=np.float64)
    cap = np.array([r["capture_rate"] for r in rows], dtype=np.float64)
    return {
        "n_cells": len(rows),
        "n_full_capture": int((cap == 1.0).sum()),
        "min_capture_rate": float(cap.min()),
        "cvar95": {k: round(float(np.percentile(cvar, q)), 1) for k, q in (("min", 0), ("q25", 25), ("median", 50), ("q75", 75), ("max", 100))},
        "mean_rel_err": round(float(np.mean([r["rel_err"] for r in rows])), 4),
    }


def run(champion_dir: Path, toml: str, noise_seeding: str, bits: int, tensor_policy: str, n_sims: int, eps_grid: list[float]) -> dict[str, Any]:
    from aerocapture.training.ablation import _load_cost_kwargs
    from aerocapture.training.seeds import HEADLINE_REQUOTE_SEED_OFFSET

    model_path = champion_dir / "best_model.json"
    model = json.loads(model_path.read_text())
    keys = [k for k, *_ in qz._quantizable_tensors(model, tensor_policy)]
    seeds, pool = qz._resolve_pool(toml, HEADLINE_REQUOTE_SEED_OFFSET, n_sims)
    scaff = qz._scaffolding_overrides(champion_dir, require=True)
    cost_kwargs = _load_cost_kwargs(toml, cost_transform="linear")
    baseline = qz._score_variant(toml, model_path, seeds, cost_kwargs, scaff, 120.0, noise_seeding)
    print(f"fp baseline: capture {baseline['capture_rate']:.3f} cvar95 {baseline['dv_cvar95']:.1f}", flush=True)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        cell = Path(tmp) / "model.json"
        for gran in GRANULARITIES:
            for eps in eps_grid:
                quantized = qz.quantize_model_weights(model, bits, gran, tensor_policy, scale_factor=1.0 + eps)
                cell.write_text(json.dumps(quantized))
                m = qz._score_variant(toml, cell, seeds, cost_kwargs, scaff, 120.0, noise_seeding)
                rows.append({"granularity": gran, "eps": eps, "rel_err": round(_rel_err(model, quantized, keys), 4), **m})
                print(f"{gran:<12} eps {eps:.3f}  capture {m['capture_rate']:.3f}  cvar95 {m['dv_cvar95']:6.1f}  rel_err {rows[-1]['rel_err']:.3f}", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    return {
        "commit": commit,
        "toml": toml,
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "noise_seeding": noise_seeding,
        "bits": bits,
        "tensor_policy": tensor_policy,
        "tensors": keys,
        "pool": pool,
        "baseline": baseline,
        "rows": rows,
        "spread": {gran: _spread([r for r in rows if r["granularity"] == gran]) for gran in GRANULARITIES},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PTQ scale-jitter sweep (#178)")
    parser.add_argument("--champion-dir", type=Path, default=Path("training_output/ou_marginal/hl_mamba_p962"))
    parser.add_argument("--toml", default="configs/training/ou_marginal/hl_mamba_p962.toml")
    parser.add_argument("--noise-seeding", default="per_draw", choices=("legacy", "per_draw"))
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--tensor-policy", default="proj_only", choices=("all", "proj_only"))
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--eps-max", type=float, default=0.10)
    parser.add_argument("--n-eps", type=int, default=21)
    args = parser.parse_args(argv)
    eps_grid = [round(float(e), 6) for e in np.linspace(0.0, args.eps_max, args.n_eps)]
    result = run(args.champion_dir, args.toml, args.noise_seeding, args.bits, args.tensor_policy, args.n_sims, eps_grid)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    for gran, s in result["spread"].items():
        c = s["cvar95"]
        print(
            f"{gran}: full capture {s['n_full_capture']}/{s['n_cells']} (min {s['min_capture_rate']:.3f}), "
            f"cvar95 median {c['median']} IQR {c['q25']}-{c['q75']} range {c['min']}-{c['max']}, mean rel_err {s['mean_rel_err']}"
        )
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
