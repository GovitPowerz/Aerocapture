"""Far-tail confirmatory pools under PER-SCENARIO noise (the Appendix E re-run).

Reuses the paper's confirmatory machinery (`_eval_cell`, `make_confirmatory_pools`
-- same 10 x 100k pre-registered pools, same estimators) with one extra override,
`monte_carlo.noise_seeding = per_draw`, and writes to its OWN results file so the
frozen-regime `confirmatory_eval.json` is never mixed with marginal rows.

Cells come from a manifest (one `label|toml[|model_dir]` per line, `#` comments;
model_dir defaults to training_output/<label>, unlike quote_marginal.py where an
omitted model_dir means a classical cell) or from `--cells label:toml[:model_dir]`;
with neither, the default manifest `confirmatory_cells.txt` (the two fine-tuned
champions, the frozen-trained headline champion, FNPAG, the Section 5 PPO cells).
A cell already in confirmatory_marginal.json is skipped; a cell without its
final_eval.parquet is not quotable yet and is skipped too.

Stop and resume: Ctrl-C (or a crash, or a shutdown) at any point, then rerun the
same command. Every finished replicate is on disk under
confirmatory_marginal.json.partial/<label>/ and is loaded, not flown again; the
cell is assembled once all its replicates exist, written into the results file,
and its partial store deleted. The store refuses a replicate flown under another
merged TOML, override set, model or pool, compared by content (delete it to re-fly).

Usage: uv run python -u experiments/ou_marginal/confirmatory_marginal.py [--manifest FILE | --cells ...] [--n 100000]
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
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "confirmatory_cells.txt"


def read_manifest(path: Path) -> list[tuple[str, str, str | None]]:
    """(label, toml, model_dir | None) per `label|toml[|model_dir]` line; blank lines and `#` comments skipped."""
    rows: list[tuple[str, str, str | None]] = []
    for n, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split("|")]
        if len(fields) not in (2, 3) or not all(fields[:2]):
            raise SystemExit(f"{path}:{n}: expected 'label|toml[|model_dir]', got {raw!r}")
        rows.append((fields[0], fields[1], fields[2] if len(fields) == 3 and fields[2] else None))
    return rows


def parse_cells(specs: list[str]) -> list[tuple[str, str, str | None]]:
    """(label, toml, model_dir | None) per `label:toml[:model_dir]` spec."""
    rows: list[tuple[str, str, str | None]] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) not in (2, 3) or not all(parts[:2]):
            raise SystemExit(f"--cells: expected 'label:toml[:model_dir]', got {spec!r}")
        rows.append((parts[0], parts[1], parts[2] if len(parts) == 3 else None))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--cells", nargs="+", default=[], metavar="LABEL:TOML[:MODEL_DIR]")
    parser.add_argument("--manifest", type=Path, help=f"one 'label|toml[|model_dir]' per line (default {DEFAULT_MANIFEST.name} when no --cells)")
    args = parser.parse_args(argv)

    import confirmatory_eval as ce  # type: ignore[import-not-found]  # articles/paper/scripts via sys.path
    from aerocapture.training.seeds import make_confirmatory_pools
    from aerocapture.training.toml_utils import load_toml_with_bases

    manifest = args.manifest or (None if args.cells else DEFAULT_MANIFEST)
    cells = parse_cells(args.cells) + (read_manifest(manifest) if manifest else [])

    # Same pools as the paper's confirmatory: every cell must share the base MC seed
    # or the paired replicate deltas would silently break.
    seeds = {label: load_toml_with_bases(REPO / toml).get("monte_carlo", {}).get("seed", 42) for label, toml, _ in cells}
    assert len(set(seeds.values())) == 1, f"base_mc_seed differs across cells: {seeds}"
    pools = make_confirmatory_pools(next(iter(seeds.values())), args.replicates, args.n)

    existing: dict = json.loads(OUT.read_text()) if OUT.exists() else {}
    by_label: dict = {c["label"]: c for c in existing.get("cells", [])}
    if existing:
        assert existing.get("n_replicates") == args.replicates and existing.get("n_per_replicate") == args.n, (
            f"pool shape mismatch vs existing {OUT.name} ({existing.get('n_replicates')}x{existing.get('n_per_replicate')})"
        )
    freeze_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    for label, toml, model_dir in cells:
        if label in by_label:
            print(f"{label}: already done, skipping")
            continue
        cell_dir = REPO / (model_dir or f"training_output/{label}")
        # Quotable only once its final eval ran (an RL dir carries a best_model.json from the
        # first promotion, long before the run is over).
        if not (cell_dir / "final_eval.parquet").exists():
            print(f"{label}: no final_eval.parquet yet, skipping")
            continue
        print(f"== {label} ({toml})", flush=True)
        by_label[label] = ce._eval_cell(
            label, toml, pools, None, {}, cell_dir=cell_dir, sim_timeout=5.0, noise_seeding="per_draw", store=ce._partial_store(OUT, label)
        )
        by_label[label]["eval_commit"] = freeze_commit  # the file-level freeze_commit is the first run's; rows added later record their own
        ce._write_atomic(
            OUT,
            json.dumps(
                {
                    "regime": "per_draw (marginal noise)",
                    "noise_seeding": "per_draw",
                    "freeze_commit": existing.get("freeze_commit", freeze_commit),
                    "n_replicates": args.replicates,
                    "n_per_replicate": args.n,
                    "cells": [by_label[k] for k in sorted(by_label)],
                },
                indent=1,
            ),
        )
        ce._drop_partial(OUT, label)
        print(f"  written {OUT.name}", flush=True)
    for label in sorted(by_label):
        p = by_label[label]["pooled"]
        print(f"{label:<28} cap {by_label[label]['replicate_stats']['capture_pct']['mean']:6.2f}%  cvar95 {p['cvar95']:7.2f}  cvar999 {p['cvar999']:7.2f}")


if __name__ == "__main__":
    main()
