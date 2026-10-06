"""Far-tail confirmatory pools under PER-SCENARIO noise (the Appendix E re-run).

Reuses the paper's confirmatory machinery (`_eval_cell`, `make_confirmatory_pools`
-- same 10 x 100k pre-registered pools, same estimators) with one extra override,
`monte_carlo.noise_seeding = per_draw`, and writes to its OWN results file so the
frozen-regime `confirmatory_eval.json` is never mixed with marginal rows.

Cells come from `--cells label:toml[:model_dir]` and/or a manifest (one
`label|toml[|model_dir]` per line, `#` comments; model_dir defaults to
training_output/<label>, unlike quote_marginal.py where an omitted model_dir means
a classical cell); with neither, the default manifest `confirmatory_cells.txt` (the
two fine-tuned champions, the frozen-trained headline champion, FNPAG, the Section 5
PPO cells); v4's rows are `confirmatory_cells_v4.txt`. A cell already in
confirmatory_marginal.json is skipped; a run that has not finished (neither its
final_eval.parquet nor a population run's end-only final_selection.json) is not
quotable yet and is skipped too; either marker must postdate the run's last
checkpoint. Violations are counted exactly per replicate next to the 2-decimal rates.

Non-captures: once scored, a cell's recorded failed_seeds (the first 50 per
replicate) are re-flown with no wall-clock limit, and their terminal outcomes
(crash, pending crash, hyperbolic, timeout at the simulation's own max_time, or a
capture: the non-capture was the scorer's 5 s timeout) land in the cell as
`non_captures`. The re-fly refuses a best_model.json whose bytes differ from the
row's recorded `model_sha256` (rows scored before the field re-fly at the current
model, as RESULTS.md notes).

Stop and resume: Ctrl-C (or a crash, or a shutdown) at any point, then rerun the
same command. Every finished replicate is on disk under
confirmatory_marginal.json.partial/<label>/ and is loaded, not flown again; the
cell is assembled once all its replicates exist, written into the results file,
and its partial store deleted. The store refuses a replicate flown under another
merged TOML, override set, model or pool, compared by content (delete it to re-fly).
A cell is classified right after it is saved, so an interrupted re-fly reruns on its own.

`--extra-override K=V` (repeatable) flies every cell given under it and is recorded in the row
(`extra_overrides`, reused by the non-capture re-fly); a scored row whose overrides differ stops the
run (#176's reset-state cell: guidance.neural_network.reset_state_every_tick=true).

`--table` flies nothing: it prints the manifest's scored cells as the 10^6 table
(capture from the pooled counts) and evaluates the #173 recipe rule as amended and the #176
controls rule.

Usage: uv run python -u experiments/ou_marginal/confirmatory_marginal.py [--cells ...] [--manifest FILE] [--n 100000] [--extra-override K=V] [--table]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "articles/paper/scripts"))
OUT = Path(__file__).resolve().parent / "confirmatory_marginal.json"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "confirmatory_cells.txt"
# The #173 recipe rule's three-seed cells: S, scratch at 512 x 2; F, the fine-tune of the shared-path champion.
RECIPE_S = tuple(f"ou_marginal/hl_mamba_p962{s}" for s in ("", "_s2", "_s3"))
RECIPE_F = tuple(f"ou_marginal/ft_mamba_p962{s}" for s in ("", "_s2", "_s3"))
# The #176 mechanism controls, retrained at the champion's allocation; read against RECIPE_S's seed range.
CONTROLS = ("ou_marginal/ctrl_window_p970", "ou_marginal/ctrl_mamba_p962_nodv")
PER_DRAW = {"monte_carlo.noise_seeding": "per_draw"}  # every flight's regime; a row's extra_overrides go on top


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
        rows.append((parts[0], parts[1], parts[2] if len(parts) == 3 and parts[2] else None))
    return rows


def table(cells: list[dict]) -> str:
    """The 10^6 table: capture from the pooled counts (the per-replicate 2-decimal capture_pct rounds
    five non-captures per 100,000 to 100.00), violation rates by constraint, non-captures by outcome."""
    lines = ["| cell | capture | non-captures | viol % any (flux / g / heat load) | CVaR95 | CVaR99.9 +- se | max |", "|---|---|---|---|---|---|---|"]
    for c in cells:
        p, nc = c["pooled"], c.get("non_captures")
        n_non = p["n"] - p["n_captured"]
        if not n_non:
            non_captures = "0"
        elif nc is None:
            non_captures = f"{n_non}: unclassified"
        else:
            non_captures = f"{n_non}: " + ", ".join(f"{v} {k}" for k, v in nc["outcomes"].items() if v)
            if nc["n_reflown"] < n_non:
                non_captures += f" ({nc['n_reflown']} re-flown)"
        if "viol_n" in p:
            viol = (
                f"{100 * p['viol_n'] / p['n']:.4f} ("
                + " / ".join(f"{100 * p[k] / p['n']:.4f}" for k in ("heat_flux_viol_n", "g_load_viol_n", "heat_load_viol_n"))
                + ")"
            )
        else:  # scored before exact counts: 2-decimal rates, and before ADR-0005 only the any-constraint and heat-load ones
            viol = (
                f"{p['viol_pct']:.2f} ("
                + " / ".join(f"{p[k]:.2f}" if k in p else "-" for k in ("heat_flux_viol_pct", "g_load_viol_pct", "heat_load_viol_pct"))
                + ")"
            )
        cvar999 = f"{p['cvar999']:.1f} +- {c['replicate_stats']['cvar999']['se']:.1f}"
        lines.append(f"| {c['label']} | {100 * p['n_captured'] / p['n']:.4f}% | {non_captures} | {viol} | {p['cvar95']:.1f} | {cvar999} | {p['max']:.0f} |")
    return "\n".join(lines)


def recipe_rule(by_label: dict[str, dict]) -> dict | None:
    """#173's rule as amended on 2026-10-01, before any 10^6 score of the S cells: every trainer seed
    counts (each passed the same 0% validation gate; its violations are quoted next to it in the
    table). Single-stage scratch iff S <= F + sqrt((sd_S^2 + sd_F^2) / 3) on pooled CVaR99.9, S and F
    the three-seed means. None until all six cells are scored."""
    if not all(k in by_label for k in RECIPE_S + RECIPE_F):
        return None
    s, f = ([by_label[k]["pooled"]["cvar999"] for k in labels] for labels in (RECIPE_S, RECIPE_F))
    threshold = statistics.mean(f) + math.sqrt((statistics.variance(s) + statistics.variance(f)) / 3)
    return {
        "S": round(statistics.mean(s), 2),
        "sd_S": round(statistics.stdev(s), 2),
        "F": round(statistics.mean(f), 2),
        "sd_F": round(statistics.stdev(f), 2),
        "threshold": round(threshold, 2),
        "single_stage": statistics.mean(s) <= threshold,
    }


def controls_rule(by_label: dict[str, dict]) -> dict | None:
    """#176's rule, pre-registered before either control ran: a control whose pooled CVaR99.9 lands
    inside the intact champion's three-seed range (RECIPE_S, min to max) gets seeds 2 and 3 before
    the paper reads it; one outside needs no repeats. None until the three champion seeds are scored."""
    if not all(k in by_label for k in RECIPE_S):
        return None
    seeds = [by_label[k]["pooled"]["cvar999"] for k in RECIPE_S]
    lo, hi = min(seeds), max(seeds)
    controls = {}
    for k in CONTROLS:
        if k in by_label:
            v = by_label[k]["pooled"]["cvar999"]
            inside = lo <= v <= hi
            controls[k] = {"cvar999": v, "inside_range": inside, "seeds_2_3_required": inside}
    return {"champion_range": [lo, hi], "controls": controls, "not_scored": [k for k in CONTROLS if k not in by_label]}


def classify_non_captures(cell: dict, cell_dir: Path, commit: str) -> dict:
    """Terminal outcomes of a cell's recorded `failed_seeds` (the first 50 per replicate) re-flown
    with no wall-clock limit: ifinal 1 / 4 a crash / pending crash, 2 a timeout (the simulation's own
    max_time now), 3 an exit, hyperbolic or a capture (the non-capture was the scorer's 5 s timeout).
    Refuses a model whose bytes differ from the row's `model_sha256` (absent on rows scored before #174)."""
    import confirmatory_eval as ce  # type: ignore[import-not-found]
    from aerocapture.training import cell_eval
    from aerocapture.training.cell_eval import FR_ECC, FR_IFINAL

    toml = REPO / cell["toml"]
    sha = ce._model_sha256(cell_eval._resolve_cell(cell_dir, toml, None)[1].get("data.neural_network"))
    if cell.get("model_sha256", sha) != sha:
        raise SystemExit(f"{cell['label']}: {cell_dir}/best_model.json is not the model the row was scored with: audit it or delete the row to re-score")
    seeds = [s for rep in cell["replicates"] for s in rep.get("failed_seeds", [])]
    overrides = {**PER_DRAW, **(cell.get("extra_overrides") or {})}  # the row's own flight, minus the wall clock
    res = cell_eval.evaluate_cell(cell_dir, toml, seeds, extra_overrides=overrides, sim_timeout_secs=None)
    ifinal, ecc = res.final_records[:, FR_IFINAL], res.final_records[:, FR_ECC]
    outcomes = {"crash": ifinal == 1, "pending_crash": ifinal == 4, "hyperbolic": (ifinal == 3) & (ecc >= 1.0), "timeout": ifinal == 2, "capture": res.captured}
    return {"n_reflown": len(seeds), "outcomes": {k: int(m.sum()) for k, m in outcomes.items()}, "eval_commit": commit, "model_sha256": sha}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--cells", nargs="+", default=[], metavar="LABEL:TOML[:MODEL_DIR]")
    parser.add_argument("--manifest", type=Path, help=f"one 'label|toml[|model_dir]' per line (default {DEFAULT_MANIFEST.name} when no --cells)")
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        metavar="K=V",
        help="applied to every sim of every cell given (#176's reset-state cell: guidance.neural_network.reset_state_every_tick=true)",
    )
    parser.add_argument("--table", action="store_true", help="print the scored cells' 10^6 table and the #173 / #176 rules; fly nothing")
    args = parser.parse_args(argv)

    import confirmatory_eval as ce  # type: ignore[import-not-found]  # articles/paper/scripts via sys.path
    from aerocapture.training.seeds import make_confirmatory_pools
    from aerocapture.training.toml_utils import load_toml_with_bases

    manifest = args.manifest or (None if args.cells else DEFAULT_MANIFEST)
    cells = parse_cells(args.cells) + (read_manifest(manifest) if manifest else [])
    if not cells:
        raise SystemExit(f"no cells: {manifest} has no rows and no --cells given")
    if args.table:
        scored = {c["label"]: c for c in json.loads(OUT.read_text())["cells"]}
        print(table([scored[label] for label, _, _ in cells if label in scored]))
        if missing := [label for label, _, _ in cells if label not in scored]:
            print(f"not scored yet: {', '.join(missing)}")
        rule = recipe_rule(scored)
        print(f"#173 recipe rule: {rule}" if rule else "#173 recipe rule: not evaluable until the three hl_mamba_p962 and three ft_mamba_p962 seeds are scored")
        controls = controls_rule(scored)
        print(f"#176 controls rule: {controls}" if controls else "#176 controls rule: not evaluable until the three hl_mamba_p962 seeds are scored")
        return
    extra = ce._parse_extra_overrides(args.extra_override)

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

    def save(label: str, cell: dict) -> None:
        # Another invocation may have saved a cell since startup: merge from disk, or this save would drop it.
        if OUT.exists():
            by_label.update({c["label"]: c for c in json.loads(OUT.read_text()).get("cells", [])})
        by_label[label] = cell
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
        print(f"  written {OUT.name}", flush=True)

    for label, toml, model_dir in cells:
        cell_dir = REPO / (model_dir or f"training_output/{label}")
        if label in by_label:
            # Compared as flown: the oldest rows recorded the regime itself as their extra_overrides.
            scored_under = by_label[label].get("extra_overrides")
            if {**PER_DRAW, **(scored_under or {})} != {**PER_DRAW, **extra}:
                raise SystemExit(f"{label}: scored under extra_overrides {scored_under}, not {extra or None}: delete the row to re-score")
            print(f"{label}: already done, skipping")
        else:
            # Quotable only once the run is over (an RL dir carries a best_model.json from the first
            # promotion): its final eval, or a population run's end-only final selection, written after
            # its last checkpoint (a run extended past its target keeps the old final_selection.json).
            last_checkpoint = max((f.stat().st_mtime_ns for f in cell_dir.glob("checkpoint*")), default=0)
            if not any((cell_dir / f).exists() and (cell_dir / f).stat().st_mtime_ns > last_checkpoint for f in ("final_eval.parquet", "final_selection.json")):
                print(f"{label}: run not finished (no final_eval.parquet or final_selection.json after its last checkpoint), skipping")
                continue
            print(f"== {label} ({toml})", flush=True)
            store = ce._partial_store(OUT, label)
            cell = ce._eval_cell(label, toml, pools, None, extra, cell_dir=cell_dir, sim_timeout=5.0, noise_seeding="per_draw", store=store)
            cell["eval_commit"] = freeze_commit  # the file-level freeze_commit is the first run's; rows added later record their own
            save(label, cell)
            ce._drop_partial(OUT, label)
        cell = by_label[label]
        if "non_captures" not in cell and cell["pooled"]["n_captured"] < cell["pooled"]["n"]:
            print(f"== {label}: re-flying the recorded non-captures without the sim timeout", flush=True)
            save(label, {**cell, "non_captures": classify_non_captures(cell, cell_dir, freeze_commit)})
    for label in sorted(by_label):
        p = by_label[label]["pooled"]
        print(f"{label:<28} cap {100 * p['n_captured'] / p['n']:8.4f}%  cvar95 {p['cvar95']:7.2f}  cvar999 {p['cvar999']:7.2f}")


if __name__ == "__main__":
    main()
