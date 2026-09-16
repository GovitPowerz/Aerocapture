"""Schema check for articles/paper/data/results.json (make -C articles/paper check-schema).

Fails loudly on the ways aggregate_results.py can degrade: a run whose parquet is
in the bundle but missing from the file, a run summarized without its run.jsonl.gz
(null best_val_rms / no actual_sims) when the log is present, a paired table or
the headline re-quote marked missing, a tail sigma_run group short of its three
seeds, or a run without the noise-regime flag (ADR-0003). Pure stdlib.

Usage: uv run python articles/paper/scripts/check_results_schema.py [results.json]
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO / "articles/paper/data/runs"
DEFAULT = REPO / "articles/paper/data/results.json"

RUN_KEYS = (
    "key",
    "legacy_prefix_regime",
    "n",
    "capture_pct",
    "dv_mean",
    "dv_p50",
    "dv_p95",
    "dv_p99",
    "dv_p999",
    "dv_cvar95",
    "dv_cvar99",
    "dv_max_descriptive",
    "dv_mean_ci",
    "dv_p95_ci",
    "dv_cvar95_ci",
    "dv_p99_ci",
    "dv_cvar99_ci",
    "heat_flux_p95",
    "g_load_p95",
    "best_val_rms_within_transform_only",
)
PAIRED_KEYS = ("a", "b", "delta_p95", "delta_p95_ci", "delta_cvar95", "delta_cvar95_ci")
TAIL_GROUPS = ("mamba_p962", "lstm_p1082", "dense_p515")


def check(path: Path) -> list[str]:
    d = json.loads(path.read_text())
    errors: list[str] = []
    for key in ("runs", "paired", "sigma_run", "headline", "headline_fresh_pool"):
        if key not in d:
            errors.append(f"top-level key missing: {key}")
    if errors:
        return errors

    runs = d["runs"]
    bundle = sorted(str(p.parent.relative_to(RUNS_DIR)) for p in RUNS_DIR.rglob("final_eval.parquet"))
    if sorted(runs) != bundle:
        errors.append(f"runs != bundle parquets: only-in-file {sorted(set(runs) - set(bundle))}, only-in-bundle {sorted(set(bundle) - set(runs))}")
    logs_present = any(RUNS_DIR.rglob("run.jsonl.gz"))
    for key, run in runs.items():
        if run.get("missing"):
            errors.append(f"{key}: marked missing")
            continue
        for k in RUN_KEYS:
            if k not in run:
                errors.append(f"{key}: missing {k}")
        if not isinstance(run.get("legacy_prefix_regime"), bool):
            errors.append(f"{key}: legacy_prefix_regime is not a bool (ADR-0003: the regime is part of the number)")
        if not isinstance(run.get("n"), int) or run["n"] <= 0:
            errors.append(f"{key}: n must be a positive int, got {run.get('n')!r}")
        has_log = (RUNS_DIR / key / "run.jsonl.gz").exists()
        if logs_present and has_log != ("actual_sims" in run):
            errors.append(f"{key}: run.jsonl.gz present={has_log} but actual_sims present={'actual_sims' in run}")
        if run.get("best_val_rms_within_transform_only") is not None and "actual_sims" not in run:
            errors.append(f"{key}: best_val_rms set without actual_sims (log-derived fields must travel together)")

    if d["headline"] not in runs:
        errors.append(f"headline {d['headline']!r} not in runs")
    if d["headline_fresh_pool"].get("missing"):
        errors.append("headline_fresh_pool marked missing")
    for k in ("pool", "n", "capture_pct", "dv_cvar95"):
        if k not in d["headline_fresh_pool"]:
            errors.append(f"headline_fresh_pool: missing {k}")

    for label, p in d["paired"].items():
        if p.get("missing"):
            errors.append(f"paired {label}: marked missing")
            continue
        for k in PAIRED_KEYS:
            if k not in p:
                errors.append(f"paired {label}: missing {k}")
        for k in ("delta_p95_ci", "delta_cvar95_ci"):
            if not (isinstance(p.get(k), list) and len(p[k]) == 2):
                errors.append(f"paired {label}: {k} is not a 2-list")
        for side in ("a", "b"):
            if p.get(side) not in runs:
                errors.append(f"paired {label}: {side}={p.get(side)!r} not in runs")

    tail = d["sigma_run"].get("tail_groups", {})
    for g in TAIL_GROUPS:
        if g not in tail:
            errors.append(f"sigma_run.tail_groups missing {g}")
        elif tail[g].get("n") != 3:
            errors.append(f"sigma_run.tail_groups[{g}].n = {tail[g].get('n')}, expected 3 seeds")
    return errors


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    errors = check(path)
    if errors:
        sys.exit(f"{path}: {len(errors)} schema violation(s)\n  " + "\n  ".join(errors))
    n = len(json.loads(path.read_text())["runs"])
    print(f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path}: schema ok ({n} runs)")


if __name__ == "__main__":
    main()
