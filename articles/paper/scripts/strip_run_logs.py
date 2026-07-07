#!/usr/bin/env python3
"""Shrink the committed run.jsonl.gz training logs by dropping the per-generation
population cost arrays that no paper output needs at full resolution.

Each record is ~96% `all_costs` (the full population's per-individual cost every
generation). The only consumer of the array values is `charts.chart_cost_distribution`
(a box plot that samples ~10 generations), and `paper_stats`/`aggregate_results`
only read `len(all_costs)` for n_pop -- so keeping the array on every Nth generation
is lossless for the paper while removing ~40x of the bytes. Also drops any key whose
value is JSON null (the always-null best_params/gen_best_params in NN runs; populated
ones in classical runs are kept).

Every other field is preserved byte-for-byte. Default is a DRY RUN -- pass --apply to
rewrite. Each rewrite is verified against the real consumers before it replaces the
original; a file that fails verification is left untouched.

    python articles/paper/scripts/strip_run_logs.py            # dry run, all tracked logs
    python articles/paper/scripts/strip_run_logs.py --apply    # rewrite in place
    python articles/paper/scripts/strip_run_logs.py --apply --keep-every 100
    python articles/paper/scripts/strip_run_logs.py --apply --drop-all-costs  # remove entirely
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO / "articles/paper/data/runs"


def _tracked_logs() -> list[Path]:
    """Git-tracked run.jsonl.gz files under the runs bundle."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "articles/paper/data/runs/**run.jsonl.gz"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO / line for line in out.splitlines() if line.strip()]


def _strip_record(rec: dict, keep_costs: bool) -> dict:
    """Return a copy with null-valued keys removed; all_costs kept iff keep_costs."""
    out = {}
    for k, v in rec.items():
        if v is None:
            continue
        if k == "all_costs" and not keep_costs:
            continue
        out[k] = v
    return out


def _keep_costs_indices(records: list[dict], keep_every: int) -> set[int]:
    """Indices whose all_costs is retained: every Nth generation, PLUS the single
    record that defines n_pop (longest all_costs) so max(len(all_costs)) is invariant
    even for short resume fragments whose generations miss every Nth. Under
    --drop-all-costs (keep_every <= 0) only the n_pop anchor survives."""
    keep = set()
    if keep_every > 0:
        keep = {i for i, r in enumerate(records) if r.get("all_costs") and r.get("generation", 0) % keep_every == 0}
    anchor = max(
        (i for i, r in enumerate(records) if r.get("all_costs")),
        key=lambda i: len(records[i]["all_costs"]),
        default=None,
    )
    if anchor is not None:
        keep.add(anchor)
    return keep


def _n_pop(records: list[dict]) -> int:
    return max((len(r.get("all_costs", [])) for r in records), default=0)


def _val_rms(records: list[dict]) -> list[float]:
    return [r["validation"]["rms_cost"] for r in records if r.get("validation")]


def _verify(original: list[dict], stripped: list[dict]) -> None:
    """Assert the stripped records preserve everything the paper consumes.
    Raises AssertionError on any regression."""
    assert len(stripped) == len(original), f"line count changed: {len(stripped)} != {len(original)}"

    # n_pop is recovered via max(len(all_costs)) -- must be unchanged.
    assert _n_pop(stripped) == _n_pop(original), "n_pop (max all_costs length) changed"

    # Validation RMS history feeds aggregate_results._best_val_rms.
    assert _val_rms(stripped) == _val_rms(original), "validation rms_cost history changed"

    # Every non-touched field must be byte-identical per record.
    for o, s in zip(original, stripped, strict=True):
        for k, v in o.items():
            if k == "all_costs" or v is None:
                continue
            assert s.get(k) == v, f"field {k!r} changed at gen {o.get('generation')}"
        # No key survived with a null value, and no new key appeared.
        assert set(s) <= set(o), "unexpected new key introduced"
        for k, v in s.items():
            assert v is not None, f"null value survived for {k!r}"

    # The cost-distribution box plot must still render (samples up to 10 gens).
    try:
        import matplotlib

        matplotlib.use("Agg")
        from aerocapture.training.charts import chart_cost_distribution

        with tempfile.TemporaryDirectory() as td:
            svg = Path(td) / "cost_dist.svg"
            ok = chart_cost_distribution(stripped, svg)
            had = any(r.get("all_costs") for r in original)
            assert ok == had, "chart_cost_distribution renderability changed"
            if ok:
                assert svg.exists() and svg.stat().st_size > 0, "cost-distribution SVG empty"
    except ImportError:
        pass  # matplotlib/aerocapture not importable here -> skip the render check only


def _rewrite(path: Path, keep_every: int, apply: bool) -> tuple[int, int]:
    """Return (old_bytes, new_bytes). Rewrites in place when apply=True and the
    stripped file passes verification."""
    old_bytes = path.stat().st_size
    with gzip.open(path, "rt") as f:
        original = [json.loads(line) for line in f if line.strip()]

    keep = _keep_costs_indices(original, keep_every)
    stripped = [_strip_record(r, i in keep) for i, r in enumerate(original)]
    _verify(original, stripped)

    # Write to a temp .gz to measure the real compressed size (level 9, matching
    # collect_runs.py) and to swap atomically.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".strip_", suffix=".gz")
    try:
        with gzip.GzipFile(fileobj=os.fdopen(fd, "wb"), mode="wb", compresslevel=9, mtime=0) as gz:
            for r in stripped:
                gz.write((json.dumps(r) + "\n").encode())
        new_bytes = os.path.getsize(tmp_name)
        if apply:
            os.replace(tmp_name, path)
        else:
            os.unlink(tmp_name)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return old_bytes, new_bytes


def _fmt(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="specific run.jsonl.gz files (default: all git-tracked ones)")
    ap.add_argument("--apply", action="store_true", help="rewrite in place (default: dry run)")
    ap.add_argument("--keep-every", type=int, default=50, metavar="N", help="keep all_costs on every Nth generation (default 50)")
    ap.add_argument("--drop-all-costs", action="store_true", help="keep all_costs only on the single n_pop-defining record (overrides --keep-every)")
    args = ap.parse_args()

    keep_every = 0 if args.drop_all_costs else args.keep_every
    files = [p.resolve() for p in args.paths] if args.paths else _tracked_logs()
    if not files:
        sys.exit("no run.jsonl.gz files found")

    mode = "APPLYING" if args.apply else "DRY RUN (no writes; pass --apply to rewrite)"
    policy = "drop all_costs entirely" if keep_every == 0 else f"keep all_costs every {keep_every} gens"
    print(f"{mode} -- {policy}, drop null-valued keys\n")

    total_old = total_new = 0
    failures = []
    for path in files:
        try:
            old, new = _rewrite(path, keep_every, args.apply)
        except Exception as e:  # noqa: BLE001 -- report and continue
            failures.append((path, e))
            print(f"  FAIL  {path.relative_to(REPO)}: {e}")
            continue
        total_old += old
        total_new += new
        pct = 100 * (1 - new / old) if old else 0
        print(f"  {_fmt(old):>9} -> {_fmt(new):>9}  (-{pct:4.1f}%)  {path.relative_to(REPO)}")

    if total_old:
        pct = 100 * (1 - total_new / total_old)
        print(f"\ntotal: {_fmt(total_old)} -> {_fmt(total_new)}  saved {_fmt(total_old - total_new)} ({pct:.1f}%)")
    else:
        print("\nnothing processed")
    if failures:
        print(f"\n{len(failures)} file(s) failed verification and were left untouched.")
        sys.exit(1)
    if not args.apply:
        print("\nDry run only. Re-run with --apply to rewrite the files in place.")


if __name__ == "__main__":
    main()
