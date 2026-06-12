"""Collect the quotable artifacts of every completed paper run into the committed bundle.

Walks the campaign output trees and, for each run dir holding a final_eval.parquet,
copies {best_model.json, best_params.json, final_eval.parquet, final_selection.json}
and gzips the newest run_*.jsonl into articles/paper/data/runs/<study>/<cell>/.
Idempotent: a destination file is rewritten only when the source is newer.
"""

import argparse
import gzip
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "articles/paper/data/runs"
TRAINING = REPO / "training_output"

ARTIFACTS = ("best_model.json", "best_params.json", "final_eval.parquet", "final_selection.json")
CLASSICAL = ("piecewise_constant", "ftc", "equilibrium_glide", "energy_controller", "pred_guid", "fnpag")


def _run_dirs() -> list[tuple[Path, Path]]:
    """(source run dir, bundle destination) pairs for every completed run."""
    pairs: list[tuple[Path, Path]] = []
    paper = TRAINING / "paper"
    if paper.is_dir():
        for parquet in sorted(paper.rglob("final_eval.parquet")):
            src = parquet.parent
            pairs.append((src, OUT / src.relative_to(paper)))
    for scheme in CLASSICAL:
        src = TRAINING / scheme
        if (src / "final_eval.parquet").exists():
            pairs.append((src, OUT / "classical_baselines" / scheme))
    for src in sorted(TRAINING.glob("sweep_*")):
        if (src / "final_eval.parquet").exists():
            pairs.append((src, OUT / "architecture_sweep" / src.name))
    return pairs


def _copy_if_newer(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _gzip_newest_jsonl(run_dir: Path, dst: Path) -> bool:
    logs = sorted(run_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return False
    src = logs[-1]
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    shutil.copystat(src, dst)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list what would be collected without writing")
    args = parser.parse_args(argv)

    pairs = _run_dirs()
    if not pairs:
        sys.exit(f"No completed runs found under {TRAINING}")
    n_new = 0
    for src, dst in pairs:
        copied: list[str] = []
        if not args.dry_run:
            copied += [a for a in ARTIFACTS if (src / a).exists() and _copy_if_newer(src / a, dst / a)]
            if _gzip_newest_jsonl(src, dst / "run.jsonl.gz"):
                copied.append("run.jsonl.gz")
        status = "would collect" if args.dry_run else (f"updated {', '.join(copied)}" if copied else "up to date")
        print(f"  {src.relative_to(TRAINING)} -> {dst.relative_to(REPO)}  [{status}]")
        n_new += bool(copied)
    print(f"\n{len(pairs)} runs in bundle, {n_new} updated. Remember to `git add articles/paper/data/runs`.")


if __name__ == "__main__":
    main()
