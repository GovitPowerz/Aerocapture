"""Prune old GA/PSO training checkpoints, keeping the most recent N.

The bulk of a `training_output/<scheme>/` directory's disk usage is
`checkpoint_g{NNNNN}.npz` files (full population state, often 10-15 MB each
for stateful NN architectures). Only the latest checkpoint is needed to
resume training; older ones are useful only for rollback or replaying the
training animation.

The per-generation JSONL log (`run_*.jsonl`) is the source of truth for
convergence charts and is NOT touched -- analysis (`report.py`,
`compare_guidance.py`) keeps working after pruning.

Per-scheme artifacts that are also preserved:
  - best_model.json / best_params.json (deployed weights)
  - corridor_boundaries.npz, ref_trajectory.dat (piecewise_constant)
  - warm_start_chromosome.npy + warm_start_cache_key.json + warm_start_loss.json
  - warm_start_baseline.json
  - final_eval.parquet, report.pdf
  - run_*.jsonl (per-generation log)

CLI:

    # Dry-run on one directory:
    python -m aerocapture.training.cleanup_checkpoints training_output/neural_network_gru_pso --keep-last 10 --dry-run

    # Apply across all subdirs of training_output/:
    python -m aerocapture.training.cleanup_checkpoints training_output/ --recursive --keep-last 10

    # Apply on a single dir:
    python -m aerocapture.training.cleanup_checkpoints training_output/neural_network_gru_pso --keep-last 10
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_CHECKPOINT_RE = re.compile(r"^checkpoint_g(\d+)\.(json|npz)$")


@dataclass
class PruneResult:
    """Summary of a prune pass on a single directory."""

    save_dir: Path
    kept: list[int]  # generation numbers retained
    deleted: list[int]  # generation numbers deleted
    bytes_freed: int  # cumulative size of deleted .json + .npz files


def _find_checkpoint_generations(save_dir: Path) -> dict[int, list[Path]]:
    """Return {generation: [json_path, npz_path]} for all checkpoint pairs.

    A generation is included if AT LEAST one of (json, npz) exists; partial
    pairs are still tracked so prune can clean them up.
    """
    by_gen: dict[int, list[Path]] = {}
    for entry in save_dir.iterdir():
        if not entry.is_file():
            continue
        m = _CHECKPOINT_RE.match(entry.name)
        if not m:
            continue
        gen = int(m.group(1))
        by_gen.setdefault(gen, []).append(entry)
    return by_gen


def prune_checkpoints(
    save_dir: Path,
    keep_last: int,
    *,
    dry_run: bool = False,
) -> PruneResult:
    """Delete all but the `keep_last` most recent checkpoint pairs in `save_dir`.

    Args:
        save_dir: training output directory containing checkpoint_g*.{json,npz}.
        keep_last: number of most-recent generations to retain. Must be >= 1.
        dry_run: if True, report what WOULD be deleted but don't touch the
            filesystem.

    Returns:
        PruneResult with the kept/deleted generation lists and bytes freed.

    Raises:
        ValueError: if `keep_last < 1`.
        FileNotFoundError: if `save_dir` does not exist.
    """
    if keep_last < 1:
        raise ValueError(f"keep_last must be >= 1, got {keep_last}")
    if not save_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {save_dir}")

    by_gen = _find_checkpoint_generations(save_dir)
    if not by_gen:
        return PruneResult(save_dir=save_dir, kept=[], deleted=[], bytes_freed=0)

    sorted_gens = sorted(by_gen.keys())
    to_keep = set(sorted_gens[-keep_last:])
    to_delete = sorted(set(sorted_gens) - to_keep)

    bytes_freed = 0
    for gen in to_delete:
        for path in by_gen[gen]:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                size = 0
            if not dry_run:
                path.unlink(missing_ok=True)
            bytes_freed += size

    return PruneResult(
        save_dir=save_dir,
        kept=sorted(to_keep),
        deleted=to_delete,
        bytes_freed=bytes_freed,
    )


def _has_checkpoints(d: Path) -> bool:
    """Cheap predicate: does this dir contain any checkpoint pairs?"""
    return any(_CHECKPOINT_RE.match(p.name) for p in d.iterdir() if p.is_file())


def _format_bytes(n: int) -> str:
    """Human-friendly byte size (binary units)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n //= 1024
    return f"{n} B"  # unreachable; pacifies type checker


def _walk_targets(root: Path, recursive: bool) -> list[Path]:
    """Resolve the list of directories to prune.

    When `recursive`, descends one level and picks subdirectories that contain
    any checkpoint files. (Two-level deep dirs aren't supported -- the actual
    training output hierarchy is flat.)
    """
    if not recursive:
        return [root]
    targets: list[Path] = []
    if _has_checkpoints(root):
        targets.append(root)
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and _has_checkpoints(sub):
            targets.append(sub)
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prune old training checkpoints while preserving analysis artifacts.")
    parser.add_argument("save_dir", type=Path, help="Training output dir, or parent dir when --recursive.")
    parser.add_argument("--keep-last", type=int, default=10, help="Number of most-recent checkpoint pairs to retain (default: 10).")
    parser.add_argument("--recursive", action="store_true", help="Walk one level deep and prune every subdir that contains checkpoints.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted without touching the filesystem.")
    args = parser.parse_args(argv)

    if args.keep_last < 1:
        print(f"ERROR: --keep-last must be >= 1, got {args.keep_last}", file=sys.stderr)
        return 2
    if not args.save_dir.is_dir():
        print(f"ERROR: not a directory: {args.save_dir}", file=sys.stderr)
        return 2

    targets = _walk_targets(args.save_dir, args.recursive)
    if not targets:
        print(f"No checkpoint directories found under {args.save_dir}.")
        return 0

    total_freed = 0
    total_deleted = 0
    for d in targets:
        try:
            result = prune_checkpoints(d, keep_last=args.keep_last, dry_run=args.dry_run)
        except (FileNotFoundError, ValueError) as e:
            print(f"  {d}: SKIP ({e})")
            continue
        verb = "would delete" if args.dry_run else "deleted"
        print(f"  {d}: {verb} {len(result.deleted)} checkpoint(s), kept {len(result.kept)}, freed {_format_bytes(result.bytes_freed)}")
        if result.deleted:
            first, last = result.deleted[0], result.deleted[-1]
            print(f"    deleted generations: g{first:05d}..g{last:05d}")
        if result.kept:
            kept_first, kept_last = result.kept[0], result.kept[-1]
            print(f"    kept generations: g{kept_first:05d}..g{kept_last:05d}")
        total_freed += result.bytes_freed
        total_deleted += len(result.deleted)

    if len(targets) > 1:
        suffix = " (dry-run)" if args.dry_run else ""
        print(f"Total: {total_deleted} checkpoints, {_format_bytes(total_freed)} freed{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
