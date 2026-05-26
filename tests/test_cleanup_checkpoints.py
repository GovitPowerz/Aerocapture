"""Tests for the checkpoint pruning helper + CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from aerocapture.training.cleanup_checkpoints import (
    PruneResult,
    main,
    prune_checkpoints,
)


def _make_ckpt(d: Path, gen: int, npz_size: int = 1024) -> tuple[Path, Path]:
    """Create a checkpoint_g{gen}.json + .npz pair with `npz_size` bytes payload."""
    j = d / f"checkpoint_g{gen:05d}.json"
    n = d / f"checkpoint_g{gen:05d}.npz"
    j.write_text("{}")
    n.write_bytes(b"\x00" * npz_size)
    return j, n


def test_keeps_last_n_and_deletes_older(tmp_path: Path) -> None:
    for gen in (10, 20, 30, 40, 50):
        _make_ckpt(tmp_path, gen)
    result = prune_checkpoints(tmp_path, keep_last=2)
    assert result.kept == [40, 50]
    assert result.deleted == [10, 20, 30]
    # Files actually removed
    assert not (tmp_path / "checkpoint_g00010.json").exists()
    assert not (tmp_path / "checkpoint_g00030.npz").exists()
    assert (tmp_path / "checkpoint_g00040.json").exists()
    assert (tmp_path / "checkpoint_g00050.npz").exists()


def test_dry_run_does_not_delete(tmp_path: Path) -> None:
    for gen in (10, 20, 30):
        _make_ckpt(tmp_path, gen)
    result = prune_checkpoints(tmp_path, keep_last=1, dry_run=True)
    assert result.deleted == [10, 20]
    assert result.kept == [30]
    # All files still exist on disk
    for gen in (10, 20, 30):
        assert (tmp_path / f"checkpoint_g{gen:05d}.json").exists()
        assert (tmp_path / f"checkpoint_g{gen:05d}.npz").exists()


def test_keep_last_exceeds_available_keeps_all(tmp_path: Path) -> None:
    for gen in (10, 20):
        _make_ckpt(tmp_path, gen)
    result = prune_checkpoints(tmp_path, keep_last=10)
    assert result.kept == [10, 20]
    assert result.deleted == []


def test_empty_dir_no_error(tmp_path: Path) -> None:
    result = prune_checkpoints(tmp_path, keep_last=5)
    assert result == PruneResult(save_dir=tmp_path, kept=[], deleted=[], bytes_freed=0)


def test_partial_pair_is_pruned(tmp_path: Path) -> None:
    """A checkpoint with only .json (no .npz) still counts as a generation
    and gets deleted when it falls out of the keep_last window."""
    _make_ckpt(tmp_path, 10)
    _make_ckpt(tmp_path, 20)
    # Orphan json at gen 5 (e.g. interrupted write)
    (tmp_path / "checkpoint_g00005.json").write_text("{}")
    result = prune_checkpoints(tmp_path, keep_last=2)
    assert result.kept == [10, 20]
    assert result.deleted == [5]
    assert not (tmp_path / "checkpoint_g00005.json").exists()


def test_preserves_analysis_artifacts(tmp_path: Path) -> None:
    """The pruner only touches checkpoint_g*.{json,npz}; everything else stays."""
    for gen in (10, 20):
        _make_ckpt(tmp_path, gen)
    keep_these = [
        "best_model.json",
        "best_params.json",
        "run_000_20260101T000000.jsonl",
        "warm_start_chromosome.npy",
        "warm_start_cache_key.json",
        "warm_start_loss.json",
        "warm_start_baseline.json",
        "corridor_boundaries.npz",
        "ref_trajectory.dat",
        "final_eval.parquet",
        "report.pdf",
    ]
    for name in keep_these:
        (tmp_path / name).write_text("kept")
    prune_checkpoints(tmp_path, keep_last=1)
    for name in keep_these:
        assert (tmp_path / name).exists(), f"pruner wrongly deleted {name}"


def test_bytes_freed_reports_correct_total(tmp_path: Path) -> None:
    _make_ckpt(tmp_path, 10, npz_size=2048)
    _make_ckpt(tmp_path, 20, npz_size=2048)
    result = prune_checkpoints(tmp_path, keep_last=1)
    # gen 10 deleted: 2-byte json ("{}") + 2048-byte npz
    assert result.bytes_freed == 2 + 2048
    assert result.deleted == [10]


def test_rejects_keep_last_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last must be >= 1"):
        prune_checkpoints(tmp_path, keep_last=0)


def test_rejects_nonexistent_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prune_checkpoints(tmp_path / "does_not_exist", keep_last=2)


def test_cli_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    for gen in (10, 20, 30):
        _make_ckpt(tmp_path, gen)
    rc = main([str(tmp_path), "--keep-last", "1", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would delete 2" in out
    assert "kept 1" in out
    # No files were actually deleted
    assert (tmp_path / "checkpoint_g00010.npz").exists()


def test_cli_recursive(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    sub_a = tmp_path / "scheme_a"
    sub_a.mkdir()
    sub_b = tmp_path / "scheme_b"
    sub_b.mkdir()
    no_ckpts = tmp_path / "no_ckpts"
    no_ckpts.mkdir()
    (no_ckpts / "best_model.json").write_text("{}")
    for gen in (10, 20):
        _make_ckpt(sub_a, gen)
        _make_ckpt(sub_b, gen)
    rc = main([str(tmp_path), "--recursive", "--keep-last", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    # Both subdirs pruned, no_ckpts skipped
    assert "scheme_a" in out
    assert "scheme_b" in out
    assert "no_ckpts" not in out
    # gen 10 deleted, gen 20 kept in each
    assert not (sub_a / "checkpoint_g00010.npz").exists()
    assert (sub_a / "checkpoint_g00020.npz").exists()
    assert not (sub_b / "checkpoint_g00010.npz").exists()
    assert (sub_b / "checkpoint_g00020.npz").exists()


def test_cli_invalid_keep_last(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rc = main([str(tmp_path), "--keep-last", "0"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--keep-last must be >= 1" in err


def test_cli_invalid_dir(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rc = main([str(tmp_path / "missing"), "--keep-last", "5"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a directory" in err
