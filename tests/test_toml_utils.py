"""Tests for aerocapture.training.toml_utils — TOML base inheritance."""

from pathlib import Path

import pytest

from aerocapture.training.toml_utils import load_toml_with_bases


def _write(path: Path, content: str) -> Path:
    """Write a TOML string to *path* and return the path."""
    path.write_text(content)
    return path


def test_single_base(tmp_path: Path) -> None:
    """One base file, child overlays a key and adds a new one."""
    _write(tmp_path / "base.toml", '[mission]\nplanet = "Mars"\ntype = "MSR"\n')
    _write(tmp_path / "child.toml", 'base = "base.toml"\n[mission]\ntype = "ESR"\n[sim]\ndt = 0.1\n')

    data = load_toml_with_bases(tmp_path / "child.toml")
    assert data["mission"]["planet"] == "Mars"  # inherited
    assert data["mission"]["type"] == "ESR"  # overridden
    assert data["sim"]["dt"] == 0.1  # child-only


def test_multiple_bases_merge_order(tmp_path: Path) -> None:
    """Left-to-right merge: later bases win, child wins over all."""
    _write(tmp_path / "a.toml", '[section]\nx = 1\ny = 10\n')
    _write(tmp_path / "b.toml", '[section]\nx = 2\nz = 30\n')
    _write(tmp_path / "child.toml", 'base = ["a.toml", "b.toml"]\n[section]\nw = 99\n')

    data = load_toml_with_bases(tmp_path / "child.toml")
    assert data["section"]["x"] == 2  # b wins over a
    assert data["section"]["y"] == 10  # from a
    assert data["section"]["z"] == 30  # from b
    assert data["section"]["w"] == 99  # from child


def test_recursive_base(tmp_path: Path) -> None:
    """Grandparent -> parent -> child chain."""
    _write(tmp_path / "gp.toml", '[a]\nval = 1\n[b]\nval = 2\n')
    _write(tmp_path / "parent.toml", 'base = "gp.toml"\n[b]\nval = 20\n[c]\nval = 3\n')
    _write(tmp_path / "child.toml", 'base = "parent.toml"\n[c]\nval = 30\n')

    data = load_toml_with_bases(tmp_path / "child.toml")
    assert data["a"]["val"] == 1  # from grandparent
    assert data["b"]["val"] == 20  # parent overrode grandparent
    assert data["c"]["val"] == 30  # child overrode parent


def test_cycle_detection(tmp_path: Path) -> None:
    """A -> B -> A raises ValueError."""
    _write(tmp_path / "a.toml", 'base = "b.toml"\nx = 1\n')
    _write(tmp_path / "b.toml", 'base = "a.toml"\ny = 2\n')

    with pytest.raises(ValueError, match="Cycle detected"):
        load_toml_with_bases(tmp_path / "a.toml")


def test_missing_base(tmp_path: Path) -> None:
    """Missing base file raises FileNotFoundError."""
    _write(tmp_path / "child.toml", 'base = "nonexistent.toml"\nx = 1\n')

    with pytest.raises(FileNotFoundError):
        load_toml_with_bases(tmp_path / "child.toml")


def test_no_base_passthrough(tmp_path: Path) -> None:
    """No base key = identity (data returned as-is)."""
    _write(tmp_path / "plain.toml", '[section]\nkey = "value"\n')

    data = load_toml_with_bases(tmp_path / "plain.toml")
    assert data == {"section": {"key": "value"}}


def test_base_single_string(tmp_path: Path) -> None:
    """base = \"file.toml\" (single string, not array) works."""
    _write(tmp_path / "base.toml", "x = 1\n")
    _write(tmp_path / "child.toml", 'base = "base.toml"\ny = 2\n')

    data = load_toml_with_bases(tmp_path / "child.toml")
    assert data["x"] == 1
    assert data["y"] == 2


def test_deep_merge_nested(tmp_path: Path) -> None:
    """Deep nesting merges correctly at all levels."""
    _write(
        tmp_path / "base.toml",
        "[a]\n[a.b]\n[a.b.c]\nval = 1\nother = 10\n",
    )
    _write(
        tmp_path / "child.toml",
        'base = "base.toml"\n[a.b.c]\nval = 99\n[a.b.d]\nnew = 42\n',
    )

    data = load_toml_with_bases(tmp_path / "child.toml")
    assert data["a"]["b"]["c"]["val"] == 99  # overridden
    assert data["a"]["b"]["c"]["other"] == 10  # inherited deep
    assert data["a"]["b"]["d"]["new"] == 42  # child-only deep
