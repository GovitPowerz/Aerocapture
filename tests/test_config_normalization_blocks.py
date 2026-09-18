"""The committed `[network] normalization = [...]` blocks carry a `# N name` comment
column that nothing else checks. Every block must be 35 wide with sequential indices
and names matching the candidate-input contract (Rust `NN_INPUT_NAMES` when the
extension is present, the config.py fallback tuple otherwise -- no importorskip, so
the pure-Python CI job runs it too)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from aerocapture.training.config import _RUNTIME_CANDIDATE_WIDTH, candidate_input_names

REPO = Path(__file__).resolve().parent.parent
_BLOCK_RE = re.compile(r"^normalization\s*=\s*\[(.*?)^\]", re.MULTILINE | re.DOTALL)
_COMMENT_RE = re.compile(r"\}\s*,\s*#\s*(\d+)\s+(\S+)\s*$")

TOMLS = sorted(p for p in (REPO / "configs").rglob("*.toml") if "normalization = [" in p.read_text())


def test_blocks_are_discovered() -> None:
    assert TOMLS, "no config declares a normalization block; the guard is dead"


@pytest.mark.parametrize("path", TOMLS, ids=lambda p: str(p.relative_to(REPO)))
def test_normalization_block_matches_contract(path: Path) -> None:
    names = candidate_input_names()
    text = path.read_text()

    block = tomllib.loads(text)["network"]["normalization"]
    assert len(block) == _RUNTIME_CANDIDATE_WIDTH
    assert all(set(e) == {"transform", "scale", "center"} for e in block)

    m = _BLOCK_RE.search(text)
    assert m, "normalization block not found as text"
    comments = [_COMMENT_RE.search(line) for line in m.group(1).splitlines() if line.strip()]
    assert all(comments), "every entry must end with a `# N name` comment"
    pairs = [(int(c.group(1)), c.group(2)) for c in comments if c]
    assert [i for i, _ in pairs] == list(range(_RUNTIME_CANDIDATE_WIDTH))
    assert [n for _, n in pairs] == names
