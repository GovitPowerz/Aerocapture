"""Shared pytest fixtures for the Aerocapture test suite."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "src" / "rust" / "target" / "release" / "aerocapture"


@pytest.fixture(scope="session")
def rust_binary() -> Path:
    """Build the Rust simulator once per session, return the binary path."""
    if not BINARY.exists():
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=ROOT / "src" / "rust",
            check=True,
        )
    return BINARY


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Per-test temporary directory for simulation outputs."""
    return tmp_path
