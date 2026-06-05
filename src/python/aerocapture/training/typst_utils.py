"""Shared Typst PDF-compilation helpers (single source for the 3 report modules)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def check_typst() -> bool:
    """True if the `typst` CLI is on PATH."""
    return shutil.which("typst") is not None


def compile_typst(
    template_path: Path,
    output_pdf: Path,
    *,
    extra_args: list[str] | None = None,
    label: str = "report",
    timeout: float = 120.0,
) -> bool:
    """Compile a .typ template to PDF. Returns True on success.

    Logs (does not raise) on non-zero exit or timeout so report generation
    degrades gracefully. Caller is responsible for check_typst() / printing
    the typst-absent message before calling here, and for writing/cleaning the
    template file.

    Args:
        template_path: Path to the .typ template.
        output_pdf: Destination PDF path.
        extra_args: Additional arguments inserted between template_path and
            output_pdf, e.g. ``["--root", "/", "--input", "dir=/tmp/x"]``.
        label: Short identifier used in log messages (e.g. ``"report"``).
        timeout: Subprocess timeout in seconds; returns False on expiry.
    """
    cmd = ["typst", "compile", str(template_path)] + (extra_args or []) + [str(output_pdf)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [{label}] typst compile timed out after {timeout:.0f}s")
        return False
    if result.returncode != 0:
        print(f"  [{label}] typst compile failed (rc={result.returncode}): {result.stderr.strip()[:300]}")
        return False
    return True
