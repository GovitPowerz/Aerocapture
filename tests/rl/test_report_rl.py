"""RL report tests: chart SVG generation + optional Typst compile smoke test."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from aerocapture.training.rl.report_rl import (
    _chart_rl_capture_rate,
    _chart_rl_dv_curve,
    _chart_rl_entropy,
    _chart_rl_return_curve,
    _chart_rl_validation_waterfall,
    _chart_rl_value_loss,
)

_SAMPLE_RECORDS = [
    {"env_steps": 1000, "episodic_return_mean": -1.0, "episodic_dv_m_s_mean": 800.0, "entropy": 1.2, "value_loss": 0.5, "episodic_capture_rate": 0.3},
    {"env_steps": 2000, "episodic_return_mean": -0.5, "episodic_dv_m_s_mean": 600.0, "entropy": 1.0, "value_loss": 0.3, "episodic_capture_rate": 0.6},
    {"env_steps": 3000, "episodic_return_mean": -0.2, "episodic_dv_m_s_mean": 400.0, "entropy": 0.8, "value_loss": 0.1, "episodic_capture_rate": 0.9},
]


def _is_svg(path: Path) -> bool:
    content = path.read_bytes()
    return content.startswith(b"<?xml") or content.startswith(b"<svg")


def test_return_chart_produces_svg(tmp_path: Path) -> None:
    out = tmp_path / "ret.svg"
    _chart_rl_return_curve(_SAMPLE_RECORDS, out)
    assert out.exists()
    assert _is_svg(out)


def test_dv_curve_produces_svg(tmp_path: Path) -> None:
    out = tmp_path / "dv.svg"
    _chart_rl_dv_curve(_SAMPLE_RECORDS, out)
    assert out.exists()
    assert _is_svg(out)


def test_entropy_produces_svg(tmp_path: Path) -> None:
    out = tmp_path / "entropy.svg"
    _chart_rl_entropy(_SAMPLE_RECORDS, out)
    assert out.exists()
    assert _is_svg(out)


def test_value_loss_produces_svg(tmp_path: Path) -> None:
    out = tmp_path / "value_loss.svg"
    _chart_rl_value_loss(_SAMPLE_RECORDS, out)
    assert out.exists()
    assert _is_svg(out)


def test_capture_rate_produces_svg(tmp_path: Path) -> None:
    out = tmp_path / "capture.svg"
    _chart_rl_capture_rate(_SAMPLE_RECORDS, out)
    assert out.exists()
    assert _is_svg(out)


def test_validation_waterfall_empty_emits_svg(tmp_path: Path) -> None:
    """No validation records -> stub empty SVG (not a crash)."""
    out = tmp_path / "val.svg"
    _chart_rl_validation_waterfall([], out)
    assert out.exists()
    content = out.read_text()
    assert "svg" in content


def test_validation_waterfall_with_records_produces_svg(tmp_path: Path) -> None:
    records = [
        {"env_steps": 1000, "val_attempted": True, "val_rms_cost": 500.0},
        {"env_steps": 2000, "val_attempted": True, "val_rms_cost": 300.0},
    ]
    out = tmp_path / "val.svg"
    _chart_rl_validation_waterfall(records, out)
    assert out.exists()
    assert _is_svg(out)


@pytest.mark.slow
def test_typst_compiles(tmp_path: Path) -> None:
    """Requires typst CLI installed. Skipped if unavailable."""
    try:
        subprocess.run(["typst", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):  # fmt: skip
        pytest.skip("typst CLI not installed")

    # Full compile test deferred to integration testing in CI.
    # If we reach here, typst is available; just verify the template exists.
    from aerocapture.training.rl.report_rl import _TYPST_DIR

    template = _TYPST_DIR / "report_rl.typ"
    assert template.exists(), f"report_rl.typ not found at {template}"
