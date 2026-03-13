"""Tests for training report generation."""

from __future__ import annotations

import json
from pathlib import Path

from aerocapture.training.report import generate_comparison_report, generate_single_report, load_run_data


def _write_fixture_jsonl(path: Path, n_gens: int = 20) -> Path:
    """Write a synthetic JSONL file for testing."""
    jsonl_path = path / "equilibrium_glide" / "run_000_20260311T120000.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    best = 1e5
    with open(jsonl_path, "w") as f:
        for gen in range(1, n_gens + 1):
            best = best * 0.9  # Improving cost
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T12:00:{gen:02d}Z",
                "best_cost": best,
                "mean_cost": best * 3,
                "worst_cost": best * 10,
                "median_cost": best * 2,
                "std_cost": best * 1.5,
                "capture_rate": min(0.5 + gen * 0.025, 1.0),
                "population_diversity": max(0.5 - gen * 0.02, 0.05),
                "best_params": {"k_hdot_scale": 0.3, "v_ratio_threshold": 1.1},
                "improvement": gen <= 15,
                "scheme": "equilibrium_glide",
                "config_hash": "abc123",
            }
            f.write(json.dumps(record) + "\n")
    return jsonl_path.parent


class TestLoadRunData:
    def test_loads_all_records(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        data = load_run_data(scheme_dir)
        assert len(data) == 20
        assert data[0]["generation"] == 1

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        scheme_dir = tmp_path / "empty_scheme"
        scheme_dir.mkdir()
        data = load_run_data(scheme_dir)
        assert data == []


class TestSingleReport:
    def test_generates_html_file(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        generate_single_report(scheme_dir)
        report_path = scheme_dir / "report.html"
        assert report_path.exists()
        content = report_path.read_text()
        assert "plotly" in content.lower()
        assert "convergence" in content.lower() or "Convergence" in content

    def test_report_contains_all_sections(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "best_cost" in content or "Best" in content
        assert "diversity" in content.lower() or "Diversity" in content


def _write_multi_scheme_fixtures(base_dir: Path) -> None:
    """Write fixture JSONL for two schemes."""
    _write_fixture_jsonl(base_dir, n_gens=10)  # equilibrium_glide
    # Add a second scheme
    ftc_dir = base_dir / "ftc"
    ftc_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = ftc_dir / "run_000_20260311T120000.jsonl"
    best = 2e5
    with open(jsonl_path, "w") as f:
        for gen in range(1, 11):
            best = best * 0.85
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T12:00:{gen:02d}Z",
                "best_cost": best,
                "mean_cost": best * 4,
                "worst_cost": best * 12,
                "median_cost": best * 2.5,
                "std_cost": best * 2,
                "capture_rate": 0.6 + gen * 0.04,
                "population_diversity": 0.4 - gen * 0.03,
                "best_params": {"capture_damping": 0.7},
                "improvement": gen <= 8,
                "scheme": "ftc",
                "config_hash": "def456",
            }
            f.write(json.dumps(record) + "\n")


class TestComparisonReport:
    def test_generates_comparison_html(self, tmp_path: Path) -> None:
        _write_multi_scheme_fixtures(tmp_path)
        generate_comparison_report(tmp_path)
        report_path = tmp_path / "comparison_report.html"
        assert report_path.exists()
        content = report_path.read_text()
        assert "plotly" in content.lower()

    def test_filters_by_scheme(self, tmp_path: Path) -> None:
        _write_multi_scheme_fixtures(tmp_path)
        generate_comparison_report(tmp_path, schemes=["ftc"])
        content = (tmp_path / "comparison_report.html").read_text()
        assert "ftc" in content.lower() or "FTC" in content
