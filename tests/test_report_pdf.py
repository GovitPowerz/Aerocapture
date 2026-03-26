"""Integration tests for PDF report generation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from aerocapture.training.report import _check_typst, generate_comparison_report, generate_report


class TestCheckTypst:
    def test_returns_true_when_available(self) -> None:
        if shutil.which("typst"):
            assert _check_typst() is True

    def test_returns_false_when_missing(self) -> None:
        with patch("shutil.which", return_value=None):
            assert _check_typst() is False


class TestGenerateReport:
    @pytest.fixture()
    def scheme_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "equilibrium_glide"
        d.mkdir()
        records = [
            {
                "generation": i,
                "best_cost": 100.0 * (0.9**i),
                "mean_cost": 150.0 * (0.95**i),
                "worst_cost": 200.0,
                "capture_rate": 0.8 + 0.02 * i,
                "population_diversity": 0.5 - 0.04 * i,
                "improvement": i % 2 == 0,
                "best_params": {"gain": 0.5 + 0.01 * i},
                "config_hash": "test123",
                "scheme": "equilibrium_glide",
            }
            for i in range(5)
        ]
        (d / "run_000_test.jsonl").write_text("\n".join(json.dumps(r) for r in records))
        return d

    def test_generates_charts_to_temp_dir(self, scheme_dir: Path) -> None:
        with patch("aerocapture.training.report._check_typst", return_value=False):
            generate_report(scheme_dir, toml_path=None, skip_final_eval=True)


class TestGenerateComparisonReport:
    def test_comparison_report_no_data(self, tmp_path: Path) -> None:
        with patch("aerocapture.training.report._check_typst", return_value=False):
            result = generate_comparison_report(tmp_path)
        assert result is None

    def test_comparison_report_with_data(self, tmp_path: Path) -> None:
        for scheme in ["eq_glide", "ftc"]:
            d = tmp_path / scheme
            d.mkdir()
            records = [
                {
                    "generation": i,
                    "best_cost": 100 - i,
                    "mean_cost": 150,
                    "worst_cost": 200,
                    "capture_rate": 0.9,
                    "population_diversity": 0.3,
                    "scheme": scheme,
                }
                for i in range(3)
            ]
            (d / "run_000.jsonl").write_text("\n".join(json.dumps(r) for r in records))
        with patch("aerocapture.training.report._check_typst", return_value=False):
            result = generate_comparison_report(tmp_path)
        assert result is None
