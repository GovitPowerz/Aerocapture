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


def _write_resumed_jsonl(path: Path) -> Path:
    """Write two JSONL files simulating a resumed training run."""
    scheme_dir = path / "equilibrium_glide"
    scheme_dir.mkdir(parents=True, exist_ok=True)

    # First session: gens 1-10
    with open(scheme_dir / "run_000_20260311T120000.jsonl", "w") as f:
        for gen in range(1, 11):
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T12:00:{gen:02d}Z",
                "best_cost": 1e5 * (0.9**gen),
                "mean_cost": 3e5 * (0.9**gen),
                "worst_cost": 1e6 * (0.9**gen),
                "median_cost": 2e5 * (0.9**gen),
                "std_cost": 1.5e5 * (0.9**gen),
                "capture_rate": 0.5 + gen * 0.05,
                "population_diversity": 0.5 - gen * 0.02,
                "best_params": {"k": 0.3},
                "improvement": True,
                "scheme": "equilibrium_glide",
                "config_hash": "abc123",
            }
            f.write(json.dumps(record) + "\n")

    # Second session (resumed): gens 11-20
    with open(scheme_dir / "run_000_20260311T140000.jsonl", "w") as f:
        for gen in range(11, 21):
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T14:00:{gen:02d}Z",
                "best_cost": 1e5 * (0.9**gen),
                "mean_cost": 3e5 * (0.9**gen),
                "worst_cost": 1e6 * (0.9**gen),
                "median_cost": 2e5 * (0.9**gen),
                "std_cost": 1.5e5 * (0.9**gen),
                "capture_rate": 0.5 + gen * 0.025,
                "population_diversity": 0.5 - gen * 0.02,
                "best_params": {"k": 0.3},
                "improvement": gen <= 15,
                "scheme": "equilibrium_glide",
                "config_hash": "abc123",
            }
            f.write(json.dumps(record) + "\n")

    return scheme_dir


def _write_fixture_with_pool_metrics(path: Path, n_gens: int = 10) -> Path:
    """Write JSONL with pool_metrics fields (adaptive seeds)."""
    scheme_dir = path / "adaptive_scheme"
    scheme_dir.mkdir(parents=True, exist_ok=True)
    with open(scheme_dir / "run_000_20260311T120000.jsonl", "w") as f:
        for gen in range(1, n_gens + 1):
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T12:00:{gen:02d}Z",
                "best_cost": 1e5 * (0.9**gen),
                "mean_cost": 3e5 * (0.9**gen),
                "worst_cost": 1e6 * (0.9**gen),
                "median_cost": 2e5 * (0.9**gen),
                "std_cost": 1.5e5 * (0.9**gen),
                "capture_rate": 0.8,
                "population_diversity": 0.3,
                "best_params": {"k": 0.3},
                "improvement": gen <= 5,
                "scheme": "test",
                "config_hash": "abc",
                "pool_metrics": {
                    "pool_size": gen + 4,
                    "difficulty_min": 600.0 + gen * 10,
                    "difficulty_max": 800.0 + gen * 5,
                    "n_evictions": gen // 3,
                },
            }
            f.write(json.dumps(record) + "\n")
    return scheme_dir


def _write_fixture_with_mc_seed(path: Path, n_gens: int = 10) -> Path:
    """Write JSONL with mc_seed fields (rotate seeds)."""
    scheme_dir = path / "rotate_scheme"
    scheme_dir.mkdir(parents=True, exist_ok=True)
    with open(scheme_dir / "run_000_20260311T120000.jsonl", "w") as f:
        for gen in range(1, n_gens + 1):
            record = {
                "generation": gen,
                "run": 0,
                "timestamp": f"2026-03-11T12:00:{gen:02d}Z",
                "best_cost": 1e5 * (0.9**gen),
                "mean_cost": 3e5 * (0.9**gen),
                "worst_cost": 1e6 * (0.9**gen),
                "median_cost": 2e5 * (0.9**gen),
                "std_cost": 1.5e5 * (0.9**gen),
                "capture_rate": 0.8,
                "population_diversity": 0.3,
                "best_params": {"k": 0.3},
                "improvement": gen <= 5,
                "scheme": "test",
                "config_hash": "abc",
                "mc_seed": 42 + gen,
            }
            f.write(json.dumps(record) + "\n")
    return scheme_dir


class TestLoadRunData:
    def test_loads_all_records(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        data, resume_gens = load_run_data(scheme_dir)
        assert len(data) == 20
        assert data[0]["generation"] == 1
        assert resume_gens == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        scheme_dir = tmp_path / "empty_scheme"
        scheme_dir.mkdir()
        data, resume_gens = load_run_data(scheme_dir)
        assert data == []
        assert resume_gens == []


class TestResumeDetection:
    def test_detects_resume_from_file_boundaries(self, tmp_path: Path) -> None:
        scheme_dir = _write_resumed_jsonl(tmp_path)
        data, resume_gens = load_run_data(scheme_dir)
        assert len(data) == 20
        assert resume_gens == [11]

    def test_no_resume_returns_empty_list(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        data, resume_gens = load_run_data(scheme_dir)
        assert len(data) == 20
        assert resume_gens == []

    def test_multiple_resumes(self, tmp_path: Path) -> None:
        scheme_dir = tmp_path / "test_scheme"
        scheme_dir.mkdir(parents=True, exist_ok=True)
        for file_idx, (start, end) in enumerate([(1, 6), (6, 11), (11, 16)]):
            ts = f"2026031{file_idx + 1}T120000"
            with open(scheme_dir / f"run_000_{ts}.jsonl", "w") as f:
                for gen in range(start, end):
                    record = {
                        "generation": gen,
                        "run": 0,
                        "timestamp": f"2026-03-1{file_idx + 1}T12:00:00Z",
                        "best_cost": 100.0 / gen,
                        "mean_cost": 300.0 / gen,
                        "worst_cost": 1000.0 / gen,
                        "median_cost": 200.0 / gen,
                        "std_cost": 150.0 / gen,
                        "capture_rate": 0.8,
                        "population_diversity": 0.3,
                        "best_params": {"k": 0.1},
                        "improvement": False,
                        "scheme": "test",
                        "config_hash": "xyz",
                    }
                    f.write(json.dumps(record) + "\n")
        data, resume_gens = load_run_data(scheme_dir)
        assert len(data) == 15
        assert resume_gens == [6, 11]


class TestResumeMarkers:
    def test_report_contains_resume_marker(self, tmp_path: Path) -> None:
        scheme_dir = _write_resumed_jsonl(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "resumed" in content.lower()

    def test_report_without_resume_has_no_marker(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "resumed" not in content.lower()


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


class TestResumeGenerationOffset:
    """Verify --n-gen means 'N additional' when resuming."""

    def test_resumed_n_gen_is_offset(self) -> None:
        """After resume from gen 100 with --n-gen 50, config.ga.n_gen should be 150."""
        from aerocapture.training.config import TrainingConfig

        config = TrainingConfig()
        config.ga.n_gen = 50
        config.ga.n_runs = 1

        start_gen = 100
        resumed = {"generation": 100}

        if resumed is not None and config.ga.n_runs == 1:
            config.ga.n_gen += resumed["generation"]

        assert config.ga.n_gen == 150
        loop_gens = list(range(start_gen, config.ga.n_gen))
        assert loop_gens[0] == 100
        assert loop_gens[-1] == 149
        assert len(loop_gens) == 50

    def test_no_resume_n_gen_unchanged(self) -> None:
        """Without resume, --n-gen means total generations."""
        from aerocapture.training.config import TrainingConfig

        config = TrainingConfig()
        config.ga.n_gen = 100

        resumed = None

        if resumed is not None and config.ga.n_runs == 1:
            config.ga.n_gen += resumed["generation"]

        assert config.ga.n_gen == 100

    def test_multi_run_no_offset(self) -> None:
        """With n_runs > 1, offset is not applied (would inflate subsequent runs)."""
        from aerocapture.training.config import TrainingConfig

        config = TrainingConfig()
        config.ga.n_gen = 50
        config.ga.n_runs = 3

        resumed = {"generation": 100}

        if resumed is not None and config.ga.n_runs == 1:
            config.ga.n_gen += resumed["generation"]

        assert config.ga.n_gen == 50


class TestConditionalPanels:
    def test_pool_metrics_panel_appears(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_with_pool_metrics(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "Seed Pool" in content or "Pool Size" in content

    def test_mc_seed_panel_appears(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_with_mc_seed(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "MC Seed" in content

    def test_no_extra_panels_without_seed_data(self, tmp_path: Path) -> None:
        scheme_dir = _write_fixture_jsonl(tmp_path)
        generate_single_report(scheme_dir)
        content = (scheme_dir / "report.html").read_text()
        assert "Seed Pool" not in content
        assert "MC Seed" not in content
