"""Tests for NN input ablation analysis."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from aerocapture.training.ablation import _DV_TOTAL_COL, NN_INPUT_NAMES
from aerocapture.training.charts_ablation import chart_ablation_bar


def test_input_names_length() -> None:
    """35 inputs: 16 baseline + 4 ref + 1 exit-teacher + 4 lateral telemetry + 6 (sin,cos) pairs + periapsis_alt + 3 predicted_dv."""
    assert len(NN_INPUT_NAMES) == 35
    assert "periapsis_alt" in NN_INPUT_NAMES


def test_input_names_unique() -> None:
    """No duplicate input names."""
    assert len(set(NN_INPUT_NAMES)) == len(NN_INPUT_NAMES)


def test_input_names_no_empty() -> None:
    """No empty strings in input names."""
    for name in NN_INPUT_NAMES:
        assert name.strip(), f"Empty input name at index {NN_INPUT_NAMES.index(name)}"


def test_dv_total_col() -> None:
    """DV total column index matches FINAL_CSV_COLUMNS layout (sim_number at 0)."""
    # Verified against output.rs FINAL_CSV_COLUMNS and results.rs final_record[41] comment.
    assert _DV_TOTAL_COL == 41


def test_ablation_chart_produces_svg() -> None:
    """Chart function produces an SVG file."""
    ranked = [{"name": f"input_{i}", "delta": 0.1 * (10 - i), "index": i, "rank": i + 1} for i in range(10)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "test.svg")
        chart_ablation_bar(ranked, path)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "<svg" in content


def test_ablation_chart_negative_deltas() -> None:
    """Chart handles negative deltas (blue bars) without error."""
    ranked = [{"name": f"input_{i}", "delta": -0.05 * i, "index": i, "rank": i + 1} for i in range(5)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "neg.svg")
        chart_ablation_bar(ranked, path)
        assert Path(path).exists()


@pytest.mark.slow
def test_run_flip_ablation_structure() -> None:
    model = "training_output/neural_network_scaledpi_pso/best_model.json"
    if not os.path.exists(model):
        pytest.skip("scaledpi model not present")
    import aerocapture_rs  # noqa: F401  (skip cleanly if binding missing)
    from aerocapture.training.ablation import run_flip_ablation

    out = run_flip_ablation("configs/training/msr_aller_nn_scaledpi_train.toml", n_sims=8, flip_indices=(15,))
    fv = sorted(r["frozen_value"] for r in out["results"])
    assert fv == [-1.0, 1.0]  # bounce_flag frozen at both phases
    assert all(isinstance(r["delta"], float) for r in out["results"])


def test_nn_input_names_has_35_with_dv() -> None:
    from aerocapture.training.ablation import NN_INPUT_NAMES

    assert len(NN_INPUT_NAMES) == 35
    assert NN_INPUT_NAMES[32] == "predicted_dv1"
    assert NN_INPUT_NAMES[33] == "predicted_dv2"
    assert NN_INPUT_NAMES[34] == "predicted_dv3"
