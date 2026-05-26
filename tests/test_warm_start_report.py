"""Warm-start report: charts + metadata + (optional) PDF render."""

from __future__ import annotations

import json
from pathlib import Path

from aerocapture.training.warm_start_report import (
    EXPECTED_SIDECARS,
    _build_metadata,
    _layer_summary,
    _load_artifacts,
    _slab_label,
    chart_bound_widening,
    chart_supervised_mse,
    chart_supervisor_selection,
    render_report,
)


def _write_artifacts(d: Path) -> None:
    """Write a synthetic but realistic set of warm-start sidecars."""
    (d / "warm_start_loss.json").write_text(
        json.dumps(
            [
                {"epoch": 0, "mean_mse": 1.0e-1, "n_chunks": 200},
                {"epoch": 1, "mean_mse": 7.5e-2, "n_chunks": 200},
                {"epoch": 2, "mean_mse": 5.2e-2, "n_chunks": 200},
                {"epoch": 3, "mean_mse": 3.8e-2, "n_chunks": 200},
            ]
        )
    )
    (d / "warm_start_baseline.json").write_text(
        json.dumps(
            {
                "n_sims": 1000,
                "n_returned": 1000,
                "capture_rate": 0.91,
                "rms_cost": 145.0,
                "mean_cost": 110.0,
                "median_cost": 95.0,
                "p95_cost": 380.0,
                "worst_cost": 3100.0,
            }
        )
    )
    (d / "warm_start_bounds.json").write_text(
        json.dumps(
            # Two layer slabs: 4 params at bound=2.0, 3 params at bound=0.8
            [{"name": f"w0_{i}", "p_min": -2.0, "p_max": 2.0, "default": 0.0, "log_scale": False, "is_integer": False} for i in range(4)]
            + [{"name": f"w1_{i}", "p_min": -0.8, "p_max": 0.8, "default": 0.0, "log_scale": False, "is_integer": False} for i in range(3)]
        )
    )
    (d / "warm_start_selection.json").write_text(
        json.dumps(
            {
                "n_warm_seeds": 200,
                "n_selected_total": 187,
                "min_corpus_required": 50,
                "per_scheme": {
                    "ftc": {
                        "n_supervised": 200,
                        "n_captured": 192,
                        "capture_rate": 0.96,
                        "n_selected": 130,
                        "mean_dv_captured": 105.0,
                        "median_dv_captured": 95.0,
                    },
                    "fnpag": {
                        "n_supervised": 200,
                        "n_captured": 142,
                        "capture_rate": 0.71,
                        "n_selected": 57,
                        "mean_dv_captured": 130.0,
                        "median_dv_captured": 112.0,
                    },
                },
            }
        )
    )
    (d / "warm_start_cache_key.json").write_text(
        json.dumps(
            {
                "architecture": [
                    {"type": "dense", "input_size": 23, "output_size": 32, "activation": "tanh"},
                    {"type": "gru", "input_size": 32, "hidden_size": 32},
                    {"type": "dense", "input_size": 32, "output_size": 2, "activation": "linear"},
                ],
                "supervisor_schemes": ["ftc", "fnpag"],
                "bptt_length": 32,
                "n_warm_seeds": 200,
                "n_epochs": 4,
                "bound_multiplier": 4.0,
                "adaptive_bounds": True,
                "mode": "full_neural",
                "output_parameterization": "atan2_signed",
                "base_mc_seed": 42,
            }
        )
    )


def test_load_artifacts_missing_returns_none(tmp_path: Path) -> None:
    artifacts = _load_artifacts(tmp_path)
    assert artifacts == {"loss": None, "baseline": None, "bounds": None, "selection": None, "cache_key": None}


def test_load_artifacts_all_present(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    artifacts = _load_artifacts(tmp_path)
    assert artifacts["loss"] is not None and len(artifacts["loss"]) == 4
    assert artifacts["baseline"]["capture_rate"] == 0.91
    assert len(artifacts["bounds"]) == 7
    assert artifacts["selection"]["n_selected_total"] == 187


def test_chart_mse_writes_svg(tmp_path: Path) -> None:
    out = tmp_path / "mse.svg"
    chart_supervised_mse([{"epoch": i, "mean_mse": 0.5**i, "n_chunks": 100} for i in range(6)], out)
    assert out.exists() and out.stat().st_size > 0
    assert b"<svg" in out.read_bytes()[:200]


def test_chart_mse_empty(tmp_path: Path) -> None:
    """n_epochs=0 fallback: chart writes but has the 'no records' annotation."""
    out = tmp_path / "mse_empty.svg"
    chart_supervised_mse([], out)
    assert out.exists() and out.stat().st_size > 0


def test_chart_supervisor_selection_writes_svg(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    out = tmp_path / "sel.svg"
    chart_supervisor_selection(json.loads((tmp_path / "warm_start_selection.json").read_text()), out)
    assert out.exists() and out.stat().st_size > 0


def test_chart_supervisor_selection_empty(tmp_path: Path) -> None:
    out = tmp_path / "sel.svg"
    chart_supervisor_selection({}, out)
    assert out.exists()


def test_chart_bound_widening_groups_slabs(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    bounds = json.loads((tmp_path / "warm_start_bounds.json").read_text())
    out = tmp_path / "bounds.svg"
    chart_bound_widening(bounds, out)
    assert out.exists() and out.stat().st_size > 0


def test_chart_bound_widening_empty(tmp_path: Path) -> None:
    out = tmp_path / "bounds_empty.svg"
    chart_bound_widening([], out)
    assert out.exists()


def test_slab_label_strips_trailing_indices() -> None:
    assert _slab_label("w0_5_3") == "w0"
    assert _slab_label("bias1_2") == "bias1"
    assert _slab_label("w_ih0_5") == "w_ih0"
    assert _slab_label("a_log1_3_5") == "a_log1"


def test_layer_summary_per_type() -> None:
    assert "Dense(23->32" in _layer_summary({"type": "dense", "input_size": 23, "output_size": 32, "activation": "tanh"})
    assert "GRU(32->16" in _layer_summary({"type": "gru", "input_size": 32, "hidden_size": 16})
    assert "Mamba(in=8" in _layer_summary({"type": "mamba", "input_size": 8, "d_state": 4})
    assert _layer_summary({"type": "unknown_layer"}) == "unknown_layer"


def test_build_metadata_assembles_all_fields(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    artifacts = _load_artifacts(tmp_path)
    meta = _build_metadata(artifacts, tmp_path)
    assert meta["scheme"] == tmp_path.name
    assert "Dense(23->32" in meta["arch_summary"]
    assert "GRU(32->32" in meta["arch_summary"]
    assert "1.0000e-01" in meta["loss_summary"] and "+62.0%" in meta["loss_summary"]
    assert meta["config"]["mode"] == "full_neural"
    assert meta["config"]["adaptive_bounds"] is True
    assert meta["baseline"]["capture_rate"] == "91%"
    assert meta["baseline"]["rms_cost"] == "1.4500e+02"
    assert len(meta["supervisors"]) == 2
    assert meta["supervisors"][0]["scheme"] == "ftc"
    assert meta["supervisors"][0]["n_selected"] == 130
    assert meta["n_selected_total"] == 187


def test_build_metadata_handles_missing_artifacts(tmp_path: Path) -> None:
    """If sidecars are missing, metadata still builds with sane defaults."""
    artifacts = _load_artifacts(tmp_path)
    meta = _build_metadata(artifacts, tmp_path)
    assert meta["arch_summary"] == "n/a"
    assert "n_epochs=0" in meta["loss_summary"]
    assert meta["supervisors"] == []
    assert meta["baseline"]["rms_cost"] == "n/a"


def test_render_report_writes_charts(tmp_path: Path) -> None:
    """render_report MUST produce the SVG charts + metadata.json even when
    Typst is unavailable. The PDF return is None in that case but the charts
    are still usable on their own."""
    _write_artifacts(tmp_path)
    pdf = render_report(tmp_path)
    report_dir = tmp_path / "warm_start_report"
    assert (report_dir / "mse_convergence.svg").exists()
    assert (report_dir / "supervisor_selection.svg").exists()
    assert (report_dir / "bound_widening.svg").exists()
    assert (report_dir / "metadata.json").exists()
    # pdf may be None if typst is not installed on the CI host -- both branches are valid.
    if pdf is not None:
        assert pdf.exists() and pdf.suffix == ".pdf"


def test_expected_sidecars_lists_all_loader_keys() -> None:
    """Documentation aid: EXPECTED_SIDECARS should mention every key _load_artifacts produces."""
    artifact_keys = {"warm_start_loss.json", "warm_start_baseline.json", "warm_start_bounds.json", "warm_start_selection.json", "warm_start_cache_key.json"}
    assert set(EXPECTED_SIDECARS.keys()) == artifact_keys
