"""Per-template asset contract + compile gate for the one Typst render spine.

Every report kind has a fixture writer producing a minimal but COMPLETE asset
set (every SVG the template can reference, every JSON field it reads, every
optional flag on). The contract test needs no typst and runs in CI; the
compile tests run wherever the CLI is installed.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from aerocapture.training.report_render import TEMPLATES_DIR, render_pdf, staged_assets

_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"></svg>'
_STATIC_ASSET_RE = re.compile(r'dir \+ "/([^"]+)"')
_PDF_PAGE_RE = re.compile(rb"/Type\s*/Page[^s]")

_MISSION_SVGS = [
    "corridor_pdyn",
    "corridor_inclination",
    "corridor_bank",
    "altitude_time",
    "heat_flux_time",
    "gload_time",
    "bank_angle_time",
    "nav_density_ratio",
    "cost_objective",
    "dv_distribution",
    "dv_individual_burns",
    "entry_conditions",
    "exit_conditions",
    "dispersion_grid",
    "morris_scatter",
    "sobol_bars",
    "sobol_heatmap",
]
_SENSITIVITY_FLAGS = {"has_sensitivity": True, "has_morris": True, "has_sobol": True, "has_sobol_heatmap": True}


def _write_svgs(assets: Path, names: list[str]) -> None:
    for name in names:
        (assets / f"{name}.svg").write_text(_SVG)


def _write_json(assets: Path, name: str, payload: object) -> None:
    (assets / name).write_text(json.dumps(payload, indent=2))


def _summary_row(name: str) -> list[str]:
    return [name, "1.00", "0.10", "0.80", "0.85", "0.95", "1.00", "1.05", "1.15", "1.20"]


def _write_mission_tables(assets: Path) -> None:
    _write_json(
        assets,
        "summary_table.json",
        {
            "rows": [_summary_row("Max G-load (g)"), _summary_row("Total DV (m/s)")],
            "violation_rows": [_summary_row("Objective cost, all sims -- RMS=1.0"), ["Capture rate: 100.0% (2/2)", "", "", "", "", "", "", "", "", ""]],
        },
    )
    _write_json(assets, "morris_table.json", {"rows": [["1", "density_bias", "12.3", "4.5", "0.6"], ["2", "entry_fpa", "3.2", "1.1", "0.2"]]})


def write_report_fixture(assets: Path, *, has_islands: bool = True) -> None:
    training = ["convergence", "diversity_cost", "cost_distribution", "parameter_evolution", "island_convergence", "migration_timeline"]
    _write_svgs(assets, [*training, *_MISSION_SVGS])
    _write_json(
        assets,
        "metadata.json",
        {
            "scheme": "equilibrium_glide",
            "mission": "msr_aller",
            "date": "2026-09-19",
            "best_cost": "1.2345e+02",
            "capture_rate": "100%",
            "total_generations": "20",
            "n_sims": "1000",
            "config_hash": "abc123",
            "has_trajectories": True,
            "has_final_eval": True,
            "has_cost_distribution": True,
            "has_islands": has_islands,
            **_SENSITIVITY_FLAGS,
        },
    )
    _write_mission_tables(assets)


def write_report_rl_fixture(assets: Path) -> None:
    _write_svgs(assets, ["rl_return", "rl_dv", "rl_entropy", "rl_value_loss", "rl_capture", "rl_val", *_MISSION_SVGS])
    _write_json(
        assets,
        "metadata.json",
        {
            "scheme": "neural_network_rl",
            "algorithm": "PPO",
            "mission": "RL Training",
            "date": "2026-09-19",
            "n_updates": "50",
            "final_eval_n_sims": "1000",
            "has_trajectories": True,
            "has_final_eval": True,
            **_SENSITIVITY_FLAGS,
        },
    )
    _write_mission_tables(assets)


def write_comparison_fixture(assets: Path) -> None:
    _write_svgs(assets, ["comparison_convergence"])
    _write_json(assets, "metadata.json", {"date": "2026-09-19", "schemes": ["equilibrium_glide", "ftc"]})
    _write_json(
        assets,
        "comparison_table.json",
        {
            "headers": ["Scheme", "Best Cost", "Generations", "Capture %", "Conv. Speed"],
            "rows": [["equilibrium_glide", "1.23e+02", "20", "100%", "5"], ["ftc", "1.10e+02", "20", "100%", "7"]],
        },
    )


def write_warm_start_report_fixture(assets: Path) -> None:
    panels = ["corridor_pdyn", "corridor_inclination", "corridor_bank", "altitude_time", "heat_flux_time"]
    panel_files = [f"compare_train_supervisor_{p}.svg" for p in panels]
    _write_svgs(assets, ["mse_convergence", "supervisor_selection", "bound_widening", *(f[:-4] for f in panel_files)])
    _write_json(
        assets,
        "metadata.json",
        {
            "save_dir": str(assets),
            "scheme": "neural_network_joint",
            "arch_summary": "Dense(17->32,swish) -> Dense(32->1,tanh)",
            "loss_summary": "1.0000e-01 -> 2.0000e-02  (-80.0%, 10 epochs)",
            "n_chunks": 640,
            "config": {
                "supervisor_schemes": ["ftc", "fnpag"],
                "bptt_length": 32,
                "n_warm_seeds": 200,
                "n_epochs": 10,
                "bound_multiplier": 4.0,
                "adaptive_bounds": True,
                "mode": "magnitude_only",
                "output_parameterization": "acos_tanh",
                "base_mc_seed": 42,
            },
            "baseline": {
                "n_sims": 1000,
                "capture_rate": "99%",
                "rms_cost": "1.2000e+02",
                "mean_cost": "1.0000e+02",
                "median_cost": "9.0000e+01",
                "p95_cost": "2.0000e+02",
                "worst_cost": "3.0000e+03",
            },
            "supervisors": [
                {"scheme": "ftc", "n_supervised": 200, "n_captured": 198, "capture_rate": "99%", "n_selected": 150, "median_dv": "133.0"},
                {"scheme": "fnpag", "n_supervised": 200, "n_captured": 190, "capture_rate": "95%", "n_selected": 40, "median_dv": "138.0"},
            ],
            "n_selected_total": 190,
            "min_corpus_required": 50,
            "eval_summary_lines": ["Final evaluation (1000 sims):", "    Capture rate:       990/1000 (99.0%)"],
            "compare": {
                "has_data": True,
                "primary_supervisor": "ftc",
                "panels": panels,
                "side_labels": {"supervisor": "FTC (supervisor)", "nn": "warm-started NN"},
                "rows": [
                    {
                        "pool": "train",
                        "side": "supervisor",
                        "side_label": "FTC (supervisor)",
                        "n_sims": 200,
                        "n_captured": 198,
                        "capture_rate_pct": "99.0%",
                        "panels": panel_files,
                    },
                    {
                        "pool": "train",
                        "side": "nn",
                        "side_label": "warm-started NN",
                        "n_sims": 200,
                        "n_captured": 0,
                        "capture_rate_pct": "error",
                        "error": "NN write failed",
                        "panels": [],
                    },
                ],
            },
        },
    )


def write_nn_input_report_fixture(assets: Path) -> None:
    inputs = []
    for index, name, in_mask in ((0, "a", True), (1, "b", False)):
        time_svg = f"nn_input_{index:02d}_{name}_time.svg"
        energy_svg = f"nn_input_{index:02d}_{name}_energy.svg"
        (assets / time_svg).write_text(_SVG)
        (assets / energy_svg).write_text(_SVG)
        inputs.append(
            {
                "index": index,
                "name": name,
                "in_mask": in_mask,
                "frac_out_of_range": 0.1,
                "separation": 0.5,
                "p1": -1.0,
                "p50": 0.0,
                "p99": 1.0,
                "time_svg": time_svg,
                "energy_svg": energy_svg,
            }
        )
    _write_json(assets, "summary.json", {"scheme": "test_scheme", "dv_threshold": 200.0, "n_sims": 2, "n_blue": 1, "n_red": 1, "inputs": inputs})


# kind -> (template name, fixture writer)
REPORT_KINDS: dict[str, tuple[str, Callable[[Path], None]]] = {
    "report": ("report", write_report_fixture),
    "report_rl": ("report_rl", write_report_rl_fixture),
    "comparison": ("comparison", write_comparison_fixture),
    "warm_start_report": ("warm_start_report", write_warm_start_report_fixture),
    "nn_input_report": ("nn_input_report", write_nn_input_report_fixture),
}

_needs_typst = pytest.mark.skipif(shutil.which("typst") is None, reason="typst not installed")


def _count_pages(pdf: Path) -> int:
    return len(_PDF_PAGE_RE.findall(pdf.read_bytes()))


def test_templates_dir_holds_every_template() -> None:
    for template, _writer in REPORT_KINDS.values():
        path = TEMPLATES_DIR / f"{template}.typ"
        assert path.exists(), path
        assert path.read_text().startswith('#import "lib.typ": *'), path


@pytest.mark.parametrize("kind", sorted(REPORT_KINDS))
def test_static_asset_contract(kind: str, tmp_path: Path) -> None:
    """Every `dir + "/<file>"` the template names must be in the fixture.

    Runs without typst: this is what catches a driver dropping an asset a
    template still references (or a template referencing one no driver writes).
    """
    template, writer = REPORT_KINDS[kind]
    writer(tmp_path)
    static = _STATIC_ASSET_RE.findall((TEMPLATES_DIR / f"{template}.typ").read_text())
    assert static, "template names no static asset"
    missing = [name for name in static if not (tmp_path / name).exists()]
    assert not missing, missing


@pytest.mark.parametrize("kind", sorted(REPORT_KINDS))
def test_render_pdf_without_typst(kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template, writer = REPORT_KINDS[kind]
    writer(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setattr("aerocapture.training.report_render.check_typst", lambda: False)
    assert render_pdf(template, tmp_path, tmp_path / "out.pdf", label=kind) is None
    assert sorted(p.name for p in tmp_path.iterdir()) == before


@_needs_typst
@pytest.mark.parametrize("kind", sorted(REPORT_KINDS))
def test_render_pdf_compiles(kind: str, tmp_path: Path) -> None:
    template, writer = REPORT_KINDS[kind]
    assets = tmp_path / "assets"
    assets.mkdir()
    writer(assets)
    out = tmp_path / "out.pdf"
    assert render_pdf(template, assets, out, label=kind) == out
    assert out.exists() and out.stat().st_size > 0


@_needs_typst
def test_report_islands_flag_adds_pages(tmp_path: Path) -> None:
    """has_islands=True must reach the PDF: the two islands panels were once
    rendered into the temp dir and deleted with it, never referenced by report.typ."""
    pages: dict[bool, int] = {}
    for has_islands in (False, True):
        assets = tmp_path / f"islands_{has_islands}"
        assets.mkdir()
        write_report_fixture(assets, has_islands=has_islands)
        out = tmp_path / f"islands_{has_islands}.pdf"
        assert render_pdf("report", assets, out, label="report") == out
        pages[has_islands] = _count_pages(out)
    assert pages[False] > 0
    assert pages[True] > pages[False], pages


@_needs_typst
def test_render_pdf_resolves_relative_assets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A RELATIVE assets dir (nn_input_report's out_dir can be one) must compile:
    with --root / an unresolved sys.inputs dir would resolve against src/typst/."""
    monkeypatch.chdir(tmp_path)
    assets = Path("rep")
    assets.mkdir()
    write_nn_input_report_fixture(assets)
    out = assets / "nn_input_report.pdf"
    assert render_pdf("nn_input_report", assets, out, label="nn_input_report") == out
    assert out.exists() and out.stat().st_size > 0


def test_staged_assets_temp_removed_unless_keep(capsys: pytest.CaptureFixture[str]) -> None:
    with staged_assets() as tmp:
        assert tmp.is_dir()
        (tmp / "x.svg").write_text(_SVG)
    assert not tmp.exists()

    with staged_assets(keep=True, prefix="aerocapture_test_keep_") as kept:
        (kept / "x.svg").write_text(_SVG)
    try:
        assert kept.is_dir() and (kept / "x.svg").exists()
        assert f"Chart artifacts kept at: {kept}" in capsys.readouterr().out
    finally:
        shutil.rmtree(kept, ignore_errors=True)
