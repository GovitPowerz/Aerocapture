"""Asset contract + compile gate for the one Typst render spine.

Two layers. The DRIVER contract runs the real report drivers with `render_pdf`
faked and checks every SVG they stage against the template's static
`dir + "/<name>"` set in both directions (a chart the template never names, a
chart the template needs that the driver dropped); it needs neither typst nor
aerocapture_rs. The COMPILE gate feeds each template a hand-written fixture
with every optional flag on and compiles it for real wherever typst is
installed (CI's python-test job installs it).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest
from aerocapture.training import report, report_render, warm_start_report
from aerocapture.training.report_render import TEMPLATES_DIR, render_pdf, staged_assets
from aerocapture.training.rl import report_rl

from tests.test_training_report import _write_fixture_jsonl, _write_multi_scheme_fixtures
from tests.test_warm_start_report import _write_artifacts

_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"></svg>'
_STATIC_ASSET_RE = re.compile(r'dir \+ "/([^"]+)"')
# The templates gate optional panels with one-line `#if meta.at("flag", default: false) {` /
# `} else {` / `}` blocks; the driver contract walks them to know which static
# names are live under the flags a driver actually wrote.
_IF_RE = re.compile(r'^\s*#?if meta\.at\("(\w+)", default: false\) \{\s*$')
_ELSE_RE = re.compile(r"^\s*\} else \{\s*$")
_CLOSE_RE = re.compile(r"^\s*\}\s*$")

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


# kind == template name -> fixture writer (every optional flag on)
REPORT_KINDS: dict[str, Callable[[Path], None]] = {
    "report": write_report_fixture,
    "report_rl": write_report_rl_fixture,
    "comparison": write_comparison_fixture,
    "warm_start_report": write_warm_start_report_fixture,
    "nn_input_report": write_nn_input_report_fixture,
}

_needs_typst = pytest.mark.skipif(shutil.which("typst") is None, reason="typst not installed")


def _template_assets(template: str, meta: dict) -> tuple[set[str], set[str]]:
    """(every static asset name in the template, the ones live under `meta`'s flags)."""
    all_names: set[str] = set()
    active: set[str] = set()
    stack: list[bool] = []
    for line in (TEMPLATES_DIR / f"{template}.typ").read_text().splitlines():
        if m := _IF_RE.match(line):
            stack.append(bool(meta.get(m.group(1), False)))
            continue
        if _ELSE_RE.match(line) and stack:
            stack[-1] = not stack[-1]
            continue
        if _CLOSE_RE.match(line) and stack:
            stack.pop()
            continue
        for name in _STATIC_ASSET_RE.findall(line):
            all_names.add(name)
            if all(stack):
                active.add(name)
    return all_names, active


def _placed_images(template: str, assets: Path) -> set[str]:
    """Basenames of every image typst places when compiling `template` over `assets`."""
    cmd = [
        "typst",
        "eval",
        "query(image).map(it => it.source)",
        "--in",
        str(TEMPLATES_DIR / f"{template}.typ"),
        "--root",
        "/",
        "--input",
        f"dir={assets.resolve()}",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return {Path(src).name for src in json.loads(out)}


def test_templates_dir_holds_every_template() -> None:
    for template in REPORT_KINDS:
        path = TEMPLATES_DIR / f"{template}.typ"
        assert path.exists(), path
        assert path.read_text().startswith('#import "lib.typ": *'), path


# ---------------------------------------------------------------------------
# Driver contract: what the real writers stage vs what the templates name
# ---------------------------------------------------------------------------
class _FakeRender:
    """Stands in for render_pdf: records (template, staged names, metadata) instead of compiling."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, set[str], dict]] = []

    def __call__(self, template: str, assets: Path, out_pdf: Path) -> Path | None:
        names = {p.name for p in assets.iterdir()}
        meta = json.loads((assets / "metadata.json").read_text())
        self.calls.append((template, names, meta))
        return None


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch) -> _FakeRender:
    fake = _FakeRender()
    # Each driver binds render_pdf by name at import; patch the module attributes.
    for module in (report, report_rl, warm_start_report):
        monkeypatch.setattr(module, "render_pdf", fake)
    return fake


def _check_driver_contract(fake: _FakeRender, expected_template: str) -> tuple[set[str], dict]:
    ((template, names, meta),) = fake.calls
    assert template == expected_template
    all_names, active = _template_assets(template, meta)
    staged_svgs = {n for n in names if n.endswith(".svg")}
    unnamed = staged_svgs - all_names
    assert not unnamed, f"{template}: driver staged SVGs the template never names: {sorted(unnamed)}"
    missing = {n for n in active if n.endswith(".svg")} - staged_svgs
    assert not missing, f"{template}: template needs (under the driver's flags) but driver did not stage: {sorted(missing)}"
    return names, meta


def _write_islands_jsonl(scheme_dir: Path) -> None:
    scheme_dir.mkdir()
    with open(scheme_dir / "run_0.jsonl", "w") as f:
        for gen in range(5):
            for i, island in enumerate(("pso", "ga", "de")):
                rec = {
                    "generation": gen,
                    "best_cost": 1.0 + gen + i * 0.1,
                    "mean_cost": 2.0,
                    "worst_cost": 4.0,
                    "median_cost": 2.0,
                    "std_cost": 0.5,
                    "capture_rate": 0.5,
                    "population_diversity": 0.3,
                    "best_params": {"k": 0.1},
                    "improvement": False,
                    "scheme": "neural_network",
                    "config_hash": "abc",
                    "island_name": island,
                }
                f.write(json.dumps(rec) + "\n")


def test_driver_contract_report(fake_render: _FakeRender, tmp_path: Path) -> None:
    scheme_dir = _write_fixture_jsonl(tmp_path)
    assert report.generate_report(scheme_dir, skip_final_eval=True) is None
    _names, meta = _check_driver_contract(fake_render, "report")
    assert meta["has_islands"] is False


def test_driver_contract_report_islands(fake_render: _FakeRender, tmp_path: Path) -> None:
    """The #104 driver half: an islands run stages both islands panels AND flags them for report.typ.

    The cover stats must come from the winning island's slice, not the raw list
    that interleaves 3 records per generation."""
    scheme_dir = tmp_path / "islands_scheme"
    _write_islands_jsonl(scheme_dir)
    assert report.generate_report(scheme_dir, skip_final_eval=True) is None
    names, meta = _check_driver_contract(fake_render, "report")
    assert meta["has_islands"] is True
    assert {"island_convergence.svg", "migration_timeline.svg"} <= names
    assert meta["total_generations"] == "5"
    assert meta["best_cost"] == f"{5.0:.4e}"  # pso, the lowest-cost island, at its last gen


def test_driver_contract_comparison(fake_render: _FakeRender, tmp_path: Path) -> None:
    _write_multi_scheme_fixtures(tmp_path)
    assert report.generate_comparison_report(tmp_path) is None
    _check_driver_contract(fake_render, "comparison")


def test_driver_contract_report_rl(fake_render: _FakeRender, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A None sys.modules entry makes `import aerocapture_rs` raise ImportError:
    # the driver's no-extension branch, the only one that needs no simulation.
    monkeypatch.setitem(sys.modules, "aerocapture_rs", None)
    out_dir = tmp_path / "rl"
    out_dir.mkdir()
    recs = [
        {"env_steps": 1000, "episodic_return_mean": -1.0, "episodic_dv_m_s_mean": 800.0, "entropy": 1.2, "value_loss": 0.5, "episodic_capture_rate": 0.3},
        {"env_steps": 2000, "episodic_return_mean": -0.5, "episodic_dv_m_s_mean": 600.0, "entropy": 1.0, "value_loss": 0.3, "episodic_capture_rate": 0.6},
    ]
    (out_dir / "rl_training_000.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    toml = tmp_path / "rl.toml"
    toml.write_text('[rl]\nalgorithm = "ppo"\n')
    assert report_rl.generate_report(out_dir, toml) is None
    _names, meta = _check_driver_contract(fake_render, "report_rl")
    assert meta["has_final_eval"] is False
    assert meta["final_eval_n_sims"] == "1000"


def test_driver_contract_warm_start_report(fake_render: _FakeRender, tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    assert warm_start_report.render_report(tmp_path) is None
    _check_driver_contract(fake_render, "warm_start_report")


# ---------------------------------------------------------------------------
# Compile gate: every template over a fixture with every optional flag on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", sorted(REPORT_KINDS))
def test_fixture_covers_static_assets(kind: str, tmp_path: Path) -> None:
    """The compile fixture must carry every static asset, or the compile test proves less than it claims."""
    REPORT_KINDS[kind](tmp_path)
    static = _STATIC_ASSET_RE.findall((TEMPLATES_DIR / f"{kind}.typ").read_text())
    assert static, "template names no static asset"
    missing = [name for name in static if not (tmp_path / name).exists()]
    assert not missing, missing


@pytest.mark.parametrize("kind", sorted(REPORT_KINDS))
def test_render_pdf_without_typst(kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    REPORT_KINDS[kind](tmp_path)
    monkeypatch.setattr("aerocapture.training.report_render.check_typst", lambda: False)
    assert render_pdf(kind, tmp_path, tmp_path / "out.pdf") is None
    assert not (tmp_path / "out.pdf").exists()


def test_render_pdf_passes_absolute_assets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With --root / a relative `dir=` would resolve against src/typst/, so
    render_pdf must hand typst an absolute path even for a relative out_dir."""
    captured: list[list[str]] = []

    def fake_compile(template_path: Path, output_pdf: Path, *, extra_args: list[str] | None = None, label: str = "") -> bool:
        captured.append(extra_args or [])
        return True

    monkeypatch.setattr(report_render, "check_typst", lambda: True)
    monkeypatch.setattr(report_render, "compile_typst", fake_compile)
    monkeypatch.chdir(tmp_path)
    Path("rep").mkdir()
    render_pdf("nn_input_report", Path("rep"), Path("rep/out.pdf"))
    (extra_args,) = captured
    assert extra_args[:3] == ["--root", "/", "--input"]
    assert Path(extra_args[3].removeprefix("dir=")) == (tmp_path / "rep").resolve()


@_needs_typst
@pytest.mark.parametrize("kind", sorted(REPORT_KINDS))
def test_render_pdf_compiles(kind: str, tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    REPORT_KINDS[kind](assets)
    out = tmp_path / "out.pdf"
    assert render_pdf(kind, assets, out) == out
    assert out.exists() and out.stat().st_size > 0


@_needs_typst
def test_report_islands_flag_places_panels(tmp_path: Path) -> None:
    """has_islands=True must reach the PDF: the two islands panels were once
    rendered into the temp dir and deleted with it, never referenced by report.typ."""
    islands = {"island_convergence.svg", "migration_timeline.svg"}
    for has_islands in (False, True):
        assets = tmp_path / f"islands_{has_islands}"
        assets.mkdir()
        write_report_fixture(assets, has_islands=has_islands)
        assert (islands <= _placed_images("report", assets)) is has_islands


@_needs_typst
def test_render_pdf_resolves_relative_assets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assets = Path("rep")
    assets.mkdir()
    write_nn_input_report_fixture(assets)
    out = assets / "nn_input_report.pdf"
    assert render_pdf("nn_input_report", assets, out) == out
    assert out.exists() and out.stat().st_size > 0


@_needs_typst
def test_warm_start_report_interpolates_supervisor(tmp_path: Path) -> None:
    """The supervisor name is Typst code, not a raw span: no `#meta...` source text may reach the PDF."""
    write_warm_start_report_fixture(tmp_path)
    cmd = ["typst", "eval", "query(raw).map(it => it.text)", "--in", str(TEMPLATES_DIR / "warm_start_report.typ"), "--root", "/", "--input", f"dir={tmp_path}"]
    raw_spans = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    assert not [span for span in raw_spans if span.startswith("#")], raw_spans


def test_staged_assets_temp_removed_unless_keep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with staged_assets() as tmp:
        assert tmp.is_dir()
        (tmp / "x.svg").write_text(_SVG)
    assert not tmp.exists()

    with staged_assets(keep=True) as kept:
        (kept / "x.svg").write_text(_SVG)
    assert kept.is_dir() and (kept / "x.svg").exists()
    assert f"Chart artifacts kept at: {kept}" in capsys.readouterr().out
