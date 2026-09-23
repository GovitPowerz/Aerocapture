"""The paper-figure driver contract (articles/paper/Makefile + scripts/figlib.py).

Pure Python: no simulator, no run logs. CI's `paper` job proves the figures are
byte-identical to git; these tests pin the pieces that make that possible.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "articles/paper"
SCRIPTS = PAPER / "scripts"


@pytest.fixture
def figlib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import figlib as fl  # type: ignore[import-not-found]

    monkeypatch.setattr(fl, "FIGDIR", tmp_path)
    monkeypatch.setattr(fl, "REPO", tmp_path)  # save() prints the path relative to REPO
    assert isinstance(fl, ModuleType)  # mypy: the untyped import is Any
    return fl


def _save_demo(fl: ModuleType, name: str) -> bytes:
    import matplotlib.pyplot as plt  # figlib already forced the Agg backend

    fl.style()
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [1, 3, 2], label="series")
    ax.set_title("bold title", fontweight="bold")
    ax.annotate("italic", (1, 3), style="italic")
    ax.legend()
    return bytes(fl.save(fig, name).read_bytes())  # bytes(): mypy sees Any from the untyped module


def test_save_is_byte_reproducible_and_undated(figlib: ModuleType) -> None:
    first = _save_demo(figlib, "demo")
    second = _save_demo(figlib, "demo")
    assert first == second
    assert b"<dc:date>" not in first
    # ids are hashed from the fixed salt, never uuid4
    assert re.search(rb'id="[a-z]+[0-9a-f]{10}"', first)


def test_save_registers_the_vendored_font(figlib: ModuleType) -> None:
    import matplotlib.font_manager as fm

    svg = _save_demo(figlib, "demo").decode()
    assert 'id="STIXTwoText-' in svg  # glyph outlines come from STIX Two Text, not a fallback serif
    for name in figlib._VENDORED_FONTS:
        assert (figlib.FONTS / name).is_file()
    # The vendored files are what resolves, not a same-named system font (macOS ships
    # one), and repeated style() calls do not pile up duplicate entries.
    figlib.style()
    for style in ("normal", "italic"):
        resolved = Path(fm.findfont(fm.FontProperties(family=figlib._FAMILY, style=style)))
        assert resolved.parent == figlib.FONTS, resolved
    assert sum(f.name == figlib._FAMILY for f in fm.fontManager.ttflist) == len(figlib._VENDORED_FONTS)


def test_missing_vendored_font_is_an_error(figlib: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(figlib, "FONTS", tmp_path / "no-fonts")
    with pytest.raises(FileNotFoundError, match="vendored figure font missing"):
        figlib.style()


def test_makefile_lists_every_figure_script() -> None:
    text = (PAPER / "Makefile").read_text()
    m = re.search(r"^FIG_NAMES := (.*?)^(?:\S)", text, flags=re.S | re.M)
    assert m is not None
    names = set(m.group(1).replace("\\\n", " ").split())
    scripts = {p.stem.removeprefix("fig_") for p in SCRIPTS.glob("fig_*.py")}
    assert names == scripts


def test_makefile_figure_inputs_exist() -> None:
    """Every data prerequisite the Makefile names for a figure is in the committed bundle."""
    text = (PAPER / "Makefile").read_text()
    for target, prereqs in re.findall(r"^\$\(FIGS\)/(fig_\w+\.svg):(.*)$", text, flags=re.M):
        for dep in prereqs.split():
            if dep.startswith("$(SWEEP_"):
                continue  # wildcard / absolute; fig_pareto.py errors on an absent cell
            path = PAPER / dep.replace("$(DATA)", "data").replace("$(RUNS)", "data/runs")
            assert not path.name.startswith("$("), f"{target}: unexpanded variable in {dep}"
            assert path.is_file(), f"{target}: {path}"


def test_check_target_requires_the_frozen_files() -> None:
    """FROZEN is defined before `check` names it: make expands prerequisite lists when it
    reads the rule, so a later definition would leave the guard silently empty."""
    out = subprocess.run(["make", "-C", str(PAPER), "-p", "-n", "check"], capture_output=True, text=True)
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("check:"))
    assert "data/plateau.json" in line and "data/quant/ticks_per_sim.json" in line, line


def test_frozen_block_names_every_orphan_data_file() -> None:
    """The 7 committed data files with no producer in the tree are declared FROZEN."""
    text = (PAPER / "Makefile").read_text()
    m = re.search(r"^FROZEN := (.*?)^\$\(FROZEN\)", text, flags=re.S | re.M)
    assert m is not None
    frozen = m.group(1).replace("\\\n", " ").split()
    names = {f.replace("$(DATA)/", "") for f in frozen}
    assert names == {
        "plateau.json",
        "sigma_extras.json",
        "selection_gate.json",
        "nominal_floor.json",
        "fnpag_failure_classification.json",
        "s2_failure_classification.json",
        "quant/ticks_per_sim.json",
    }
    for n in names:
        assert (PAPER / "data" / n).is_file()


def test_results_schema_check_passes_on_the_committed_bundle() -> None:
    out = subprocess.run([sys.executable, str(SCRIPTS / "check_results_schema.py")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_results_schema_check_rejects_a_dropped_run(tmp_path: Path) -> None:
    d = json.loads((PAPER / "data/results.json").read_text())
    d["runs"].pop(d["headline"])
    tampered = tmp_path / "results.json"
    tampered.write_text(json.dumps(d))
    out = subprocess.run([sys.executable, str(SCRIPTS / "check_results_schema.py"), str(tampered)], capture_output=True, text=True)
    assert out.returncode != 0
    assert "only-in-bundle" in out.stderr


def test_confirmatory_marginal_check_passes_then_rejects_a_drifted_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`make check`'s extract gate: current on the committed pair, a failure once the source moves."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import extract_confirmatory_marginal as ecm  # type: ignore[import-not-found]

    src = tmp_path / "experiments/ou_marginal/confirmatory_marginal.json"
    out = tmp_path / "articles/paper/data/confirmatory_marginal.json"
    src.parent.mkdir(parents=True)
    out.parent.mkdir(parents=True)
    shutil.copy(ecm.SRC, src)
    shutil.copy(ecm.OUT, out)
    monkeypatch.setattr(ecm, "REPO", tmp_path)
    monkeypatch.setattr(ecm, "SRC", src)
    monkeypatch.setattr(ecm, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["extract_confirmatory_marginal.py", "--check"])
    ecm.main()

    d = json.loads(src.read_text())
    next(c for c in d["cells"] if c["label"] == "fnpag")["pooled"]["cvar999"] += 1.0
    src.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="is not what"):
        ecm.main()


def test_aggregate_fails_without_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bundle without run.jsonl.gz is an exit, not a degraded results.json."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aggregate_results as agg  # type: ignore[import-not-found]

    bundle = tmp_path / "runs/study/cell"
    bundle.mkdir(parents=True)
    shutil.copy(PAPER / "data/runs/headline/mamba_p962/final_eval.parquet", bundle / "final_eval.parquet")
    monkeypatch.setattr(agg, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(agg, "OUT", tmp_path / "results.json")
    monkeypatch.setattr(sys, "argv", ["aggregate_results.py"])
    with pytest.raises(SystemExit, match="No run.jsonl.gz"):
        agg.main()
    assert not (tmp_path / "results.json").exists()
