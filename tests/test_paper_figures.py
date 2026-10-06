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

    # The recovery the message prescribes: regenerate, then --check passes again.
    before = out.read_text()
    monkeypatch.setattr(sys, "argv", ["extract_confirmatory_marginal.py"])
    ecm.main()
    assert out.read_text() != before
    monkeypatch.setattr(sys, "argv", ["extract_confirmatory_marginal.py", "--check"])
    ecm.main()

    # A partially re-collected source names the missing cell instead of a KeyError.
    d["cells"] = [c for c in d["cells"] if c["label"] != "fnpag"]
    src.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="lacks the quoted cell.*fnpag"):
        ecm.main()


def test_confirmatory_extract_cells_are_the_scorer_manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #179: the extract's cell list is the scorer's manifests (every arxiv-v3 and v4 row), read
    from the 'label|toml|model_dir' lines, comments and blanks skipped, duplicates once, in order."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import extract_confirmatory_marginal as ecm  # type: ignore[import-not-found]

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("# comment\nx|cfg/x.toml|out/x  # trailing comment\n\n  y|cfg/y.toml\n")
    b.write_text("y|cfg/y2.toml|out/y\nz|cfg/z.toml|out/z\n")
    assert ecm.manifest_labels((a, b)) == ("x", "y", "z")

    labels = ecm.manifest_labels()
    committed = {c["label"] for c in json.loads(ecm.OUT.read_text())["cells"]}
    assert set(labels) == committed
    assert "ou_marginal/hl_mamba_p962" in labels and "ou_marginal/classical/ftc_joint" in labels and "fnpag" in labels
    assert json.loads(ecm.OUT.read_text())["manifests"] == list(ecm.MANIFEST_REL)


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


def test_quote_marginal_check_passes_then_rejects_a_drifted_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`make check`'s n = 1000 per-scenario extract gate (issue #157): current on the committed
    pair, a failure once the source moves, the prescribed regeneration, a missing cell named."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import extract_quote_marginal as eqm  # type: ignore[import-not-found]

    src = tmp_path / "experiments/ou_marginal/quote_results.json"
    out = tmp_path / "articles/paper/data/quote_marginal.json"
    src.parent.mkdir(parents=True)
    out.parent.mkdir(parents=True)
    shutil.copy(eqm.SRC, src)
    shutil.copy(eqm.OUT, out)
    committed_src = src.read_text()
    monkeypatch.setattr(eqm, "REPO", tmp_path)
    monkeypatch.setattr(eqm, "SRC", src)
    monkeypatch.setattr(eqm, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["extract_quote_marginal.py", "--check"])
    eqm.main()

    d = json.loads(src.read_text())
    d["cells"]["fnpag/marginal"]["dv_cvar95"] += 1.0
    src.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="is not what"):
        eqm.main()

    before = out.read_text()
    monkeypatch.setattr(sys, "argv", ["extract_quote_marginal.py"])
    eqm.main()
    assert out.read_text() != before
    monkeypatch.setattr(sys, "argv", ["extract_quote_marginal.py", "--check"])
    eqm.main()

    # Every cell of the source is extracted (issue #179: the v4 rows are the whole campaign), the
    # quoted fields only, keys sorted; an empty source is an error, never an empty bundle file.
    monkeypatch.setattr(sys, "argv", ["extract_quote_marginal.py"])
    eqm.main()
    extracted = json.loads(out.read_text())["cells"]
    assert list(extracted) == sorted(d["cells"])
    assert set(extracted["ou_hl_mamba_p962/marginal"]) == set(eqm.FIELDS)
    d["cells"] = {}
    src.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="has no cells"):
        eqm.main()

    # The regime pair and the seed pool are the source's own record (issue #166): copied, never
    # restated, so an edited source record reaches the extract.
    d = json.loads(committed_src)
    d["regimes"]["marginal"]["per_seed_override"] = "simulation.random_seed = 1 + 2 i"
    d["seed_pool"]["seed"] = 1
    src.write_text(json.dumps(d))
    monkeypatch.setattr(sys, "argv", ["extract_quote_marginal.py"])
    eqm.main()
    assert json.loads(out.read_text())["regimes"] == d["regimes"]
    assert json.loads(out.read_text())["seed_pool"] == d["seed_pool"]
    del d["regimes"]
    src.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="carries no protocol record"):
        eqm.main()


def test_heat_load_slope_check_passes_then_rejects_a_drifted_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`make check`'s heat-load ceiling extract gate (issues #192, #179): current on the committed
    pair, a failure once the source moves, the prescribed regeneration, a source without its
    protocol record refused."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import extract_heat_load_slope as ehs  # type: ignore[import-not-found]

    src = tmp_path / "experiments/ou_marginal/heat_load_slope.json"
    out = tmp_path / "articles/paper/data/heat_load_slope.json"
    src.parent.mkdir(parents=True)
    out.parent.mkdir(parents=True)
    shutil.copy(ehs.SRC, src)
    shutil.copy(ehs.OUT, out)
    monkeypatch.setattr(ehs, "REPO", tmp_path)
    monkeypatch.setattr(ehs, "SRC", src)
    monkeypatch.setattr(ehs, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["extract_heat_load_slope.py", "--check"])
    ehs.main()

    d = json.loads(src.read_text())
    d["rows"]["hs_mamba_p962_q30/own"]["dv_cvar95"] += 1.0
    src.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="is not what"):
        ehs.main()

    before = out.read_text()
    monkeypatch.setattr(sys, "argv", ["extract_heat_load_slope.py"])
    ehs.main()
    assert out.read_text() != before
    assert json.loads(out.read_text())["rows"] == d["rows"]
    monkeypatch.setattr(sys, "argv", ["extract_heat_load_slope.py", "--check"])
    ehs.main()

    del d["regime"]
    src.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="carries no protocol record"):
        ehs.main()


def test_results_schema_check_rejects_a_mislabelled_regime(tmp_path: Path) -> None:
    """A per-scenario study's run flagged legacy (or a shared-path run flagged per_draw) is a
    schema violation: results.typ's regime accessors would refuse the row (issue #179)."""
    d = json.loads((PAPER / "data/results.json").read_text())
    assert d["runs"]["ou_marginal/hl_mamba_p962"]["noise_seeding"] == "per_draw"
    assert d["runs"]["headline/mamba_p962"]["noise_seeding"] == "legacy"
    d["runs"]["ou_marginal/hl_mamba_p962"]["noise_seeding"] = "legacy"
    d["runs"]["headline/mamba_p962"]["noise_seeding"] = "per_draw"
    tampered = tmp_path / "results.json"
    tampered.write_text(json.dumps(d))
    out = subprocess.run([sys.executable, str(SCRIPTS / "check_results_schema.py"), str(tampered)], capture_output=True, text=True)
    assert out.returncode != 0
    assert "ou_marginal/hl_mamba_p962: noise_seeding 'legacy' contradicts" in out.stderr
    assert "headline/mamba_p962: noise_seeding 'per_draw' contradicts" in out.stderr


def test_aggregate_training_n_sims_per_study(monkeypatch: pytest.MonkeyPatch) -> None:
    """actual_sims reconstructs training evals as n_gen x n_pop x training_n_sims: the #173 cells
    (ou_marginal/hl_*) trained at 512 x 2 like the headline runs, every other ou_marginal cell at 10."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import aggregate_results as agg  # type: ignore[import-not-found]

    assert agg._infer_training_n_sims("ou_marginal/hl_mamba_p962_s2") == 2
    assert agg._infer_training_n_sims("headline/mamba_p962") == 2
    assert agg._infer_training_n_sims("ou_marginal/ft_dense_p515") == 10
    assert agg._infer_training_n_sims("ou_marginal/classical/ftc") == 10
    assert agg._infer_training_n_sims("training_n_sims/ga_50") == 50


def test_fetch_run_logs_reads_the_asset_from_provenance() -> None:
    """One place names the Release asset (write_provenance.RELEASE_TAG): the fetch script reads
    the committed provenance.json's run_logs_asset, so the tag moves with the provenance."""
    prov = json.loads((PAPER / "data/provenance.json").read_text())
    script = (SCRIPTS / "fetch_run_logs.sh").read_text()
    assert "run_logs_asset" in script and "releases/download" not in script
    assert prov["run_logs_asset"].endswith(f"/releases/download/{prov['release_tag']}/paper_run_logs.tar")
    url = subprocess.run(
        ["sed", "-n", r's/^ *"run_logs_asset": "\(.*\)",*$/\1/p', str(PAPER / "data/provenance.json")], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert url == prov["run_logs_asset"]


def test_results_typ_asserts_the_regime_of_every_run_accessor() -> None:
    """results.typ exposes a results.json run to the tables only through a regime-asserting
    accessor: legacy_regime() for the development regime, per_draw_regime() for the main body."""
    text = (PAPER / "results.typ").read_text()
    assert re.search(r"#let legacy_regime\(key\) = \{\n  let r = run\(key\)\n  assert\(r\.noise_seeding == \"legacy\"", text)
    assert re.search(r"#let per_draw_regime\(key\) = \{\n  let r = run\(key\)\n  assert\(r\.noise_seeding == \"per_draw\"", text)
    assert '#let heat_load = json("data/heat_load_slope.json")' in text
    assert re.search(r"#assert\(heat_load\.regime\.noise_seeding == \"legacy\"", text)
