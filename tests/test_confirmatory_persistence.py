"""The confirmatory scorer's per-replicate persistence and cell manifests (issue #171).

No simulator: `evaluate_cell` is stubbed with a deterministic toy flight keyed on the pool.
"""

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from aerocapture.training import cell_eval, report
from aerocapture.training.cell_eval import FR_DV_TOTAL, FR_ECC, FR_IFINAL, CellResult
from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES

REPO = Path(__file__).resolve().parents[1]
# The toy cell's record as the pre-#171 `_eval_cell` assembled it (one flight per replicate, no store).
FIXTURE = REPO / "tests/reference_data/confirmatory_toy_cell.json"
TOY_TOML = "configs/training/ou_marginal/mamba_p962.toml"
TOY_POOLS = [list(range(2**31, 2**31 + 40)), list(range(2**31 + 40, 2**31 + 80))]


def toy_flight(seeds: list[int]) -> CellResult:
    rng = np.random.default_rng(seeds[0])
    n = len(seeds)
    fr = np.zeros((n, 52))
    fr[:, FR_IFINAL] = np.where(rng.random(n) < 0.9, 3.0, 1.0)
    fr[:, FR_ECC] = 0.5
    fr[:, FR_DV_TOTAL] = rng.gamma(4.0, 30.0, n)
    col = dict(zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True))
    fr[:, col["max_heat_flux_kw_m2"]] = rng.uniform(0.0, 2.0, n)
    fr[:, col["max_load_factor_g"]] = rng.uniform(0.0, 2.0, n)
    fr[:, col["integrated_flux_mj_m2"]] = rng.uniform(0.0, 1.1, n)
    return CellResult(
        final_records=fr,
        dispersions=np.zeros((n, 26)),
        trajectories=None,
        seeds=list(seeds),
        toml_path=Path("toy.toml"),
        overrides={"data.neural_network": "toy/best_model.json"},
    )


class Interrupted(Exception):
    pass


class ToySim:
    """Stand-in for `evaluate_cell`: logs every pool flown; raises on flight `fail_at` (0-based)."""

    def __init__(self) -> None:
        self.flown: list[list[int]] = []
        self.calls = 0
        self.fail_at: int | None = None

    def __call__(self, cell_dir: Path | None, base_toml: Path, seeds: list[int], **_kw: Any) -> CellResult:
        self.calls += 1
        if self.fail_at is not None and self.calls - 1 == self.fail_at:
            raise Interrupted
        self.flown.append(list(seeds))
        return toy_flight(seeds)


@pytest.fixture
def sim(monkeypatch: pytest.MonkeyPatch) -> ToySim:
    s = ToySim()
    monkeypatch.setattr(cell_eval, "evaluate_cell", s)
    monkeypatch.setattr(report, "_read_constraint_limits", lambda _toml: (1.9, 1.9, 1000.0))
    return s


@pytest.fixture
def ce(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(REPO / "articles/paper/scripts"))
    import confirmatory_eval  # type: ignore[import-not-found]

    yield confirmatory_eval


@pytest.fixture
def cm(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(REPO / "experiments/ou_marginal"))
    import confirmatory_marginal  # type: ignore[import-not-found]

    yield confirmatory_marginal


@pytest.fixture
def qm(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(REPO / "experiments/ou_marginal"))
    import quote_marginal  # type: ignore[import-not-found]

    yield quote_marginal


def test_assembly_matches_the_pre_change_record(tmp_path: Path, sim: ToySim, ce: ModuleType) -> None:
    """Flown, then reassembled from the store alone: both byte-identical to the pre-#171 record."""
    kw = {"cell_dir": tmp_path, "sim_timeout": 5.0, "noise_seeding": "per_draw", "store": tmp_path / "store"}
    flown = ce._eval_cell("toy", TOY_TOML, TOY_POOLS, None, {}, **kw)
    assert json.dumps(flown, indent=1) == FIXTURE.read_text()
    assert len(sim.flown) == 2

    reloaded = ce._eval_cell("toy", TOY_TOML, TOY_POOLS, None, {}, **kw)
    assert json.dumps(reloaded, indent=1) == FIXTURE.read_text()
    assert len(sim.flown) == 2


def test_store_refuses_a_replicate_flown_under_another_config(tmp_path: Path, sim: ToySim, ce: ModuleType) -> None:
    kw = {"cell_dir": tmp_path, "sim_timeout": 5.0, "store": tmp_path / "store"}
    (tmp_path / "best_model.json").write_text('{"v": 1}')
    ce._eval_cell("toy", TOY_TOML, TOY_POOLS, None, {}, noise_seeding="per_draw", **kw)
    with pytest.raises(SystemExit, match="r00 was flown under another"):
        ce._eval_cell("toy", TOY_TOML, TOY_POOLS, None, {}, noise_seeding="legacy", **kw)
    with pytest.raises(SystemExit, match="r00 was flown under another"):
        ce._eval_cell("toy", TOY_TOML, [p[:-1] for p in TOY_POOLS], None, {}, noise_seeding="per_draw", **kw)
    with pytest.raises(SystemExit, match="r00 was flown under another"):  # a timed-out sim scores as a failure
        ce._eval_cell("toy", TOY_TOML, TOY_POOLS, None, {}, noise_seeding="per_draw", **{**kw, "sim_timeout": 30.0})
    (tmp_path / "best_model.json").write_text('{"v": 2}')  # a model repaired in place: same path, other bytes
    with pytest.raises(SystemExit, match="r00 was flown under another"):
        ce._eval_cell("toy", TOY_TOML, TOY_POOLS, None, {}, noise_seeding="per_draw", **kw)


def test_store_compares_the_flown_toml_by_content(tmp_path: Path, sim: ToySim, ce: ModuleType) -> None:
    toml = tmp_path / "toy.toml"
    toml.write_text("[guidance]\ntype = 'fnpag'\n")
    kw = {"cell_dir": tmp_path / "cell", "sim_timeout": 5.0, "noise_seeding": "per_draw", "store": tmp_path / "store"}
    ce._eval_cell("toy", str(toml), TOY_POOLS, None, {}, **kw)
    ce._eval_cell("toy", str(toml), TOY_POOLS, None, {}, **kw)
    assert len(sim.flown) == 2
    toml.write_text("[guidance]\ntype = 'fnpag'\n[simulation]\nmax_time = 1.0\n")  # same path, edited in place
    with pytest.raises(SystemExit, match="r00 was flown under another"):
        ce._eval_cell("toy", str(toml), TOY_POOLS, None, {}, **kw)


def test_interrupted_cell_resumes_without_reflying_a_replicate(tmp_path: Path, sim: ToySim, cm: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    cell = tmp_path / "toy"
    cell.mkdir()
    (cell / "final_eval.parquet").touch()
    argv = ["--cells", f"ou_marginal/toy:{TOY_TOML}:{cell}", "--replicates", "5", "--n", "40"]

    monkeypatch.setattr(cm, "OUT", tmp_path / "straight.json")
    cm.main(argv)
    assert not (tmp_path / "straight.json.partial").exists()

    out = tmp_path / "resumed.json"
    monkeypatch.setattr(cm, "OUT", out)
    sim.fail_at = 6 + 3  # five replicates and the loss re-fly, then the interrupted run's r0-r2 land, r3 raises
    with pytest.raises(Interrupted):
        cm.main(argv)
    assert not out.exists()
    store = tmp_path / "resumed.json.partial/ou_marginal/toy"
    assert sorted(p.name for p in store.iterdir()) == [f"r{r:02d}.{ext}" for r in range(3) for ext in ("json", "npy")]

    sim.fail_at = None
    cm.main(argv)
    assert sim.flown[9:] == sim.flown[3:6]  # only r3, r4 and the loss re-fly flew again
    assert out.read_bytes() == (tmp_path / "straight.json").read_bytes()
    assert not (tmp_path / "resumed.json.partial").exists()

    cm.main(argv)  # a cell present in the results file is skipped
    assert len(sim.flown) == 12


def test_confirmatory_eval_skips_a_saved_cell_unless_forced(tmp_path: Path, sim: ToySim, ce: ModuleType) -> None:
    out = tmp_path / "confirmatory.json"
    argv = ["--cells", f"joint_reference/toy:{TOY_TOML}", "--replicates", "2", "--n", "30", "--out", str(out), "--noise-seeding", "per_draw"]
    ce.main(argv)
    assert [c["label"] for c in json.loads(out.read_text())["cells"]] == ["joint_reference/toy"]
    assert not (tmp_path / "confirmatory.json.partial").exists()
    ce.main(argv)
    assert len(sim.flown) == 2
    ce.main([*argv, "--force"])
    assert len(sim.flown) == 4


def test_manifest_parsing(tmp_path: Path, cm: ModuleType) -> None:
    m = tmp_path / "cells.txt"
    m.write_text("# label|toml|model_dir\n\nA|a.toml\n  B | b.toml | training_output/b  # trailing comment\nC|c.toml|\n")
    assert cm.read_manifest(m) == [("A", "a.toml", None), ("B", "b.toml", "training_output/b"), ("C", "c.toml", None)]
    assert cm.parse_cells(["A:a.toml", "B:b.toml:training_output/b", "C:c.toml:"]) == [
        ("A", "a.toml", None),
        ("B", "b.toml", "training_output/b"),
        ("C", "c.toml", None),
    ]

    m.write_text("A|a.toml\n\nB\n")
    with pytest.raises(SystemExit, match=r"cells.txt:3: expected 'label\|toml\[\|model_dir\]'"):
        cm.read_manifest(m)
    m.write_text("A|a.toml|x|y\n")
    with pytest.raises(SystemExit, match="cells.txt:1"):
        cm.read_manifest(m)
    with pytest.raises(SystemExit, match="--cells: expected"):
        cm.parse_cells(["A:"])
    m.write_text("# nothing left\n")
    with pytest.raises(SystemExit, match="no cells"):
        cm.main(["--manifest", str(m)])


def test_a_save_keeps_cells_another_invocation_wrote_meanwhile(
    tmp_path: Path, sim: ToySim, ce: ModuleType, cm: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two invocations sharing one results file (the slow FNPAG cell runs separately): a cell the
    other one saved after this one started must survive this one's save."""
    out = tmp_path / "confirmatory.json"
    foreign = {
        "label": "other",
        "replicates": [],
        "pooled": {"n": 30, "n_captured": 30, "capture_pct": 1.0, "cvar95": 1.0, "cvar999": 1.0},
        "replicate_stats": {"capture_pct": {"mean": 1.0}},
    }

    def other_invocation_saves(seeds: list[int]) -> CellResult:
        out.write_text(json.dumps({"n_replicates": 2, "n_per_replicate": 30, "noise_seeding": "per_draw", "cells": [foreign]}))
        return toy_flight(seeds)

    monkeypatch.setattr(cell_eval, "evaluate_cell", lambda _d, _t, seeds, **_kw: other_invocation_saves(seeds))
    ce.main(["--cells", f"joint_reference/toy:{TOY_TOML}", "--replicates", "2", "--n", "30", "--out", str(out), "--noise-seeding", "per_draw"])
    assert [c["label"] for c in json.loads(out.read_text())["cells"]] == ["joint_reference/toy", "other"]

    cell = tmp_path / "toy"
    cell.mkdir()
    (cell / "final_eval.parquet").touch()
    monkeypatch.setattr(cm, "OUT", out)
    out.unlink()
    cm.main(["--cells", f"ou_marginal/toy:{TOY_TOML}:{cell}", "--replicates", "2", "--n", "30"])
    assert [c["label"] for c in json.loads(out.read_text())["cells"]] == ["other", "ou_marginal/toy"]


def test_default_manifest_names_committed_tomls(cm: ModuleType) -> None:
    rows = cm.read_manifest(cm.DEFAULT_MANIFEST)
    assert rows
    assert len({label for label, _, _ in rows}) == len(rows)
    for _, toml, _ in rows:
        assert (REPO / toml).is_file(), toml


def test_a_finished_run_is_scored_without_its_final_eval(tmp_path: Path, sim: ToySim, cm: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """The 60 x 10 campaign trained with --skip-report: a final_selection.json written after the last
    checkpoint marks a population run finished. A run with no marker is not scored, nor one extended
    past its target, whose old sidecar predates the checkpoints of the generations it is adding."""
    monkeypatch.setattr(cm, "OUT", tmp_path / "out.json")
    for name in ("done", "running", "extended"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "checkpoint_g00010.json").write_text("{}")
    (tmp_path / "done/final_selection.json").write_text("{}")
    (tmp_path / "running/best_model.json").write_text("{}")
    (tmp_path / "extended/final_selection.json").write_text("{}")
    os.utime(tmp_path / "done/checkpoint_g00010.json", ns=(1, 1))
    os.utime(tmp_path / "extended/final_selection.json", ns=(1, 1))
    cells = [f"ou_marginal/{name}:{TOY_TOML}:{tmp_path / name}" for name in ("done", "running", "extended")]
    cm.main(["--cells", *cells, "--replicates", "2", "--n", "30"])
    assert [c["label"] for c in json.loads(cm.OUT.read_text())["cells"]] == ["ou_marginal/done"]
    assert len(sim.flown) == 2 + 1  # two replicates and the non-capture re-fly


class OutcomeSim:
    """Stand-in for `evaluate_cell` keyed on each seed alone, so a re-fly reproduces it. s % 10 = 0
    times out under the scorer's wall clock and captures without it, 1 crashes, 2 is a pending
    crash, 3 exits on a hyperbola, the rest capture; 4 exceeds the heat-flux and heat-load limits,
    5 the heat-load limit only."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], float | None, dict]] = []

    def __call__(self, _cell_dir: Path, _toml: Path, seeds: list[int], *, sim_timeout_secs: float | None, extra_overrides: dict, **_kw: Any) -> CellResult:
        self.calls.append((list(seeds), sim_timeout_secs, dict(extra_overrides)))
        fr = np.zeros((len(seeds), 52))
        r = np.asarray(seeds) % 10
        fr[:, FR_IFINAL] = np.select([r == 0, r == 1, r == 2], [2.0 if sim_timeout_secs is not None else 3.0, 1.0, 4.0], 3.0)
        fr[:, FR_ECC] = np.where(r == 3, 1.2, 0.5)
        fr[:, FR_DV_TOTAL] = 100.0 + r
        col = dict(zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True))
        fr[:, col["max_heat_flux_kw_m2"]] = np.where(r == 4, 2.0, 0.0)
        fr[:, col["integrated_flux_mj_m2"]] = np.where((r == 4) | (r == 5), 0.01, 0.0)
        return CellResult(final_records=fr, dispersions=np.zeros((len(seeds), 26)), trajectories=None, seeds=list(seeds), toml_path=Path("t"), overrides={})


@pytest.fixture
def outcomes(tmp_path: Path, cm: ModuleType, monkeypatch: pytest.MonkeyPatch) -> tuple[OutcomeSim, list[str]]:
    """The seed-keyed sim on two 40-seed pools (100-139, 140-179) and the argv scoring one finished toy cell."""
    s = OutcomeSim()
    monkeypatch.setattr(cell_eval, "evaluate_cell", s)
    monkeypatch.setattr(report, "_read_constraint_limits", lambda _toml: (1.0, 1.0, 1.0))
    monkeypatch.setattr("aerocapture.training.seeds.make_confirmatory_pools", lambda _seed, _r, _n: [list(range(100, 140)), list(range(140, 180))])
    monkeypatch.setattr(cm, "OUT", tmp_path / "out.json")
    cell = tmp_path / "toy"
    cell.mkdir()
    (cell / "final_selection.json").write_text("{}")
    return s, ["--cells", f"ou_marginal/toy:{TOY_TOML}:{cell}", "--replicates", "2", "--n", "40"]


def test_violations_are_counted_exactly(cm: ModuleType, outcomes: tuple[OutcomeSim, list[str]]) -> None:
    """Per-replicate rates round to 2 decimals (fewer than 5 violations per 100,000 read 0.00): the
    counts ride along, per replicate and summed over the pool."""
    _, argv = outcomes
    cm.main(argv)
    (rec,) = json.loads(cm.OUT.read_text())["cells"]
    counts = {"viol_n": 8, "heat_flux_viol_n": 4, "g_load_viol_n": 0, "heat_load_viol_n": 8}
    for rep in rec["replicates"]:
        assert {k: rep[k] for k in counts} == counts
        assert rep["heat_flux_viol_pct"] == 10.0
    assert {k: rec["pooled"][k] for k in counts} == {k: 2 * v for k, v in counts.items()}


def test_non_captures_are_reflown_without_the_sim_timeout_and_classified(cm: ModuleType, outcomes: tuple[OutcomeSim, list[str]]) -> None:
    sim, argv = outcomes
    cm.main(argv)
    (rec,) = json.loads(cm.OUT.read_text())["cells"]
    assert rec["pooled"]["n_captured"] == 48
    non_captured = [s for s in range(100, 180) if s % 10 in (0, 1, 2, 3)]
    assert sim.calls[2] == (non_captured, None, {"monte_carlo.noise_seeding": "per_draw"})
    assert rec["non_captures"]["n_reflown"] == 32
    assert rec["non_captures"]["outcomes"] == {"crash": 8, "pending_crash": 8, "hyperbolic": 8, "timeout": 0, "capture": 8}

    cm.main(argv)  # scored and classified: nothing flies
    assert len(sim.calls) == 3


def test_non_capture_reflight_refuses_a_changed_model(tmp_path: Path, cm: ModuleType, outcomes: tuple[OutcomeSim, list[str]]) -> None:
    """The row records the scored model's bytes: a best_model.json repaired or retrained since
    (audit_deployed_models.py --repair, a rerun into the dir) must not classify as that row."""
    sim, argv = outcomes
    model = tmp_path / "toy/best_model.json"
    model.write_text('{"v": 1}')
    cm.main(argv)
    d = json.loads(cm.OUT.read_text())
    (rec,) = d["cells"]
    assert rec["model_sha256"] == rec["non_captures"]["model_sha256"] == hashlib.sha256(b'{"v": 1}').hexdigest()
    del rec["non_captures"]
    cm.OUT.write_text(json.dumps(d))
    model.write_text('{"v": 2}')
    with pytest.raises(SystemExit, match="is not the model the row was scored with"):
        cm.main(argv)
    assert len(sim.calls) == 3  # refused before flying
    model.write_text('{"v": 1}')
    cm.main(argv)
    assert len(sim.calls) == 4 and sim.calls[3][1] is None


def _scored(label: str, n_captured: int, cvar999: float, viol: tuple[float, float, float] = (0.0, 0.0, 0.0), **extra: Any) -> dict:
    flux, g, hl = viol
    pooled = {"n": 1_000_000, "n_captured": n_captured, "cvar95": 138.66, "cvar999": cvar999, "max": 249.4}
    pooled |= {"heat_flux_viol_pct": flux, "g_load_viol_pct": g, "heat_load_viol_pct": hl, "viol_pct": max(viol)}
    return {"label": label, "toml": "t.toml", "pooled": pooled, "replicate_stats": {"cvar999": {"se": 0.29}}, **extra}


def test_table_quotes_capture_and_violations_from_counts(
    tmp_path: Path, cm: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    long_outcomes = {"crash": 480, "pending_crash": 20, "hyperbolic": 0, "timeout": 0, "capture": 0}
    cells = [
        _scored("ou_marginal/ft_mamba_p962", 999_995, 163.02, non_captures={"n_reflown": 5, "outcomes": {**long_outcomes, "crash": 5, "pending_crash": 0}}),
        _scored("mamba_p962_long", 979_330, 221.27, (0.0, 0.0, 0.91), non_captures={"n_reflown": 500, "outcomes": long_outcomes}),
        _scored("ou_marginal/pre_adr5", 1_000_000, 200.0, (0.0, 0.0, 0.02)),
        _scored("ou_marginal/hl_counted", 1_000_000, 150.0, (0.0, 0.0, 0.0)),
        _scored("fnpag", 993_679, 236.66),
    ]
    for k in ("heat_flux_viol_pct", "g_load_viol_pct"):  # scored before ADR-0005 split the rate by constraint
        del cells[2]["pooled"][k]
    cells[3]["pooled"] |= {"viol_n": 37, "heat_flux_viol_n": 2, "g_load_viol_n": 0, "heat_load_viol_n": 36}
    monkeypatch.setattr(cm, "OUT", tmp_path / "out.json")
    cm.OUT.write_text(json.dumps({"n_replicates": 10, "n_per_replicate": 100_000, "cells": cells}))
    m = tmp_path / "cells.txt"
    rows = ("mamba_p962_long", "ou_marginal/ft_mamba_p962", "ou_marginal/pre_adr5", "ou_marginal/hl_counted", "fnpag", "ou_marginal/hl_x")
    m.write_text("".join(f"{label}|t.toml|d\n" for label in rows))
    cm.main(["--manifest", str(m), "--table"])
    out = capsys.readouterr().out
    assert "| mamba_p962_long | 97.9330% | 20670: 480 crash, 20 pending_crash (500 re-flown) | 0.91 (0.00 / 0.00 / 0.91) | 138.7 | 221.3 +- 0.3 | 249 |" in out
    assert "| ou_marginal/ft_mamba_p962 | 99.9995% | 5: 5 crash | 0.00 (0.00 / 0.00 / 0.00) | 138.7 | 163.0 +- 0.3 | 249 |" in out
    assert "| ou_marginal/pre_adr5 | 100.0000% | 0 | 0.02 (- / - / 0.02) | 138.7 | 200.0 +- 0.3 | 249 |" in out
    assert "| ou_marginal/hl_counted | 100.0000% | 0 | 0.0037 (0.0002 / 0.0000 / 0.0036) | 138.7 | 150.0 +- 0.3 | 249 |" in out
    assert "| fnpag | 99.3679% | 6321: unclassified | 0.00 (0.00 / 0.00 / 0.00) | 138.7 | 236.7 +- 0.3 | 249 |" in out
    assert out.index("| mamba_p962_long |") < out.index("| ou_marginal/ft_mamba_p962 |")  # manifest order
    assert "not scored yet: ou_marginal/hl_x" in out


def test_recipe_rule_compares_the_three_seed_means(cm: ModuleType) -> None:
    """#173's rule as amended on 2026-10-01: every trainer seed counts, so scratch (S, the hl_mamba
    seeds) wins iff S <= F + sqrt((sd_S^2 + sd_F^2) / 3), F the ft_mamba seeds. Worked examples."""
    f = [_scored("ou_marginal/ft_mamba_p962", 999_995, 163.0), _scored("ou_marginal/ft_mamba_p962_s2", 999_970, 164.6)]
    f.append(_scored("ou_marginal/ft_mamba_p962_s3", 999_920, 161.9))
    labels = ("ou_marginal/hl_mamba_p962", "ou_marginal/hl_mamba_p962_s2", "ou_marginal/hl_mamba_p962_s3")

    spread = [_scored(label, 999_990, v, (0.0, 0.0, 0.05)) for label, v in zip(labels, (150.0, 175.0, 176.0), strict=True)]
    rule = cm.recipe_rule({c["label"]: c for c in spread + f})
    assert rule == {"S": 167.0, "sd_S": 14.73, "F": 163.17, "sd_F": 1.36, "threshold": 171.71, "single_stage": True}

    tight = [_scored(label, 999_990, v) for label, v in zip(labels, (175.0, 176.0, 177.0), strict=True)]
    rule = cm.recipe_rule({c["label"]: c for c in tight + f})
    assert rule == {"S": 176.0, "sd_S": 1.0, "F": 163.17, "sd_F": 1.36, "threshold": 164.14, "single_stage": False}

    assert cm.recipe_rule({c["label"]: c for c in tight[1:] + f}) is None  # a seed not scored yet


def test_v4_manifest_lists_every_paper_row_once(cm: ModuleType) -> None:
    """#174's rows: the #173 cells, the 60 x 10 scratch repeats and fine-tunes, the nine classical
    rows (#188: the ungated reruns for piecewise_constant and equilibrium_glide), the shared-path
    champion, the four PPO cells, their four v4 retrains (#175) and the v4 PPO-scratch seed repeats;
    every TOML committed, every row already scored flown with it."""
    rows = cm.read_manifest(cm.REPO / "experiments/ou_marginal/confirmatory_cells_v4.txt")
    labels = [label for label, _, _ in rows]
    assert len(set(labels)) == len(labels)
    nets = [f"{c}{s}" for c in ("mamba_p962", "dense_p515", "lstm_p1082", "gru_p1014", "dense_p972") for s in ("", "_s2", "_s3")]
    hl = ["hl_mamba_p962", "hl_mamba_p962_s2", "hl_mamba_p962_s3", "hl_dense_p515", "hl_dense_p515_s2", "hl_dense_p515_s3"]
    hl += ["hl_lstm_p1082", "hl_gru_p1014", "hl_dense_p972"]
    ft = ["ft_dense_p515", "ft_dense_p515_s2", "ft_dense_p515_s3", "ft_mamba_p962", "ft_mamba_p962_s2", "ft_mamba_p962_s3"]
    ft += ["ft_gru_p1014", "ft_dense_p972", "ft_lstm_p1082"]
    classical = [f"classical/{c}" for c in ("ftc", "energy_controller", "pred_guid", "ftc_joint", "energy_controller_joint", "pred_guid_joint", "fnpag")]
    classical += ["classical_ungated/piecewise_constant", "classical_ungated/equilibrium_glide"]
    rl = [f"paper/rl/{p}{c}_ppo_{w}" for p in ("", "hl_") for c in ("dense_p515", "gru_p1014") for w in ("scratch", "warm")]
    rl += [f"paper/rl/hl_{c}_ppo_scratch_s{k}" for c in ("dense_p515", "gru_p1014") for k in (2, 3)]
    assert set(labels) == {f"ou_marginal/{c}" for c in nets + hl + ft + classical} | {"mamba_p962_long", *rl}
    for _, toml, model_dir in rows:
        assert (REPO / toml).is_file(), toml
        assert model_dir, toml
    scored = {c["label"]: c["toml"] for c in json.loads(cm.OUT.read_text())["cells"]}
    assert {label: toml for label, toml, _ in rows if label in scored} == {label: scored[label] for label in labels if label in scored}


def test_quote_marginal_skips_quoted_cells_unless_forced(tmp_path: Path, qm: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    scored: list[str] = []

    def fake_score(toml: str, model_dir: str | None, seeds: np.ndarray, regime: str) -> dict:
        scored.append(f"{toml}/{regime}")
        return {k: 1.0 for k in ("capture_pct", "dv_p50", "dv_p95", "dv_p99", "dv_cvar95", "heat_load_viol_pct", "heat_flux_viol_pct", "g_load_viol_pct")}

    monkeypatch.setattr(qm, "score", fake_score)
    monkeypatch.setattr(qm, "OUT", tmp_path / "quote_results.json")
    monkeypatch.setattr(qm, "REPO", tmp_path)
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.toml").touch()
    m = tmp_path / "cells.txt"
    m.write_text("A|a.toml\nB|b.toml\nD|d.toml\n")  # d.toml is not written yet: skipped, not flown
    argv = ["--manifest", str(m), "--n-sims", "4"]

    qm.main(argv)
    assert scored == ["a.toml/frozen", "a.toml/marginal", "b.toml/frozen", "b.toml/marginal"]
    first = qm.OUT.read_text()
    assert set(json.loads(first)["cells"]) == {"A/frozen", "A/marginal", "B/frozen", "B/marginal"}

    scored.clear()
    qm.main(argv)
    assert scored == []
    assert qm.OUT.read_text() == first
    with pytest.raises(SystemExit, match="nothing to re-quote"):  # an explicit re-score of the unwritten row is an error, not a skip
        qm.main([*argv, "--only", "D"])
    assert scored == []

    m.write_text("A|a.toml\nB|b.toml\nC|c.toml\n")
    qm.main(argv)
    assert scored == ["c.toml/frozen", "c.toml/marginal"]

    scored.clear()
    qm.main([*argv, "--only", "A"])  # --only re-scores a quoted cell and keeps the others
    assert scored == ["a.toml/frozen", "a.toml/marginal"]
    assert len(json.loads(qm.OUT.read_text())["cells"]) == 6

    scored.clear()
    m.write_text("A|a.toml\n")
    qm.main([*argv, "--force"])  # --force re-quotes the listed cells into a fresh file
    assert scored == ["a.toml/frozen", "a.toml/marginal"]
    assert set(json.loads(qm.OUT.read_text())["cells"]) == {"A/frozen", "A/marginal"}

    with pytest.raises(SystemExit, match="another n_sims / protocol"):
        qm.main(["--manifest", str(m), "--n-sims", "5"])
    with pytest.raises(SystemExit, match="another n_sims / protocol"):
        qm.main(["--manifest", str(m), "--n-sims", "5", "--only", "A"])


def test_extra_override_flies_every_replicate_and_the_reflight(cm: ModuleType, outcomes: tuple[OutcomeSim, list[str]]) -> None:
    """#176's reset-state cell: the override reaches every replicate, the non-capture re-fly, and the row."""
    sim, argv = outcomes
    cm.main([*argv, "--extra-override", "guidance.neural_network.reset_state_every_tick=true"])
    (rec,) = json.loads(cm.OUT.read_text())["cells"]
    flown = {"monte_carlo.noise_seeding": "per_draw", "guidance.neural_network.reset_state_every_tick": True}
    assert [c[2] for c in sim.calls] == [flown, flown, flown]
    assert rec["extra_overrides"] == {"guidance.neural_network.reset_state_every_tick": True}
    assert rec["non_captures"]["n_reflown"] == 32


def test_scored_row_under_other_overrides_stops_instead_of_skipping(cm: ModuleType, outcomes: tuple[OutcomeSim, list[str]]) -> None:
    """A label scored without the reset-state override must not pass as the reset-state cell: the skip
    compares the row's recorded flight (the regime-only extra_overrides of the oldest rows included)."""
    sim, argv = outcomes
    cm.main(argv)
    n_calls = len(sim.calls)
    with pytest.raises(SystemExit, match="delete the row to re-score"):
        cm.main([*argv, "--extra-override", "guidance.neural_network.reset_state_every_tick=true"])
    data = json.loads(cm.OUT.read_text())
    data["cells"][0]["extra_overrides"] = {"monte_carlo.noise_seeding": "per_draw"}
    cm.OUT.write_text(json.dumps(data))
    cm.main(argv)
    assert len(sim.calls) == n_calls


def test_controls_rule_reads_the_champion_seed_range(cm: ModuleType) -> None:
    """#176's pre-registered rule: a control whose CVaR99.9 lands inside the intact champion's three-seed
    range needs seeds 2 and 3 before the paper reads it; one outside needs no repeats."""
    champion = [_scored(k, 1_000_000, v) for k, v in zip(cm.RECIPE_S, (173.9, 159.3, 142.3), strict=True)]
    window, nodv = cm.CONTROLS
    inside, outside = _scored(window, 1_000_000, 160.0), _scored(nodv, 1_000_000, 190.0)
    window_s2 = _scored(f"{window}_s2", 1_000_000, 150.0)
    rule = cm.controls_rule({c["label"]: c for c in [*champion, inside, outside, window_s2]})
    assert rule == {
        "champion_range": [142.3, 173.9],
        "controls": {
            window: {"cvar999": 160.0, "inside_range": True, "seeds_2_3_required": True, "repeats": {f"{window}_s2": 150.0, f"{window}_s3": None}},
            nodv: {"cvar999": 190.0, "inside_range": False, "seeds_2_3_required": False, "repeats": {f"{nodv}_s2": None, f"{nodv}_s3": None}},
        },
        "not_scored": [],
    }
    rule = cm.controls_rule({c["label"]: c for c in [*champion, outside]})
    assert rule is not None and rule["not_scored"] == [window] and list(rule["controls"]) == [nodv]
    assert cm.controls_rule({c["label"]: c for c in champion[1:]}) is None  # a champion seed not scored yet
