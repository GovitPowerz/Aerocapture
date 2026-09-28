"""The confirmatory scorer's per-replicate persistence and cell manifests (issue #171).

No simulator: `evaluate_cell` is stubbed with a deterministic toy flight keyed on the pool.
"""

import json
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
    (tmp_path / "best_model.json").write_text('{"v": 2}')  # a model repaired in place: same path, other bytes
    with pytest.raises(SystemExit, match="r00 was flown under another"):
        ce._eval_cell("toy", TOY_TOML, TOY_POOLS, None, {}, noise_seeding="per_draw", **kw)


def test_interrupted_cell_resumes_without_reflying_a_replicate(tmp_path: Path, sim: ToySim, cm: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    cell = tmp_path / "toy"
    cell.mkdir()
    (cell / "final_eval.parquet").touch()
    argv = ["--cells", f"ou_marginal/toy:{TOY_TOML}:{cell}", "--replicates", "5", "--n", "40"]

    monkeypatch.setattr(cm, "OUT", tmp_path / "straight.json")
    cm.main(argv)
    straight = json.loads((tmp_path / "straight.json").read_text())["cells"]
    assert not (tmp_path / "straight.json.partial").exists()

    out = tmp_path / "resumed.json"
    monkeypatch.setattr(cm, "OUT", out)
    sim.fail_at = 5 + 3  # the interrupted run's replicates r0-r2 land, r3 raises
    with pytest.raises(Interrupted):
        cm.main(argv)
    assert not out.exists()
    store = tmp_path / "resumed.json.partial/ou_marginal/toy"
    assert sorted(p.name for p in store.iterdir()) == [f"r{r:02d}.{ext}" for r in range(3) for ext in ("json", "npy")]

    sim.fail_at = None
    cm.main(argv)
    assert sim.flown[8:] == sim.flown[3:5]  # only r3 and r4 flew again
    assert json.dumps(json.loads(out.read_text())["cells"]) == json.dumps(straight)
    assert not (tmp_path / "resumed.json.partial").exists()

    cm.main(argv)  # a cell present in the results file is skipped
    assert len(sim.flown) == 10


def test_confirmatory_eval_drops_the_store_once_the_cell_is_saved(tmp_path: Path, sim: ToySim, ce: ModuleType) -> None:
    out = tmp_path / "confirmatory.json"
    ce.main(["--cells", f"joint_reference/toy:{TOY_TOML}", "--replicates", "2", "--n", "30", "--out", str(out), "--noise-seeding", "per_draw"])
    assert [c["label"] for c in json.loads(out.read_text())["cells"]] == ["joint_reference/toy"]
    assert not (tmp_path / "confirmatory.json.partial").exists()


def test_manifest_parsing(tmp_path: Path, cm: ModuleType) -> None:
    m = tmp_path / "cells.txt"
    m.write_text("# label|toml|model_dir\n\nA|a.toml\n  B | b.toml | training_output/b  # trailing comment\nC|c.toml|\n")
    assert cm.read_manifest(m) == [("A", "a.toml", None), ("B", "b.toml", "training_output/b"), ("C", "c.toml", None)]
    assert cm.parse_cells(["A:a.toml", "B:b.toml:training_output/b"]) == [("A", "a.toml", None), ("B", "b.toml", "training_output/b")]

    m.write_text("A|a.toml\n\nB\n")
    with pytest.raises(SystemExit, match=r"cells.txt:3: expected 'label\|toml\[\|model_dir\]'"):
        cm.read_manifest(m)
    m.write_text("A|a.toml|x|y\n")
    with pytest.raises(SystemExit, match="cells.txt:1"):
        cm.read_manifest(m)


def test_default_manifest_names_committed_tomls(cm: ModuleType) -> None:
    rows = cm.read_manifest(cm.DEFAULT_MANIFEST)
    assert rows
    assert len({label for label, _, _ in rows}) == len(rows)
    for _, toml, _ in rows:
        assert (REPO / toml).is_file(), toml


def test_quote_marginal_skips_quoted_cells_unless_forced(tmp_path: Path, qm: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    scored: list[str] = []

    def fake_score(toml: str, model_dir: str | None, seeds: np.ndarray, regime: str) -> dict:
        scored.append(f"{toml}/{regime}")
        return {k: 1.0 for k in ("capture_pct", "dv_p50", "dv_p95", "dv_p99", "dv_cvar95", "heat_load_viol_pct")}

    monkeypatch.setattr(qm, "score", fake_score)
    monkeypatch.setattr(qm, "OUT", tmp_path / "quote_results.json")
    m = tmp_path / "cells.txt"
    m.write_text("A|a.toml\nB|b.toml\n")
    argv = ["--manifest", str(m), "--n-sims", "4"]

    qm.main(argv)
    assert scored == ["a.toml/frozen", "a.toml/marginal", "b.toml/frozen", "b.toml/marginal"]
    first = qm.OUT.read_text()
    assert set(json.loads(first)["cells"]) == {"A/frozen", "A/marginal", "B/frozen", "B/marginal"}

    scored.clear()
    qm.main(argv)
    assert scored == []
    assert qm.OUT.read_text() == first

    m.write_text("A|a.toml\nB|b.toml\nC|c.toml\n")
    qm.main(argv)
    assert scored == ["c.toml/frozen", "c.toml/marginal"]

    scored.clear()
    qm.main([*argv, "--force"])
    assert len(scored) == 6

    with pytest.raises(SystemExit, match="another n_sims / protocol"):
        qm.main(["--manifest", str(m), "--n-sims", "5"])
