"""The #176 deployment controls on the v4 champion: the stress-depth scorer, the fresh-pool
re-quote's regime flag and the runner that pre-registers the controls' decision rule.

No simulator: `evaluate_cell` is stubbed with a deterministic toy flight keyed on the pool.
"""

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from aerocapture.training import cell_eval
from aerocapture.training.cell_eval import FR_DV_TOTAL, FR_ECC, FR_IFINAL, CellResult

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "articles/paper/scripts"
RUNNER = REPO / "experiments/paper/15_state_controls.sh"


class ToySim:
    """Stand-in for `evaluate_cell`: records the overrides of every flight; the same pool flies the same scenarios."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, _cell_dir: Path | None, _toml: Path, seeds: list[int] | None = None, *, pool: tuple[int, int] | None = None, **kw: Any) -> CellResult:
        self.calls.append(dict(kw.get("extra_overrides") or {}))
        seed_list = list(seeds) if seeds is not None else list(range(pool[0], pool[0] + pool[1]))  # type: ignore[index]
        rng = np.random.default_rng(seed_list[0] + len(self.calls))
        n = len(seed_list)
        fr = np.zeros((n, 52))
        fr[:, FR_IFINAL] = np.where(rng.random(n) < 0.9, 3.0, 1.0)
        fr[:, FR_ECC] = 0.5
        fr[:, FR_DV_TOTAL] = rng.gamma(4.0, 30.0, n)
        return CellResult(final_records=fr, dispersions=np.zeros((n, 26)), trajectories=None, seeds=seed_list, toml_path=Path("toy.toml"), overrides={})


@pytest.fixture
def sim(monkeypatch: pytest.MonkeyPatch) -> ToySim:
    s = ToySim()
    monkeypatch.setattr(cell_eval, "evaluate_cell", s)
    return s


@pytest.fixture
def sd(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import stress_depth_eval  # type: ignore[import-not-found]

    yield stress_depth_eval


@pytest.fixture
def fpr(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import fresh_pool_requote  # type: ignore[import-not-found]

    yield fresh_pool_requote


def test_stress_depth_cells_are_the_champion_and_the_four_retuned_classicals(sd: ModuleType) -> None:
    assert [label for label, _, _ in sd.CELLS] == ["NN", "joint-FTC", "FTC-fixed", "PredGuid", "FNPAG"]
    assert sd.CELLS[0][1] == "ou_marginal/hl_mamba_p962"
    for _, run_dir, toml in sd.CELLS:
        assert (REPO / toml).is_file(), toml
        assert run_dir.startswith("ou_marginal/"), run_dir
    assert sd.REGIME == "per_draw"
    assert sd.OUT == REPO / "articles/paper/data/stress_depth.json"
    assert "mc-stress-depth" in (REPO / "articles/paper/Makefile").read_text()


def test_stress_depth_pairs_the_champion_with_every_classical_under_per_draw(
    tmp_path: Path, sim: ToySim, sd: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _, run_dir, _ in sd.CELLS:
        (tmp_path / run_dir).mkdir(parents=True)
        (tmp_path / run_dir / "best_params.json").write_text("{}")
    monkeypatch.setattr(sd, "TRAINING", tmp_path)
    monkeypatch.setattr(sd, "OUT", tmp_path / "stress_depth.json")
    monkeypatch.setattr(sd, "N_BOOT", 50)
    sd.main(["--n-sims", "40"])
    out = json.loads((tmp_path / "stress_depth.json").read_text())
    assert out["regime"] == "per_draw" and out["noise_seeding"] == "per_draw" and out["n_sims"] == 40
    assert [c["label"] for c in out["cells"]] == ["NN", "joint-FTC", "FTC-fixed", "PredGuid", "FNPAG"]
    assert [(p["a"], p["b"]) for p in out["paired"]] == [("NN", b) for b in ("joint-FTC", "FTC-fixed", "PredGuid", "FNPAG")]
    for p in out["paired"]:
        assert {"delta_capture_pts", "delta_capture_pts_ci", "delta_cvar95", "delta_cvar95_ci"} <= set(p)
    for c in out["cells"]:
        assert {"capture_pct", "capture_pct_ci", "dv_mean", "dv_cvar95", "dv_cvar95_ci"} <= set(c)
    stress = {"monte_carlo.noise_seeding": "per_draw", **sd.STRESS_OVERRIDES}
    assert sim.calls == [stress] * 5


def test_stress_depth_requires_every_cell_deployed(tmp_path: Path, sd: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sd, "TRAINING", tmp_path)
    with pytest.raises(SystemExit, match="not deployed"):
        sd.main(["--n-sims", "40"])


@pytest.mark.parametrize(("flag", "regime"), [([], "legacy"), (["--noise-seeding", "per_draw"], "per_draw")])
def test_fresh_pool_requote_states_its_regime(tmp_path: Path, sim: ToySim, fpr: ModuleType, flag: list[str], regime: str) -> None:
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    (run_dir / "best_model.json").write_text("{}")
    fpr.main([str(run_dir), "--toml", "configs/training/ou_marginal/hl_mamba_p962.toml", "--n-sims", "40", *flag])
    assert sim.calls == [{"monte_carlo.noise_seeding": regime}]
    out = json.loads((run_dir / "fresh_pool_requote.json").read_text())
    assert out["noise_seeding"] == regime and out["n"] == 40


def test_runner_registers_the_v4_controls_and_the_reset_state_cell() -> None:
    text = RUNNER.read_text()
    assert "jobs_controls.txt" in text
    assert "ou_marginal/v4_reset_state" in text
    assert "guidance.neural_network.reset_state_every_tick=true" in text
    assert subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True).returncode == 0
