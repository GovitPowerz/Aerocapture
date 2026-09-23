"""The one deploy-side evaluation path (cell_eval): override shape, scaffolding merge,
model pin, captured mask, pool seeds -- against a fake `aerocapture_rs`, plus one
slow real run on the committed demo cell."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from aerocapture.training import cell_eval
from aerocapture.training.cell_eval import CellResult, evaluate_cell, fly_mc, fly_nominal, is_captured, reserved_pool
from aerocapture.training.reference import _MC_DISPERSION_DOMAINS
from aerocapture.training.seeds import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds

REPO = Path(__file__).resolve().parents[1]
DEMO_TOML = REPO / "configs/training/sweep/mamba_p962.toml"
DEMO_CELL = REPO / "models/demo/ft_mamba_962"

N_COLS = 52


def _records(n: int) -> np.ndarray:
    fr = np.zeros((n, N_COLS))
    fr[:, cell_eval.FR_IFINAL] = 3.0
    fr[:, cell_eval.FR_ECC] = 0.5
    fr[:, cell_eval.FR_DV_TOTAL] = np.arange(n, dtype=float) + 100.0
    return fr


class FakeRs:
    """Records every run_batch / run_mc call; returns synthetic BatchResults."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_batch(
        self,
        toml_path: str,
        overrides_list: list[dict],
        n_threads: int | None = None,
        include_trajectories: bool = False,
        sim_timeout_secs: float | None = None,
    ) -> Any:
        self.calls.append(
            {
                "fn": "run_batch",
                "toml_path": toml_path,
                "overrides_list": overrides_list,
                "n_threads": n_threads,
                "include_trajectories": include_trajectories,
                "sim_timeout_secs": sim_timeout_secs,
            }
        )
        return self._results(len(overrides_list), include_trajectories)

    def run_mc(self, toml_path: str, overrides: dict | None = None, include_trajectories: bool = False, sim_timeout_secs: float | None = None) -> Any:
        self.calls.append(
            {"fn": "run_mc", "toml_path": toml_path, "overrides": overrides, "include_trajectories": include_trajectories, "sim_timeout_secs": sim_timeout_secs}
        )
        n = int((overrides or {}).get("simulation.n_sims", 4))
        return self._results(n, include_trajectories)

    @staticmethod
    def _results(n: int, include_trajectories: bool) -> Any:
        return SimpleNamespace(
            final_records=_records(n),
            dispersions=np.zeros((n, 26)),
            trajectories=[np.ones((5, 17)) * i for i in range(n)] if include_trajectories else [np.empty((0, 17)) for _ in range(n)],
        )


@pytest.fixture
def fake_rs(monkeypatch: pytest.MonkeyPatch) -> FakeRs:
    fake = FakeRs()
    monkeypatch.setitem(sys.modules, "aerocapture_rs", fake)
    return fake


@pytest.fixture
def nn_cell(tmp_path: Path) -> tuple[Path, Path]:
    """An NN cell (best_model.json + best_params.json) next to a base TOML."""
    base = tmp_path / "base.toml"
    base.write_text("[monte_carlo]\nseed = 7\n")
    cell = tmp_path / "mamba"
    cell.mkdir()
    (cell / "best_model.json").write_text("{}")
    (cell / "best_params.json").write_text(json.dumps({"nav.density_filter_gain": 0.9, "shaping.max_bank_acceleration": 5.0, "w0": 1.0}))
    return base, cell


SCAFFOLDING = {"navigation.density_filter_gain": 0.9, "guidance.command_shaping.max_bank_acceleration": 5.0, "guidance.command_shaping.enabled": True}


# ---------------------------------------------------------------------------
# evaluate_cell: the override dict every caller used to hand-build
# ---------------------------------------------------------------------------
def test_evaluate_cell_override_shape(fake_rs: FakeRs, nn_cell: tuple[Path, Path]) -> None:
    base, cell = nn_cell
    res = evaluate_cell(cell, base, [11, 22], include_trajectories=True, sim_timeout_secs=5.0, n_threads=1)
    [call] = fake_rs.calls
    assert call["fn"] == "run_batch"
    assert call["toml_path"] == str(base.resolve())
    assert call["include_trajectories"] is True and call["sim_timeout_secs"] == 5.0 and call["n_threads"] == 1
    expected_base = {"simulation.n_sims": 1, **SCAFFOLDING, "data.neural_network": str((cell / "best_model.json").resolve())}
    assert call["overrides_list"] == [{**expected_base, "monte_carlo.seed": 11}, {**expected_base, "monte_carlo.seed": 22}]
    assert res.seeds == [11, 22]
    assert res.toml_path == base.resolve()
    assert res.overrides == expected_base


def test_optimized_toml_wins_with_no_scaffolding_and_no_pin(fake_rs: FakeRs, tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text("")
    cell = tmp_path / "ftc"
    cell.mkdir()
    (cell / "optimized_ftc.toml").write_text("")
    (cell / "best_params.json").write_text(json.dumps({"nav.density_filter_gain": 0.9}))
    res = evaluate_cell(cell, base, [1])
    [call] = fake_rs.calls
    assert call["toml_path"] == str((cell / "optimized_ftc.toml").resolve())
    assert call["overrides_list"] == [{"simulation.n_sims": 1, "monte_carlo.seed": 1}]
    assert res.toml_path == (cell / "optimized_ftc.toml").resolve()


def test_explicit_model_beats_the_cell_model(fake_rs: FakeRs, nn_cell: tuple[Path, Path], tmp_path: Path) -> None:
    base, cell = nn_cell
    other = tmp_path / "bundle.json"
    other.write_text("{}")
    evaluate_cell(cell, base, [1], model=other)
    assert fake_rs.calls[0]["overrides_list"][0]["data.neural_network"] == str(other.resolve())


def test_missing_explicit_model_raises(fake_rs: FakeRs, nn_cell: tuple[Path, Path], tmp_path: Path) -> None:
    base, cell = nn_cell
    with pytest.raises(FileNotFoundError, match="does not exist"):
        evaluate_cell(cell, base, [1], model=tmp_path / "nope.json")
    assert fake_rs.calls == []


def test_no_cell_dir_means_base_toml_and_nothing_else(fake_rs: FakeRs, tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text("")
    evaluate_cell(None, base, [3])
    assert fake_rs.calls[0]["toml_path"] == str(base.resolve())
    assert fake_rs.calls[0]["overrides_list"] == [{"simulation.n_sims": 1, "monte_carlo.seed": 3}]


def test_extra_overrides_win_over_scaffolding_but_not_over_the_seed(fake_rs: FakeRs, nn_cell: tuple[Path, Path]) -> None:
    base, cell = nn_cell
    extra = {"navigation.density_filter_gain": 0.1, "monte_carlo.noise_seeding": "legacy", "monte_carlo.seed": 999}
    evaluate_cell(cell, base, [5], extra_overrides=extra)
    [ov] = fake_rs.calls[0]["overrides_list"]
    assert ov["navigation.density_filter_gain"] == 0.1
    assert ov["monte_carlo.noise_seeding"] == "legacy"
    assert ov["monte_carlo.seed"] == 5


def test_per_seed_overrides_apply_by_index(fake_rs: FakeRs, tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text("")
    evaluate_cell(None, base, [1, 2], per_seed_overrides=[{"simulation.random_seed": 1000.0}, {"simulation.random_seed": 1007.0}])
    ovs = fake_rs.calls[0]["overrides_list"]
    assert [o["simulation.random_seed"] for o in ovs] == [1000.0, 1007.0]
    assert [o["monte_carlo.seed"] for o in ovs] == [1, 2]


def test_per_seed_overrides_length_mismatch_raises(fake_rs: FakeRs, tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text("")
    with pytest.raises(ValueError, match="per_seed_overrides"):
        evaluate_cell(None, base, [1, 2], per_seed_overrides=[{}])


def test_seeds_are_plain_ints(fake_rs: FakeRs, tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text("")
    res = evaluate_cell(None, base, np.array([4, 5], dtype=np.int64))
    assert all(type(s) is int for s in res.seeds)
    assert all(type(o["monte_carlo.seed"]) is int for o in fake_rs.calls[0]["overrides_list"])


# ---------------------------------------------------------------------------
# CellResult: what every caller re-derived
# ---------------------------------------------------------------------------
def test_result_captured_dv_and_trajectories(fake_rs: FakeRs, tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text("")
    res = evaluate_cell(None, base, [1, 2, 3], include_trajectories=True)
    assert res.final_records.shape == (3, N_COLS) and res.dispersions.shape == (3, 26)
    assert res.trajectories is not None and len(res.trajectories) == 3
    assert res.n == 3
    assert res.captured.dtype == np.bool_ and res.captured.all()
    np.testing.assert_array_equal(res.dv, [100.0, 101.0, 102.0])


def test_trajectories_none_when_not_requested(fake_rs: FakeRs, tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text("")
    assert evaluate_cell(None, base, [1]).trajectories is None


def test_is_captured_needs_exit_and_bound_orbit() -> None:
    fr = _records(4)
    fr[1, cell_eval.FR_IFINAL] = 1.0  # crash
    fr[2, cell_eval.FR_ECC] = 1.2  # hyperbolic exit
    fr[3, cell_eval.FR_ECC] = 1.0  # parabolic: not bound
    np.testing.assert_array_equal(is_captured(fr), [True, False, False, False])
    res = CellResult(final_records=fr, dispersions=np.zeros((4, 26)), trajectories=None, seeds=[0, 1, 2, 3], toml_path=Path("x"), overrides={})
    np.testing.assert_array_equal(res.dv, [100.0])


def test_charts_is_captured_is_the_same_predicate() -> None:
    from aerocapture.training import charts

    assert charts.is_captured is is_captured


def test_column_constants_match_the_rust_map() -> None:
    aero = pytest.importorskip("aerocapture_rs")
    idx = aero.final_record_indices()
    assert (idx["ifinal"], idx["ecc"], idx["dv_total_ms"]) == (cell_eval.FR_IFINAL, cell_eval.FR_ECC, cell_eval.FR_DV_TOTAL)


# ---------------------------------------------------------------------------
# fly_mc / fly_nominal: the config's own Monte Carlo through the same resolution
# ---------------------------------------------------------------------------
def test_fly_mc_uses_run_mc_with_n_sims(fake_rs: FakeRs, nn_cell: tuple[Path, Path]) -> None:
    base, cell = nn_cell
    res = fly_mc(cell, base, n_sims=6, extra_overrides={"guidance.type": "neural_network"}, include_trajectories=True, sim_timeout_secs=2.0)
    [call] = fake_rs.calls
    assert call["fn"] == "run_mc"
    assert call["toml_path"] == str(base.resolve())
    assert call["overrides"] == {
        "simulation.n_sims": 6,
        **SCAFFOLDING,
        "data.neural_network": str((cell / "best_model.json").resolve()),
        "guidance.type": "neural_network",
    }
    assert call["include_trajectories"] is True and call["sim_timeout_secs"] == 2.0
    assert res.n == 6 and res.seeds == []


def test_fly_mc_without_n_sims_keeps_the_config_value(fake_rs: FakeRs, tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text("")
    fly_mc(None, base, extra_overrides={"guidance.type": "ftc"})
    assert fake_rs.calls[0]["overrides"] == {"guidance.type": "ftc"}


def test_fly_nominal_turns_every_dispersion_domain_off(fake_rs: FakeRs, nn_cell: tuple[Path, Path]) -> None:
    base, cell = nn_cell
    res = fly_nominal(cell, base, extra_overrides={"monte_carlo.noise_seeding": "legacy"})
    [call] = fake_rs.calls
    assert call["fn"] == "run_mc" and call["include_trajectories"] is True
    ov = call["overrides"]
    assert ov["simulation.n_sims"] == 1
    assert all(ov[f"monte_carlo.{d}.level"] == "off" for d in _MC_DISPERSION_DOMAINS)
    assert ov["data.neural_network"] == str((cell / "best_model.json").resolve())
    assert ov["monte_carlo.noise_seeding"] == "legacy"
    assert res.trajectories is not None and len(res.trajectories) == 1


# ---------------------------------------------------------------------------
# reserved_pool: the TOML's base seed + a registered offset
# ---------------------------------------------------------------------------
def test_pool_seeds_are_keyed_by_the_toml_actually_flown(fake_rs: FakeRs, tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text("[monte_carlo]\nseed = 7\n")
    cell = tmp_path / "ftc"
    cell.mkdir()
    (cell / "optimized_ftc.toml").write_text("[monte_carlo]\nseed = 9\n")
    res = evaluate_cell(cell, base, pool=(FINAL_EVAL_SEED_OFFSET, 4))
    assert res.seeds == make_reserved_seeds(9, FINAL_EVAL_SEED_OFFSET, 4)
    assert [o["monte_carlo.seed"] for o in fake_rs.calls[0]["overrides_list"]] == res.seeds


def test_seeds_and_pool_are_exclusive(fake_rs: FakeRs, tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    base.write_text("")
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_cell(None, base, [1], pool=(FINAL_EVAL_SEED_OFFSET, 1))
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_cell(None, base)


def test_reserved_pool_reads_the_toml_seed(tmp_path: Path) -> None:
    toml = tmp_path / "t.toml"
    toml.write_text("[monte_carlo]\nseed = 7\n")
    assert reserved_pool(toml, FINAL_EVAL_SEED_OFFSET, 5) == make_reserved_seeds(7, FINAL_EVAL_SEED_OFFSET, 5)


def test_reserved_pool_defaults_to_42(tmp_path: Path) -> None:
    toml = tmp_path / "t.toml"
    toml.write_text("")
    assert reserved_pool(toml, FINAL_EVAL_SEED_OFFSET, 3) == make_reserved_seeds(42, FINAL_EVAL_SEED_OFFSET, 3)


# ---------------------------------------------------------------------------
# The real thing: the committed demo cell on three seeds
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_evaluate_cell_real_run_matches_a_direct_run_batch() -> None:
    aero = pytest.importorskip("aerocapture_rs")
    seeds = reserved_pool(DEMO_TOML, FINAL_EVAL_SEED_OFFSET, 3)
    res = evaluate_cell(DEMO_CELL, DEMO_TOML, seeds, include_trajectories=True, sim_timeout_secs=30.0)
    assert res.final_records.shape == (3, N_COLS) and res.dispersions.shape == (3, 26)
    assert res.trajectories is not None and len(res.trajectories) == 3 and all(t.shape[1] == 17 for t in res.trajectories)
    assert res.captured.shape == (3,)
    assert "data.neural_network" in res.overrides and "navigation.density_filter_gain" in res.overrides
    direct = aero.run_batch(str(DEMO_TOML.resolve()), [{**res.overrides, "monte_carlo.seed": s} for s in seeds], sim_timeout_secs=30.0)
    assert np.array_equal(direct.final_records, res.final_records)
    assert np.array_equal(direct.dispersions, res.dispersions)
    np.testing.assert_array_equal(direct.captured, res.captured)
