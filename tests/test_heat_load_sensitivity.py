"""The #192 heat-load ceiling sensitivity: six fine-tune legs, their jobs file, the slope scorer.

`campaign.sh experiments/ou_marginal/jobs_heat_load.txt` continues the two #173 seed-1 champions
(`hl_mamba_p962`, `hl_dense_p515`, g20000) 2000 gens at the fine-tune recipe (GA 60 x 10) under
three `[flight.constraints] max_heat_load` ceilings; the 25000 leg is the control.
`heat_load_slope.py` flies every leg on the paired marginal pool at its own ceiling and again
under the v4 limit, named as an explicit override. Pure Python except the one slow seam test.
"""

from __future__ import annotations

import json
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from aerocapture.training.cell_eval import FR_DV_TOTAL, FR_ECC, FR_IFINAL
from aerocapture.training.deploy_overrides import LEGACY_NOISE_REGIME
from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES
from aerocapture.training.toml_utils import load_toml_with_bases

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/training/ou_marginal"
JOBS = REPO / "experiments/ou_marginal/jobs_heat_load.txt"
FR_INTEGRATED_FLUX = FINAL_RECORD_INDICES[FINAL_COLUMNS.index("integrated_flux_mj_m2")]

CELLS = ("mamba_p962", "dense_p515")
LEGS = {"q25": 25000.0, "q27": 27500.0, "q30": 30000.0}
REGISTERED_JOBS = [(f"hs_{cell}_{leg}", 22000, 1, f"training_output/ou_marginal/hl_{cell}") for cell in CELLS for leg in LEGS]
HS_NAMES = [name for name, *_ in REGISTERED_JOBS]


def read_jobs(path: Path) -> list[tuple[str, int, int, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, target, seed, src = line.split("|")
        rows.append((name, int(target), int(seed), src))
    return rows


def split(name: str) -> tuple[str, str]:
    cell, leg = name.removeprefix("hs_").rsplit("_", 1)
    return cell, leg


# ---------------------------------------------------------------------------
# Configs and jobs file
# ---------------------------------------------------------------------------
def test_jobs_file_is_the_registered_list() -> None:
    assert read_jobs(JOBS) == REGISTERED_JOBS


@pytest.mark.parametrize("name", HS_NAMES)
def test_hs_leaf_states_recipe_ceiling_and_deploy_path(name: str) -> None:
    cell, leg = split(name)
    leaf = tomllib.loads((CONFIG_DIR / f"{name}.toml").read_text())
    assert leaf["base"] == [f"hl_{cell}.toml"]
    assert leaf["optimizer"] == {"n_pop": 60, "training_n_sims": 10}
    assert leaf["flight"] == {"constraints": {"max_heat_load": LEGS[leg]}}
    assert leaf["data"] == {"neural_network": f"training_output/ou_marginal/{name}/best_model.json"}
    assert set(leaf) == {"base", "optimizer", "flight", "data"}


@pytest.mark.parametrize("name", HS_NAMES)
def test_hs_resolves_to_its_source_otherwise(name: str) -> None:
    cell, leg = split(name)
    resolved = load_toml_with_bases(CONFIG_DIR / f"{name}.toml")
    source = load_toml_with_bases(CONFIG_DIR / f"hl_{cell}.toml")
    # The leaf's partial [flight.constraints] table deep-merges: the other three limits survive.
    assert resolved["flight"]["constraints"] == {**source["flight"]["constraints"], "max_heat_load": LEGS[leg]}
    assert source["flight"]["constraints"]["max_heat_load"] == LEGS["q25"]
    o = resolved["optimizer"]
    assert (o["algorithm"], o["seed_strategy"], o["n_pop"], o["training_n_sims"]) == ("ga", "adaptive", 60, 10)
    assert resolved["monte_carlo"]["noise_seeding"] == "per_draw"
    for cfg in (resolved, source):
        cfg["flight"]["constraints"].pop("max_heat_load")
        cfg["optimizer"].pop("n_pop")
        cfg["optimizer"].pop("training_n_sims")
        cfg["data"].pop("neural_network")
    assert resolved == source


# ---------------------------------------------------------------------------
# heat_load_slope.py against a fake aerocapture_rs
# ---------------------------------------------------------------------------
@pytest.fixture
def hs(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(REPO / "experiments/ou_marginal"))
    import heat_load_slope  # type: ignore[import-not-found]

    yield heat_load_slope


class FakeRs:
    """Records every run_batch call; every scenario captures at 26 MJ/m2 integrated flux."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_batch(self, toml_path: str, overrides_list: list[dict[str, Any]], **kw: Any) -> Any:
        self.calls.append({"toml_path": toml_path, "overrides_list": overrides_list})
        n = len(overrides_list)
        fr = np.zeros((n, 52))
        fr[:, FR_IFINAL] = 3.0
        fr[:, FR_ECC] = 0.5
        fr[:, FR_DV_TOTAL] = 100.0 + np.arange(n)
        fr[:, FR_INTEGRATED_FLUX] = 26.0
        return SimpleNamespace(final_records=fr, dispersions=np.zeros((n, 26)), trajectories=[np.empty((0, 17)) for _ in range(n)])


@pytest.fixture
def fake_rs(monkeypatch: pytest.MonkeyPatch) -> FakeRs:
    fake = FakeRs()
    monkeypatch.setitem(sys.modules, "aerocapture_rs", fake)
    return fake


@pytest.fixture
def leg(tmp_path: Path) -> tuple[Path, Path]:
    """A q30 leg: a TOML whose ceiling is 30000 and a run dir holding a model."""
    toml = tmp_path / "hs_q30.toml"
    toml.write_text("[monte_carlo]\nseed = 7\n[flight.constraints]\nmax_heat_flux = 200.0\nmax_load_factor = 4.0\nmax_heat_load = 30000.0\n")
    run = tmp_path / "hs_q30"
    run.mkdir()
    (run / "best_model.json").write_text("{}")
    (run / "best_params.json").write_text(json.dumps({"nav.density_filter_gain": 0.9}))
    return toml, run


def test_v4_ceiling_is_the_control_leg_and_the_mission_limit(hs: ModuleType) -> None:
    assert hs.V4_MAX_HEAT_LOAD == LEGS["q25"] == 25000.0
    assert hs.CEILING_KEY == "flight.constraints.max_heat_load"
    assert load_toml_with_bases(REPO / "configs/missions/mars.toml")["flight"]["constraints"]["max_heat_load"] == hs.V4_MAX_HEAT_LOAD


def test_fly_at_own_ceiling_adds_no_override_and_scores_against_the_toml(hs: ModuleType, fake_rs: FakeRs, leg: tuple[Path, Path]) -> None:
    toml, run = leg
    row = hs.fly(toml, run, np.array([5, 6, 7]), ceiling=None)
    [call] = fake_rs.calls
    assert call["toml_path"] == str(toml.resolve())
    for i, ov in enumerate(call["overrides_list"]):
        assert hs.CEILING_KEY not in ov
        assert ov["monte_carlo.noise_seeding"] == LEGACY_NOISE_REGIME["monte_carlo.noise_seeding"]
        assert ov["simulation.random_seed"] == 1000.0 + 7.0 * i  # quote_marginal's marginal regime
        assert ov["data.neural_network"] == str((run / "best_model.json").resolve())
    assert row["ceiling_kj_m2"] == 30000.0
    assert row["heat_load_viol_pct"] == 0.0  # 26 MJ/m2 is under the TOML's 30 MJ/m2
    assert (row["heat_load_p95_mj_m2"], row["heat_load_max_mj_m2"]) == (26.0, 26.0)
    assert (row["dv_cvar95"], row["dv_max"], row["capture_pct"]) == (102.0, 102.0, 100.0)


def test_fly_at_v4_names_the_ceiling_in_every_override_and_in_the_violation_limit(hs: ModuleType, fake_rs: FakeRs, leg: tuple[Path, Path]) -> None:
    toml, run = leg
    row = hs.fly(toml, run, np.array([5, 6]), ceiling=hs.V4_MAX_HEAT_LOAD)
    [call] = fake_rs.calls
    assert all(ov[hs.CEILING_KEY] == 25000.0 for ov in call["overrides_list"])
    assert row["ceiling_kj_m2"] == 25000.0
    assert row["heat_load_viol_pct"] == 100.0  # the same 26 MJ/m2 flights, scored at the v4 limit


def test_summarize_reads_the_slope_against_the_source_seed_spread(hs: ModuleType) -> None:
    rows = {
        "hs_mamba_p962_q25/own": {"dv_cvar95": 120.0},
        "hs_mamba_p962_q27/own": {"dv_cvar95": 116.0},
        "hs_mamba_p962_q30/own": {"dv_cvar95": 110.0},
        "hl_mamba_p962/own": {"dv_cvar95": 119.7},
        "hl_mamba_p962_s2/own": {"dv_cvar95": 132.0},
        "hl_mamba_p962_s3/own": {"dv_cvar95": 122.9},
    }
    s = hs.summarize(rows)
    assert list(s) == ["mamba_p962"]
    m = s["mamba_p962"]
    assert m["cvar95_q30_minus_q25"] == -10.0
    assert m["slope_m_s_per_mj_m2"] == -2.0
    assert m["source_seed_cvar95"] == [119.7, 132.0, 122.9]
    assert m["source_seed_sd"] == pytest.approx(6.38, abs=0.01)  # ddof=1, the RESULTS.md convention
    assert m["outside_seed_spread"] is True


# ---------------------------------------------------------------------------
# The seam: the ceiling reaches the simulator both ways, and moves the flight
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_ceiling_override_matches_the_toml_path_and_moves_the_flight(tmp_path: Path) -> None:
    pytest.importorskip("aerocapture_rs")
    from aerocapture.training.cell_eval import evaluate_cell

    base = REPO / "configs/training/sweep/mamba_p962.toml"
    cell = REPO / "models/demo/ft_mamba_962"
    leaf = tmp_path / "q30.toml"
    leaf.write_text(f'base = ["{base}"]\n\n[flight.constraints]\nmax_heat_load = 30000.0\n')
    seeds = [11, 12, 13]
    via_toml = evaluate_cell(cell, leaf, seeds, extra_overrides=LEGACY_NOISE_REGIME, sim_timeout_secs=30.0)
    via_override = evaluate_cell(cell, base, seeds, extra_overrides={**LEGACY_NOISE_REGIME, "flight.constraints.max_heat_load": 30000.0}, sim_timeout_secs=30.0)
    at_v4 = evaluate_cell(cell, base, seeds, extra_overrides=LEGACY_NOISE_REGIME, sim_timeout_secs=30.0)
    assert np.array_equal(via_toml.final_records, via_override.final_records)
    # NN input 7 is cumulative_heat_load / max_heat_load: a raised ceiling rescales it from the first tick.
    assert not np.array_equal(at_v4.final_records, via_override.final_records)
