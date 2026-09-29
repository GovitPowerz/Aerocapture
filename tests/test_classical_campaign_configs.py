"""The #172 classical retune cells differ from their shared-path parents only in regime and allocation.

Each `configs/training/ou_marginal/classical/<cell>.toml` states `per_draw` and the
full GA allocation in its own file (a config that base-inherits it does not inherit
CLI flags), resolves to its parent everywhere else, and is driven by
`experiments/ou_marginal/classical_campaign.sh` and scored through
`experiments/ou_marginal/classical_cells.txt`. The #188 `classical_ungated/<cell>.toml`
reruns differ from their #172 cell only in the opened feasibility ceiling. Pure Python (no bindings).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from aerocapture.training.toml_utils import load_toml_with_bases

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO / "configs/training/ou_marginal"
CELL_DIR = CAMPAIGN_ROOT / "classical"
UNGATED_DIR = CAMPAIGN_ROOT / "classical_ungated"
RUNNER = REPO / "experiments/ou_marginal/classical_campaign.sh"
MANIFEST = REPO / "experiments/ou_marginal/classical_cells.txt"

# cell -> (shared-path parent config, n_gen)
CELLS = {
    "ftc": ("msr_aller_ftc_train", 2000),
    "energy_controller": ("msr_aller_energy_controller_train", 2000),
    "pred_guid": ("msr_aller_pred_guid_train", 2000),
    "ftc_joint": ("msr_aller_ftc_joint_ref_train", 2000),
    "energy_controller_joint": ("msr_aller_energy_controller_joint_ref_train", 2000),
    "pred_guid_joint": ("msr_aller_pred_guid_joint_ref_train", 2000),
    "equilibrium_glide": ("msr_aller_eqglide_train", 2000),
    "piecewise_constant": ("msr_aller_piecewise_constant_train", 2000),
    "fnpag": ("msr_aller_fnpag_train", 300),
}
ALLOCATION_KEYS = ("algorithm", "n_gen", "n_pop", "training_n_sims")
UNGATED = ("piecewise_constant", "equilibrium_glide")
CAMPAIGNS = ("classical", "classical_ungated")


def test_cell_dir_holds_exactly_the_cells() -> None:
    assert sorted(p.stem for p in CELL_DIR.glob("*.toml")) == sorted(CELLS)


@pytest.mark.parametrize("cell", CELLS)
def test_leaf_states_regime_and_allocation(cell: str) -> None:
    leaf = tomllib.loads((CELL_DIR / f"{cell}.toml").read_text())
    assert leaf["monte_carlo"] == {"noise_seeding": "per_draw"}
    assert leaf["optimizer"] == {"algorithm": "ga", "n_gen": CELLS[cell][1], "n_pop": 300, "training_n_sims": 10}


@pytest.mark.parametrize("cell", CELLS)
def test_cell_resolves_to_its_parent_otherwise(cell: str) -> None:
    resolved = load_toml_with_bases(CELL_DIR / f"{cell}.toml")
    parent = load_toml_with_bases(REPO / "configs/training" / f"{CELLS[cell][0]}.toml")
    assert resolved["optimizer"]["seed_strategy"] == "adaptive"
    for cfg in (resolved, parent):
        cfg["monte_carlo"].pop("noise_seeding", None)
        for key in ALLOCATION_KEYS:
            cfg["optimizer"].pop(key, None)
    assert resolved == parent


def test_ungated_dir_holds_exactly_the_reruns() -> None:
    assert sorted(p.stem for p in UNGATED_DIR.glob("*.toml")) == sorted(UNGATED)


@pytest.mark.parametrize("cell", UNGATED)
def test_ungated_leaf_opens_only_the_ceiling(cell: str) -> None:
    leaf = tomllib.loads((UNGATED_DIR / f"{cell}.toml").read_text())
    assert leaf == {"base": [f"../classical/{cell}.toml"], "optimizer": {"max_violation_rate": 1.0}}


@pytest.mark.parametrize("cell", UNGATED)
def test_ungated_resolves_to_the_gated_cell_otherwise(cell: str) -> None:
    ungated = load_toml_with_bases(UNGATED_DIR / f"{cell}.toml")
    gated = load_toml_with_bases(CELL_DIR / f"{cell}.toml")
    assert (ungated["optimizer"].pop("max_violation_rate"), gated["optimizer"].pop("max_violation_rate")) == (1.0, 0.0)
    assert ungated == gated


def test_runner_trains_every_campaign_cell() -> None:
    campaigns = {name: cells.split() for name, cells in re.findall(r'^\s*(\w+)\) CELLS="([^"]*)"', RUNNER.read_text(), re.MULTILINE)}
    assert sorted(campaigns) == sorted(CAMPAIGNS)
    for name, cells in campaigns.items():
        assert cells[0] == "piecewise_constant"
        assert sorted(cells) == sorted(p.stem for p in (CAMPAIGN_ROOT / name).glob("*.toml"))


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_manifest_scores_every_campaign_cell(campaign: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO / "experiments/ou_marginal"))
    from confirmatory_marginal import read_manifest  # type: ignore[import-not-found]

    prefix = f"ou_marginal/{campaign}/"
    rows = {label.removeprefix(prefix): toml for label, toml, _ in read_manifest(MANIFEST) if label.startswith(prefix)}
    assert sorted(rows) == sorted(p.stem for p in (CAMPAIGN_ROOT / campaign).glob("*.toml"))
    for cell, toml in rows.items():
        scheme = load_toml_with_bases(CAMPAIGN_ROOT / campaign / f"{cell}.toml")["guidance"]["type"]
        assert toml == f"training_output/{prefix}{cell}/optimized_{scheme}.toml"
