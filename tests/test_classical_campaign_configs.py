"""The #172 classical retune cells differ from their shared-path parents only in regime and allocation.

Each `configs/training/ou_marginal/classical/<cell>.toml` states `per_draw` and the
full GA allocation in its own file (a config that base-inherits it does not inherit
CLI flags), resolves to its parent everywhere else, and is driven by
`experiments/ou_marginal/classical_campaign.sh` and scored through
`experiments/ou_marginal/classical_cells.txt`. Pure Python (no bindings).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from aerocapture.training.toml_utils import load_toml_with_bases

REPO = Path(__file__).resolve().parents[1]
CELL_DIR = REPO / "configs/training/ou_marginal/classical"
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


def test_runner_trains_every_cell() -> None:
    match = re.search(r'^CELLS="([^"]*)"', RUNNER.read_text(), re.MULTILINE)
    assert match is not None
    assert sorted(match.group(1).split()) == sorted(CELLS)


def test_manifest_scores_every_retuned_cell() -> None:
    rows = [line.split("|") for line in MANIFEST.read_text().splitlines() if line.strip() and not line.startswith("#")]
    retuned = {label.removeprefix("ou_marginal/classical/"): toml for label, toml in rows if label.startswith("ou_marginal/classical/")}
    assert sorted(retuned) == sorted(CELLS)
    for cell, toml in retuned.items():
        scheme = load_toml_with_bases(CELL_DIR / f"{cell}.toml")["guidance"]["type"]
        assert toml == f"training_output/ou_marginal/classical/{cell}/optimized_{scheme}.toml"
