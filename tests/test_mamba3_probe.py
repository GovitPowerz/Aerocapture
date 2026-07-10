"""Unit tests for the Mamba-3 probe driver (2x2 at the 962-dim anchor)."""

from __future__ import annotations

from pathlib import Path

from aerocapture.training.config import _layer_n_params
from aerocapture.training.experiments.mamba3_probe import ARMS, BASE_SEED, BASELINE, TREATMENTS
from aerocapture.training.experiments.probe_common import leaf_toml


def test_arms_cover_the_2x2() -> None:
    combos = {(arch[1]["discretization"], arch[1]["state_mode"]) for arch in ARMS.values()}
    assert combos == {("euler", "real"), ("trapezoidal", "real"), ("euler", "complex"), ("trapezoidal", "complex")}
    assert BASELINE == "baseline"
    assert TREATMENTS == ["trapz", "complex", "both"]


def test_budgets_match_the_962_anchor() -> None:
    totals = {arm: sum(_layer_n_params(e) for e in arch) for arm, arch in ARMS.items()}
    # euler+real == the deployed Mamba_962 cell; axis costs +16 (trapz) / +192 (complex).
    assert totals == {"baseline": 962, "trapz": 978, "complex": 1154, "both": 1170}


def test_mamba3_leaf_toml_carries_flags_and_seed() -> None:
    toml = leaf_toml("mamba3_probe", "both", ARMS["both"], BASE_SEED + 2, BASE_SEED, Path("training_output/mamba3_probe/both_s2"), 5000, 2)
    assert 'base = ["../msr_aller_nn_atan2_train.toml"]' in toml
    assert "n_pop = 300" in toml
    # algorithm / seed_strategy / curation inherit the sweep's ga + adaptive + max.
    assert "algorithm = " not in toml
    assert "seed_strategy = " not in toml
    assert 'type = "mamba3"' in toml
    assert 'discretization = "trapezoidal"' in toml
    assert 'state_mode = "complex"' in toml
    assert f"seed = {BASE_SEED + 2}" in toml
    assert ".mamba3_probe_both_s2" in toml
