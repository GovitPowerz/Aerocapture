"""Unit tests for the xLSTM probe driver."""

from __future__ import annotations

from pathlib import Path

from aerocapture.training.config import _layer_n_params
from aerocapture.training.experiments.probe_common import leaf_toml
from aerocapture.training.experiments.xlstm_probe import ARMS, BASE_SEED, BASELINE, TREATMENTS


def test_xlstm_arms_and_budgets_within_2pct() -> None:
    assert set(ARMS) == {"lstm", "slstm", "mlstm"}
    assert BASELINE == "lstm"
    assert TREATMENTS == ["slstm", "mlstm"]
    totals = {arm: sum(_layer_n_params(e) for e in arch) for arm, arch in ARMS.items()}
    assert totals["lstm"] == 9090  # 576 + 8448 + 66 (17-input atan2 mask inherited from the base)
    assert totals["slstm"] == 8962  # 576 + 8320 + 66
    assert totals["mlstm"] == 9220  # 576 + 8514 + 130
    for arm in TREATMENTS:
        assert abs(totals[arm] - totals["lstm"]) / totals["lstm"] < 0.02


def test_mlstm_head_reads_64_wide() -> None:
    head = ARMS["mlstm"][-1]
    assert head["input_size"] == 64  # mlstm H=64 for budget parity


def test_xlstm_leaf_toml_carries_layer_and_seed() -> None:
    toml = leaf_toml("xlstm_probe", "slstm", ARMS["slstm"], BASE_SEED, BASE_SEED, Path("training_output/xlstm_probe/slstm_s0"), 500, 10)
    assert 'base = ["../msr_aller_nn_atan2_train.toml"]' in toml
    assert "n_pop = 300" in toml
    assert "algorithm = " not in toml
    assert "seed_strategy = " not in toml
    assert 'type = "slstm"' in toml
    assert f"seed = {BASE_SEED}" in toml
    assert ".xlstm_probe_slstm_s0" in toml
