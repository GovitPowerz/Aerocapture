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
    assert totals["lstm"] == 1082  # 180 + 880 + 22 == the sweep cell lstm_p1082, verbatim
    assert totals["slstm"] == 1042  # same H=10; -40 = the single-bias delta inherent to the cell
    assert totals["mlstm"] == 1078  # 180 + 858 + 40 (H=19, no recurrent matrices)
    assert abs(totals["mlstm"] - totals["lstm"]) / totals["lstm"] < 0.02
    # slstm is exempt from the 2% gate: at matched H=10 its 4H fewer bias params
    # (single bias vs LSTM's double) are an axis cost, not a sizing miss.
    assert abs(totals["slstm"] - totals["lstm"]) / totals["lstm"] < 0.05


def test_mlstm_head_reads_19_wide() -> None:
    head = ARMS["mlstm"][-1]
    assert head["input_size"] == 19  # mlstm H=19 for budget parity at the 1082 anchor


def test_xlstm_leaf_toml_carries_layer_and_seed() -> None:
    toml = leaf_toml("xlstm_probe", "slstm", ARMS["slstm"], BASE_SEED, BASE_SEED, Path("training_output/xlstm_probe/slstm_s0"), 500, 10)
    assert 'base = ["../msr_aller_nn_atan2_train.toml"]' in toml
    assert "n_pop = 300" in toml
    assert "algorithm = " not in toml
    assert "seed_strategy = " not in toml
    assert 'type = "slstm"' in toml
    assert f"seed = {BASE_SEED}" in toml
    assert ".xlstm_probe_slstm_s0" in toml
