"""RL terminal cost must honor TOML cost_settings (D4)."""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.cost import CostSettings

pytest.importorskip("aerocapture_rs")

from aerocapture.training.rl.rewards import compute_terminal_cost  # noqa: E402


def test_terminal_cost_respects_cost_transform() -> None:
    fr = np.zeros(52, dtype=np.float64)
    fr[41] = 1500.0  # dv_total_m_s, above the 1000 dv_threshold knee
    linear = compute_terminal_cost(fr, cost_settings=CostSettings(cost_transform="linear", dv_threshold=1000.0))
    log = compute_terminal_cost(fr, cost_settings=CostSettings(cost_transform="log", dv_threshold=1000.0))
    assert linear != log  # the transform must actually be applied


def test_terminal_cost_is_the_one_record_rms() -> None:
    """One record's terminal cost is `compute_cost` of that record alone."""
    from aerocapture.training.cost import compute_cost

    fr = np.zeros(52, dtype=np.float64)
    fr[41] = 1500.0
    assert compute_terminal_cost(fr, cost_settings=CostSettings()) == float(compute_cost(fr.reshape(1, -1), CostSettings()))
