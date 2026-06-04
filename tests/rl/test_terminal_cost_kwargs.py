"""RL terminal cost must honor TOML cost_kwargs (D4)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")

from aerocapture.training.rl.rewards import compute_terminal_cost  # noqa: E402


def test_terminal_cost_respects_cost_transform() -> None:
    fr = np.zeros(52, dtype=np.float64)
    fr[41] = 1500.0  # dv_total_m_s, above the 1000 dv_threshold knee
    linear = compute_terminal_cost(fr, cost_kwargs={"cost_transform": "linear", "dv_threshold": 1000.0})
    log = compute_terminal_cost(fr, cost_kwargs={"cost_transform": "log", "dv_threshold": 1000.0})
    assert linear != log  # the transform must actually be applied


def test_terminal_cost_defaults_unchanged() -> None:
    """No cost_kwargs -> identical to the legacy defaults call (backward compatible)."""
    from aerocapture.training.evaluate import compute_cost

    fr = np.zeros(52, dtype=np.float64)
    fr[41] = 1500.0
    assert compute_terminal_cost(fr) == float(compute_cost(fr.reshape(1, -1)))
