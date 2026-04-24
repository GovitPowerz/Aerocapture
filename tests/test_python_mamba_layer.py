"""Unit test for the Python MambaLayer torch mirror.

Validates the forward contract in isolation (Python-side only). The full
cross-language equivalence test vs Rust runtime lives in
test_rust_python_mamba_equivalence.py (Task 14).
"""

from __future__ import annotations

import math

import pytest
import torch
from aerocapture.training.rl.layers.mamba import MambaLayer


@pytest.fixture
def tiny_layer():
    layer = MambaLayer(input_size=2, d_state=2, dt_rank=1)
    layer.double()
    # Match the Rust hand-verified test fixture exactly.
    with torch.no_grad():
        layer.x_proj_w.copy_(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float64,
            )
        )
        layer.dt_proj_w.copy_(torch.zeros(2, 1, dtype=torch.float64))
        # inv_softplus(0.5) = log(e^0.5 - 1)
        b_val = math.log(math.exp(0.5) - 1.0)
        layer.dt_proj_b.copy_(torch.full((2,), b_val, dtype=torch.float64))
        layer.a_log.zero_()  # A = -exp(0) = -1
        layer.d_skip.zero_()  # no skip
    return layer


def test_mamba_forward_step_zero_state(tiny_layer):
    x = torch.tensor([1.0, 0.0], dtype=torch.float64)
    h = torch.zeros(2, 2, dtype=torch.float64)
    y, h_new = tiny_layer(x, h)

    # Matches Rust hand-verified expectations.
    assert abs(y[0].item() - 0.3934693402873666) < 1e-12
    assert abs(y[1].item() - 0.0) < 1e-15


def test_mamba_forward_two_step_state_evolution(tiny_layer):
    x1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    h0 = torch.zeros(2, 2, dtype=torch.float64)
    y1, h1 = tiny_layer(x1, h0)

    x2 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    y2, h2 = tiny_layer(x2, h1)

    assert abs(y2[0].item() - 0.0) < 1e-15
    assert abs(y2[1].item() - 0.3934693402873666) < 1e-12
    # State h[0, 0] decays by exp(-0.5) from step 1's ~0.39347
    # Expected mathematically: exp(-0.5) * 0.3934693402873666 = 0.2386512185411911
    assert abs(h2[0, 0].item() - 0.2386512185411911) < 1e-12


def test_mamba_new_state_dtype_matches_parameters():
    layer = MambaLayer(input_size=4, d_state=3, dt_rank=1)
    layer.double()
    state = layer.new_state()
    assert state.dtype == torch.float64
    assert state.shape == (4, 3)
    assert bool(torch.all(state == 0.0))


def test_mamba_deterministic_under_repeated_input():
    torch.manual_seed(0)
    layer = MambaLayer(input_size=4, d_state=3, dt_rank=1)
    layer.double()
    x = torch.randn(4, dtype=torch.float64)
    h_a = layer.new_state()
    h_b = layer.new_state()
    for _ in range(5):
        y_a, h_a = layer(x, h_a)
        y_b, h_b = layer(x, h_b)
        assert torch.allclose(y_a, y_b, atol=0.0)
