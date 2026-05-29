import math

import pytest
import torch

pytest.importorskip("aerocapture_rs")  # warm_start raises ImportError at module load without it

from aerocapture.training.warm_start import encode_supervised_target  # noqa: E402


def test_scaled_pi_target_is_y_over_n_pi_clamped() -> None:
    y = torch.tensor([0.0, math.pi / 2, math.pi])
    out = encode_supervised_target("scaled_pi", y, prev_realized=None, scaled_pi_n=2.0, delta_max=0.0)
    expected = torch.clamp(y / (2.0 * math.pi), -1.0, 1.0)
    assert torch.allclose(out, expected, atol=1e-12)


def test_delta_target_is_wrapped_diff_over_max_clamped() -> None:
    y = torch.tensor([1.0, 1.0])
    prev = torch.tensor([0.9, 1.5])
    out = encode_supervised_target("delta", y, prev_realized=prev, scaled_pi_n=0.0, delta_max=0.2)
    diff = torch.tensor([0.1, -0.5])  # wrap_to_pi(y - prev), both small so no wrap
    expected = torch.clamp(diff / 0.2, -1.0, 1.0)
    assert torch.allclose(out, expected, atol=1e-9)


def test_delta_target_wraps_across_pi() -> None:
    y = torch.tensor([-math.pi + 0.1])
    prev = torch.tensor([math.pi - 0.1])  # raw diff = -2π+0.2, wraps to +0.2
    out = encode_supervised_target("delta", y, prev_realized=prev, scaled_pi_n=0.0, delta_max=1.0)
    assert torch.allclose(out, torch.tensor([0.2]), atol=1e-6)
