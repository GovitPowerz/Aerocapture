"""MambaLayer.forward accepts batched (B, input_size) and matches unbatched."""

import pytest
import torch
from aerocapture.training.rl.layers import MambaLayer


@pytest.fixture
def mamba() -> MambaLayer:
    torch.manual_seed(0)
    layer = MambaLayer(input_size=8, d_state=4, dt_rank=2).double()
    # Randomize params so the test exercises non-trivial weights
    with torch.no_grad():
        for p in layer.parameters():
            p.uniform_(-0.1, 0.1)
    return layer


def test_batched_forward_matches_unbatched(mamba: MambaLayer) -> None:
    B = 3
    xs = torch.randn(B, 8, dtype=torch.float64)
    h_batched = mamba.new_state(batch_size=B)
    assert h_batched.shape == (B, 8, 4)

    # Unbatched: loop one at a time
    expected_y = []
    expected_h_new = []
    for b in range(B):
        # Use a fresh unbatched state matching the batched zero-init
        y_b, h_b_new = mamba.forward_unbatched(xs[b], h_batched[b])
        expected_y.append(y_b)
        expected_h_new.append(h_b_new)
    expected_y_stack = torch.stack(expected_y, dim=0)
    expected_h_stack = torch.stack(expected_h_new, dim=0)

    # Batched call
    y_batched, h_new_batched = mamba.forward(xs, h_batched)
    assert y_batched.shape == (B, 8)
    assert h_new_batched.shape == (B, 8, 4)
    assert torch.allclose(y_batched, expected_y_stack, atol=1e-14)
    assert torch.allclose(h_new_batched, expected_h_stack, atol=1e-14)
