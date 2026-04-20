"""_zero_state_where_done handles LSTM tuple state."""

from __future__ import annotations

import pytest
import torch
from aerocapture.training.rl.policy import _zero_state_where_done


def test_zero_state_where_done_tuple_zeros_both_tensors_on_done_rows() -> None:
    h = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    c = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
    done_mask = torch.tensor([False, True, False, True])

    state = [(h, c)]  # list of per-layer states; single LSTM layer
    new_state = _zero_state_where_done(state, done_mask)
    assert len(new_state) == 1
    new_h, new_c = new_state[0]

    # Non-done rows (indices 0, 2) unchanged
    assert torch.equal(new_h[0], h[0])
    assert torch.equal(new_c[0], c[0])
    assert torch.equal(new_h[2], h[2])
    assert torch.equal(new_c[2], c[2])

    # Done rows (indices 1, 3) zeroed in both tensors
    assert torch.all(new_h[1] == 0.0)
    assert torch.all(new_c[1] == 0.0)
    assert torch.all(new_h[3] == 0.0)
    assert torch.all(new_c[3] == 0.0)


def test_zero_state_where_done_raises_on_non_tensor_non_tuple_entry() -> None:
    done_mask = torch.tensor([False])
    with pytest.raises(TypeError, match="unsupported state entry type"):
        _zero_state_where_done([object()], done_mask)


def test_zero_state_where_done_passes_through_none_and_tensor_entries_alongside_tuple() -> None:
    tensor_state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    h = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    c = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    done_mask = torch.tensor([True, False])

    state = [None, tensor_state, (h, c)]
    new_state = _zero_state_where_done(state, done_mask)

    assert new_state[0] is None
    assert torch.all(new_state[1][0] == 0.0)  # done
    assert torch.equal(new_state[1][1], tensor_state[1])  # not done
    new_h, new_c = new_state[2]
    assert torch.all(new_h[0] == 0.0)
    assert torch.all(new_c[0] == 0.0)
    assert torch.equal(new_h[1], h[1])
    assert torch.equal(new_c[1], c[1])
