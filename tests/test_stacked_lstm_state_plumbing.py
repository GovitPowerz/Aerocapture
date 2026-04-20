"""Regression: stacked-LSTM architectures round-trip through the PPO state
packing helpers (`_np_state_to_torch` / `_torch_state_to_np`) without
cross-contamination between layer slots.

The review flagged that the `(B, 2, H)` LSTM convention is documented only
for single-LSTM architectures; this test locks in the contract for two
LSTMs separated by a dense, verifying each layer keeps its own slab and
the roundtrip is lossless.
"""

from __future__ import annotations

import numpy as np
import torch
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec, LstmSpec
from aerocapture.training.rl.train import _np_state_to_torch, _torch_state_to_np


def test_stacked_lstm_state_roundtrip_is_lossless() -> None:
    """Two LSTM layers separated by a dense: per-layer (h, c) survives numpy
    packing / unpacking roundtrip bit-for-bit (up to f32 precision).
    """
    torch.manual_seed(0)
    arch: list[DenseSpec | LstmSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        LstmSpec(type="lstm", input_size=4, hidden_size=6),
        DenseSpec(type="dense", input_size=6, output_size=5, activation="tanh"),
        LstmSpec(type="lstm", input_size=5, hidden_size=8),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=arch, output_interpretation="atan2", input_mask=None)
    B = 4

    # Seed each layer state with distinguishable values so cross-slot leakage
    # would be visible in the roundtrip.
    state = policy.new_state(B, "cpu")
    # state[0] None (Dense0), state[1] tuple for Lstm1, state[2] None, state[3] tuple for Lstm3, state[4] None
    assert state[0] is None and state[2] is None and state[4] is None
    assert isinstance(state[1], tuple) and len(state[1]) == 2
    assert isinstance(state[3], tuple) and len(state[3]) == 2

    # Fill with slot-tagged numbers: Lstm1 h/c get values starting at 1.0/2.0;
    # Lstm3 h/c start at 10.0/20.0.
    with torch.no_grad():
        h1, c1 = state[1]
        h3, c3 = state[3]
        h1.fill_(1.5)
        c1.fill_(2.5)
        h3.fill_(10.5)
        c3.fill_(20.5)

    # Round-trip: torch -> np -> torch.
    np_state = _torch_state_to_np(state)
    reconstructed = _np_state_to_torch(np_state)

    # Structural: list length + None slots preserved.
    assert len(reconstructed) == 5
    assert reconstructed[0] is None and reconstructed[2] is None and reconstructed[4] is None
    assert isinstance(reconstructed[1], tuple) and len(reconstructed[1]) == 2
    assert isinstance(reconstructed[3], tuple) and len(reconstructed[3]) == 2

    # Value: each layer's (h, c) survives, and no cross-contamination between
    # the two LSTM slots.
    h1_r, c1_r = reconstructed[1]
    h3_r, c3_r = reconstructed[3]
    assert h1_r.shape == (B, 6)
    assert c1_r.shape == (B, 6)
    assert h3_r.shape == (B, 8)
    assert c3_r.shape == (B, 8)
    assert torch.allclose(h1_r, torch.full_like(h1_r, 1.5))
    assert torch.allclose(c1_r, torch.full_like(c1_r, 2.5))
    assert torch.allclose(h3_r, torch.full_like(h3_r, 10.5))
    assert torch.allclose(c3_r, torch.full_like(c3_r, 20.5))


def test_stacked_lstm_done_mask_zeros_both_layers_independently() -> None:
    """Numpy boolean-row zeroing on a `(B, 2, H)` slab zeros both h and c for
    the masked rows. With two LSTM layers, each slot is zeroed independently.

    This mirrors the `h_next_np[li][done] = 0.0` idiom in the PPO rollout
    collect loop at train.py around line 682.
    """
    B, H1, H2 = 4, 6, 8
    # Construct a stacked state manually mirroring what _torch_state_to_np
    # would produce for a two-LSTM architecture.
    np_state = [
        None,
        np.full((B, 2, H1), 1.5, dtype=np.float32),  # Lstm1
        None,
        np.full((B, 2, H2), 10.5, dtype=np.float32),  # Lstm3
        None,
    ]

    # Env 0 and env 2 just terminated.
    done = np.array([True, False, True, False], dtype=bool)

    for _, slab in enumerate(np_state):
        if slab is not None:
            slab[done] = 0.0

    slab_l1 = np_state[1]
    slab_l3 = np_state[3]
    assert slab_l1 is not None and slab_l3 is not None
    # Env 0 and 2 zeroed in BOTH layers (h and c both fall under axis-0 slicing).
    assert np.all(slab_l1[0] == 0.0)
    assert np.all(slab_l1[2] == 0.0)
    assert np.all(slab_l3[0] == 0.0)
    assert np.all(slab_l3[2] == 0.0)

    # Env 1 and 3 untouched on both layers.
    assert np.all(slab_l1[1] == 1.5)
    assert np.all(slab_l1[3] == 1.5)
    assert np.all(slab_l3[1] == 10.5)
    assert np.all(slab_l3[3] == 10.5)


def test_stacked_lstm_forward_keeps_layers_separate() -> None:
    """End-to-end: a forward pass through two LSTM layers on random weights
    produces (h, c) state for each layer with independent magnitudes.
    Cross-contamination would show up as one layer's state copying into
    the other's.
    """
    torch.manual_seed(7)
    arch: list[DenseSpec | LstmSpec] = [
        LstmSpec(type="lstm", input_size=3, hidden_size=4),
        LstmSpec(type="lstm", input_size=4, hidden_size=6),
        DenseSpec(type="dense", input_size=6, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=arch, output_interpretation="atan2", input_mask=None)
    with torch.no_grad():
        # Scale LSTM1 weights up, LSTM2 weights down, so the two layers' state
        # magnitudes should end up very different after a few steps.
        for p in policy.layers[0].parameters():
            p.data = torch.randn_like(p.data) * 1.0
        for p in policy.layers[1].parameters():
            p.data = torch.randn_like(p.data) * 0.05

    state = policy.new_state(1, "cpu")
    for _ in range(10):
        x = torch.randn(1, 3)
        _, state = policy(x, state)

    lstm1_h, lstm1_c = state[0]
    lstm2_h, lstm2_c = state[1]
    # LSTM1 saw input magnitude 1.0 weights -> its h/c should have grown.
    # LSTM2 saw input from LSTM1 scaled by 0.05 -> much smaller magnitudes.
    assert float(lstm1_h.detach().abs().max()) > 0.1
    assert float(lstm1_c.detach().abs().max()) > 0.1
    assert float(lstm2_h.detach().abs().max()) < float(lstm1_h.detach().abs().max())
    assert float(lstm2_c.detach().abs().max()) < float(lstm1_c.detach().abs().max())
