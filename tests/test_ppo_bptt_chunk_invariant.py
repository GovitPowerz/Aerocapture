"""Chunk-size equivalence: one-chunk vs multi-chunk BPTT produce identical forward outputs."""

from __future__ import annotations

from typing import Any

import torch
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec, GruSpec, LstmSpec


def test_bptt_chunk_size_invariant_forward_outputs() -> None:
    """For the same policy + same rollout, one-chunk BPTT (`bptt_length = T`) and
    multi-chunk BPTT (`bptt_length = T/k`) re-evaluate the sequence via
    V2Policy.evaluate. The chunk-boundary detach() does not change the forward
    values; only gradients differ.
    """
    torch.manual_seed(42)
    arch: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        GruSpec(type="gru", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=arch, output_interpretation="atan2", input_mask=None)
    T, B = 16, 2
    obs_seq = torch.randn(T, B, 3)
    raw_seq = torch.randn(T, B, 2)
    dones_seq = torch.zeros(T, B, dtype=torch.bool)
    state_0 = p.new_state(B, "cpu")

    # One-chunk: evaluate the entire T-step sequence in a single call.
    lp_one, ent_one = p.evaluate(obs_seq, state_0, dones_seq, raw_seq)

    # Multi-chunk: evaluate in chunks of length 4; detach state at boundaries.
    bptt = 4
    lp_multi = torch.zeros_like(lp_one)
    ent_multi = torch.zeros_like(ent_one)
    state_c = state_0
    for c in range(T // bptt):
        lo, hi = c * bptt, (c + 1) * bptt
        state_c_detached = [None if s is None else s.detach() for s in state_c]
        lp_c, ent_c = p.evaluate(obs_seq[lo:hi], state_c_detached, dones_seq[lo:hi], raw_seq[lo:hi])
        lp_multi[lo:hi] = lp_c.detach()
        ent_multi[lo:hi] = ent_c.detach()
        # Advance state: run the forward once more with no_grad to get the chunk-end state.
        with torch.no_grad():
            s = state_c_detached
            for t in range(bptt):
                _, s = p.forward(obs_seq[lo + t], s)
            state_c = s

    # Forward values must be identical between one-chunk and multi-chunk BPTT.
    torch.testing.assert_close(lp_one.detach(), lp_multi, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ent_one.detach(), ent_multi, rtol=1e-6, atol=1e-6)


def _detach_state_entry(s: Any) -> Any:
    """Recursive state detachment handling None, Tensor, and tuple-of-Tensors."""
    if s is None:
        return None
    if isinstance(s, torch.Tensor):
        return s.detach()
    if isinstance(s, tuple):
        return tuple(_detach_state_entry(sub) for sub in s)
    raise TypeError(f"Unsupported state entry: {type(s).__name__}")


def test_bptt_chunk_size_invariant_forward_outputs_lstm() -> None:
    """LSTM variant: chunk-boundary detach on a tuple (h, c) state must not
    change forward values. Exercises the Task 8 tuple dispatch end-to-end
    through V2Policy.evaluate.
    """
    torch.manual_seed(1337)
    arch: list[DenseSpec | LstmSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        LstmSpec(type="lstm", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=arch, output_interpretation="atan2", input_mask=None)
    T, B = 16, 2
    obs_seq = torch.randn(T, B, 3)
    raw_seq = torch.randn(T, B, 2)
    dones_seq = torch.zeros(T, B, dtype=torch.bool)
    state_0 = p.new_state(B, "cpu")

    # Verify that the LSTM layer's new_state returns a tuple (h, c).
    lstm_state = state_0[1]  # index 1 is the LstmLayer
    assert isinstance(lstm_state, tuple), f"expected tuple for LSTM state, got {type(lstm_state).__name__}"
    assert len(lstm_state) == 2, f"expected (h, c) tuple of length 2, got {len(lstm_state)}"

    # One-chunk
    lp_one, ent_one = p.evaluate(obs_seq, state_0, dones_seq, raw_seq)

    # Multi-chunk with tuple-aware detach
    bptt = 4
    lp_multi = torch.zeros_like(lp_one)
    ent_multi = torch.zeros_like(ent_one)
    state_c = state_0
    for c in range(T // bptt):
        lo, hi = c * bptt, (c + 1) * bptt
        state_c_detached = [_detach_state_entry(s) for s in state_c]
        lp_c, ent_c = p.evaluate(obs_seq[lo:hi], state_c_detached, dones_seq[lo:hi], raw_seq[lo:hi])
        lp_multi[lo:hi] = lp_c.detach()
        ent_multi[lo:hi] = ent_c.detach()
        with torch.no_grad():
            s = state_c_detached
            for t in range(bptt):
                _, s = p.forward(obs_seq[lo + t], s)
            state_c = s

    # Forward values must be identical one-chunk vs multi-chunk.
    torch.testing.assert_close(lp_one.detach(), lp_multi, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ent_one.detach(), ent_multi, rtol=1e-6, atol=1e-6)


def test_bptt_chunk_size_invariant_forward_outputs_lstm_with_dones() -> None:
    """LSTM chunk-invariance under mid-chunk done events: one-chunk BPTT and
    multi-chunk BPTT must still agree when `_zero_state_where_done` is hit
    inside `V2Policy.evaluate` for both tensor and tuple state entries.

    This covers the review gap where previous chunk-invariance tests used an
    all-False dones mask and never exercised the tuple-state zeroing path via
    V2Policy.evaluate.
    """
    torch.manual_seed(271828)
    arch: list[DenseSpec | LstmSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        LstmSpec(type="lstm", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=arch, output_interpretation="atan2", input_mask=None)
    T, B = 16, 3
    obs_seq = torch.randn(T, B, 3)
    raw_seq = torch.randn(T, B, 2)
    # Done pattern that straddles chunk boundaries when bptt=4:
    #   env 0: done at t=2 (chunk 0), t=10 (chunk 2)
    #   env 1: done at t=6 (chunk 1 boundary)
    #   env 2: never done
    dones_seq = torch.zeros(T, B, dtype=torch.bool)
    dones_seq[2, 0] = True
    dones_seq[10, 0] = True
    dones_seq[6, 1] = True
    state_0 = p.new_state(B, "cpu")

    # One-chunk (T=16 in a single evaluate call; dones applied inside).
    lp_one, ent_one = p.evaluate(obs_seq, state_0, dones_seq, raw_seq)

    # Multi-chunk (bptt=4, four chunks; dones applied inside each).
    bptt = 4
    lp_multi = torch.zeros_like(lp_one)
    ent_multi = torch.zeros_like(ent_one)
    state_c = state_0
    for c in range(T // bptt):
        lo, hi = c * bptt, (c + 1) * bptt
        state_c_detached = [_detach_state_entry(s) for s in state_c]
        lp_c, ent_c = p.evaluate(obs_seq[lo:hi], state_c_detached, dones_seq[lo:hi], raw_seq[lo:hi])
        lp_multi[lo:hi] = lp_c.detach()
        ent_multi[lo:hi] = ent_c.detach()
        # Advance state with the same dones-aware zeroing V2Policy.evaluate uses,
        # so the next chunk's state_0 matches the one-chunk pass.
        from aerocapture.training.rl.policy import _zero_state_where_done

        with torch.no_grad():
            s = state_c_detached
            for t in range(bptt):
                t_abs = lo + t
                _, s = p.forward(obs_seq[t_abs], s)
                if bool(dones_seq[t_abs].any()):
                    s = _zero_state_where_done(s, dones_seq[t_abs])
            state_c = s

    torch.testing.assert_close(lp_one.detach(), lp_multi, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ent_one.detach(), ent_multi, rtol=1e-6, atol=1e-6)
