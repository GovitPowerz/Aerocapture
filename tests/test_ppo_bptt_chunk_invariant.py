"""Chunk-size equivalence: one-chunk vs multi-chunk BPTT produce identical forward outputs."""

from __future__ import annotations

import torch
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec, GruSpec


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
