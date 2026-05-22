"""V2Policy.forward_seq_means computes (T, B, out_dim) mean predictions
for supervised warm-start, with done-mask state zeroing matching evaluate."""

import torch
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import LayerSpec
from pydantic import TypeAdapter


def _build_policy(arch):
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    return V2Policy(architecture=validated, input_mask=None).double()


def test_forward_seq_means_dense_shape_and_finite():
    arch = [{"type": "dense", "input_size": 4, "output_size": 2, "activation": "tanh"}]
    policy = _build_policy(arch)
    T, B = 5, 3
    obs = torch.randn(T, B, 4, dtype=torch.float64)
    state_0 = policy.new_state(batch_size=B, device=None)
    dones = torch.zeros(T, B, dtype=torch.bool)
    means = policy.forward_seq_means(obs, state_0, dones)
    assert means.shape == (T, B, 2)
    assert torch.isfinite(means).all()


def test_forward_seq_means_gru_state_propagates():
    arch = [
        {"type": "gru", "input_size": 4, "hidden_size": 8},
        {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
    ]
    policy = _build_policy(arch)
    T, B = 4, 2
    obs = torch.randn(T, B, 4, dtype=torch.float64)
    state_0 = policy.new_state(batch_size=B, device=None)
    dones = torch.zeros(T, B, dtype=torch.bool)
    means = policy.forward_seq_means(obs, state_0, dones)
    assert means.shape == (T, B, 2)
    assert torch.isfinite(means).all()


def test_forward_seq_means_done_zeros_state():
    """When done[t]=True, the GRU hidden state at t+1 is zeroed."""
    arch = [
        {"type": "gru", "input_size": 2, "hidden_size": 4},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
    ]
    policy = _build_policy(arch)
    T, B = 3, 1
    obs = torch.ones(T, B, 2, dtype=torch.float64)

    # done at t=1: state at t=2 should be the same as starting fresh
    state_0 = policy.new_state(batch_size=B, device=None)
    dones_with = torch.tensor([[False], [True], [False]])
    means_with = policy.forward_seq_means(obs, state_0, dones_with)

    # If we replay only t=2 with fresh state, we should get the same output
    state_fresh = policy.new_state(batch_size=B, device=None)
    means_fresh = policy.forward_seq_means(obs[2:3], state_fresh, torch.zeros(1, B, dtype=torch.bool))
    assert torch.allclose(means_with[2], means_fresh[0], atol=1e-14)
