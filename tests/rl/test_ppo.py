"""Unit tests for PPO update internals."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aerocapture.training.rl.policy import V2Policy, ValueNetwork  # noqa: E402
from aerocapture.training.rl.ppo import RolloutBuffer, compute_gae, ppo_update  # noqa: E402
from aerocapture.training.rl.schemas import Activation, DenseSpec  # noqa: E402


def _make_v2_policy(input_dim: int, layer_sizes: list[int], activations: list[Activation]) -> V2Policy:
    """Build a dense-only V2Policy from (input_dim, layer_sizes, activations) like GaussianPolicy."""
    specs: list[DenseSpec] = []
    prev = input_dim
    for out_dim, act in zip(layer_sizes, activations, strict=True):
        specs.append(DenseSpec(type="dense", input_size=prev, output_size=out_dim, activation=act))
        prev = out_dim
    return V2Policy(
        architecture=specs,
        output_interpretation="atan2",
        input_mask=list(range(input_dim)),
    )


def test_gae_known_values() -> None:
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    next_values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dones = np.array([False, False, True], dtype=np.bool_)
    adv, ret = compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95)
    assert adv.shape == (3,)
    assert np.isfinite(adv).all()
    assert np.isfinite(ret).all()


def test_gae_truncation_keeps_bootstrap() -> None:
    """Truncation sets done=False + provides V(terminal_obs) in next_values so
    the advantage uses r + gamma*V(term) - V(s) instead of masking to 0."""
    rewards = np.array([0.0], dtype=np.float32)
    values = np.array([5.0], dtype=np.float32)
    next_values = np.array([10.0], dtype=np.float32)  # V(terminal_obs)
    dones = np.array([False], dtype=np.bool_)
    adv, _ = compute_gae(rewards, values, next_values, dones, gamma=1.0, lam=1.0)
    # delta = 0 + 1.0 * 10.0 - 5.0 = 5.0
    assert adv[0] == 5.0


def test_gae_true_termination_zeros_bootstrap() -> None:
    """Termination sets done=True so the bootstrap is masked, no matter what
    V(next) is; advantage = r - V(s)."""
    rewards = np.array([0.0], dtype=np.float32)
    values = np.array([5.0], dtype=np.float32)
    next_values = np.array([10.0], dtype=np.float32)
    dones = np.array([True], dtype=np.bool_)
    adv, _ = compute_gae(rewards, values, next_values, dones, gamma=1.0, lam=1.0)
    assert adv[0] == -5.0


def test_rollout_buffer_create() -> None:
    buf = RolloutBuffer.create(n_steps=8, n_envs=4, obs_dim=16)
    assert buf.obs.shape == (8, 4, 16)
    assert buf.raw_actions.shape == (8, 4, 2)
    assert buf.log_probs.shape == (8, 4)
    assert buf.rewards.shape == (8, 4)
    assert buf.values.shape == (8, 4)
    assert buf.dones.shape == (8, 4)


def test_ppo_update_runs_without_crashing() -> None:
    torch.manual_seed(0)
    policy = _make_v2_policy(16, [32, 32, 2], ["tanh", "tanh", "linear"])
    value = ValueNetwork(16, [32, 32], ["tanh", "tanh", "linear"])
    optim = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-4)

    n = 256
    obs = torch.randn(n, 16)
    raw_actions = torch.randn(n, 2)
    old_log_probs = torch.randn(n) * 0.1
    advantages = torch.randn(n)
    returns = torch.randn(n)

    metrics = ppo_update(
        policy,
        value,
        optim,
        obs,
        raw_actions,
        old_log_probs,
        advantages,
        returns,
        clip_range=0.2,
        update_epochs=2,
        minibatches=4,
        entropy_coef=0.0,
        value_coef=0.5,
        max_grad_norm=0.5,
    )
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "approx_kl" in metrics
    assert "clip_frac" in metrics


def test_target_kl_early_stops_epochs() -> None:
    """When policy updates cause mean approx_kl to exceed target_kl, the outer
    epoch loop breaks early and `epochs_run` is less than the configured budget.
    Forcing a large KL per update by combining zero clip_range with a huge
    learning rate reliably trips the early-stop."""
    torch.manual_seed(0)
    policy = _make_v2_policy(8, [16, 2], ["tanh", "linear"])
    value = ValueNetwork(8, [16], ["tanh", "linear"])
    optim = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=1.0)
    n = 64
    obs = torch.randn(n, 8)
    raw = torch.randn(n, 2)
    old_lp = torch.randn(n) * 0.1
    adv = torch.randn(n)
    ret = torch.randn(n)
    metrics = ppo_update(
        policy,
        value,
        optim,
        obs,
        raw,
        old_lp,
        adv,
        ret,
        clip_range=0.2,
        update_epochs=10,
        minibatches=4,
        entropy_coef=0.0,
        value_coef=0.5,
        max_grad_norm=1.0,
        target_kl=0.001,  # trivially small threshold
    )
    assert "epochs_run" in metrics
    assert metrics["epochs_run"] < 10, "target_kl should have triggered early stop"


def test_value_network_gradient_flows() -> None:
    """Verify ValueNetwork.forward() preserves autograd graph (not detached)."""
    value = ValueNetwork(4, [8], ["tanh", "linear"])
    obs = torch.randn(2, 4, requires_grad=False)
    out = value(obs)
    loss = out.sum()
    loss.backward()
    for p in value.parameters():
        assert p.grad is not None, "ValueNetwork parameter has no gradient"
        assert p.grad.abs().sum() > 0, "ValueNetwork gradient is all zeros"
