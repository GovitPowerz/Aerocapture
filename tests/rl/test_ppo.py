"""Unit tests for PPO update internals."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aerocapture.training.rl.policy import ValueNetwork  # noqa: E402
from aerocapture.training.rl.ppo import RolloutBuffer, compute_gae, critic_warmup_update, ppo_update_bptt  # noqa: E402
from aerocapture.training.torch_mirror.policy import V2Policy  # noqa: E402
from aerocapture.training.torch_mirror.schemas import Activation, DenseSpec  # noqa: E402


def _make_v2_policy(input_dim: int, layer_sizes: list[int], activations: list[Activation]) -> V2Policy:
    """Build a dense-only V2Policy from (input_dim, layer_sizes, activations) like GaussianPolicy."""
    specs: list[DenseSpec] = []
    prev = input_dim
    for out_dim, act in zip(layer_sizes, activations, strict=True):
        specs.append(DenseSpec(type="dense", input_size=prev, output_size=out_dim, activation=act))
        prev = out_dim
    return V2Policy(
        architecture=specs,
        input_mask=list(range(input_dim)),
    )


def test_gae_known_values() -> None:
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    next_values = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dones = np.array([False, False, True], dtype=np.bool_)
    adv, ret = compute_gae(rewards, values, next_values, terminated=dones, dones=dones, gamma=0.99, lam=0.95)
    assert adv.shape == (3,)
    assert np.isfinite(adv).all()
    assert np.isfinite(ret).all()


def test_gae_truncation_keeps_bootstrap() -> None:
    """Truncation is done but not terminated + provides V(terminal_obs) in next_values
    so the advantage uses r + gamma*V(term) - V(s) instead of masking to 0."""
    rewards = np.array([0.0], dtype=np.float32)
    values = np.array([5.0], dtype=np.float32)
    next_values = np.array([10.0], dtype=np.float32)  # V(terminal_obs)
    adv, _ = compute_gae(rewards, values, next_values, terminated=np.array([False]), dones=np.array([True]), gamma=1.0, lam=1.0)
    # delta = 0 + 1.0 * 10.0 - 5.0 = 5.0
    assert adv[0] == 5.0


def test_gae_true_termination_zeros_bootstrap() -> None:
    """Termination sets done=True so the bootstrap is masked, no matter what
    V(next) is; advantage = r - V(s)."""
    rewards = np.array([0.0], dtype=np.float32)
    values = np.array([5.0], dtype=np.float32)
    next_values = np.array([10.0], dtype=np.float32)
    dones = np.array([True], dtype=np.bool_)
    adv, _ = compute_gae(rewards, values, next_values, terminated=dones, dones=dones, gamma=1.0, lam=1.0)
    assert adv[0] == -5.0


@pytest.mark.parametrize("terminated_at_1", [False, True])
def test_gae_episode_end_cuts_the_trace(terminated_at_1: bool) -> None:
    """One env column: episode A ends at t=1 (truncated or terminated), episode B
    starts at t=2. Step 1's advantage must not carry B's advantages, and A's
    advantages must equal those of A's steps alone."""
    rewards = np.array([1.0, 2.0, 100.0, 50.0], dtype=np.float32)
    values = np.array([0.5, 1.0, -3.0, 4.0], dtype=np.float32)
    next_values = np.array([1.0, 7.0, 4.0, 9.0], dtype=np.float32)  # t=1: V(terminal_obs of A)
    terminated = np.array([False, terminated_at_1, False, False])
    dones = np.array([False, True, False, False])
    gamma, lam = 0.9, 0.8

    adv, ret = compute_gae(rewards, values, next_values, terminated=terminated, dones=dones, gamma=gamma, lam=lam)

    adv_a, _ = compute_gae(rewards[:2], values[:2], next_values[:2], terminated=terminated[:2], dones=dones[:2], gamma=gamma, lam=lam)
    adv_b, _ = compute_gae(rewards[2:], values[2:], next_values[2:], terminated=terminated[2:], dones=dones[2:], gamma=gamma, lam=lam)
    np.testing.assert_array_equal(adv, np.concatenate([adv_a, adv_b]))
    bootstrap = 0.0 if terminated_at_1 else gamma * 7.0
    assert adv[1] == pytest.approx(2.0 + bootstrap - 1.0)
    np.testing.assert_allclose(ret, adv + values)


def test_gae_over_env_columns_matches_per_column() -> None:
    """A (T, N) call is exactly the N per-column calls (bit-identical, float32 chain kept)."""
    rng = np.random.default_rng(3)
    T, N = 12, 5
    rewards = rng.standard_normal((T, N)).astype(np.float32)
    values = rng.standard_normal((T, N)).astype(np.float32)
    next_values = rng.standard_normal((T, N)).astype(np.float32)
    dones = rng.random((T, N)) < 0.3
    terminated = dones & (rng.random((T, N)) < 0.5)
    adv, ret = compute_gae(rewards, values, next_values, terminated=terminated, dones=dones, gamma=0.99, lam=0.95)
    assert adv.shape == ret.shape == (T, N)
    for e in range(N):
        adv_e, ret_e = compute_gae(rewards[:, e], values[:, e], next_values[:, e], terminated=terminated[:, e], dones=dones[:, e], gamma=0.99, lam=0.95)
        np.testing.assert_array_equal(adv[:, e], adv_e)
        np.testing.assert_array_equal(ret[:, e], ret_e)


def test_rollout_buffer_create() -> None:
    buf = RolloutBuffer.create(n_steps=8, n_envs=4, obs_dim=16)
    assert buf.obs.shape == (8, 4, 16)
    assert buf.raw_actions.shape == (8, 4, 2)
    assert buf.log_probs.shape == (8, 4)
    assert buf.rewards.shape == (8, 4)
    assert buf.values.shape == (8, 4)
    assert buf.dones.shape == (8, 4)
    assert buf.terminated.shape == (8, 4)


def _fill_buffer_random(buf: RolloutBuffer, rng: np.random.Generator) -> None:
    """Fill a RolloutBuffer with random data (feedforward-path smoke input)."""
    buf.obs[:] = rng.standard_normal(buf.obs.shape).astype(np.float32)
    buf.raw_actions[:] = rng.standard_normal(buf.raw_actions.shape).astype(np.float32)
    buf.log_probs[:] = (rng.standard_normal(buf.log_probs.shape) * 0.1).astype(np.float32)


def test_ppo_update_bptt_runs_without_crashing() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    policy = _make_v2_policy(16, [32, 32, 2], ["tanh", "tanh", "linear"])
    value = ValueNetwork(16, [32, 32], ["tanh", "tanh", "linear"])
    optim = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-4)

    T, N = 32, 8  # T * N = 256 samples total, matching the pre-BPTT test scale.
    # Dense-only policy -> per-layer hidden_shapes = [None, None, None].
    buf = RolloutBuffer.create(n_steps=T, n_envs=N, obs_dim=16, hidden_shapes=[None] * len(policy.layers))
    _fill_buffer_random(buf, rng)
    advantages = rng.standard_normal((T, N)).astype(np.float32)
    returns = rng.standard_normal((T, N)).astype(np.float32)

    metrics = ppo_update_bptt(
        policy,
        value,
        optim,
        buf,
        advantages,
        returns,
        bptt_length=T,  # single chunk = feedforward equivalent
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
    Forcing a large KL per update by combining a huge learning rate with a tiny
    target_kl reliably trips the early-stop."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    policy = _make_v2_policy(8, [16, 2], ["tanh", "linear"])
    value = ValueNetwork(8, [16], ["tanh", "linear"])
    optim = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=1.0)

    T, N = 16, 4  # T * N = 64 samples, matching the pre-BPTT test scale.
    buf = RolloutBuffer.create(n_steps=T, n_envs=N, obs_dim=8, hidden_shapes=[None] * len(policy.layers))
    _fill_buffer_random(buf, rng)
    advantages = rng.standard_normal((T, N)).astype(np.float32)
    returns = rng.standard_normal((T, N)).astype(np.float32)

    metrics = ppo_update_bptt(
        policy,
        value,
        optim,
        buf,
        advantages,
        returns,
        bptt_length=T,
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


def _run_seeded_update(seed: int) -> list[np.ndarray]:
    """Run one fixed ppo_update_bptt with a separately-seeded rng; return updated
    policy params. Identical torch seed + identical buffer => only the rng-driven
    minibatch shuffle order varies between calls."""
    torch.manual_seed(0)
    policy = _make_v2_policy(8, [16, 2], ["tanh", "linear"])
    value = ValueNetwork(8, [16], ["tanh", "linear"])
    optim = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=1e-2)

    T, N = 16, 4
    buf = RolloutBuffer.create(n_steps=T, n_envs=N, obs_dim=8, hidden_shapes=[None] * len(policy.layers))
    _fill_buffer_random(buf, np.random.default_rng(0))
    advantages = np.random.default_rng(1).standard_normal((T, N)).astype(np.float32)
    returns = np.random.default_rng(2).standard_normal((T, N)).astype(np.float32)

    ppo_update_bptt(
        policy,
        value,
        optim,
        buf,
        advantages,
        returns,
        bptt_length=8,  # 2 chunks -> chunk shuffle is exercised
        clip_range=0.2,
        update_epochs=3,
        minibatches=4,  # envs_per_minibatch=1 -> env shuffle is exercised
        entropy_coef=0.0,
        value_coef=0.5,
        max_grad_norm=0.5,
        rng=np.random.default_rng(seed),
    )
    return [p.detach().clone().numpy() for p in policy.parameters()]


def test_ppo_update_bptt_rng_reproducible() -> None:
    """Two updates seeded with the same rng produce bit-identical policy params;
    a different seed (which permutes minibatch order) produces different params."""
    params_a = _run_seeded_update(123)
    params_b = _run_seeded_update(123)
    for a, b in zip(params_a, params_b, strict=True):
        assert np.array_equal(a, b)

    params_c = _run_seeded_update(999)
    assert any(not np.array_equal(a, c) for a, c in zip(params_a, params_c, strict=True))


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


def test_critic_warmup_freezes_policy_and_fits_value() -> None:
    """critic_warmup_update fits the value net to returns while leaving the policy frozen
    (the warm-start invariant: the cold critic must not move the warm-started policy)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    policy = _make_v2_policy(16, [32, 32, 2], ["tanh", "tanh", "linear"])
    value = ValueNetwork(16, [32, 32], ["tanh", "tanh", "linear"])
    optim = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=1e-3)

    T, N = 32, 8
    buf = RolloutBuffer.create(n_steps=T, n_envs=N, obs_dim=16, hidden_shapes=[None] * len(policy.layers))
    _fill_buffer_random(buf, rng)
    returns = np.full((T, N), 5.0, dtype=np.float32)  # constant target the critic should fit

    policy_before = [p.detach().clone() for p in policy.parameters()]
    value_before = [p.detach().clone() for p in value.parameters()]
    obs_t = torch.from_numpy(buf.obs.reshape(-1, buf.obs_dim)).float()
    with torch.no_grad():
        init_loss = float(0.5 * ((value(obs_t).reshape(-1) - 5.0) ** 2).mean())

    final_loss = critic_warmup_update(value, optim, buf, returns, update_epochs=5, minibatches=4, obs_norm=None, rng=rng)

    # Key invariant: value-only loss must not touch the (warm-started) policy.
    for before, p in zip(policy_before, policy.parameters(), strict=True):
        assert torch.equal(before, p), "critic warmup must not change policy params"
    # Critic moved toward the constant target.
    assert any(not torch.equal(b, p) for b, p in zip(value_before, value.parameters(), strict=True))
    assert np.isfinite(final_loss)
    assert final_loss < init_loss
