"""PPO update rule and rollout buffer for aerocapture RL training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from aerocapture.training.rl.policy import GaussianPolicy, ValueNetwork


@dataclass
class RolloutBuffer:
    """Fixed-size per-env rollout buffer; (n_steps, n_envs, ...) tensors.

    ``raw_actions`` stores the 2D Gaussian sample (n_steps, n_envs, 2) so that
    ``ppo_update`` replays the *exact* point at which ``old_log_probs`` were
    evaluated. The scalar bank angle sent to the env is atan2(raw[0], raw[1]).
    """

    n_steps: int
    n_envs: int
    obs_dim: int
    obs: npt.NDArray[np.float32]
    raw_actions: npt.NDArray[np.float32]
    log_probs: npt.NDArray[np.float32]
    rewards: npt.NDArray[np.float32]
    values: npt.NDArray[np.float32]
    dones: npt.NDArray[np.bool_]

    @classmethod
    def create(cls, n_steps: int, n_envs: int, obs_dim: int) -> RolloutBuffer:
        return cls(
            n_steps=n_steps,
            n_envs=n_envs,
            obs_dim=obs_dim,
            obs=np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32),
            raw_actions=np.zeros((n_steps, n_envs, 2), dtype=np.float32),
            log_probs=np.zeros((n_steps, n_envs), dtype=np.float32),
            rewards=np.zeros((n_steps, n_envs), dtype=np.float32),
            values=np.zeros((n_steps, n_envs), dtype=np.float32),
            dones=np.zeros((n_steps, n_envs), dtype=np.bool_),
        )


def compute_gae(
    rewards: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    dones: npt.NDArray[np.bool_],
    gamma: float,
    lam: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Standard GAE-lambda. `values` has length n_steps+1 (trailing bootstrap)."""
    n = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        not_done = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret


def ppo_update(
    policy: GaussianPolicy,
    value: ValueNetwork,
    optim: torch.optim.Optimizer,
    obs: torch.Tensor,  # (N, obs_dim)
    raw_actions: torch.Tensor,  # (N, 2) original Gaussian samples
    old_log_probs: torch.Tensor,  # (N,)
    advantages: torch.Tensor,  # (N,)
    returns: torch.Tensor,  # (N,)
    clip_range: float,
    update_epochs: int,
    minibatches: int,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """PPO clipped-surrogate update. Mean metrics across all minibatches returned."""
    n = obs.shape[0]
    batch_size = max(1, n // minibatches)
    indices = np.arange(n)

    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    metrics_acc: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_frac": [],
    }

    for _ in range(update_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            mb = indices[start : start + batch_size]
            mb_obs = obs[mb]
            mb_raw = raw_actions[mb]
            mb_old_lp = old_log_probs[mb]
            mb_adv = adv_norm[mb]
            mb_ret = returns[mb]

            mean, log_std = policy.forward_mean_logstd(mb_obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            new_lp = dist.log_prob(mb_raw).sum(-1)
            ratio = (new_lp - mb_old_lp).exp()

            s1 = ratio * mb_adv
            s2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv
            policy_loss = -torch.min(s1, s2).mean()

            v_pred = value(mb_obs)
            value_loss = 0.5 * ((v_pred - mb_ret) ** 2).mean()

            entropy = dist.entropy().sum(-1).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), max_grad_norm)
            optim.step()

            with torch.no_grad():
                approx_kl = (mb_old_lp - new_lp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_range).float().mean().item()
            metrics_acc["policy_loss"].append(policy_loss.item())
            metrics_acc["value_loss"].append(value_loss.item())
            metrics_acc["entropy"].append(entropy.item())
            metrics_acc["approx_kl"].append(approx_kl)
            metrics_acc["clip_frac"].append(clip_frac)

    return {k: float(np.mean(v)) for k, v in metrics_acc.items()}
