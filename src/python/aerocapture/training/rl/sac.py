"""SAC (Soft Actor-Critic) over the 2D Gaussian latent.

The policy samples `raw = (out0, out1)` from a diagonal Gaussian; the
environment sees `bank = atan2(raw[0], raw[1])`. SAC's critic, replay
buffer, and entropy objective all operate on `raw` -- the space the
policy density actually lives on. That way `target_entropy = -2` (the
latent dimension) is consistent with what alpha auto-tuning regularizes.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from aerocapture.training.rl.policy import GaussianPolicy, build_mlp


class _QNetwork(nn.Module):
    """Q(obs, raw_action) with raw_action in R^2 (pre-atan2 Gaussian sample)."""

    def __init__(self, obs_dim: int, hidden_sizes: list[int], activations: list[str]) -> None:
        super().__init__()
        input_dim = obs_dim + 2
        layer_sizes = list(hidden_sizes) + [1]
        act_list = list(activations) + ["linear"]
        self.net = build_mlp(input_dim, layer_sizes, act_list)

    def forward(self, obs: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, raw_action], dim=-1)
        return self.net(x).squeeze(-1)  # type: ignore[no-any-return]


class ReplayBuffer:
    """Fixed-size ring buffer storing (obs, raw_action, reward, next_obs, done, truncated)."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self._cap = capacity
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, 2), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=bool)
        self._ptr = 0
        self._size = 0

    def push(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        n = len(obs)
        idxs = (self._ptr + np.arange(n)) % self._cap
        self._obs[idxs] = obs
        self._actions[idxs] = raw_action
        self._rewards[idxs] = reward
        self._next_obs[idxs] = next_obs
        self._dones[idxs] = done
        self._ptr = (self._ptr + n) % self._cap
        self._size = min(self._size + n, self._cap)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idxs = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self._obs[idxs]),
            torch.from_numpy(self._actions[idxs]),
            torch.from_numpy(self._rewards[idxs]),
            torch.from_numpy(self._next_obs[idxs]),
            torch.from_numpy(self._dones[idxs]),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "cap": self._cap,
            "ptr": self._ptr,
            "size": self._size,
            "obs": self._obs,
            "actions": self._actions,
            "rewards": self._rewards,
            "next_obs": self._next_obs,
            "dones": self._dones,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        if d["cap"] != self._cap:
            raise ValueError(f"replay buffer capacity mismatch: ckpt={d['cap']} current={self._cap}")
        self._ptr = int(d["ptr"])
        self._size = int(d["size"])
        self._obs[:] = d["obs"]
        self._actions[:] = d["actions"]
        self._rewards[:] = d["rewards"]
        self._next_obs[:] = d["next_obs"]
        self._dones[:] = d["dones"]

    def __len__(self) -> int:
        return self._size


class SACAgent:
    """SAC with twin Q networks, soft target updates, and alpha auto-tuning."""

    def __init__(
        self,
        obs_dim: int,
        layer_sizes: list[int],
        activations: list[str],
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        learning_rate: float = 3.0e-4,
        target_entropy: str | float = "auto",
        initial_alpha: float = 0.2,
    ) -> None:
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.policy = GaussianPolicy(obs_dim, layer_sizes, activations)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        hidden = layer_sizes[:-1]
        hidden_acts = activations[:-1]
        self.q1 = _QNetwork(obs_dim, hidden, hidden_acts)
        self.q2 = _QNetwork(obs_dim, hidden, hidden_acts)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        self.q_optim = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate)

        # target_entropy = -dim(A); A is the 2D Gaussian latent that the
        # policy density lives on, so -2 is correct.
        self.target_entropy: float = -2.0 if target_entropy == "auto" else float(target_entropy)
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(initial_alpha))))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=learning_rate)

        self.replay_buffer = ReplayBuffer(buffer_size, obs_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(
        self,
        obs: torch.Tensor,
        raw_actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, Any]:
        obs = obs.float()
        raw_actions = raw_actions.float()
        rewards = rewards.float()
        next_obs = next_obs.float()
        done_float = dones.float()

        with torch.no_grad():
            _, next_raw, next_log_prob = self.policy.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_raw)
            q2_next = self.q2_target(next_obs, next_raw)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1.0 - done_float) * (q_next - self.alpha.detach() * next_log_prob)

        q1_pred = self.q1(obs, raw_actions)
        q2_pred = self.q2(obs, raw_actions)
        q_loss = nn.functional.mse_loss(q1_pred, target_q) + nn.functional.mse_loss(q2_pred, target_q)
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        _, raw_new, log_prob = self.policy.sample(obs)
        q1_pi = self.q1(obs, raw_new)
        q2_pi = self.q2(obs, raw_new)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_target.parameters(), strict=True):
                pt.data.mul_(1.0 - self.tau).add_(p.data * self.tau)
            for p, pt in zip(self.q2.parameters(), self.q2_target.parameters(), strict=True):
                pt.data.mul_(1.0 - self.tau).add_(p.data * self.tau)

        return {
            "q_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "mean_log_prob": float(log_prob.mean().item()),
        }
