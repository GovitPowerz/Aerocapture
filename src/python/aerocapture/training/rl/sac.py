"""SAC (Soft Actor-Critic) — experimental, parallel track to PPO.

UNCONVENTIONAL DESIGN NOTE:
    This SAC uses the same GaussianPolicy as PPO (2-output atan2 head,
    NOT the textbook tanh-squashed-Gaussian). The policy samples in
    unconstrained R^2 space and maps to bank via atan2. Consequently:
    - log_prob is the Gaussian log-prob on the 2D raw output (no tanh correction).
    - target_entropy = -2.0 (action dimension in raw space, not -1 for bank).
    - Export to best_model.json is lossless and identical to PPO.

    This is unconventional but keeps the export path identical between algorithms.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from aerocapture.training.rl.policy import GaussianPolicy, _build_mlp


class _QNetwork(nn.Module):
    """Q(obs, action): input = [obs, sin(action), cos(action)]."""

    def __init__(self, obs_dim: int, hidden_sizes: list[int], activations: list[str]) -> None:
        super().__init__()
        input_dim = obs_dim + 2
        layer_sizes = list(hidden_sizes) + [1]
        act_list = list(activations) + ["linear"]
        self.net = _build_mlp(input_dim, layer_sizes, act_list)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, torch.sin(action).unsqueeze(-1), torch.cos(action).unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)  # type: ignore[no-any-return]


class ReplayBuffer:
    """Fixed-size ring buffer storing (obs, action, reward, next_obs, done)."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self._cap = capacity
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=bool)
        self._ptr = 0
        self._size = 0

    def push(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, next_obs: np.ndarray, done: np.ndarray) -> None:
        n = len(obs)
        idxs = (self._ptr + np.arange(n)) % self._cap
        self._obs[idxs] = obs
        self._actions[idxs] = action
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

    def __len__(self) -> int:
        return self._size


class SACAgent:
    """SAC agent with twin Q networks, soft target updates, and alpha auto-tuning.

    Uses GaussianPolicy with atan2 head (see module docstring for the
    unconventional log_prob convention).
    """

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

        # Policy uses the full layer_sizes with output dim 2 (atan2).
        self.policy = GaussianPolicy(obs_dim, layer_sizes, activations)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Q nets: hidden sizes = layer_sizes[:-1], activations[:-1]
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

        # Alpha: entropy regularisation coefficient.
        # target_entropy = -2.0 because the raw action space is 2-D.
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
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, Any]:
        """One SAC gradient step on the provided batch."""
        obs = obs.float()
        actions = actions.float()
        rewards = rewards.float()
        next_obs = next_obs.float()
        done_float = dones.float()

        with torch.no_grad():
            next_bank, next_log_prob = self.policy.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_bank)
            q2_next = self.q2_target(next_obs, next_bank)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1.0 - done_float) * (q_next - self.alpha.detach() * next_log_prob)

        # Q loss
        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        q_loss = nn.functional.mse_loss(q1_pred, target_q) + nn.functional.mse_loss(q2_pred, target_q)
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        # Policy loss
        bank, log_prob = self.policy.sample(obs)
        q1_pi = self.q1(obs, bank)
        q2_pi = self.q2(obs, bank)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Soft target update
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
