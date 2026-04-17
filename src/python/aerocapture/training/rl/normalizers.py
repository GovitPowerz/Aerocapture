"""Return and observation normalizers for RL training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch


def _chan_update(
    mean: npt.NDArray[np.float64],
    m2: npt.NDArray[np.float64],
    count: int,
    batch: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Chan's parallel Welford update for a batch of rows (shape (n, d))."""
    n = batch.shape[0]
    if n == 0:
        return mean, m2, count
    batch_mean = batch.mean(axis=0)
    batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)
    if count == 0:
        return batch_mean, batch_m2, n
    new_count = count + n
    delta = batch_mean - mean
    new_mean = mean + delta * (n / new_count)
    new_m2 = m2 + batch_m2 + delta**2 * (count * n / new_count)
    return new_mean, new_m2, new_count


class ReturnNormalizer:
    """Normalize rewards by the running std of discounted returns.

    Follows Stable-Baselines3: maintain a running estimate of
    R_t = gamma * R_{t-1} + r_t  (per env), then divide per-step rewards by
    std(R) so the value-function target keeps a stable scale across rollouts.
    Only std is applied; the running mean is tracked but not subtracted.
    """

    def __init__(self, gamma: float = 0.99, warmup_steps: int = 1000, epsilon: float = 1e-8) -> None:
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.epsilon = epsilon
        self._count = 0
        self._mean = np.zeros(1, dtype=np.float64)
        self._m2 = np.zeros(1, dtype=np.float64)
        self._returns: npt.NDArray[np.float64] | None = None

    @property
    def std(self) -> float:
        if self._count < 2:
            return 1.0
        return float(max(np.sqrt(self._m2[0] / self._count), self.epsilon))

    def update(self, rewards: npt.NDArray[np.float64], dones: npt.NDArray[np.bool_]) -> None:
        """Update running return stats. `rewards`/`dones` are per-env for one step."""
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=np.bool_)
        n = rewards.shape[0]
        if self._returns is None or self._returns.shape[0] != n:
            self._returns = np.zeros(n, dtype=np.float64)
        self._returns[:] = self.gamma * self._returns * (~dones).astype(np.float64) + rewards
        batch = self._returns.reshape(-1, 1)
        self._mean, self._m2, self._count = _chan_update(self._mean, self._m2, self._count, batch)

    def normalize(self, rewards: npt.NDArray[np.float64], std: float | None = None) -> npt.NDArray[np.float64]:
        """Divide by running std. Pass an explicit `std` (e.g. a snapshot taken at
        rollout end) to avoid non-stationarity when `update()` is called between
        a rollout's collection and its normalization."""
        if self._count < self.warmup_steps:
            return rewards
        return rewards / (std if std is not None else self.std)

    def state_dict(self) -> dict[str, Any]:
        return {
            "count": self._count,
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
            "warmup_steps": self.warmup_steps,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "returns": None if self._returns is None else self._returns.tolist(),
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._count = int(d["count"])
        self._mean = np.array(d["mean"], dtype=np.float64)
        self._m2 = np.array(d["m2"], dtype=np.float64)
        self.warmup_steps = int(d["warmup_steps"])
        self.gamma = float(d.get("gamma", 0.99))
        self.epsilon = float(d.get("epsilon", 1e-8))
        self._returns = None if d.get("returns") is None else np.array(d["returns"], dtype=np.float64)


class ObsNormalizer:
    """Per-feature running normalization for observation vectors.

    Uses Chan's parallel Welford update over a batch. Clips normalized output.
    Can bake the affine transform into a nn.Linear layer for export.
    """

    def __init__(self, obs_dim: int, warmup_steps: int = 0, clip: float = 10.0) -> None:
        self.obs_dim = obs_dim
        self.warmup_steps = warmup_steps
        self.clip = clip
        self._count = 0
        self._mean = np.zeros(obs_dim, dtype=np.float64)
        self._m2 = np.zeros(obs_dim, dtype=np.float64)

    @property
    def std(self) -> npt.NDArray[np.float64]:
        if self._count < 2:
            return np.ones(self.obs_dim, dtype=np.float64)
        return np.maximum(np.sqrt(self._m2 / self._count), 1e-8)

    def update(self, obs_batch: npt.NDArray[np.float32]) -> None:
        """Update running stats with a batch of observations (n, obs_dim)."""
        batch = np.asarray(obs_batch, dtype=np.float64)
        self._mean, self._m2, self._count = _chan_update(self._mean, self._m2, self._count, batch)

    def normalize(self, obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if self._count < max(self.warmup_steps, 1):
            return obs
        normed = (obs.astype(np.float64) - self._mean) / self.std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def bake_into_linear(self, linear: torch.nn.Linear) -> None:
        """Absorb normalization into the first linear layer weights.

        After baking: linear(raw_obs) == original_linear(normalized_obs).
        W_new = W / std,  b_new = b - W @ (mean / std)
        """
        import torch

        mean = torch.from_numpy(self._mean).float()
        std = torch.from_numpy(self.std).float()
        with torch.no_grad():
            linear.bias.data -= linear.weight.data @ (mean / std)
            linear.weight.data /= std.unsqueeze(0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "count": self._count,
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
            "obs_dim": self.obs_dim,
            "warmup_steps": self.warmup_steps,
            "clip": self.clip,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._count = int(d["count"])
        self._mean = np.array(d["mean"], dtype=np.float64)
        self._m2 = np.array(d["m2"], dtype=np.float64)
        self.obs_dim = int(d["obs_dim"])
        self.warmup_steps = int(d["warmup_steps"])
        self.clip = float(d["clip"])
