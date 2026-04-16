"""Return and observation normalizers for RL training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch


class ReturnNormalizer:
    """Normalize rewards by running standard deviation of episode returns.

    Uses Welford's online algorithm. During warmup, returns are unscaled.
    Only variance is normalized (not mean) to preserve reward sign.
    """

    def __init__(self, warmup_episodes: int = 64) -> None:
        self.warmup_episodes = warmup_episodes
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    @property
    def std(self) -> float:
        if self._count < 2:
            return 1.0
        return float(max(np.sqrt(self._m2 / self._count), 1e-8))

    def update_episode_return(self, episode_return: float) -> None:
        self._count += 1
        delta = episode_return - self._mean
        self._mean += delta / self._count
        delta2 = episode_return - self._mean
        self._m2 += delta * delta2

    def normalize(self, rewards: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self._count < self.warmup_episodes:
            return rewards
        return rewards / self.std

    def state_dict(self) -> dict[str, Any]:
        return {"count": self._count, "mean": self._mean, "m2": self._m2, "warmup_episodes": self.warmup_episodes}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._count = d["count"]
        self._mean = d["mean"]
        self._m2 = d["m2"]
        self.warmup_episodes = d["warmup_episodes"]


class ObsNormalizer:
    """Per-feature running normalization for observation vectors.

    Uses Welford's online algorithm per feature. Clips normalized output.
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
        for row in obs_batch.astype(np.float64):
            self._count += 1
            delta = row - self._mean
            self._mean += delta / self._count
            delta2 = row - self._mean
            self._m2 += delta * delta2

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
        self._count = d["count"]
        self._mean = np.array(d["mean"], dtype=np.float64)
        self._m2 = np.array(d["m2"], dtype=np.float64)
        self.obs_dim = d["obs_dim"]
        self.warmup_steps = d["warmup_steps"]
        self.clip = d["clip"]
