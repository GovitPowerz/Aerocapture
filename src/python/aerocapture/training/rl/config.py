"""[rl] TOML section parser.

Uses the existing `toml_utils.load_toml_with_bases` resolver to apply base
inheritance, then plucks the [rl] subtree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aerocapture.training.toml_utils import load_toml_with_bases

_VALID_ALGOS: tuple[str, ...] = ("ppo", "sac")


@dataclass
class RewardConfig:
    # Potential-based shaping weights (capture phase: corridor + energy; shared: constraint)
    corridor_weight: float = 0.1
    energy_rate_weight: float = 0.05
    constraint_weight: float = 0.2
    # Exit-phase shaping weights
    apoapsis_weight: float = 0.2
    eccentricity_weight: float = 0.1
    energy_scale: float = 1.0e6
    # Running normalization
    normalize_returns: bool = True
    normalize_obs: bool = True
    # Return normalizer warms up over this many per-step updates (per-env, summed).
    norm_warmup_steps: int = 1000


@dataclass
class PPOConfig:
    learning_rate: float = 3.0e-4
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    initial_log_std: float = -0.5
    min_log_std: float = -2.0
    lr_anneal_start: float = 0.7
    # Early-stop the update epoch when mean approx_kl exceeds this threshold.
    # None disables early-stop.
    target_kl: float | None = 0.03


@dataclass
class SACConfig:
    learning_rate: float = 3.0e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    train_every: int = 1
    gradient_steps: int = 1
    target_entropy: str | float = "auto"
    initial_alpha: float = 0.2
    warmup_steps: int = 50_000


@dataclass
class RLConfig:
    algorithm: Literal["ppo", "sac"] = "ppo"
    total_env_steps: int = 5_000_000
    n_envs: int = 64
    seed_base: int = 3_000_000
    validation_n_sims: int = 1000
    validation_interval_updates: int = 20
    checkpoint_interval_updates: int = 50
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    raw_toml: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_toml(
        cls,
        path: Path,
        overrides: dict[str, Any] | None = None,
        ppo_overrides: dict[str, Any] | None = None,
    ) -> RLConfig:
        resolved = load_toml_with_bases(path)
        rl = resolved.get("rl", {})
        if overrides:
            rl = {**rl, **overrides}
        algo = rl.get("algorithm", "ppo")
        if algo not in _VALID_ALGOS:
            raise ValueError(f"[rl] algorithm must be one of {_VALID_ALGOS}, got {algo!r}")
        reward = RewardConfig(**rl.get("reward", {}))
        ppo_src = rl.get("ppo", {})
        if ppo_overrides:
            ppo_src = {**ppo_src, **ppo_overrides}
        ppo = PPOConfig(**ppo_src)
        sac = SACConfig(**rl.get("sac", {}))
        return cls(
            algorithm=algo,
            total_env_steps=rl.get("total_env_steps", 5_000_000),
            n_envs=rl.get("n_envs", 64),
            seed_base=rl.get("seed_base", 3_000_000),
            validation_n_sims=rl.get("validation_n_sims", 1000),
            validation_interval_updates=rl.get("validation_interval_updates", 20),
            checkpoint_interval_updates=rl.get("checkpoint_interval_updates", 50),
            reward=reward,
            ppo=ppo,
            sac=sac,
            raw_toml=resolved,
        )
