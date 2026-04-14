"""Factory functions for building test objects with sensible defaults."""

from __future__ import annotations

import numpy as np
from aerocapture.training.config import NetworkConfig, OptimizerConfig, SimConfig, TrainingConfig


def make_training_config(guidance_type: str = "equilibrium_glide") -> TrainingConfig:
    """Build a minimal TrainingConfig for the given guidance type."""
    return TrainingConfig(
        network=NetworkConfig(),
        optimizer=OptimizerConfig(seed_strategy="adaptive"),
        sim=SimConfig(
            executable="dummy",
            nn_param_file="dummy.json",
            final_file="final.csv",
        ),
        save_dir="dummy",
        guidance_type=guidance_type,
    )


def make_chromosome(length: int, *, strategy: str = "mid") -> np.ndarray:
    """Generate a binary chromosome (kept for backward compat in TOML patching tests).

    Strategies:
        mid   -- alternating 0/1 (mid-range parameter values)
        zeros -- all zeros (minimum parameter values)
        ones  -- all ones (maximum parameter values)
        random -- uniformly random bits (seed=42)
    """
    if strategy == "mid":
        return np.array([i % 2 for i in range(length)], dtype=np.int8)
    if strategy == "zeros":
        return np.zeros(length, dtype=np.int8)
    if strategy == "ones":
        return np.ones(length, dtype=np.int8)
    if strategy == "random":
        return np.random.default_rng(42).integers(0, 2, size=length, dtype=np.int8)
    msg = f"Unknown strategy: {strategy}"
    raise ValueError(msg)


def make_normalized_individual(n_params: int, *, strategy: str = "mid") -> np.ndarray:
    """Generate a normalized [0, 1] individual for real-valued optimization.

    Strategies:
        mid    -- 0.5 for all params
        zeros  -- 0.0 for all params (minimum values)
        ones   -- 1.0 for all params (maximum values)
        random -- uniform random in [0, 1] (seed=42)
    """
    if strategy == "mid":
        return np.full(n_params, 0.5, dtype=np.float64)
    if strategy == "zeros":
        return np.zeros(n_params, dtype=np.float64)
    if strategy == "ones":
        return np.ones(n_params, dtype=np.float64)
    if strategy == "random":
        return np.random.default_rng(42).random(n_params)
    msg = f"Unknown strategy: {strategy}"
    raise ValueError(msg)
