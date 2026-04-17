# RL Reward Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PBRS with phase-aware per-step rewards, add return normalization, and add observation normalization with bake-into-weights export.

**Architecture:** New `StepRewardCalculator` computes per-step rewards from obs + aux using phase-gated components. `ReturnNormalizer` tracks running return std via Welford's algorithm. `ObsNormalizer` tracks per-feature stats and bakes the affine transform into the first layer at export time. All three are wired into both PPO and SAC training loops. PBRS is removed.

**Tech Stack:** Python (numpy, torch), existing PyO3 `BatchedSimulation` aux channel.

**Spec:** `docs/superpowers/specs/2026-04-16-rl-reward-redesign.md`

---

### Task 1: Update RewardConfig dataclass and TOML defaults

**Files:**
- Modify: `src/python/aerocapture/training/rl/config.py`
- Modify: `configs/training/rl_common.toml`

- [ ] **Step 1: Update RewardConfig dataclass**

Replace the PBRS fields with the new per-step reward config:

```python
@dataclass
class RewardConfig:
    # Capture phase weights
    corridor_weight: float = 0.1
    energy_rate_weight: float = 0.05
    constraint_weight: float = 0.2
    # Exit phase weights
    apoapsis_weight: float = 0.2
    eccentricity_weight: float = 0.1
    # Normalization scales
    energy_scale: float = 1.0e6
    # Return and obs normalization
    normalize_returns: bool = True
    normalize_obs: bool = True
    norm_warmup_episodes: int = 64
```

In `config.py`, replace the existing `RewardConfig` class.

- [ ] **Step 2: Update rl_common.toml**

Replace the `[rl.reward]` section:

```toml
[rl.reward]
corridor_weight     = 0.1
energy_rate_weight  = 0.05
constraint_weight   = 0.2
apoapsis_weight     = 0.2
eccentricity_weight = 0.1
energy_scale        = 1.0e6
normalize_returns   = true
normalize_obs       = true
norm_warmup_episodes = 64
```

- [ ] **Step 3: Run config tests**

Run: `uv run pytest tests/rl/test_config.py -v`
Expected: PASS (RewardConfig is parsed from `rl_common.toml` via base inheritance)

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/rl/config.py configs/training/rl_common.toml
git commit -m "feat(rl): update RewardConfig for phase-aware per-step rewards"
```

---

### Task 2: Implement StepRewardCalculator

**Files:**
- Rewrite: `src/python/aerocapture/training/rl/rewards.py`
- Rewrite: `tests/rl/test_rewards.py`

- [ ] **Step 1: Write tests for StepRewardCalculator**

```python
"""Tests for phase-aware per-step reward calculator."""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.rl.rewards import StepRewardCalculator, RewardConfig, compute_terminal_cost


@pytest.fixture
def default_calc() -> StepRewardCalculator:
    return StepRewardCalculator(
        input_mask=list(range(23)),
        corridor_weight=0.1,
        energy_rate_weight=0.05,
        constraint_weight=0.2,
        apoapsis_weight=0.2,
        eccentricity_weight=0.1,
        energy_scale=1.0e6,
    )


def test_capture_phase_corridor_penalty(default_calc: StepRewardCalculator) -> None:
    """Non-zero pdyn_error during capture produces negative reward."""
    n = 4
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 15] = -1.0  # bounce_flag = pre-bounce (capture phase)
    obs[:, 19] = 0.5   # pdyn_error (normalized)
    aux_cur = np.zeros((n, 2), dtype=np.float32)
    aux_next = np.zeros((n, 2), dtype=np.float32)
    r = default_calc.step_reward(obs, aux_cur, aux_next)
    assert np.all(r < 0), "corridor penalty should be negative"


def test_exit_phase_apoapsis_penalty(default_calc: StepRewardCalculator) -> None:
    """Non-zero sma_error during exit produces negative reward."""
    n = 4
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 15] = 1.0   # bounce_flag = post-bounce (exit phase)
    obs[:, 13] = 0.5   # sma_error (normalized)
    aux_cur = np.zeros((n, 2), dtype=np.float32)
    aux_next = np.zeros((n, 2), dtype=np.float32)
    r = default_calc.step_reward(obs, aux_cur, aux_next)
    assert np.all(r < 0), "apoapsis penalty should be negative"


def test_zero_obs_gives_zero_reward(default_calc: StepRewardCalculator) -> None:
    """All-zero obs and aux produces zero reward (no deviation = no penalty)."""
    n = 2
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 15] = -1.0  # capture phase
    aux = np.zeros((n, 2), dtype=np.float32)
    r = default_calc.step_reward(obs, aux, aux)
    assert np.allclose(r, 0.0, atol=1e-10)


def test_energy_dissipation_reward(default_calc: StepRewardCalculator) -> None:
    """Negative energy change (dissipation) during capture gives non-negative contribution."""
    n = 2
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 15] = -1.0  # capture phase
    aux_cur = np.array([[5e6, 0.0]] * n, dtype=np.float32)
    aux_next = np.array([[4.9e6, 0.0]] * n, dtype=np.float32)  # energy decreased
    r = default_calc.step_reward(obs, aux_cur, aux_next)
    # Energy rate = (4.9e6 - 5e6) / 1e6 = -0.1, clamped to 0 for max(), so reward component = 0
    # Actually: -weight * max(delta/scale, 0). delta is negative, max(neg, 0) = 0. So no penalty.
    # Energy dissipation should NOT add penalty. Gaining energy adds penalty.
    assert np.all(r >= -1e-10), "energy dissipation should not penalize"


def test_energy_gain_penalized(default_calc: StepRewardCalculator) -> None:
    """Positive energy change (gaining energy) during capture gives negative reward."""
    n = 2
    obs = np.zeros((n, 23), dtype=np.float32)
    obs[:, 15] = -1.0  # capture phase
    aux_cur = np.array([[4.9e6, 0.0]] * n, dtype=np.float32)
    aux_next = np.array([[5e6, 0.0]] * n, dtype=np.float32)  # energy increased
    r = default_calc.step_reward(obs, aux_cur, aux_next)
    assert np.all(r < 0), "energy gain should be penalized"


def test_constraint_penalty_scales_quadratically(default_calc: StepRewardCalculator) -> None:
    """Constraint penalty grows quadratically with heat flux fraction."""
    n = 1
    obs_low = np.zeros((n, 23), dtype=np.float32)
    obs_low[:, 15] = -1.0
    obs_low[:, 6] = 0.0  # heat_flux_frac in obs space: frac*2-1, so 0.0 = frac=0.5
    obs_high = np.zeros((n, 23), dtype=np.float32)
    obs_high[:, 15] = -1.0
    obs_high[:, 6] = 0.8  # frac = 0.9
    aux = np.zeros((n, 2), dtype=np.float32)
    r_low = default_calc.step_reward(obs_low, aux, aux)
    r_high = default_calc.step_reward(obs_high, aux, aux)
    assert r_high[0] < r_low[0], "higher heat flux fraction should give more penalty"


def test_phase_gating(default_calc: StepRewardCalculator) -> None:
    """Capture-only terms inactive during exit, exit-only terms inactive during capture."""
    n = 1
    # Capture phase with corridor error but no exit errors
    obs_cap = np.zeros((n, 23), dtype=np.float32)
    obs_cap[:, 15] = -1.0  # capture
    obs_cap[:, 19] = 1.0   # pdyn_error
    obs_cap[:, 13] = 0.0   # no sma_error
    # Exit phase with sma_error but no corridor error
    obs_exit = np.zeros((n, 23), dtype=np.float32)
    obs_exit[:, 15] = 1.0  # exit
    obs_exit[:, 19] = 0.0  # no pdyn_error
    obs_exit[:, 13] = 1.0  # sma_error
    aux = np.zeros((n, 2), dtype=np.float32)
    r_cap = default_calc.step_reward(obs_cap, aux, aux)
    r_exit = default_calc.step_reward(obs_exit, aux, aux)
    # Both should be negative (penalties active in their respective phases)
    assert r_cap[0] < 0
    assert r_exit[0] < 0


def test_terminal_cost_matches_evaluate_module() -> None:
    from aerocapture.training.evaluate import compute_cost
    fc = np.zeros((1, 52))
    fc[0, 41] = 100.0
    fc[0, 17] = 5.0
    fc[0, 16] = 150.0
    fc[0, 28] = 10.0
    expected = compute_cost(fc)
    actual = compute_terminal_cost(fc[0])
    assert abs(actual - expected) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rl/test_rewards.py -v`
Expected: FAIL (StepRewardCalculator not yet defined)

- [ ] **Step 3: Implement StepRewardCalculator in rewards.py**

Rewrite `rewards.py`:

```python
"""Phase-aware per-step reward calculator and terminal cost for RL training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerocapture.training.evaluate import compute_cost

# Full-input indices from build_nn_input (neural.rs).
# These are indices into the 23-element full input vector.
_IDX_ECC_EXCESS = 0
_IDX_HEAT_FLUX_FRAC = 6
_IDX_HEAT_LOAD_FRAC = 7
_IDX_SMA_ERROR = 13
_IDX_BOUNCE_FLAG = 15
_IDX_PDYN_ERROR = 19


@dataclass
class StepRewardCalculator:
    """Phase-aware per-step reward from obs + aux.

    Obs indices refer to the full 23-element input vector. When a subset
    input_mask is used, the constructor builds a reverse lookup so that
    obs[:, mapped_idx] corresponds to the correct full-input index.
    """

    input_mask: list[int]
    corridor_weight: float = 0.1
    energy_rate_weight: float = 0.05
    constraint_weight: float = 0.2
    apoapsis_weight: float = 0.2
    eccentricity_weight: float = 0.1
    energy_scale: float = 1.0e6

    def __post_init__(self) -> None:
        # Build reverse lookup: full_input_idx -> obs column index.
        self._rev: dict[int, int] = {v: i for i, v in enumerate(self.input_mask)}
        required = [_IDX_ECC_EXCESS, _IDX_HEAT_FLUX_FRAC, _IDX_HEAT_LOAD_FRAC,
                     _IDX_SMA_ERROR, _IDX_BOUNCE_FLAG, _IDX_PDYN_ERROR]
        missing = [r for r in required if r not in self._rev]
        if missing:
            raise ValueError(f"input_mask missing required indices: {missing}")

    def _col(self, full_idx: int) -> int:
        return self._rev[full_idx]

    def step_reward(
        self,
        obs: npt.NDArray[np.float32],
        aux_cur: npt.NDArray[np.float32],
        aux_next: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float64]:
        """Compute per-step reward for all envs. Shape: (n_envs,)."""
        n = obs.shape[0]
        reward = np.zeros(n, dtype=np.float64)

        bounce = obs[:, self._col(_IDX_BOUNCE_FLAG)]  # -1 = capture, +1 = exit
        in_capture = bounce < 0
        in_exit = ~in_capture

        # -- Shared: constraint proximity (both phases) --
        # obs[6] = heat_flux_fraction * 2.0 - 1.0, so frac = (obs[6] + 1) / 2
        hf_frac = (obs[:, self._col(_IDX_HEAT_FLUX_FRAC)].astype(np.float64) + 1.0) / 2.0
        hl_frac = (obs[:, self._col(_IDX_HEAT_LOAD_FRAC)].astype(np.float64) + 1.0) / 2.0
        reward -= self.constraint_weight * (hf_frac ** 2 + hl_frac ** 2)

        # -- Capture phase --
        if np.any(in_capture):
            pdyn_err = obs[:, self._col(_IDX_PDYN_ERROR)].astype(np.float64)
            reward -= np.where(in_capture, self.corridor_weight * pdyn_err ** 2, 0.0)

            delta_e = (aux_next[:, 0] - aux_cur[:, 0]).astype(np.float64) / self.energy_scale
            reward -= np.where(in_capture, self.energy_rate_weight * np.maximum(delta_e, 0.0), 0.0)

        # -- Exit phase --
        if np.any(in_exit):
            sma_err = obs[:, self._col(_IDX_SMA_ERROR)].astype(np.float64)
            reward -= np.where(in_exit, self.apoapsis_weight * sma_err ** 2, 0.0)

            ecc_excess = obs[:, self._col(_IDX_ECC_EXCESS)].astype(np.float64)
            reward -= np.where(in_exit, self.eccentricity_weight * np.maximum(ecc_excess, 0.0) ** 2, 0.0)

        return reward


def compute_terminal_cost(final_record: npt.NDArray[np.float64]) -> float:
    """Per-episode cost matching evaluate.compute_cost on a single record."""
    return compute_cost(final_record.reshape(1, -1))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/rl/test_rewards.py -v`
Expected: PASS

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/python/aerocapture/training/rl/rewards.py tests/rl/test_rewards.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/rl/rewards.py tests/rl/test_rewards.py
git commit -m "feat(rl): phase-aware StepRewardCalculator replacing PBRS"
```

---

### Task 3: Implement ReturnNormalizer and ObsNormalizer

**Files:**
- Create: `src/python/aerocapture/training/rl/normalizers.py`
- Create: `tests/rl/test_normalizers.py`

- [ ] **Step 1: Write tests for normalizers**

```python
"""Tests for return and observation normalizers."""

from __future__ import annotations

import numpy as np
import pytest
from aerocapture.training.rl.normalizers import ReturnNormalizer, ObsNormalizer


class TestReturnNormalizer:
    def test_warmup_returns_unscaled(self) -> None:
        norm = ReturnNormalizer(warmup_episodes=10)
        for _ in range(5):
            norm.update_episode_return(-500.0)
        raw = np.array([-400.0, -600.0], dtype=np.float64)
        out = norm.normalize(raw)
        np.testing.assert_array_equal(out, raw)

    def test_post_warmup_scales_by_std(self) -> None:
        norm = ReturnNormalizer(warmup_episodes=2)
        norm.update_episode_return(-100.0)
        norm.update_episode_return(-300.0)
        # mean=-200, var=10000, std=100
        raw = np.array([-200.0], dtype=np.float64)
        out = norm.normalize(raw)
        assert abs(out[0] - (-200.0 / 100.0)) < 0.1

    def test_checkpoint_roundtrip(self) -> None:
        norm = ReturnNormalizer(warmup_episodes=2)
        for v in [-100.0, -200.0, -300.0]:
            norm.update_episode_return(v)
        state = norm.state_dict()
        norm2 = ReturnNormalizer(warmup_episodes=2)
        norm2.load_state_dict(state)
        raw = np.array([-250.0], dtype=np.float64)
        np.testing.assert_allclose(norm.normalize(raw), norm2.normalize(raw))


class TestObsNormalizer:
    def test_normalize_shape_preserved(self) -> None:
        norm = ObsNormalizer(obs_dim=4, warmup_steps=0)
        obs = np.ones((8, 4), dtype=np.float32)
        norm.update(obs)
        out = norm.normalize(obs)
        assert out.shape == (8, 4)
        assert out.dtype == np.float32

    def test_normalize_zero_mean_unit_var(self) -> None:
        rng = np.random.default_rng(42)
        norm = ObsNormalizer(obs_dim=3, warmup_steps=0)
        for _ in range(100):
            obs = rng.standard_normal((64, 3)).astype(np.float32) * 10 + 5
            norm.update(obs)
        obs = rng.standard_normal((64, 3)).astype(np.float32) * 10 + 5
        out = norm.normalize(obs)
        # After many updates, normalized output should be roughly zero-mean
        assert abs(np.mean(out)) < 2.0

    def test_clip_bounds(self) -> None:
        norm = ObsNormalizer(obs_dim=2, warmup_steps=0, clip=5.0)
        norm.update(np.array([[0.0, 0.0]], dtype=np.float32))
        extreme = np.array([[1e6, -1e6]], dtype=np.float32)
        out = norm.normalize(extreme)
        assert np.all(out <= 5.0)
        assert np.all(out >= -5.0)

    def test_bake_into_linear_layer(self) -> None:
        import torch
        norm = ObsNormalizer(obs_dim=4, warmup_steps=0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            norm.update(rng.standard_normal((32, 4)).astype(np.float32) * 10 + 5)
        linear = torch.nn.Linear(4, 8)
        torch.manual_seed(0)
        torch.nn.init.normal_(linear.weight)
        torch.nn.init.normal_(linear.bias)
        w_orig = linear.weight.data.clone()
        b_orig = linear.bias.data.clone()
        norm.bake_into_linear(linear)
        # After baking: linear(raw_input) should equal original_linear(normalized_input)
        raw = torch.from_numpy(rng.standard_normal((16, 4)).astype(np.float32) * 10 + 5)
        normalized = torch.from_numpy(norm.normalize(raw.numpy()))
        out_baked = linear(raw)
        out_manual = torch.nn.functional.linear(normalized, w_orig, b_orig)
        torch.testing.assert_close(out_baked, out_manual, atol=1e-4, rtol=1e-4)

    def test_checkpoint_roundtrip(self) -> None:
        norm = ObsNormalizer(obs_dim=3, warmup_steps=0)
        norm.update(np.ones((10, 3), dtype=np.float32) * 5)
        state = norm.state_dict()
        norm2 = ObsNormalizer(obs_dim=3, warmup_steps=0)
        norm2.load_state_dict(state)
        obs = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        np.testing.assert_allclose(norm.normalize(obs), norm2.normalize(obs))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rl/test_normalizers.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement normalizers.py**

```python
"""Return and observation normalizers for RL training."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


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
        return max(np.sqrt(self._m2 / self._count), 1e-8)

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
        return {"count": self._count, "mean": self._mean, "m2": self._m2,
                "warmup_episodes": self.warmup_episodes}

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
        if self._count < max(self.warmup_steps, 2):
            return obs
        normed = (obs.astype(np.float64) - self._mean) / self.std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def bake_into_linear(self, linear: "torch.nn.Linear") -> None:
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
        return {"count": self._count, "mean": self._mean.tolist(),
                "m2": self._m2.tolist(), "obs_dim": self.obs_dim,
                "warmup_steps": self.warmup_steps, "clip": self.clip}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._count = d["count"]
        self._mean = np.array(d["mean"], dtype=np.float64)
        self._m2 = np.array(d["m2"], dtype=np.float64)
        self.obs_dim = d["obs_dim"]
        self.warmup_steps = d["warmup_steps"]
        self.clip = d["clip"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/rl/test_normalizers.py -v`
Expected: PASS

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/python/aerocapture/training/rl/normalizers.py tests/rl/test_normalizers.py`
Expected: All checks passed

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/rl/normalizers.py tests/rl/test_normalizers.py
git commit -m "feat(rl): ReturnNormalizer and ObsNormalizer with bake-into-weights"
```

---

### Task 4: Add obs normalization bake-in to export.py

**Files:**
- Modify: `src/python/aerocapture/training/rl/export.py`

- [ ] **Step 1: Update export_policy_to_json to accept optional ObsNormalizer**

Add an optional `obs_normalizer` parameter. When provided, bake the normalization into a clone of the policy's first linear layer before extracting weights:

```python
def export_policy_to_json(
    policy: GaussianPolicy,
    output_path: Path,
    input_mask: Sequence[int],
    output_interpretation: str = "atan2",
    obs_normalizer: "ObsNormalizer | None" = None,
) -> None:
```

Before iterating over `policy.trunk`, if `obs_normalizer` is not None, clone the policy, bake the normalizer into the clone's first linear layer, and extract weights from the clone:

```python
    import copy
    if obs_normalizer is not None:
        trunk = copy.deepcopy(policy.trunk)
        for module in trunk:
            if isinstance(module, torch.nn.Linear):
                obs_normalizer.bake_into_linear(module)
                break
    else:
        trunk = policy.trunk
```

Then iterate over `trunk` instead of `policy.trunk` for weight extraction.

- [ ] **Step 2: Run existing export test**

Run: `uv run pytest tests/rl/test_export.py -v`
Expected: PASS (no obs_normalizer passed = old behavior)

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/rl/export.py
git commit -m "feat(rl): export_policy_to_json accepts optional ObsNormalizer for bake-in"
```

---

### Task 5: Wire new reward + normalizers into PPO loop

**Files:**
- Modify: `src/python/aerocapture/training/rl/train.py`

- [ ] **Step 1: Update imports**

Replace PBRS imports with new modules:

```python
from aerocapture.training.rl.rewards import StepRewardCalculator, compute_terminal_cost
from aerocapture.training.rl.normalizers import ReturnNormalizer, ObsNormalizer
```

Remove: `from aerocapture.training.rl.rewards import PBRSShaper, compute_terminal_cost, load_reference_pdyn`
Remove: `from collections.abc import Callable`

- [ ] **Step 2: Replace PBRS setup in _run_ppo with StepRewardCalculator + normalizers**

Replace the PBRS block (ref trajectory loading, PBRSShaper construction) with:

```python
    step_calc = StepRewardCalculator(
        input_mask=input_mask,
        corridor_weight=cfg.reward.corridor_weight,
        energy_rate_weight=cfg.reward.energy_rate_weight,
        constraint_weight=cfg.reward.constraint_weight,
        apoapsis_weight=cfg.reward.apoapsis_weight,
        eccentricity_weight=cfg.reward.eccentricity_weight,
        energy_scale=cfg.reward.energy_scale,
    )
    ret_norm = ReturnNormalizer(warmup_episodes=cfg.reward.norm_warmup_episodes) if cfg.reward.normalize_returns else None
    obs_norm = ObsNormalizer(obs_dim=input_dim) if cfg.reward.normalize_obs else None
```

- [ ] **Step 3: Update rollout collection to use new reward calculator**

In the rollout loop, replace PBRS calls:

```python
            # Per-step reward from obs + aux (phase-aware).
            shaped = step_calc.step_reward(obs, aux_cur, aux_next).astype(np.float32)

            for i, d in enumerate(done):
                if d:
                    fr = np.array(info[i]["final_record"], dtype=np.float64)
                    term_cost = compute_terminal_cost(fr)
                    shaped[i] += float(-term_cost)
                    episodic_returns.append(float(-term_cost))
                    if ret_norm is not None:
                        ret_norm.update_episode_return(float(-term_cost))
                    episodic_dvs.append(float(info[i].get("dv_m_s", float("nan"))))
                    episodic_captures.append(bool(info[i].get("captured", False)))
```

Note: the terminal reward is now added to `shaped[i]` (accumulating with the step reward), not replacing it. The old PBRS boundary correction is gone since we no longer use PBRS.

- [ ] **Step 4: Apply return normalization to rewards before GAE**

After the rollout loop ends and before GAE computation, normalize the rewards in the buffer:

```python
        if ret_norm is not None:
            buf.rewards = ret_norm.normalize(buf.rewards.astype(np.float64)).astype(np.float32)
```

- [ ] **Step 5: Apply obs normalization**

Update the obs before policy/value forward passes. At rollout collection:

```python
            if obs_norm is not None:
                obs_norm.update(obs)
            obs_for_policy = obs_norm.normalize(obs) if obs_norm is not None else obs
            obs_t = torch.from_numpy(obs_for_policy).float()
```

Store raw obs in the buffer (for normalizer updates) and normalize at PPO update time:

```python
        flat_obs = torch.from_numpy(buf.obs.reshape(-1, env.obs_dim)).float()
        if obs_norm is not None:
            flat_obs = torch.from_numpy(obs_norm.normalize(buf.obs.reshape(-1, env.obs_dim))).float()
```

- [ ] **Step 6: Update checkpoint save/load to include normalizer state**

In `_save_checkpoint`, add:

```python
            "ret_norm": ret_norm.state_dict() if ret_norm is not None else None,
            "obs_norm": obs_norm.state_dict() if obs_norm is not None else None,
```

In the checkpoint resume block, add after loading model state:

```python
        if ret_norm is not None and ckpt.get("ret_norm") is not None:
            ret_norm.load_state_dict(ckpt["ret_norm"])
        if obs_norm is not None and ckpt.get("obs_norm") is not None:
            obs_norm.load_state_dict(ckpt["obs_norm"])
```

- [ ] **Step 7: Update export to bake obs normalization**

In the validation export and final export calls, pass `obs_norm`:

```python
    export_policy_to_json(policy, output_dir / "best_model.json", input_mask, obs_normalizer=obs_norm)
```

- [ ] **Step 8: Run smoke test**

Run: `uv run pytest tests/rl/test_train_smoke.py -v`
Expected: PASS

- [ ] **Step 9: Lint**

Run: `uv run ruff check src/python/aerocapture/training/rl/train.py`
Expected: All checks passed

- [ ] **Step 10: Commit**

```bash
git add src/python/aerocapture/training/rl/train.py
git commit -m "feat(rl): wire StepRewardCalculator + normalizers into PPO loop"
```

---

### Task 6: Wire new reward + normalizers into SAC loop

**Files:**
- Modify: `src/python/aerocapture/training/rl/train.py`

- [ ] **Step 1: Replace PBRS setup in _run_sac**

Same pattern as PPO -- replace the PBRS block with `StepRewardCalculator` + `ReturnNormalizer` + `ObsNormalizer`. Use the same constructor as Task 5 Step 2.

- [ ] **Step 2: Update SAC step collection**

Replace the old reward computation with `step_calc.step_reward()` + terminal cost. Apply return normalization to shaped rewards before pushing to replay buffer:

```python
        shaped = step_calc.step_reward(obs, aux_cur, aux_next).astype(np.float32)
        for i, d in enumerate(done):
            if d:
                fr = np.array(info[i]["final_record"], dtype=np.float64)
                term_cost = compute_terminal_cost(fr)
                shaped[i] += float(-term_cost)
                episodic_returns.append(float(-term_cost))
                if ret_norm is not None:
                    ret_norm.update_episode_return(float(-term_cost))
                episodic_dvs.append(float(info[i].get("dv_m_s", float("nan"))))
                episodic_captures.append(bool(info[i].get("captured", False)))

        if ret_norm is not None:
            shaped = ret_norm.normalize(shaped.astype(np.float64)).astype(np.float32)
```

- [ ] **Step 3: Apply obs normalization in SAC collection and update**

Normalize obs before policy forward pass and before storing in replay buffer:

```python
        if obs_norm is not None:
            obs_norm.update(obs)
        obs_for_policy = obs_norm.normalize(obs) if obs_norm is not None else obs
        obs_t = torch.from_numpy(obs_for_policy).float()
```

Store normalized obs in the replay buffer (SAC replays from buffer, so it needs pre-normalized data):

```python
        obs_store = obs_norm.normalize(obs) if obs_norm is not None else obs
        next_obs_store = obs_norm.normalize(next_obs) if obs_norm is not None else next_obs
        agent.replay_buffer.push(obs_store, actions_np, shaped, next_obs_store, done)
```

- [ ] **Step 4: Update SAC checkpoint to include normalizer state**

Same pattern as PPO: add `ret_norm` and `obs_norm` state dicts to checkpoint save/load.

- [ ] **Step 5: Update SAC export calls**

Pass `obs_normalizer=obs_norm` to `export_policy_to_json`.

- [ ] **Step 6: Run full RL test suite**

Run: `uv run pytest tests/rl/ -v`
Expected: All PASS

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/python/aerocapture/training/rl/`
Expected: All checks passed

- [ ] **Step 8: Commit**

```bash
git add src/python/aerocapture/training/rl/train.py
git commit -m "feat(rl): wire StepRewardCalculator + normalizers into SAC loop"
```

---

### Task 7: Run full test suite and update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All pass (448+)

- [ ] **Step 2: Update CLAUDE.md RL section**

Update the reward shaping paragraph to describe the new system:

- Replace the PBRS description with the phase-aware per-step reward description
- Add note about return normalization and obs normalization with bake-into-weights
- Update the `[rl.reward]` TOML field list

- [ ] **Step 3: Commit with smart-commit skill**

Invoke the `smart-commit` skill, telling it to take the whole git branch into account.
