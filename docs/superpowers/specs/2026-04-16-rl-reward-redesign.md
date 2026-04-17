# RL Reward Redesign: Phase-Aware Per-Step Rewards + Normalization

**Date:** 2026-04-16
**Status:** Approved
**Branch:** `feature/rl-nn-guidance-spec`

## Problem

The current RL reward structure has three compounding issues that prevent convergence beyond the GA baseline:

1. **PBRS is invisible.** Step rewards are ~0.05 per step; terminal cost is -400 to -50,000. The policy gradient is dominated by a single scalar at episode end.
2. **Credit assignment over 700 steps.** With gamma=0.99, the signal from step 0 is attenuated by gamma^700 ~ 0.007. The value function must predict returns in [-50000, -100] -- a hard regression problem.
3. **No reward normalization.** Huge variance between crash episodes (-50,000) and good captures (-400) destabilizes both policy and value learning.

FTC achieves DV p50=150 m/s. The GA-trained NN achieves ~550. PPO warm-started from GA achieves ~450. The gap is not network capacity or warm-start -- it's reward signal quality.

## Design

### 1. Phase-aware per-step reward

Replace PBRS with direct per-step rewards gated by the bounce flag (obs[15]). The bounce flag transitions from -1 (capture phase) to +1 (exit phase), providing a clean phase boundary already present in the observation vector.

#### Capture phase (bounce_flag < 0)

The spacecraft is descending into the atmosphere, decelerating, and must track the reference corridor while respecting thermal/structural constraints.

| Component | Source | Formula | Rationale |
|---|---|---|---|
| Corridor tracking | obs[19] (pdyn_error) | `-corridor_weight * pdyn_error^2` | Penalize deviation from reference corridor. Quadratic for smooth gradient near zero. obs[19] is already normalized by 2e3 in `build_nn_input`. |
| Energy dissipation | aux channel | `-energy_rate_weight * max(delta_energy / energy_scale, 0)` | Reward negative energy rate (spacecraft bleeding orbital energy). `delta_energy = energy_next - energy_cur` from consecutive aux values (one GNC tick apart, ~1s). Clamp positive (gaining energy during skip-out = no penalty, avoid fighting the physics). No explicit dt normalization needed since the GNC tick is fixed and the weight absorbs the scale. |
| Constraint proximity | obs[6], obs[7] | `-constraint_weight * (heat_flux_frac^2 + heat_load_frac^2)` | Quadratic penalty as thermal quantities approach limits. obs[6] and obs[7] are already in [0, 1] as fractions of the constraint limits. |

#### Exit phase (bounce_flag > 0)

The spacecraft is ascending out of the atmosphere and must target the correct orbit.

| Component | Source | Formula | Rationale |
|---|---|---|---|
| Apoapsis targeting | obs[13] (sma_error) | `-apoapsis_weight * sma_error^2` | Drive toward target semi-major axis. obs[13] is normalized by 5e5 in `build_nn_input`. |
| Eccentricity reduction | obs[0] (ecc_excess) | `-eccentricity_weight * max(ecc_excess, 0)^2` | Penalize ecc > 1 (hyperbolic). Once ecc < 1 (captured), no penalty. |
| Constraint proximity | obs[6], obs[7] | Same as capture phase | Thermal constraints apply throughout. |

#### Terminal reward

Keep `compute_cost(final_record)` as the terminal reward (negated). This is the ground truth objective used by validation and final evaluation. The per-step rewards guide learning; the terminal cost anchors the optimum.

#### Phase transition

When `bounce_flag` transitions from -1 to +1 within a step, use the exit-phase reward for that step. No blending -- clean switch.

### 2. Running return normalization

Normalize returns by their running standard deviation to keep the value function's regression target in a learnable range.

**Algorithm:** Welford's online method tracking mean and variance of observed per-episode returns.

**Application:**
- After computing shaped rewards for an episode, divide all rewards by `max(return_std, 1e-8)`
- Normalize variance only, not mean -- shifting rewards changes the optimal policy
- Warmup: skip normalization for the first `norm_warmup_episodes` (default 64) until statistics stabilize
- Applied identically in PPO (before GAE) and SAC (before replay buffer insertion)

**Checkpoint:** Save `count`, `mean`, `M2` (Welford state) in the `.pt` checkpoint alongside model weights.

**Not normalized:** validation and final-eval costs. Those use raw `compute_cost` for cross-run comparability.

### 3. Observation normalization

Normalize the observation vector using running per-feature statistics, then bake the transform into the exported model weights so the Rust runtime needs no changes.

**Collection time:**
- Track per-feature running mean and variance (Welford's algorithm)
- Normalize: `obs_norm = (obs - mean) / max(std, 1e-8)`, clipped to [-10, 10]
- Policy and value networks see normalized observations
- Same statistics shared between policy and value networks

**Export time (bake into weights):**
- The first linear layer performs `y = W @ x + b` where `x` is raw observation
- With normalization: `y = W @ ((x - mu) / sigma) + b = (W / sigma) @ x + (b - W @ mu / sigma)`
- Set `W_new = W / sigma` (broadcast division per input feature) and `b_new = b - W @ (mu / sigma)`
- The exported JSON model produces identical outputs on raw inputs with zero Rust changes

**Checkpoint:** Save per-feature `count`, `mean`, `M2` arrays.

### 4. TOML configuration

New `[rl.reward]` section replacing the old PBRS fields:

```toml
[rl.reward]
# Capture phase weights
corridor_weight     = 0.1
energy_rate_weight  = 0.05
constraint_weight   = 0.2

# Exit phase weights
apoapsis_weight     = 0.2
eccentricity_weight = 0.1

# Normalization scales (for fixed-scale component normalization)
energy_scale        = 1.0e6   # J/kg, for energy rate normalization

# Return normalization
normalize_returns      = true
normalize_obs          = true
norm_warmup_episodes   = 64
```

Old fields removed: `shaping_enabled`, `shaping_alpha`, `energy_scale` (repurposed), `pdyn_scale`.

### 5. Implementation scope

#### Files modified

| File | Change |
|---|---|
| `rewards.py` | Rewrite: `StepRewardCalculator` (phase-aware), `ReturnNormalizer` (Welford), `ObsNormalizer` (Welford + bake) |
| `config.py` | Update `RewardConfig` dataclass for new TOML fields |
| `train.py` | Wire new reward calculator + normalizers into PPO and SAC loops |
| `export.py` | Add `bake_obs_normalization(policy, obs_normalizer)` that folds affine transform into layer 0 |
| `rl_common.toml` | New `[rl.reward]` defaults |
| `test_rewards.py` | Rewrite for new API: step reward components, normalization, bake-in correctness |

#### Files not modified

- No Rust changes. Obs normalization is baked into exported weights.
- Aux channel stays as-is (energy/pdyn needed for energy rate).
- Validation/final-eval path unchanged (raw `compute_cost`).
- `ppo.py`, `sac.py` -- reward normalization is applied before data enters these modules.

### 6. Invariants

- Validation gate and final evaluation use raw `compute_cost` -- never affected by reward shaping or normalization.
- Exported `best_model.json` is self-contained -- runs on the Rust runtime with no normalization code.
- The bake-in transform is exact (linear algebra, no approximation).
- Obs normalization statistics are frozen at export time (no drift between training and deployment).
- Per-step reward components are bounded by construction (obs features are normalized in `build_nn_input`, aux energy rate is clamped).
