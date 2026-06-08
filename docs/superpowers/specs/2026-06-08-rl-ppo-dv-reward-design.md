# DV-inferred reward for dense PPO

Date: 2026-06-08
Branch: `feature/rl-ppo-dv-reward` (off `feature/parameter_sweep`)

## Motivation

PSO empirically beats PPO/SAC on this problem. A plausible cause: the RL per-step
shaping potential is built from *proxy* quantities (corridor pdyn error, energy-gain,
apoapsis/eccentricity tracking) that only loosely correlate with the true objective,
which is correction delta-v. The reference trajectory the corridor term tracks is itself
a PSO/GA artifact, so the RL agent is being shaped toward a hand-tuned proxy rather than
toward minimum delta-v.

The simulator already computes, every tick, a smooth always-defined estimate of the
remaining correction budget: `maneuver::predicted_dv_for_nn` returns `[dv1, dv2, dv3]`
(energy-closing vis-viva burn, periapsis correction, inclination plane change) on the
*current osculating orbit*. This is, by construction, an estimate of the cost-to-go.

Using it as the shaping potential `Phi = -(weighted dv1+dv2+dv3)` makes the shaping a
potential-based approximation of `-V*` (the negative optimal value function). Per
Ng, Harada & Russell (1999), any potential-based shaping `F = gamma*Phi(s') - Phi(s)`
leaves the optimal policy set unchanged; choosing `Phi ≈ -V*` is the textbook-densest,
most informative such shaping. This directly aligns the dense reward with the terminal
objective the agent is ultimately graded on.

## Scope

Revisit PPO on a **dense** architecture only (no recurrent/attention layers). One new
reward mode, one Rust aux-channel extension, one training config. Non-destructive:
the existing phase-aware potential and `msr_aller_rl_train.toml` remain intact for A/B.

## Resolved design decisions

1. **Replace, not augment** the orbital-tracking potential. The DV potential supersedes
   the corridor / energy-gain (capture phase) and apoapsis / eccentricity (exit phase)
   terms. It does NOT replace the thermal-proximity term (see #6).
2. **Weighted sum** combination: `Phi_dv = -(w1*dv1 + w2*dv2 + w3*dv3)`, TOML knobs
   `dv1_weight`/`dv2_weight`/`dv3_weight` default `1.0` (= physical total delta-v budget,
   since all three are additive m/s). Down-weighting lets the user suppress a component
   the policy cannot influence (e.g. plane-change `dv3`).
3. **Raw DV via the aux channel.** The DV must reach the reward in raw m/s for the
   weighted-sum-of-budgets semantics to hold. The observation vector from `build_nn_input`
   is NN-normalized (asinh/affine-compressed), so the obs path would corrupt the units.
   The aux channel already carries raw physical quantities (`energy`, `pdyn`) for exactly
   this reason; we extend it. No change to obs/policy for the reward's sake.
4. **Policy observation = the 17-input atan2 mask**
   `[0,2,3,5,6,7,11,12,18,19,27,28,29,30,32,33,34]`. This mask already includes 32/33/34,
   so the policy observes the (normalized) cost-to-go it is being rewarded for reducing.
   Dense arch `17 -> 24 -> 12 -> 2`, `swish/swish/asinh`, `atan2_signed` head.
5. **Selectable mode, not deletion.** `[rl.reward] potential = "dv" | "phase_aware"`,
   default `"phase_aware"` so every existing config is bit-identical. The new config sets
   `"dv"`.
6. **Keep the dense thermal-proximity term.** The DV signal is blind to heat-flux/heat-load
   limits; relying on the terminal `compute_cost` penalty alone is a sparse, delayed teacher
   that can let the policy ride the heat-flux limit during capture. So in DV mode:

   ```
   Phi = -(w1*dv1 + w2*dv2 + w3*dv3) - constraint_weight*(hf_frac^2 + hl_frac^2)
   ```

   where `hf_frac`/`hl_frac` are the `[0,1]`-rescaled heat-flux/heat-load fractions already
   read from obs in the existing potential (indices 6, 7).
7. **Terminal cost unchanged.** `compute_terminal_cost` (= `compute_cost`, real DV +
   constraint penalties) is still added at the done step (`train.py:547`) and shares
   return normalization with the shaped stream.

## Data flow

```
SimState (per env, per tick)
  -> elements::from_spherical(nav-estimated pos/vel)        [env.rs helper]
  -> predicted_dv_for_nn(orbit, target_orbit, parking_orbit, planet) -> [dv1,dv2,dv3] (raw m/s)
  -> aux row = [energy, pdyn, dv1, dv2, dv3]                 (N,5) f32
collect_rollout (train.py)
  -> step_calc.step_reward(obs, next_obs, aux_cur, aux_next) reads aux[:,2:5]
  -> Phi_dv mode: gamma*Phi(next) - Phi(cur)
  -> on done: += -compute_terminal_cost(final_record)
  -> ret_norm.normalize(...)                                 (scale handled here)
```

PBRS invariant: the integrated shaped reward over an episode telescopes to
`gamma^T*Phi(terminal) - Phi(initial)`, a policy-independent offset, so the optimum is
preserved for any choice of `dv*_weight`/`constraint_weight`. Weights affect learning
dynamics and the density of the gradient, not the optimal policy.

## Component changes

### Rust: `src/rust/aerocapture-py/src/env.rs`

- New free fn `predicted_dv_for_state(state: &SimState, data: &Arc<SimData>, config: &SimInput) -> [f64; 3]`
  mirroring `build_obs_for_env`'s orbit construction:
  `elements::from_spherical(nav.position_estimated[0..3], nav.velocity_estimated[0..3], &config.planet)`
  then `maneuver::predicted_dv_for_nn(&orbit, &data.target_orbit, &data.parking_orbit, &config.planet)`.
  `nav = state.last_nav_output()`. This recomputes the same closed-form `build_nn_input`
  produces at indices 32-34 (pre-normalization) -- cheap, no new state plumbing.
- `step()` Rayon closure: extend the captured aux tuple from `[f64; 2]` to `[f64; 5]`:
  `[nav.energy_estimated, nav.dynamic_pressure_estimated, dv[0], dv[1], dv[2]]`, computed
  pre-reset (same as the existing energy/pdyn capture). Update the `outcomes` tuple type
  `(bool, Option<TerminalOutcome>, [f64; 5])` and all match arms.
- `step()` aux array: `PyArray2::<f32>::zeros(py, [self.n_envs, 5], false)`; fill 5 columns.
- `build_aux()` (used by `reset()`): `[self.n_envs, 5]`; fill `energy, pdyn, dv0, dv1, dv2`
  via the new helper.
- Rebuild PyO3 from repo root with `--manifest-path` (per project rule; subcrate builds go
  stale).

### Python reward: `src/python/aerocapture/training/rl/rewards.py`

- `StepRewardCalculator` gains fields: `potential: str = "phase_aware"`,
  `dv1_weight: float = 1.0`, `dv2_weight: float = 1.0`, `dv3_weight: float = 1.0`.
- `__post_init__`: when `potential == "dv"`, the required obs indices reduce to the thermal
  pair only (`_IDX_HEAT_FLUX_FRAC=6, _IDX_HEAT_LOAD_FRAC=7`). The corridor/sma/pdyn indices
  AND the bounce flag (15) are NOT required -- the DV potential is phase-agnostic
  (`predicted_dv_for_nn` is smooth across the bounce), and index 15 is not in the 17-input
  atan2 mask, so requiring it would break construction. Validate
  `potential in {"phase_aware", "dv"}`. (The existing `phase_aware` mode keeps requiring all
  six indices incl. 15, which is why it is only usable with masks that contain them, e.g. the
  old `[0..20]` mask.)
- New `_potential_dv(obs, aux)`:
  ```
  hf = (obs[:, col(6)] + 1)/2 ; hl = (obs[:, col(7)] + 1)/2
  dv1, dv2, dv3 = aux[:,2], aux[:,3], aux[:,4]
  return -(w1*dv1 + w2*dv2 + w3*dv3) - constraint_weight*(hf**2 + hl**2)
  ```
  Note: `aux` is now `(N,5)`. The DV potential is *not* phase-gated -- `predicted_dv_for_nn`
  is smooth across the `e=1` / bounce boundary, so no `in_capture`/`in_exit` split.
- `_potential` dispatches on `self.potential`. `step_reward` signature unchanged.

### Config: repurpose `configs/training/msr_aller_nn_atan2_ppo_train.toml`

The file currently does not inherit `rl_common.toml` (no `[rl]` block -> cannot drive RL),
carries the PSO-only `scaffolding = "live"` key, and has a stale "+1.5 sentinel" comment.
Repurpose it into a working RL config:

- `base = ["../missions/mars.toml", "common.toml", "rl_common.toml"]`.
- Drop `scaffolding` from `[guidance.neural_network]` (PSO concept; RL ignores it). Keep
  `mode = "full_neural"`, `output_parameterization = "atan2_signed"`.
- Fix the stale DV comment (no sentinel; smooth across `e=1`).
- Keep the 17-input mask, the `[network.normalization]` block, and the dense
  `17->24->12->2` (`swish/swish/asinh`) architecture.
- `[data] neural_network = "training_output/neural_network_atan2_rl/best_model.json"`,
  `results_suffix = ".train_nn_atan2_rl"`.
- `[rl.reward]` overrides on top of `rl_common.toml`:
  ```
  potential   = "dv"
  dv1_weight  = 1.0
  dv2_weight  = 1.0
  dv3_weight  = 1.0
  constraint_weight = 0.2
  ```
- Keep `[navigation]` and `[guidance.command_shaping]` (live under full_neural).

Deploy/scheme registration: add `neural_network_atan2_rl` to `compare_guidance.SCHEMES` +
`_NN_DEPLOY_SCHEMES` and a `train_all.sh` alias if a head-to-head is wanted (optional; can
follow in a later change).

### Wiring: `src/python/aerocapture/training/rl/config.py` + `train.py`

- `RewardConfig`: add `potential: str = "phase_aware"`, `dv1_weight/dv2_weight/dv3_weight:
  float = 1.0`. `RewardConfig(**rl.get("reward", {}))` then parses them; unknown keys still
  raise (validation preserved).
- `_build_shaper_and_norms` (`train.py:177`): pass `potential=cfg.reward.potential`,
  `dv1_weight=...`, etc. into `StepRewardCalculator`.
- SAC shares the same `step_calc` and aux -> DV mode works for SAC for free (not the focus,
  but no extra work and no breakage).
- `export_v2_policy_to_json` already writes `output_param`; confirm the deployed
  `best_model.json` carries `atan2_signed` so the Rust runtime decodes correctly.

## Testing

- `tests/rl/test_rewards.py`:
  - `potential="dv"`: telescoping identity (sum of `gamma*Phi'-Phi` over a synthetic
    trajectory equals `gamma^T*Phi_T - Phi_0`), sign (reward positive when total dv
    decreases), weight linearity (doubling `dv2_weight` doubles dv2's contribution),
    thermal term present.
  - `potential="phase_aware"` path unchanged (regression).
  - `__post_init__` index-requirement relaxation for `"dv"`; invalid `potential` raises.
- Aux shape: `(N,5)` from `reset()` and `step()`; columns 2-4 are raw, finite, and shrink
  as the orbit approaches the target (sanity on a captured seed).
- Rust unit test in `env.rs` (or an integration test): `predicted_dv_for_state` equals the
  raw indices 32-34 that `build_nn_input` computes for the same state (extract raw via a
  full-width mask, pre-normalization) so obs and reward provably agree on the DV.
- PPO smoke: a few updates on the new config produce a finite, loadable `best_model.json`
  (mirrors `test_gru_ppo_smoke.py` style; dense + atan2 + DV reward end to end).

## Risks / open items

- **Scale balance shaped-vs-terminal.** `dv1` is O(hundreds-thousands) m/s pre-capture;
  the per-step potential difference can dominate the terminal until return normalization
  warms up (`norm_warmup_steps`). Mitigation: return normalization already shares both
  streams; if needed, `dv*_weight` are the tuning handle. PBRS guarantees the optimum is
  invariant regardless.
- **DV from nav estimate, not truth.** `predicted_dv_for_state` uses the navigation-estimated
  orbit (same as the obs the policy sees). This is intentional and consistent; PBRS only
  requires `Phi` be a deterministic function of state, which a nav-derived quantity is.
- **`predicted_dv_for_nn` discontinuity sanity.** CLAUDE.md asserts dv1/dv2/dv3 are smooth
  across `e=1` with no sentinel; the Rust unit test above exercises this implicitly via the
  obs-vs-aux agreement check across capture and hyperbolic states.

## Final step

After implementation and verification, invoke the `smart-commit` skill, instructing it to
take the whole `feature/rl-ppo-dv-reward` branch into account.
