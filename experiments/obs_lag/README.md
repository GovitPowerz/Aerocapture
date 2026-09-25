# RL env vs deploy parity (2026-09-24)

Before the parity fix, `BatchedSimulation` differed from the deploy loop in two ways:

- **Observation lag.** `step(a_k)` returned nav(t_k), the navigation `a_k` had been applied at,
  instead of nav(t_{k+1}). The policy chose `a_{k+1}` from one-tick-old navigation. The telemetry
  inputs and the time channels were already current.
- **Action injected after the command shaper.** The action went straight to the pilot, skipping
  the command shaper (or the max-rate clamp when shaping is off). The `prev_bank_signed`
  telemetry inputs (22, 27, 28) recorded the raw action, while deploy records the shaped command.

`measure_obs_lag.py` measures both. All arms fly the same 1000 seeds (fresh 8M pool, legacy
regime, run_batch noise matched to the env's; see the script docstring). Paired ΔDV is per seed
against `deploy`.

## Headline dense_p515 (deployed champion flown through each env variant)

`deploy` = 120.29 mean / 115.03 p50 / 155.91 p95 / 191.37 CVaR95, 100% capture. On its own
noise path it reproduces the quoted 109.89. With shaping off, deploy moves by -0.05 ± 0.03
(paired), so the shaper itself is negligible for this cell.

| Env variant | Mean | p50 | p95 | CVaR95 | Paired ΔDV |
|---|---|---|---|---|---|
| pre-fix (lagged obs, action after shaper) | 178.99 | 175.80 | 230.51 | 253.84 | +58.7 ± 0.9 |
| lag fixed only | 182.40 | 180.87 | 224.91 | 236.36 | +62.1 ± 1.0 |
| injection fixed only (lagged obs) | 144.23 | 124.96 | 244.49 | 274.97 | +23.9 ± 1.2 |
| parity (both fixed) | 120.29 | 115.02 | 155.91 | 191.37 | 0.00 (max 2.06 on one seed: f32 obs) |

The injection mismatch dominates: the policy reads `prev_bank` telemetry it never saw in
training. With the injection fixed, the lag alone still costs +24 m/s mean and +84 m/s CVaR95.

## Pre-parity Section 5 PPO models: pre-fix env (their training env) vs deploy

These are the 2026-09-22 cells, archived at `training_output/rl_preparity_20260922/` (untracked)
and no longer bundled; the bundled `rl/<cell>` are the parity-env retrains of #154.

| Cell | Pre-fix env mean | Deploy mean |
|---|---|---|
| dense_p515_ppo_scratch | 218.89 | 234.31 |
| dense_p515_ppo_warm | 180.58 | 113.06 |
| gru_p1014_ppo_scratch | 280.74 | 289.44 |
| gru_p1014_ppo_warm | 129.28 | 127.70 |

PPO's validation gate flies deploy (`evaluate_cell`), but PPO optimizes the env. The dense warm
start is a 113 m/s policy by the gate and a 181 m/s policy in the env it trains in.

Retrained on the parity env (#154, 2026-09-25), the dense warm start starts at ~120 m/s in its
env and still walks off the champion (validation capture 83% by 17M steps on the 1M-offset
validation pool, per_draw), so the mismatch above
does not explain the walk-off. The parity-env numbers are the ones the paper quotes.

## Reproducing

The parity row and every deploy arm run on the current build. The pre-fix rows need the extension
built from `b620fe6c` (the commit before the fix). The two "fixed only" rows were intermediate
builds of the fix branch and are not reproducible from a commit.
