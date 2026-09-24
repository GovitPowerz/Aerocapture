# RL trainer (`src/python/aerocapture/training/rl/`)

Reinforcement-learning training (PPO, experimental SAC) for the `neural_network` guidance
scheme: a parallel track to the population search, shelved after the paper's Section 5 result
(population search wins decisively; see "Paper baseline" below). RL-trained weights deploy via
the same `best_model.json` the population path produces, so `compare_guidance` treats RL as just
another scheme (`neural_network_rl`, `neural_network_atan2_rl`, `neural_network_gru_ppo`,
`neural_network_lstm_ppo`) feeding the Rust `neural_network` runtime. This package depends on
`torch_mirror/` ([the NN runtime README](../../../../rust/src/data/neural/README.md)), never the
reverse: deleting `rl/` leaves the population path intact. TOML knobs: `[rl]`, `[rl.reward]`,
`[rl.ppo]`, `[rl.sac]` (shared defaults in `configs/training/rl_common.toml`; see
[configs/README.md](../../../../../configs/README.md)). Specs:
`docs/design/2026-04-15-rl-nn-guidance-design.md`, `docs/design/2026-04-16-rl-reward-redesign.md`,
`docs/design/2026-04-18-phase-1-5-ppo-gru-bptt-design.md`, `docs/design/2026-06-08-rl-ppo-dv-reward-design.md`.

```bash
# Train a PPO policy
uv run python -m aerocapture.training.rl.train configs/training/msr_aller_rl_train.toml --algorithm ppo --total-steps 5000000

# Head-to-head RL vs population search on identical MC scenarios
uv run python -m aerocapture.training.compare_guidance --n-sims 500 --schemes neural_network neural_network_rl

# All schemes (train_all.sh): nn_rl runs after piecewise_constant
./train_all.sh nn_rl
```

CLI flags: `--algorithm {ppo|sac}`, `--total-steps N`, `--n-envs N`, `--rollout-steps N`,
`--validation-n-sims N`, `--validation-interval-updates N`, `--data-neural-network PATH`,
`--from-scratch`, `--learning-rate F`, `--clip-range F`, `--entropy-coef F`, `--min-log-std F`,
`--update-epochs N`, `--lr-anneal-start F`, `--target-kl F`, `--no-tui`, `--skip-report`,
`--resume DIR`. `--from-scratch` and `--data-neural-network` (warm-start from a population
champion; clears stale checkpoints) are mutually exclusive. The output dir is derived from the
TOML's `[data] neural_network` (`_resolve_output_dir`), which must be
`training_output/<scheme>/best_model.json`.

## Environment

`BatchedSimulation` (`src/rust/aerocapture-py/src/env.rs`) is the step-able pyclass: N
`SimState`s sharing one `Arc<SimData>`, Rayon-parallel ticks through `tick::act_tick` +
`tick::sense_tick`, auto-reset on episode end, GIL released via `py.detach()`, sub-tick events via
`promote_pending_crash_if_applicable` so truncation vs termination is surfaced as
`info["truncated"]`. `step()` returns `(obs, reward, done, info, aux)`; `aux` is `(N, 7)`:
`[energy_estimated, dynamic_pressure_estimated, predicted_dv1, predicted_dv2, predicted_dv3,
heat_flux_fraction, heat_load_fraction]` per env. The two thermal fractions are RAW fractions of
the limit (the reward never inverts the model-dependent NN input normalization); the 3 raw-m/s
`predicted_dv_for_nn` components are computed per env via `predicted_dv_for_state`, mirroring
`build_obs_for_env`'s orbit construction so they equal candidate inputs 32-34 pre-normalization.
`env.py::AerocaptureVecEnv` wraps the pyclass.

The env is the deployed NN's decision process, bit for bit (gate: `src/rust/tests/rl_env_parity.rs`,
bias and EKF navigation, shaping and OU noise on). Two rules make it so. (1) Timing: `reset()`
senses tick 0 and `step(a_k)` runs `act_tick(a_k)` then the next tick's `sense_tick`, so the obs
it returns is nav(t_{k+1}), the input the deployed NN reads when it picks a_{k+1}; terminal states
are sensed too (unless non-finite), so `terminal_observation` and the terminal aux describe the
state the episode ended in. (2) Injection: the action replaces the NN forward pass inside
`guidance_step` (its `policy_bank`), so it goes through the same activation gating and command
shaping as a deployed NN output, and the NN telemetry inputs (21-24, 27-30) record the shaped
command a deployed NN would have sent. The env therefore requires `guidance.type =
"neural_network"` in `full_neural` mode. Before this (up to the Section 5 campaign), `step()`
returned the navigation the action had been applied at (a one-tick lag) and the action bypassed the
shaper, with the telemetry recording the raw action. The headline dense_p515 champion flown through
that env gave +59 m/s mean DV (paired, n = 1000) over deploy (`experiments/obs_lag/`).

## Modules

- `config.py` — `RLConfig.from_toml` (`[rl]` / `[rl.reward]` / `[rl.ppo]` / `[rl.sac]`;
  `_parse_network_config` accepts both v1 `layer_sizes + activations` and v2
  `[[network.architecture]]`; raises at parse time when `rollout_steps % bptt_length != 0`).
- `policy.py` — the v1 `GaussianPolicy` (a PyTorch MLP mirroring the v1 `NeuralNetModel` JSON,
  `atan2(out[0], out[1])` bank mapping, `sample()` returning `(bank, raw, log_prob)` so SAC can
  Q-bootstrap on the 2D latent, `load_weights_from_json()` for warm-start) and `ValueNetwork`
  (the feedforward critic). PPO uses `torch_mirror.policy.V2Policy` (state-threaded
  `forward_mean_logstd` / `sample` / `evaluate`); SAC still uses `GaussianPolicy`.
- `export.py` — the v1 `export_policy_to_json` (format_version=1, architecture + per-layer
  `w`/`b`) and the rollout state packing; v2 export is `torch_mirror/export.py`
  (`export_v2_policy_to_json` bakes the obs-normalizer affine into layer 0).
- `ppo.py` — `RolloutBuffer` (+ `h_initial`, `h_final`, `states`: per-layer `ndarray | None`,
  zero overhead for dense-only rollouts; `states[t]` is the state BEFORE step t, so chunk c+1
  reads `states[c * bptt_length]` as its detached seed), `compute_gae` (per-step `next_values`
  bootstrap so truncated episodes use `V(terminal_obs)`), `ppo_update` (clipped surrogate + value
  + entropy + optional `target_kl` early stop) and `ppo_update_bptt` (chunked truncated BPTT:
  `rollout_steps // bptt_length` chunks, hidden state detached at chunk boundaries, minibatches
  partition the ENV axis, not time-flattened `(T*N)`; feedforward PPO runs through the same loop
  with `bptt_length = rollout_steps`; LSTM tuple state reconstructed at chunk boundaries when
  `ndim == 3`).
- `sac.py` — SAC with twin Q networks on the 2D Gaussian latent (entropy target `-dim(A) = -2`,
  the density the policy actually regularizes), `ReplayBuffer` with `state_dict` /
  `load_state_dict` persisted in the checkpoint; terminal obs stored in replay rather than the
  next episode's reset obs.
- `rewards.py` — `StepRewardCalculator` and `compute_terminal_cost` (below).
- `normalizers.py` — `ReturnNormalizer` (Chan's parallel Welford over per-env discounted-return
  streams; scales per-step rewards by return std after `norm_warmup_steps`; PPO applies it DURING
  rollout collection so advantages see a stable scale at GAE time) and `ObsNormalizer`
  (per-feature mean/std, baked into the first linear layer at export: `W_new = W/std`, `b_new =
  b - W@(mean/std)`, so the Rust runtime needs no change). Both checkpoint with the weights.
- `train.py` — the CLI and outer loop (CleanRL-style): rollout collection threading per-env
  hidden state across steps (seeds `buf.h_initial`, snapshots each pre-state into `buf.states`,
  zeros state rows per env on done, mirroring the Rust auto-reset; `_derive_hidden_shapes`,
  `_np_state_to_torch` / `_torch_state_to_np` pack multi-tensor states as stacked arrays, LSTM
  `(2, H)`), the reserved-seed validation gate (promotion on `val_rms_cost` alone: there is NO
  feasibility gate, no ADR-0005 analogue, so PPO can and does promote constraint-violating
  policies), checkpoint save/resume (`checkpoint.pt`), graceful Ctrl+C, the final MC evaluation
  summary. Warm-start goes through `load_policy_from_json` + `load_state_dict` with a pre-check
  that raises on layer-count mismatch.
- `report_rl.py` — the three-part PDF (Part 1 RL convergence panels; Parts 2/3 reused from the
  population report via `src/typst/report_rl.typ`). `logger.py` / `display.py` — JSONL + Rich TUI
  mirroring the population-trainer contract.

Artifacts under `training_output/<scheme>/`: `best_model.json`, `rl_training_*.jsonl`
(per-update metrics), `config_resolved.toml`, `checkpoint.pt`, `final_eval.parquet`, `report.pdf`.
Seeds: `RL_TRAINING_SEED_OFFSET = 3_000_000` is the default `seed_base`; the validation pool is
the shared `VALIDATION_SEED_OFFSET` stream (`training/seeds.py`).

## Reward structure

Potential-based per-step shaping (Ng, Harada & Russell 1999): `r_shape = gamma * Phi(s') -
Phi(s)`, computed by `StepRewardCalculator` (`StepRewardCalculator.potential` selects the mode), so
the optimal policy is provably preserved. With `[rl.reward] potential = "phase_aware"` (default), `Phi` is gated on the bounce flag (obs[15]):
the capture-phase potential combines corridor tracking (`corridor_weight * pdyn_error^2`), an
energy-gain penalty (`energy_rate_weight * max(delta_energy, 0)`) and constraint proximity
(`constraint_weight * (heat_flux_frac^2 + heat_load_frac^2)`); the exit phase replaces the
corridor/energy terms with apoapsis targeting (`apoapsis_weight * sma_error^2`) and eccentricity
reduction (`eccentricity_weight * max(ecc_excess, 0)^2`). With `potential = "dv"`, `Phi =
-(dv1_weight*dv1 + dv2_weight*dv2 + dv3_weight*dv3) - constraint_weight*(heat_flux_frac^2 +
heat_load_frac^2)`, the raw-m/s `predicted_dv_for_nn` components from `aux[:, 2:5]`: `Phi`
approximates `-V*` (cost-to-go), the densest optimum-preserving shaping, NOT phase-gated
(predicted DV is smooth across the bounce and never reads obs[15], so the 17-input atan2 mask
that omits index 15 is valid) while keeping the thermal-proximity term (DV is blind to heat
limits). `dv*_weight` default to 1.0; return normalization rescales the combined stream but not
the dv-vs-thermal ratio, so raise `constraint_weight` (or lower `dv*_weight`) to give the thermal
term more authority. The terminal reward adds the raw `compute_terminal_cost` (DV + constraint
penalties; it uses the cost defaults and ignores the TOML `[cost_function]`) on TERMINATED
episodes only; truncated (`max_time` timeout, ifinal=2) episodes bootstrap `V(terminal_obs)` /
`Q(terminal_obs)` instead (dones masked with `& ~truncated`), since adding the timeout virtual-DV
cost on top would double-count the terminal state in the value target; `episodic_*` logging
still records every outcome. All weights are TOML-configurable in `[rl.reward]`.

## Configs and schemes

- `configs/training/msr_aller_rl_train.toml` -> `neural_network_rl` (dense v1, phase-aware
  reward; `training_output/neural_network_rl/`).
- `configs/training/msr_aller_nn_atan2_ppo_train.toml` -> `neural_network_atan2_rl`: a dense
  atan2 policy (`17->24->12->2`, `output_parameterization = "atan2_signed"`) on the 17-input atan2
  mask `[0,2,3,5,6,7,11,12,18,19,27,28,29,30,32,33,34]` with `potential = "dv"`;
  `training_output/neural_network_atan2_rl/`.
- `configs/training/msr_aller_gru_ppo_train.toml` / `msr_aller_lstm_ppo_train.toml` ->
  `neural_network_gru_ppo` / `neural_network_lstm_ppo` (v2 recurrent policies, `[rl.ppo]
  bptt_length = 32`, `rollout_steps = 2048`).

## Paper baseline (Section 5)

Four leaf configs under `configs/training/paper/rl/`, `{dense_p515,gru_p1014}_ppo_{scratch,warm}.toml`,
inherit the atan2 PPO knobs and pin the exact protocol of the per-scenario population champions
they are compared to (`training_output/ou_marginal/ft_dense_p515` = Dense 17->18->9->2,
`ft_gru_p1014` = Dense(17->11)->GRU(11)->Dense(11->2); same 17-input mask and `[network]
normalization`, `atan2_signed`, the champion's `best_params.json` nav/shaping values written into
`[navigation]` / `[guidance.command_shaping]`, `noise_seeding = "per_draw"` explicit, `[data]
neural_network` under `training_output/paper/rl/<cell>/` so the output dir is the bundle key
`rl/<cell>`). `experiments/paper/18_rl_baseline.sh` runs the two cells of a pair concurrently and
is resumable (done = `final_eval.parquet`; `checkpoint.pt` = plain resume; else `--from-scratch`
/ `--data-neural-network <champion best_model.json>`). `tests/test_paper_rl_configs.py` asserts
each RL config against the bundled champion (architecture, mask, normalization, decoder,
scaffolding, regime, `[rl]` pools/budget, seed, output dir) and each bundled `rl/<cell>` model
against its config. Result (2M pool, n = 1000, per_draw): PPO scratch 237 mean / 316 CVaR95
(dense, 4.8% heat-flux violations) and 284 / 435 (GRU, 47% heat-flux + 31% g-load) vs champions
113 / 127 and 125 / 151; PPO warm-started deploys the champion (best validation checkpoint within
the first 10-20 updates) and then walks off it.

These four cells were trained in the pre-parity env (the lag and shaper bypass above), so they
optimized a different decision process than the one their validation gate and the quoted numbers
fly. The dense warm start begins as a ~181 m/s policy in that env against 113 m/s deployed
(`experiments/obs_lag/`), which is enough on its own to explain the walk-off. The re-quote on the
parity env is open (`TODO.md`).

## Gates

`tests/test_ppo_bptt_chunk_invariant.py` (one-chunk and multi-chunk BPTT produce bit-identical
forward values, dense / GRU / LSTM; detach changes gradients only), `tests/test_gru_ppo_smoke.py`
and `tests/test_lstm_ppo_smoke.py` (5 PPO updates on reduced archs; LSTM with `bptt_length=16` ->
4 chunks, exercising tuple-state detach), `tests/test_ppo_feedforward_regression.py` (5 updates
of the dense `msr_aller_rl_train.toml` through `V2Policy` with `bptt_length = rollout_steps`),
`tests/test_atan2_rl_ppo_smoke.py`, `tests/test_rl_config_bptt.py`,
`tests/test_rl_parse_network_v2.py`, `tests/test_rl_rollout_state_reset.py`, the PPO-rejection
tests for PSO-only cells (`tests/test_{transformer,mamba,mamba3,cfc_xlstm}_ppo_rejection.py`),
and the PPO-GRU export round-trip in `tests/test_v2_rust_python_equivalence.py`.
