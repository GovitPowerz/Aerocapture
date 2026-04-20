# TODO

- [ ] check the output_interpretation field utility
- [ ] check why ppo degrades a pso solution for gru and lstm.
- [ ] explore JEPA guidance

---

## Backlog

- [ ] Add neural counterparts for navigation and control
- [ ] Develop ESR (Earth Sample Return) mission profiles

---

## LSTM / Transformer / Mamba architectures for neural guidance

**End goal:** arXiv paper follow-up to 2008 AIAA paper on NN-for-aerocapture.
Compare feedforward (2008-style) against stateful architectures on identical MC seeds.

**Experimental grid (10 cells):**

|                | MLP (baseline) | Window-MLP | GRU | LSTM | Transformer | Mamba |
|----------------|:--------------:|:----------:|:---:|:----:|:-----------:|:-----:|
| PSO            | yes (existing) |    yes     | ✅  |  ✅  |     yes     |  yes  |
| BPTT (PPO)     |       --       |     --     | ✅  |  ✅  |     yes     |  yes  |

Primary trainer is PSO (shown to outperform PPO/SAC on this problem).
BPTT axis is an ablation on the four stateful architectures.

**Staging (S1 -- MVP first, then extend. Each phase ships spec + plan + PR.):**

### Phase 0 -- Stateful NN runtime infrastructure [DONE 2026-04-17]
- [x] JSON format v2: layer-type-tagged schema (`dense | gru | lstm | attention | ssm | layer_norm | window`), v1 still loads
- [x] Rust `NeuralNetModel` becomes stateful-capable; `NnState { h, c, window }` lives outside the model
- [x] Stateful forward threaded through: CLI sim runner, `BatchedSimulation` (per-env, reset on episode boundary), PSO MC eval
- [x] `to_flat_weights` / `from_flat_weights` + `param_spaces.py` extended for new layer types (PSO chromosome round-trip)
- [x] PyTorch mirror base class with JSON v2 export / obs-norm bake-in
- [x] Unit + integration tests: Rust forward == PyTorch mirror to 1e-10; episode reset invariant; JSON v1 unchanged

Shipped on branch `feature/stateful-nn-runtime` (21 commits).
Cross-language equivalence: max abs diff = 4.4e-16 (machine epsilon).
Full spec: `docs/superpowers/specs/2026-04-17-stateful-nn-runtime-infrastructure-design.md`.

### Phase 1 -- GRU MVP (validates the Phase 0 stack on one architecture) [DONE 2026-04-18]

Shipped on branch `feature/gru-mvp` (20 commits on top of the Phase 0 merge).
486 Python tests + full Rust suite pass, 0 failures. 6/6 guidance golden regressions bit-identical.
PSO-GRU training smoke test lives in the python-pyo3 CI job.

- [x] Task 1-7: Rust-side GRU (enum layers, GruLayer, LayerState::Gru, LayerWeights, JSON v2 read/write, `[[network.architecture]]` TOML parser, `aerocapture_rs.flat_weights_to_json` PyO3 helper)
- [x] Task 8: Python GruLayer torch module + GruSpec Pydantic + restored LayerSpec discriminated union
- [x] Task 9: Python export/load/encoding Gru branches + `evaluate.write_nn_json` routes through `aerocapture_rs.flat_weights_to_json` (Rust is now the single source of truth for NN weight serialization)
- [x] Task 10: Cross-language equivalence extensions (GRU case + input_mask case; closes Phase 0 carry-over #3)
- [x] Task 11: Training config `configs/training/msr_aller_gru_pso_train.toml` + `compare_guidance` registration (`neural_network_gru_pso`)
- [x] Task 12: Training smoke test (2 PSO gens on minimal Dense->GRU->Dense arch, 586 params; verifies v2 JSON + `nn_forward` roundtrip; wired into the python-pyo3 CI job)
- [x] Task 13: Full verification + smart-commit

Enabling Task 12 also threaded `[[network.architecture]]` through the Python training pipeline (one-time v2 plumbing): `NetworkConfig.architecture` + `_layer_n_params`, `train.py` dispatch to `nn_param_specs_from_v2` when the v2 arch is set, `create_nn_initial_population` / `compute_weight_stats` skipped for v2, `evaluate.write_nn_json` passes the v2 arch list through directly. After this payment, Phase 2+ layer types land by touching only the files enumerated in the Phase 0 extensibility contract -- no more changes to `train.py`, `problem.py`, `dispatch.rs`, or `runner.rs`.

Spec: `docs/superpowers/specs/2026-04-17-phase-1-gru-mvp-design.md`.
Plan: `docs/superpowers/plans/2026-04-17-phase-1-gru-mvp-plan.md`.

**Out-of-Phase-1 carry-overs (still deferred):**
- [ ] Widen `load_policy_from_json` to accept v1 JSON (currently v2-only). Materialize when Phase 1.5+ analysis code needs a legacy-artifact loader.
- [ ] Fix pre-existing `cargo clippy --workspace` warnings in `src/rust/aerocapture-py/src/lib.rs` (2x `type_complexity`, 1x `needless_range_loop`). `check_all.sh` scopes to `-p aerocapture`; separate one-line fix.
- [x] Per-layer activation-aware initialization for GRU (closed by Phase 2a: `init_v2_population` in `training/initialization_v2.py` retroactively applies tanh-Xavier gate matrices + small bias noise to GRU).

**Not in Phase 1 (explicit non-goals, landed or landing later):**
- [x] PPO-GRU (Phase 1.5: rollout-buffer hidden-state snapshots, truncation-aware bootstrap) [DONE 2026-04-18]
- [x] LSTM (Phase 2a: PSO + PPO-BPTT + activation-aware init + forget-bias-1) [DONE 2026-04-18]
- [ ] Window-MLP / Transformer / Mamba (Phases 2b-4)

### Phase 1.5 -- PPO-GRU + truncated BPTT [DONE 2026-04-18]

Shipped on branch `feature/gru-mvp` (16 commits on top of the Phase 1 payload, 32 total on the branch).
505 Python tests + full Rust suite pass, 0 failures. 6/6 guidance golden regressions bit-identical.
PPO-GRU training smoke test + feedforward PPO regression gate wired into the python-pyo3 CI job.

- [x] Task 1: V2Policy state-threaded methods (`forward_mean_logstd`, `sample`, `evaluate`); `_zero_state_where_done` helper with TypeError guard for future multi-tensor states.
- [x] Task 2: `RolloutBuffer` gains `h_initial`, `h_final`, `states` per-layer lists (None entries = zero-overhead dense fast path).
- [x] Task 3: `[rl.ppo] bptt_length` knob (default 32) + `RLConfig.from_toml` divisibility guard.
- [x] Task 4: `_parse_network_config` returns `(input_mask, architecture, input_dim, output_interpretation)`; accepts `[[network.architecture]]` in the RL path.
- [x] Task 5: PPO seed-model / validate / train_ppo migrated to `V2Policy`; warm-start via `model_io.load_policy_from_json` + layer-count pre-check. SAC stays on `GaussianPolicy` (Phase 1.6).
- [x] Task 6: Rollout collect loop threads per-env hidden state, zeros on done (mirrors Rust auto-reset), snapshots `buf.states[t]` for chunked BPTT seeding.
- [x] Task 7: `ppo_update_bptt` chunks the rollout into `rollout_steps // bptt_length` segments, minibatches on the env axis, detaches state at chunk boundaries. Chunk-size invariant test proves forward values are bit-identical across chunk counts; only gradients differ.
- [x] Task 8: `configs/training/msr_aller_gru_ppo_train.toml` (Dense(23->32) -> Gru(32,32) -> Dense(32->2), `bptt_length = 32`, PPO+BPTT). `compare_guidance` + `train_all.sh` register `neural_network_gru_ppo` with `gru_ppo` / `nn_gru_ppo` aliases.
- [x] Task 9: Cross-language equivalence extended with PPO-GRU export roundtrip (max abs diff 5.55e-17, machine epsilon).
- [x] Task 10: `@slow` PPO-GRU smoke test (~2s wall-clock) + CI registration.
- [x] Task 11: Feedforward PPO regression gate confirms V2Policy + `bptt_length = rollout_steps` preserves the dense-only training path.
- [x] Task 12: Full verification (Rust check_all, Python lint+tests, guidance golden regressions).
- [x] Task 13: smart-commit.

Spec: `docs/superpowers/specs/2026-04-18-phase-1-5-ppo-gru-bptt-design.md`.
Plan: `docs/superpowers/plans/2026-04-18-phase-1-5-ppo-gru-bptt-plan.md`.

**Out-of-Phase-1.5 carry-overs (deferred):**
- [ ] SAC-GRU (Phase 1.6: R2D2-style sequence replay + burn-in; `_validate_deterministic_v1` twin helper deletable once SAC migrates to V2Policy).
- [ ] Recurrent critic (deferred; feedforward critic mirroring policy trunk widths is fine for GRU-at-32-hidden).
- [x] Per-layer activation-aware init for GRU and LSTM (closed by Phase 2a).
- [ ] Widen `load_policy_from_json` to accept v1 JSON (Phase 0 carry-over; still deferred).

### Phase 2a -- LSTM MVP (PSO + PPO-BPTT) + activation-aware init [DONE 2026-04-18]

Shipped on branch `feature/lstm-mvp` (13+ substantive commits + 2 hygiene commits on top of main).
Cross-language equivalence: LSTM forward matches at machine epsilon (same ~1e-16 ceiling as Phase 1 GRU).
PSO-LSTM + PPO-LSTM + BPTT chunk-invariant LSTM smoke tests wired into the python-pyo3 CI job.

- [x] Rust `LstmLayer` + `Layer::Lstm` + `LayerState::Lstm { h, c }` + `TomlLayerSpec::Lstm`
- [x] `LayerWeights for LstmLayer` 4H flat ordering + JSON v2 + PyO3 verification (no Rust change needed -- delegated through from_flat_weights_v2)
- [x] Python `LstmLayer` torch module + `LstmSpec` pydantic + `_zero_state_where_done` tuple branch
- [x] `_lstm_specs` (asymmetric bias bounds for forget slice) + `config.py::_layer_n_params` + `_layer_output_size` lstm arms + export / load Lstm branches
- [x] `init_v2_population`: dense Xavier/He/LeCun, GRU tanh-Xavier + small bias noise, LSTM tanh-Xavier + forget-bias 1.0 on bias_ih only
- [x] Training configs `msr_aller_lstm_pso_train.toml` + `msr_aller_lstm_ppo_train.toml`
- [x] Cross-language equivalence test + PSO-LSTM + PPO-LSTM smoke tests (@slow, python-pyo3 CI)
- [x] PPO rollout buffer tuple state packing (`hidden_shapes` / _np_state_to_torch / _torch_state_to_np / ppo_update_bptt ndim==3 paths)

Spec: `docs/superpowers/specs/2026-04-18-phase-2a-lstm-mvp-design.md`.
Plan: `docs/superpowers/plans/2026-04-18-phase-2a-lstm-mvp-plan.md`.

**Out-of-Phase-2a carry-overs (still deferred):**
- [ ] SAC-GRU / SAC-LSTM (Phase 1.6; SAC stays on GaussianPolicy).
- [ ] Recurrent critic (Phase 1.5 carry-over).
- [ ] Widen `load_policy_from_json` to accept v1 JSON (Phase 0 carry-over).
- [ ] Fix pre-existing clippy warnings in `src/rust/aerocapture-py/src/lib.rs` (3 warnings).

**Closed by Phase 2a:**
- [x] Per-layer activation-aware initialization for GRU and LSTM (Phase 1 carry-over).

### Phase 2b -- Window-MLP (ring buffer, no new matmul) [DOING 2026-04-20 on feature/window-mlp]
- [ ] Rust `WindowLayer` + `Layer::Window` + `LayerSpec::Window { input_size, n_steps }` + `LayerState::Window { buffer: VecDeque<Vec<f64>> }` + `TomlLayerSpec::Window`
- [ ] `LayerWeights for WindowLayer` zero-param impl + JSON v2 + PyO3 test
- [ ] Python `WindowLayer` torch module + `WindowSpec` pydantic + `build_layer` PPO-rejection guard
- [ ] `_layer_param_specs` / `_layer_n_params` / `_layer_output_size` Window arms + `init_v2_population` no-op continue
- [ ] Training config `msr_aller_window_pso_train.toml` + `compare_guidance` + `train_all.sh` registration
- [ ] Cross-language equivalence test + PSO smoke test + PPO-rejection test (@slow python-pyo3 CI + @fast main CI)

Spec: `docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md`.
Plan: `docs/superpowers/plans/2026-04-20-phase-2b-window-mlp-plan.md`.

### Phase 3 -- Transformer
- [ ] Rust multi-head attention + layer norm + sinusoidal position encoding
- [ ] Causal window attention (fixed N=64 token buffer)
- [ ] Small arch (~10k params): 1 layer, d_model=32, 4 heads, FFN 64
- [ ] PyTorch mirror uses manual attention (not `nn.MultiheadAttention`) for 1-for-1 Rust equivalence
- [ ] PSO + PPO-Transformer training configs

### Phase 4 -- Mamba (S6)
- [ ] Rust SSM layer: input-dependent A, B, C; sequential scan at inference (no parallel scan needed)
- [ ] Arch (~15k params): 1 block, d_model=32, state=16
- [ ] PyTorch mirror: naive sequential scan (correct, slow, fine for 600-step episodes)
- [ ] PSO + PPO-Mamba training configs

### Phase 5 -- Paper artifact
- [ ] Unified 10-cell comparison on identical MC seeds, same reward / cost function
- [ ] Figures: per-cell DV CDF, corridor plots, convergence curves, param-count vs DV frontier
- [ ] Per-architecture sensitivity analysis via existing `sensitivity.py`
- [ ] arXiv draft in `paper/` (LaTeX, figures sourced from `training_output/`)
