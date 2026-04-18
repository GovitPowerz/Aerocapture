# TODO

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
| PSO            | yes (existing) |    yes     | yes |  yes |     yes     |  yes  |
| BPTT (PPO)     |       --       |     --     | yes |  yes |     yes     |  yes  |

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
- [ ] Per-layer activation-aware initialization for GRU (currently uniform-in-[0,1] via ParamSpec bounds; the dense-only `create_nn_initial_population` path with Xavier/He/LeCun is bypassed for v2 arches).

**Not in Phase 1 (explicit non-goals, landing in Phase 1.5+):**
- [ ] PPO-GRU (Phase 1.5: rollout-buffer hidden-state snapshots, truncation-aware bootstrap)
- [ ] LSTM / Window-MLP / Transformer / Mamba (Phases 2-4)

### Phase 2 -- LSTM + Window-MLP (cheap extensions on Phase 0/1 infra)
- [ ] LSTM: 4 gates, h+c state; PyTorch mirror; PSO + PPO configs
- [ ] Window-MLP: ring buffer via `NnState.window`, no new matmul; window-size ablation N in {4, 8, 16}

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
