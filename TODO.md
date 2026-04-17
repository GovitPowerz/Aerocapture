# TODO

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

### Phase 1 -- GRU MVP (validates the Phase 0 stack on one architecture)
- [ ] Rust GRU layer (3 gates, h state)
- [ ] PyTorch `GruPolicy` with manual unroll (1-for-1 with Rust)
- [ ] PSO training config `configs/training/msr_aller_gru_pso_train.toml`
- [ ] PPO-GRU: rollout buffer carries hidden state; truncation-aware bootstrap uses `V(terminal_obs)` with terminal h
- [ ] Register `neural_network_gru_pso` / `neural_network_gru_rl` in `compare_guidance.py`
- [ ] Validation gate + final MC eval on reserved seeds

**Carried over from Phase 0 review (scaffolding tidy-ups to land alongside Phase 1):**
- [ ] Widen `load_policy_from_json` to accept v1 -- currently v2-only, so any Phase 1 analysis code that points it at a legacy `training_output/*/best_model.json` fails. Map v1's `layer_sizes`+`activations` arrays onto `list[DenseSpec]` internally.
- [ ] Adopt `LayerWeights` trait in the PSO write path or delete the trait. Today `evaluate.py` writes JSON v1 directly; `to_flat_weights` / `from_flat_weights` are exercised only by a unit test. Pick one: route the Python PSO chromosome through the Rust trait (unlocks PSO on GRU/LSTM/etc.), or drop the trait and extend `evaluate.py` with per-layer-type writers.
- [ ] Extend `tests/test_v2_rust_python_equivalence.py` with a non-None `input_mask` case. The mask-validation branch in `nn_forward` (`src/rust/aerocapture-py/src/lib.rs`) is uncovered; a 5-input with mask `[0, 2, 4]` routed into a 3-input first layer closes it.
- [ ] Promote `NnState::Clone` coverage from structural to behavioral once a stateful `LayerState` variant lands (Gru/Lstm/Window/Ssm). Current test only asserts `layer_states.len()` equivalence, which is meaningless while the only variant is `None`.
- [ ] Fix pre-existing `cargo clippy --workspace` warnings in `src/rust/aerocapture-py/src/lib.rs` (2x `type_complexity`, 1x `needless_range_loop`). Out of Phase 0 scope, but `check_all.sh` passes only because it scopes to `-p aerocapture`; a workspace-wide clippy gate would flag them.

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
