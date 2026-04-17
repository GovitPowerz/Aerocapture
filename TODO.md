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

### Phase 0 -- Stateful NN runtime infrastructure
- [ ] JSON format v2: layer-type-tagged schema (`dense | gru | lstm | attention | ssm | layer_norm | window`), v1 still loads
- [ ] Rust `NeuralNetModel` becomes stateful-capable; `NnState { h, c, window }` lives outside the model
- [ ] Stateful forward threaded through: CLI sim runner, `BatchedSimulation` (per-env, reset on episode boundary), PSO MC eval
- [ ] `to_flat_weights` / `from_flat_weights` + `param_spaces.py` extended for new layer types (PSO chromosome round-trip)
- [ ] PyTorch mirror base class with JSON v2 export / obs-norm bake-in
- [ ] Unit + integration tests: Rust forward == PyTorch mirror to 1e-10; episode reset invariant; JSON v1 unchanged

### Phase 1 -- GRU MVP (validates the Phase 0 stack on one architecture)
- [ ] Rust GRU layer (3 gates, h state)
- [ ] PyTorch `GruPolicy` with manual unroll (1-for-1 with Rust)
- [ ] PSO training config `configs/training/msr_aller_gru_pso_train.toml`
- [ ] PPO-GRU: rollout buffer carries hidden state; truncation-aware bootstrap uses `V(terminal_obs)` with terminal h
- [ ] Register `neural_network_gru_pso` / `neural_network_gru_rl` in `compare_guidance.py`
- [ ] Validation gate + final MC eval on reserved seeds

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
