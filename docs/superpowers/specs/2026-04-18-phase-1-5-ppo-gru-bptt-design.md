# Phase 1.5 -- PPO-GRU with Truncated BPTT

**Date:** 2026-04-18
**Status:** Design approved, ready for implementation planning
**Parent effort:** LSTM / Transformer / Mamba architectures for neural guidance (see `TODO.md`).
**Predecessors:** Phase 0 (stateful NN runtime infrastructure, PR #37), Phase 1 (PSO-GRU MVP, `feature/gru-mvp`).

## 1. Context

Phase 0 shipped the stateful infrastructure (JSON v2, `NnState`, `LayerState` enum, `V2Policy` PyTorch mirror, cross-language equivalence at machine epsilon). Phase 1 landed the first stateful layer type (GRU) and exercised it end-to-end through PSO: the Python training stack now reads `[[network.architecture]]` from TOML, dispatches `nn_param_specs_from_v2` for PSO chromosome bounds, writes v2 JSON via `aerocapture_rs.flat_weights_to_json`, and the Rust runtime consumes `Dense -> GRU -> Dense` inside the normal neural_network guidance dispatch. Per-env hidden state already resets correctly with the Rust-side `build_sim_state` on `BatchedSimulation` auto-reset.

The Phase 1 spec explicitly deferred PPO-GRU: the RL rollout pipeline assumes a stateless policy (time-flattened `RolloutBuffer`, per-step `policy.forward_mean_logstd(obs)` with no hidden-state threading, stateless value bootstrap). Phase 1.5 lifts that assumption and adds truncated backpropagation-through-time (BPTT) so a recurrent policy can be trained by PPO on identical MC seeds to the PSO-GRU baseline. The scientific purpose of the phase is to produce the second row of the paper's training-axis × architecture-axis grid: given that PSO-GRU already beats PSO-MLP (or doesn't), does BPTT close the gap further, or does the 600-step aerocapture episode length favour PSO's global search over PPO's local gradient?

## 2. Scope

**In scope:**

- Unify the RL path on `V2Policy`. Retire `GaussianPolicy` from the RL training loop (the class stays in `policy.py` for legacy v1 artifact replay; nothing in `train.py` instantiates it after this phase).
- Extend `V2Policy` with three methods that replace `GaussianPolicy`'s sampling + update surface:
  - `forward_mean_logstd(obs, state) -> (mean, log_std, new_state)` (state-threaded).
  - `sample(obs, state) -> (bank, raw, log_prob, new_state)` (replaces the inline `raw = mean + std * eps` in `train.py`).
  - `evaluate(obs_seq, state_0, dones_seq, raw_seq) -> (log_probs_seq, entropy_seq)` (BPTT forward over a chunk).
- Extend `RolloutBuffer` with three new arrays: per-env `h_initial`, per-env `h_final`, per-step `states` tracking the hidden state trajectory so chunked BPTT can seed each chunk from the correct detached state. For dense-only policies these arrays are empty `list[None]` placeholders so feedforward PPO pays zero overhead.
- Rewrite the PPO rollout collect loop to thread per-env hidden state across steps, honouring the Rust auto-reset semantics (zero the state on `done`).
- Rewrite the PPO update loop to run chunked truncated BPTT: split each env's `rollout_steps` trajectory into `rollout_steps / bptt_length` chunks, detach hidden state at chunk boundaries, and BPTT within each chunk. Minibatches partition the env axis; the time axis inside each minibatch stays intact.
- Keep `ValueNetwork` feedforward (no recurrent critic). The critic consumes the same obs as today; no hidden-state input.
- Keep `compute_gae` unchanged (advantages are computed once at rollout time from feedforward value predictions; chunking happens inside the update loop only).
- Keep `target_kl` early-stop (per-epoch mean `approx_kl` threshold).
- Config: new `[rl.ppo] bptt_length` knob (default 32). `[[network.architecture]]` in the RL config drives whether the instantiated `V2Policy` is recurrent.
- Training config `configs/training/msr_aller_gru_ppo_train.toml` mirroring the Phase 1 PSO-GRU config but with the PPO block.
- `compare_guidance` registration: new scheme `neural_network_gru_ppo` dispatching through the Rust `neural_network` runtime (same as `neural_network_rl` and `neural_network_gru_pso`).
- Test coverage: unit tests on `V2Policy.sample` / `evaluate` shapes and grad flow; hidden-state-reset-on-done test on a mocked env; chunk-size equivalence test (single-chunk BPTT vs multi-chunk BPTT must agree on the forward pass bit-for-bit, with gradients clearly differing); feedforward PPO regression test on a seed-pinned dense arch; cross-language equivalence extension confirming the exported PPO-GRU v2 JSON is bit-identical under `aerocapture_rs.nn_forward`; `@pytest.mark.slow` 5-update PPO-GRU smoke test wired into the `python-pyo3` CI job.

**Out of scope for Phase 1.5:**

- SAC-GRU. R2D2-style sequence replay is deferred. The `RolloutBuffer` hidden-state axis is designed to be mirrorable by a future `SequenceReplayBuffer`; no eager implementation.
- Recurrent critic. The feedforward critic stays.
- LSTM-PPO, Attention-PPO, SSM-PPO. Phase 2+.
- Per-layer activation-aware init for GRU (still deferred from Phase 1 as a carry-over).
- Widening `load_policy_from_json` to accept v1 JSON (still deferred from Phase 0).

## 3. Architecture

### 3.1 Policy

`V2Policy` becomes the single RL policy class. Three new methods:

```python
def forward_mean_logstd(self, obs: Tensor, state: list) -> tuple[Tensor, Tensor, list]:
    """Single-step forward. Returns (mean, log_std, new_state).
    mean, log_std shape: (batch, action_dim). log_std is clamped at min_log_std.
    """

def sample(self, obs: Tensor, state: list) -> tuple[Tensor, Tensor, Tensor, list]:
    """Reparameterized sample. Returns (bank, raw, log_prob, new_state).
    bank = atan2(raw[...,0], raw[...,1]) for atan2 output_interpretation.
    log_prob = Normal(mean, std).log_prob(raw).sum(-1).
    """

def evaluate(
    self,
    obs_seq: Tensor,            # (T, B, obs_dim)
    state_0: list,              # list of (B, H) tensors; detach()'d caller-side
    dones_seq: Tensor,          # (T, B) bool
    raw_seq: Tensor,            # (T, B, action_dim)
) -> tuple[Tensor, Tensor]:
    """BPTT forward over a time chunk. Returns (log_probs_seq, entropy_seq)
    each of shape (T, B). Within-chunk episode boundaries are handled by
    zeroing each layer's state on dones_seq[t]=True before step t+1.
    """
```

`evaluate` is the only method that unrolls a sequence. Its body is a plain Python `for t in range(T)` over the step axis, calling each layer's stateful `forward(x, s) -> (y, s')`. Autograd records the graph; the caller controls where detachment happens (state_0 is detached before passing in, chunk boundaries are the caller's responsibility).

`V2Policy.log_std` already exists as a state-independent `nn.Parameter` of shape `(action_dim,)`. It is not exported to JSON (exploration noise is training-only). The Phase 0 `export_v2_policy_to_json` contract is unchanged.

`GaussianPolicy` is not modified or deleted. `train.py` stops instantiating it; any code still referencing it for legacy analysis continues to work.

### 3.2 Critic

`ValueNetwork` is unchanged. Forward pass is stateless `value(obs) -> v_pred`. The rollout collect loop keeps calling it step-by-step as today. Truncation bootstrap uses `V(terminal_obs)` with no hidden state (reasoning: recurrent critic adds a second RNN to debug, and the dominant signal for the paper is whether recurrent *policy* helps -- the critic is an auxiliary function, not the artifact being compared).

### 3.3 Rollout buffer

Current layout: `(n_steps, n_envs, ...)` time-first. Phase 1.5 keeps the time-first layout (env is the natural minibatch axis for BPTT; splitting minibatches on the env axis keeps each sequence intact).

Three new fields:

```python
@dataclass
class RolloutBuffer:
    # existing fields unchanged:
    obs: ndarray            # (T, N, obs_dim)
    raw_actions: ndarray    # (T, N, action_dim)
    log_probs: ndarray      # (T, N)
    rewards: ndarray        # (T, N)
    values: ndarray         # (T, N)
    dones: ndarray          # (T, N) bool

    # Phase 1.5 additions (empty when policy is dense-only):
    h_initial: list[ndarray]      # per-layer state at t=0 of this rollout; list[(N, H)]
    h_final: list[ndarray]        # per-layer state at t=T (seed for next rollout)
    states: list[ndarray]         # per-layer per-step state; list[(T, N, H)]
```

The `list[...]` wrapper handles the heterogeneous per-layer shape: each layer contributes one entry. For dense layers the entry is `None` (or a zero-size sentinel); for GRU the entry is an `(T, N, H)` array. This matches `NnState` in Rust (`Vec<LayerState>`) and `V2Policy.new_state` in Python (`list[Any]`), so no new wire format is needed.

`states[t]` stores the state *before* step t was computed -- i.e. the state that was passed to `policy.sample(obs[t], state)`. Chunk k (k >= 1) reads `states[k * bptt_length]` as its detached starting state.

### 3.4 Rollout collect loop

Per rollout:

```
h_current := h_final from previous rollout  (zeros at training start)
h_initial := h_current
for t in range(rollout_steps):
    obs_policy := obs_norm.normalize(obs) if obs_norm else obs
    states[t]  := h_current                           # store BEFORE forward
    bank, raw, log_prob, h_next = policy.sample(obs_policy, h_current)
    v_pred = value(obs_policy)
    next_obs, _, done, info, aux_next = env.step(bank)
    # terminal-obs-aware PBRS, terminal cost, return normalizer: unchanged
    buf.obs[t]          := obs
    buf.raw_actions[t]  := raw
    buf.log_probs[t]    := log_prob
    buf.rewards[t]      := shaped_reward
    buf.values[t]       := v_pred
    buf.dones[t]        := done & ~truncated          # true termination only
    next_values[t]      := V(terminal_or_continuing_obs)   # feedforward, unchanged
    h_current := zeros_like(h_next) if done[env] else h_next   # per-env
    obs := next_obs
    aux_cur := aux_next
h_final := h_current
```

The `zeros_like(h_next) if done` branch mirrors the Rust auto-reset: when the Rust env auto-resets at the end of step t, it rebuilds `GuidanceState` which rebuilds `NnState::for_model` (zeros). By zeroing the Python-side state under the same condition, Python's state trajectory matches Rust's -- so an exported PPO-GRU policy produces identical inference-time behavior to the training-time policy at episode boundaries.

Advantage computation (`compute_gae`) is unchanged. The value bootstrap path stays feedforward: `V(terminal_obs)` on truncation, `V(reset_obs)` on continuing steps, `0` on true termination (handled by the existing `dones` mask).

### 3.5 BPTT update loop

Per update epoch:

```
n_chunks := rollout_steps // bptt_length     # must divide evenly
env_indices := permutation(n_envs)
for mb_start in range(0, n_envs, envs_per_minibatch):
    mb = env_indices[mb_start : mb_start + envs_per_minibatch]
    h_chunk := [h_initial[layer][mb] for layer in layers]   # seed for chunk 0
    for c in range(n_chunks):
        lo := c * bptt_length
        hi := (c + 1) * bptt_length
        obs_seq    = buf.obs[lo:hi, mb]          # (L, |mb|, obs_dim)
        dones_seq  = buf.dones[lo:hi, mb]        # (L, |mb|)
        raw_seq    = buf.raw_actions[lo:hi, mb]
        old_lp_seq = buf.log_probs[lo:hi, mb]
        adv_seq    = normalized_advantages[lo:hi, mb]
        ret_seq    = returns[lo:hi, mb]

        # Detach the seed state so gradients do not flow across chunks.
        h_chunk_detached := [s.detach() for s in h_chunk]
        new_lp_seq, entropy_seq = policy.evaluate(
            obs_seq, h_chunk_detached, dones_seq, raw_seq,
        )

        # PPO clipped surrogate, value loss, entropy bonus on the flattened (L*|mb|) axis.
        policy_loss, value_loss, entropy = ppo_losses(...)
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optim.zero_grad(); loss.backward(); clip_grad_norm_; optim.step()

        # Advance h_chunk for the next chunk, except after the last one.
        # states[t] holds the state *before* step t, so states[hi] is the
        # state after stepping through the current chunk -- which is exactly
        # the seed for chunk c+1.
        if c < n_chunks - 1:
            h_chunk := states[hi, mb]     # stored by the collect loop
```

Two alternatives for seeding chunk c+1:

- **Buffer-read (chosen):** use the `states[hi, mb]` array already stored by the collect loop. One extra axis of storage, zero extra compute.
- **Re-forward:** run the policy forward again with `torch.no_grad()` to compute the chunk-end state. Zero extra storage, one extra forward pass per chunk.

We pick buffer-read. Memory for a `(T, N, H)` state array at `T=2048, N=64, H=32` is `2048*64*32*4 B = 16.8 MB` per layer -- negligible.

**Within-chunk episode boundary:** when `dones_seq[t] = True`, the state passed to step `t+1` inside `evaluate` is zeroed (per-env). This mirrors the collect-loop behavior exactly and means the BPTT graph correctly captures the Markov reset.

### 3.6 Minibatching and epochs

`minibatches` in the current `[rl.ppo]` config partitions the flattened `(T * N)` sample axis today. Under Phase 1.5 it partitions the env axis `N`. The cap is `n_envs` (no sub-env splits, sequences stay intact). For typical `n_envs = 64` and `minibatches = 4`, each minibatch has 16 envs and `bptt_length = 32` steps, so each gradient step sees `16 * 32 = 512` samples -- same order as today.

`update_epochs` behaves as today (multiple passes over the rollout with reshuffled env indices between epochs). `target_kl` early-stop is unchanged: per-epoch mean `approx_kl` across all chunks and minibatches.

### 3.7 Advantage normalization

Advantages are computed once at rollout time (before the update loop) and normalized once globally across the full `(T, N)` sample population -- same as today. The chunked update consumes `advantages[lo:hi, mb]` slices; no per-chunk renormalization.

### 3.8 Dense-arch fast path

When the instantiated architecture is dense-only (`V2Policy.layers` has no `GruLayer`), `h_initial`, `h_final`, and `states` lists are populated with `None` sentinels. `policy.sample` and `policy.evaluate` thread `None` through every layer's `forward(x, state=None) -> (y, None)`. The BPTT loop still chunks and minibatches over envs, but with `bptt_length = rollout_steps` (one chunk), the graph reduces to the standard PPO graph. Regression test (seed-pinned) pins feedforward PPO output to the Phase 0 numbers within `1e-6`.

Setting `bptt_length = rollout_steps` at runtime is the default when no recurrent layer is present, regardless of what the TOML specifies -- one-chunk BPTT on a stateless policy is equivalent to the current time-flattened PPO update.

### 3.9 Warm-start

`--data-neural-network path/to/best_model.json` already loads v2 JSON. After Phase 1.5 it accepts both a PSO-GRU and a PPO-GRU checkpoint without code changes. This unlocks the warm-start matrix:

| warm-start FROM / TO          | PPO-MLP | PPO-GRU | SAC-GRU (Phase 1.6) |
|-------------------------------|---------|---------|---------------------|
| PSO-MLP (`training_output/neural_network/`)         | yes     | no (arch mismatch) | no |
| PSO-GRU (`training_output/neural_network_gru_pso/`) | no (arch mismatch)   | yes | yes |
| PPO-MLP (`training_output/neural_network_rl/`)      | yes     | no | no |
| PPO-GRU (`training_output/neural_network_gru_ppo/`) | no      | yes | yes |

Architecture mismatches at load time are a hard error in `load_policy_from_json` today (Pydantic shape validation). No change needed.

### 3.10 Export

`export_v2_policy_to_json` already handles Dense + GRU and bakes the obs-normalizer into layer 0 (raising `NotImplementedError` if layer 0 is GRU, per the Phase 0 section 3.5 invariant). The PPO export path in `train.py` switches from `export_policy_to_json(GaussianPolicy, ...)` to `export_v2_policy_to_json(V2Policy, ...)`. Legacy v1 JSON written by the PPO-MLP path is no longer produced; existing `best_model.json` artifacts on disk remain loadable (Rust supports both v1 and v2).

## 4. Data flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ rollout collect (per env, time-first):                               │
│                                                                      │
│   h = h_final_prev  ─────▶  [state][obs] → V2Policy.sample           │
│                                  ↓                                   │
│                          (bank, raw, log_prob, h_next)               │
│                                  ↓                                   │
│                          BatchedSimulation.step(bank)                │
│                                  ↓                                   │
│                          (obs', reward, done, info, aux)             │
│                                                                      │
│   buf.obs/actions/log_probs/values/dones/states[t] ← store           │
│   h ← zeros if done else h_next       (per-env)                      │
│                                                                      │
│   buf.next_values[t] ← ValueNetwork(terminal_or_continuing_obs)      │
│                                                                      │
│ after T steps: h_final_this ← h                                      │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ advantage computation (unchanged):                                   │
│   advantages, returns = compute_gae(rewards, values, next_values,    │
│                                     dones, gamma, lam)               │
│   adv_norm = (advantages - mean) / (std + 1e-8)                      │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ BPTT update loop (per epoch):                                        │
│   env_idx ~ Permutation(N)                                           │
│   for each env-minibatch mb:                                         │
│       h_chunk = h_initial[:, mb]                                     │
│       for c in range(n_chunks):                                      │
│           h_chunk = h_chunk.detach()                                 │
│           new_lp, ent = V2Policy.evaluate(obs[lo:hi, mb],            │
│                                           h_chunk,                   │
│                                           dones[lo:hi, mb],          │
│                                           raw[lo:hi, mb])            │
│           v_pred = ValueNetwork(obs[lo:hi, mb])       # feedforward  │
│           loss = ppo_clip + value_coef*value_mse                     │
│                  - entropy_coef*ent                                  │
│           loss.backward(); clip; optim.step()                        │
│           h_chunk = buf.states[hi, mb]         # next chunk's seed   │
│       accumulate approx_kl, clip_frac                                │
│   if mean(approx_kl_epoch) > target_kl: break                        │
└──────────────────────────────────────────────────────────────────────┘
```

## 5. Config

New knob in `[rl.ppo]`:

```toml
[rl.ppo]
# ... existing knobs unchanged ...
bptt_length = 32     # truncated BPTT window in timesteps.
                     # Must divide rollout_steps evenly.
                     # Ignored (set to rollout_steps) when the policy is dense-only.
```

New training config `configs/training/msr_aller_gru_ppo_train.toml` (mirrors `msr_aller_gru_pso_train.toml` structure):

```toml
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "neural_network"

[data]
neural_network = "training_output/neural_network_gru_ppo/best_model.json"
results_suffix = ".train_gru_ppo"

[network]
input_mask = [0, 1, ..., 22]   # match PSO-GRU for apples-to-apples comparison
output_interpretation = "atan2"

[[network.architecture]]
type = "dense"
input_size = 23
output_size = 32
activation = "tanh"

[[network.architecture]]
type = "gru"
input_size = 32
hidden_size = 32

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 2
activation = "linear"

[rl]
n_envs = 64
total_env_steps = 5_000_000

[rl.ppo]
rollout_steps = 2048
bptt_length = 32
update_epochs = 10
minibatches = 4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
entropy_coef = 0.001
value_coef = 0.5
max_grad_norm = 0.5
target_kl = 0.02
learning_rate = 3e-4
```

## 6. Scheme registration

Add `neural_network_gru_ppo` to `compare_guidance.SCHEMES` and `_NN_DEPLOY_SCHEMES` (same pattern as `neural_network_gru_pso`). Add a `train_neural_network_gru_ppo` function to `train_all.sh` with aliases `gru_ppo` / `nn_gru_ppo`. The PSO equivalent keeps its existing `gru` / `nn_gru` aliases.

## 7. Tests

| Test | Type | Purpose |
|---|---|---|
| `test_v2_policy_sample_shapes` | unit | `sample(obs, state)` returns correct shapes for dense-only and Dense->GRU->Dense |
| `test_v2_policy_evaluate_shapes_and_grad` | unit | `evaluate(...)` returns `(T, B)` outputs; gradient flows through `bptt_length` timesteps |
| `test_rollout_hidden_state_reset_on_done` | unit | Mocked env issues `done=True` at t=5; assert `h_current[t=6] == zeros` per env |
| `test_bptt_chunk_size_forward_invariant` | unit | One-chunk BPTT (`bptt_length = rollout_steps`) and multi-chunk BPTT (e.g. 32-step chunks) produce bit-identical forward outputs; gradient norms differ (pinned seed) |
| `test_feedforward_ppo_regression` | integration | Seed-pinned dense-only PPO run through V2Policy matches the pre-Phase-1.5 `GaussianPolicy` output within `1e-6` over 5 updates |
| `test_v2_rust_python_gru_ppo_equivalence` | integration | Export a PPO-GRU v2 JSON, load via `aerocapture_rs.nn_forward`, assert single-step output matches `V2Policy.forward(obs, zero_state)` < 1e-10 |
| `test_gru_ppo_smoke_5_updates` | slow, CI | 5-update PPO training on a minimal GRU arch (compact `n_envs`, `rollout_steps`); asserts `best_model.json` is v2 with `gru` present and `nn_forward` returns finite 2-tuple |

CI: append `tests/test_gru_ppo_smoke.py` to the `python-pyo3` job's pytest command.

## 8. Success criteria

1. All Phase 1 and Phase 0 tests continue to pass bit-identically.
2. Feedforward PPO regression (dense-only `neural_network_rl`) produces identical learned policy to the pre-Phase-1.5 baseline within `1e-6` over 5 seed-pinned updates.
3. `test_bptt_chunk_size_forward_invariant` passes: chunking does not change the forward computation.
4. Cross-language equivalence gate holds: PPO-GRU `best_model.json` roundtrips through `nn_forward` at machine epsilon.
5. 5-update PPO-GRU smoke test passes in the `python-pyo3` CI job, `<= 60s`.
6. Scientific gate (informal): a full `total_env_steps = 5_000_000` PPO-GRU run converges to a DV distribution within a factor of 2 of the PSO-GRU checkpoint on the reserved eval seed set. Measured by `compare_guidance --schemes neural_network_gru_pso neural_network_gru_ppo`. Not a merge-blocking gate -- informs whether the paper includes BPTT as a useful training axis or frames it as a "PSO wins" result.
7. Extensibility invariant holds: adding Phase 2 LSTM-PPO requires no changes to `train.py`, `problem.py`, `env.py`, `dispatch.rs`, or `runner.rs` -- only the layer-type-specific files enumerated in the Phase 1 extensibility contract plus zero changes to the rollout / update loops.

## 9. Non-goals (explicit)

- SAC-GRU: Phase 1.6. The `RolloutBuffer` hidden-state axis is extension-ready but no `SequenceReplayBuffer` is implemented.
- Recurrent critic: deferred.
- LSTM-PPO / Attention-PPO / SSM-PPO: Phase 2+.
- `bptt_length` that does not divide `rollout_steps` evenly: hard error at config parse time, not silently truncated.
- Per-layer activation-aware init for GRU: still deferred.
- Widening `load_policy_from_json` to accept v1 JSON: still deferred.

## 10. Risks and mitigations

**Memory blow-up during BPTT.** At `T = 2048, N = 64, H = 32, bptt_length = 32` the per-chunk forward graph holds `32 * 64 * 32 = 65536` activations per GRU gate times 3 gates plus dense activations -- tractable on a single GPU and trivially fine on CPU. At `bptt_length = 256` and larger `H`, this can grow. Mitigation: `bptt_length` is a user-tunable knob; if users pick too-large values they see OOM rather than silent correctness loss. The default 32 is safe for expected arch sizes through Phase 2.

**Chunk-boundary detach changes the gradient.** Detaching at chunk boundaries means the policy cannot learn from errors earlier than `bptt_length` steps ago. This is standard truncated BPTT and is well-known to work for recurrent policies with episode horizons comparable to `bptt_length`. Aerocapture episodes are ~600 steps, `bptt_length = 32` means ~19 chunks per episode -- aligns with R-PPO standard practice.

**Feedforward regression drift.** Small-numeric-differences risk when moving PPO from `GaussianPolicy` to `V2Policy`. Mitigated by the `test_feedforward_ppo_regression` seed-pinned gate. If forward outputs disagree by more than `1e-6` we debug the policy migration before landing the rest.

**BPTT with env-minibatching alters the gradient distribution.** Minibatching on the env axis instead of the full `(T*N)` axis changes the variance of gradient estimates. In practice R-PPO works well with this partitioning; if convergence is worse than expected the `minibatches` knob lets us reduce the partitioning.

**`bptt_length` not dividing `rollout_steps`.** Caught at config parse time (`rollout_steps % bptt_length != 0` raises `ValueError`). No silent chunk truncation. User picks compatible values (e.g. `rollout_steps = 2048, bptt_length ∈ {16, 32, 64, 128, ...}`).

## 11. Migration checklist

- [ ] `V2Policy.forward_mean_logstd / sample / evaluate` added; unit tests green.
- [ ] `RolloutBuffer` gains `h_initial`, `h_final`, `states`.
- [ ] Rollout collect loop threads state; zeroes on done.
- [ ] PPO update loop chunks; minibatches on env axis.
- [ ] `GaussianPolicy` references removed from `train.py` (class retained in `policy.py`).
- [ ] `export_policy_to_json` call sites replaced with `export_v2_policy_to_json`.
- [ ] Feedforward PPO regression gate passes.
- [ ] Cross-language equivalence gate extends to PPO-GRU.
- [ ] `msr_aller_gru_ppo_train.toml` + `compare_guidance` + `train_all.sh` registration.
- [ ] Smoke test in `python-pyo3` CI.
- [ ] CLAUDE.md + TODO.md + README.md updated.

## 12. References

- R-PPO: truncated BPTT on recurrent policies is the standard lift. See IMPALA (Espeholt et al. 2018, §4.1) for the chunked BPTT idiom and Kapturowski et al. 2019 (R2D2) for the related replay-buffer story that Phase 1.6 will follow.
- Hochreiter & Schmidhuber 1997 for BPTT foundations.
- Ng, Harada & Russell 1999 for the potential-based shaping that the existing reward structure already uses (unchanged in Phase 1.5).
- Phase 0 spec: `docs/superpowers/specs/2026-04-17-stateful-nn-runtime-infrastructure-design.md`.
- Phase 1 spec: `docs/superpowers/specs/2026-04-17-phase-1-gru-mvp-design.md`.
