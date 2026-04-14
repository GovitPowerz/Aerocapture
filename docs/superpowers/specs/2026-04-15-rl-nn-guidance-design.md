# Reinforcement Learning for Neural Network Guidance

**Status:** Design (brainstormed 2026-04-15)
**Scope:** Parallel track alongside the existing pymoo GA pipeline for the `neural_network` guidance scheme. RL-trained weights are drop-in replacements for GA-trained weights via the existing `best_model.json` format.

## Motivation

The `neural_network` guidance scheme is currently trained by evolutionary search over static weights (pymoo SBX + polynomial mutation). The simulator is inherently step-able (per-tick nav -> guidance -> control -> integrator), so a reinforcement learning formulation is natural and may exploit temporal structure that weight-space GA cannot express. This design stands up RL as a first-class sibling of the GA flow: same runtime scheme, same `best_model.json`, same `compare_guidance` entry, same PDF report structure.

## Goals

- RL-trained policies deploy via the existing Rust `neural_network` guidance runtime without any Rust guidance-side changes (bit-compatible weights).
- Training infrastructure (logger, Rich TUI, PDF reporter, final-eval Parquet, checkpoint resume, validation gate, reserved seed pools) is shared between GA and RL paths.
- Head-to-head comparison via `compare_guidance.py` requires zero additional code.
- PPO is the v1 validated baseline; SAC is built but shipped experimental.

## Non-Goals

- Replacing GA as the default for `neural_network`. That decision is out of scope for v1 and contingent on measured results.
- Recurrent policies, curriculum learning, action=delta-bank or rate-command variants, multi-mission training, distributed training, TensorBoard export, NN architecture search, animation parity.

## Design Decisions (brainstorm record)

| # | Decision | Rationale |
|---|---|---|
| Q1 | Parallel track alongside pymoo (not research spike, not replacement) | Maximizes reuse of existing training infrastructure; enables head-to-head comparison. |
| Q2 | Stateful `BatchedSimulation` PyO3 pyclass (reset/step) | Only option that provides real sequential RL at throughput comparable to batched GA. |
| Q3 | PPO first, SAC second (same TOML pattern as pymoo's GA/CMA-ES/DE/PSO) | Env is fast, so PPO's sample inefficiency is cheap; SAC trivially added later. |
| Q4 | Potential-based shaping + terminal cost | Reuses `ref_trajectory.dat` from piecewise_constant training; provably policy-invariant per Ng et al. 1999. |
| Q5 | Bit-compatible with existing NN runtime (same 23-input vector, same `input_mask`, same MLP JSON format) | Drop-in replacement; zero Rust guidance changes; fair comparison. |
| Q6 | Fresh random dispersion per episode (domain randomization); reserved-seed validation gate | Standard RL practice; GA's curated-CDF was a response to a pathology that doesn't apply to RL's millions-of-episodes regime. |
| Q7 | Rust-side batched step + CleanRL-style custom PyTorch loop | Maximum throughput, reuses Rayon + existing JSONL/TUI/report infrastructure; SB3 VecEnv subprocesses waste IPC and fight the reporter. |
| Q8 | Full citizen under `training_output/neural_network_rl/`; RL-flavored Part 1 in report, Parts 2/3 reused | The whole thesis of parallel track is shared infrastructure. |

## Architecture

```
Python (aerocapture.training.rl)
  train.py        CleanRL-style outer loop (PPO / SAC)
  ppo.py, sac.py  per-algorithm update rules (PyTorch)
  policy.py       MLP policy mirroring NeuralNetModel JSON format
  env.py          thin Gymnasium-ish wrapper around BatchedSimulation
  rewards.py      potential-based shaping phi(s) from ref_trajectory.dat
  config.py       [rl] TOML parser
  logger.py       per-update JSONL (reuses TrainingLogger contract)
  report_rl.py    RL-flavored Part 1 panels (reuses charts.py theme)
  export.py       PyTorch policy -> best_model.json

PyO3 bridge

Rust (aerocapture-py + aerocapture crate)
  BatchedSimulation pyclass
    .reset(seeds, overrides)        -> obs (N, k)
    .step(actions)                  -> (obs', reward, done, info) per-env
  One SimState per env, Rayon parallel per-tick advance, shared Arc<SimData>
  Existing run / run_mc / run_batch / run_with_draws APIs unchanged
```

### Control flow per update

1. Python calls `env.step(actions)` with a vector of N bank commands.
2. Rust advances each `SimState` one outer guidance tick using the supplied bank (bypasses dispatch's `signed_schemes` branch but reuses the same downstream pilot + integrator + event path).
3. Each env returns `(obs', reward, done, info)`. On `done=True`, Rust auto-resets to a fresh dispersion draw and returns the first obs of the new episode (Gymnasium VecEnv auto-reset convention).
4. Python buffers trajectories, performs PPO/SAC update, repeats.

## Rust env API

New pyclass `BatchedSimulation` in `src/rust/aerocapture-py/src/`. Existing APIs untouched.

**Construction:**

```python
env = aerocapture_rs.BatchedSimulation(
    toml_path="configs/training/msr_aller_rl_train.toml",
    n_envs=64,
    overrides=None,
    seed_base=3_000_000,
)
```

Resolves bases, loads `SimData` once (shared `Arc<SimData>` across envs), allocates N `SimState` instances.

**API:**

| Method | Returns | Notes |
|---|---|---|
| `reset(seeds: np.ndarray[int64] \| None)` | `obs: (N, k) f32` | If `None`, draws sequentially from `seed_base`. If provided, must be length N. |
| `step(actions: np.ndarray[f32])` | `(obs, reward, done, info)` | `actions` shape `(N,)`, bank commands in `[-pi, pi]` rad. Auto-reset on done: returned obs is first obs of next episode. Info per-env dict populated only on done. |
| `action_mask() -> (N,) bool` | | True where env is in phase 2 exit; unused in v1, reserved for future policy-gradient masking. |
| `close()` | | Drops SimStates. |

**Step semantics** - one outer guidance tick per call, applying (in order):
1. Navigation (bias or EKF, per TOML)
2. Policy action as bank command
3. Pilot + attitude control
4. Integrator substeps (DOPRI45 or Gill RK4) with event detection (bounce, atmosphere exit, crash, phase transition)
5. Photo record append

**Termination** - reward delivered on the step where atmosphere exit, crash, pending crash (`ifinal=4`), NaN/Inf state, or `max_time` fires. Virtual DV computed per existing `runner.rs` rules. Info payload on done:

```python
{
  "ifinal": int, "captured": bool, "ecc": float, "dv_m_s": float,
  "peak_heat_flux_kW_m2": float, "peak_g_load": float,
  "peak_heat_load_kJ_m2": float, "violated_constraints": bool,
}
```

**Observation** - same 23-element candidate vector as `neural_network`, built Rust-side via a newly exported `build_nn_input()` (refactored out of `nn_bank_angle`). `input_mask` applied Rust-side so obs shape is `(N, k)` with `k = len(input_mask)`; default 16 for backward compat.

**Thread model** - `step` uses `rayon::par_iter_mut` over N SimStates; Python GIL released via `py.allow_threads()` around the Rayon scope.

## Python training loop

### Policy network (`policy.py`)

Mirrors `NeuralNetModel` JSON format for lossless export:

- Input: `k` floats (= `len(input_mask)`)
- Hidden: same `layer_sizes` / `activations` as the existing `[network]` section
- Output head:
  - PPO: Gaussian with learned state-independent `log_std` (CleanRL convention); mean is `atan2(out[0], out[1])` signed bank
  - SAC: squashed Gaussian (tanh), mean + state-dependent `log_std`; deterministic bank = `tanh(mean) * pi`
- Value head (PPO only): separate MLP with same hidden sizes, scalar output
- `export.py` strips stochastic head, writes `best_model.json` matching the GA format byte-for-byte

### PPO loop (`ppo.py`)

Standard CleanRL recipe:
1. Collect `rollout_steps` per env -> `(n_envs * rollout_steps, ...)` batch
2. Compute GAE advantages (gamma=0.99, lambda=0.95 defaults)
3. `update_epochs` passes of clipped surrogate + value loss + entropy bonus over minibatches
4. Every `validation_interval_updates`, evaluate deterministic policy on reserved 1000-seed validation pool via `run_batch` (faster than stepping env for pure eval); promote `best_val_cost` on improvement

### SAC loop (`sac.py`)

Standard recipe: replay buffer, twin Q with target networks, policy update with entropy bonus, `target_entropy = -action_dim` default. Validation gate identical to PPO's.

### Shaping reward (`rewards.py`)

Potential-based, provably policy-invariant:

```
r_t  =  gamma * phi(s_{t+1}) - phi(s_t)                       if not done
r_T  =  gamma * 0 - phi(s_T) + R_terminal                      if done

R_terminal  =  -softplus_cost(dv, peak_g, peak_heat_flux, heat_load)
               (same function evaluate.py uses for GA)

phi(s)  =  -alpha * ||(E(s), pdyn(s)) - ref(E(s))||_2 / scale
```

- `ref(E)` interpolates `training_output/<mission>/ref_trajectory.dat` produced by piecewise_constant training.
- `scale = (E_scale, pdyn_scale)` commensurates axes; `alpha` TOML-tunable.
- If `ref_trajectory.dat` missing, shaping silently disabled (falls back to pure terminal reward) with a startup warning.

### Shared infrastructure (`train.py`)

- Config loading via `[rl]` section + base inheritance
- `TrainingLogger` adapter writes per-update JSONL records (schema extends the GA one with RL-specific fields; see below)
- `LiveDisplay` adapter: same Rich TUI scaffolding, RL-labeled metric panels (episodic return sparkline, entropy sparkline, validation badge, ETA on total env steps)
- Checkpoint save/resume (`torch.save`/`torch.load`); `--total-steps N` on resume means "N additional steps"
- Graceful `KeyboardInterrupt` -> save checkpoint -> exit cleanly (same contract as GA path)
- Final re-eval on reserved final-eval seed pool + PDF report via `report_rl.py`

### Reserved seed pools

Reuses `make_reserved_seeds(base, offset, n)`. Existing `VALIDATION_SEED_OFFSET = 1_000_000`, `FINAL_EVAL_SEED_OFFSET = 2_000_000`. New: `RL_TRAINING_SEED_OFFSET = 3_000_000`, the default for `seed_base`. Training seeds advance monotonically from `seed_base + episode_counter`, guaranteeing zero overlap with validation and final-eval pools.

## Observation, action, reward details

**Observation** - bit-identical to what deployed `neural_network` sees at inference; we call the same Rust `build_nn_input()`. No RL-specific normalization, no frame stacking, no history.

**Action** - mapped to bank in `[-pi, pi]` rad:

- PPO: policy outputs `(out0, out1)`; deterministic bank = `atan2(out0, out1)`. Stochastic exploration is a Gaussian on `(out0, out1)` in unconstrained space; the bank distribution is the natural pushforward.
- SAC: policy outputs `mean` in R; deterministic bank = `tanh(mean) * pi`. Stochastic = squashed Gaussian.

Rust `step` does not re-wrap, clip, or rate-limit. Rate limiting comes from `[guidance.command_shaping]`, not from the action space.

**Dispersion draws per episode** - on each auto-reset, Rust draws a fresh random dispersion using `seed_base + episode_counter`.

## Config schema

New `[rl]` section in a training TOML. Example `configs/training/msr_aller_rl_train.toml`:

```toml
base = "missions/mars.toml"

[mission]
mission_type = "msr_aller"

[guidance]
type = "neural_network"

[network]
layer_sizes = [16, 64, 64, 2]
activations = ["tanh", "tanh", "linear"]
input_mask   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

[monte_carlo]
level    = "medium"
sampling = "random"

[rl]
algorithm                   = "ppo"
total_env_steps             = 5_000_000
n_envs                      = 64
seed_base                   = 3_000_000
validation_n_sims           = 1000
validation_interval_updates = 20
checkpoint_interval_updates = 50

[rl.reward]
shaping_enabled = true
shaping_alpha   = 1.0
energy_scale    = 1.0e6      # J/kg
pdyn_scale      = 1.0e3      # Pa

[rl.ppo]
learning_rate     = 3.0e-4
rollout_steps     = 2048
update_epochs     = 10
minibatches       = 32
gamma             = 0.99
gae_lambda        = 0.95
clip_range        = 0.2
entropy_coef      = 0.0
value_coef        = 0.5
max_grad_norm     = 0.5
initial_log_std   = -0.5

[rl.sac]
learning_rate    = 3.0e-4
buffer_size      = 1_000_000
batch_size       = 256
gamma            = 0.99
tau              = 0.005
train_every      = 1
gradient_steps   = 1
target_entropy   = "auto"
initial_alpha    = 0.2
```

Defaults live in `configs/training/rl_common.toml`, pulled in via `base` inheritance.

**CLI:** `python -m aerocapture.training.rl.train <config.toml> [--algorithm ppo|sac] [--total-steps N] [--no-tui] [--skip-report] [--resume <dir>]`. Flags override TOML values when explicitly provided.

## Reporting and integration

### Directory layout

```
training_output/neural_network_rl/
  config_resolved.toml     # fully merged config
  training_log.jsonl       # one record per update
  checkpoint.pt            # torch state (policy + optimizer + replay buffer for SAC)
  best_model.json          # drop-in for Rust neural_network runtime
  gen_best.pt              # stochastic policy state at best validation
  final_eval.parquet       # 65-column Parquet with embedded config metadata
  report.pdf               # Parts 1 (RL convergence), 2 (Mission Performance), optional 3
  sensitivity/             # populated if sensitivity CLI is run separately
```

### JSONL record schema (per update)

```json
{
  "update_idx": 140,
  "env_steps": 286720,
  "episodic_return_mean": -1.23, "episodic_return_p50": -1.10, "episodic_return_p95": -3.40,
  "episodic_dv_m_s_mean": 78.4,  "episodic_capture_rate": 0.94,
  "policy_loss": 0.012, "value_loss": 0.34, "entropy": 0.71, "approx_kl": 0.018,
  "learning_rate": 3.0e-4,
  "val_attempted": true, "val_promoted": true, "val_rms_cost": 61.2, "val_capture_rate": 0.98,
  "best_val_cost": 61.2,
  "wallclock_seconds": 4312.0
}
```

The existing `TrainingLogger` consumes dict-ish records; we extend its optional-field set rather than fork.

### Report (`report_rl.py`)

Same three-part PDF structure as the GA report with Part 1 swapped.

- **Part 1 (RL Convergence, new):** episodic return mean/p50/p95 vs env_steps; episodic DV distribution vs env_steps; entropy + value-loss curves; validation-gate waterfall; capture rate vs env_steps; per-layer weight statistics (reuse `weight_stats.py`).
- **Part 2 (Mission Performance, reused verbatim):** corridor plots, altitude / heat flux / g-load / bank angle / density ratio vs time, DV distributions, entry / exit conditions, performance summary, dispersion correlations. Driven off `final_eval.parquet`.
- **Part 3 (Sensitivity, reused):** same `sensitivity_results.json` + same charts. Runs on the deterministic exported policy.

Typst template `report_rl.typ` extends `report.typ`, replacing Part 1; Parts 2/3 import from the shared `lib.typ`.

### `compare_guidance.py`

Zero changes. Iterates over `--schemes` directories loading `best_params.json` or `best_model.json`. The `neural_network_rl` scheme directory contains a `best_model.json` that the existing Rust `neural_network` runtime loads natively. Head-to-head-vs-GA runs out of the box.

### `train_all.sh`

New alias `nn_rl` running PPO with tuned defaults (`--algorithm ppo --total-steps 5_000_000 --n-envs 64`). Runs after `piecewise_constant` (depends on `ref_trajectory.dat` for shaping) and is independent of the remaining schemes.

### `animate.py`

Not supported in v1. Animation path is chromosome-specific; porting to PyTorch checkpoints is future work.

## Testing

### Rust (`src/rust/tests/` and inline `#[cfg(test)]`)

- `BatchedSimulation` unit tests: reset shape, step shape, auto-reset correctness, seed determinism under constant action, info payload on done, GIL-release scope safety
- Equivalence test: stepping one env with a constant bank command matches the existing `run()` path using `reference.rs` guidance with the same constant bank, bit-identically
- Rayon parallelism: N=64 envs with identical seeds and actions produce N identical obs sequences
- Proptest: `step` returns finite obs/reward for any action in `[-pi, pi]`, any seed in `[0, 10000)`, any step index up to done

### Python (`tests/`)

- `policy.py`: random PyTorch policy -> `best_model.json` -> Rust load -> bit-identical output on 100 random obs
- `rewards.py`: PBRS telescoping identity (sum of step rewards equals `-phi(s_0) + R_terminal` with `gamma=1`)
- `config.py`: TOML parsing, base inheritance, algorithm selection, graceful fallback when `ref_trajectory.dat` missing
- `export.py`: PyTorch -> JSON -> Rust roundtrip preserves deterministic output
- `env.py`: Gymnasium-ish API contract (obs/action shape, dtype, done semantics, auto-reset)
- Smoke test: 10000-step PPO run with `n_envs=4` on a tiny config produces a valid `best_model.json`, `training_log.jsonl`, and `final_eval.parquet`
- Report test: `report_rl.py` generates all SVG panels from canned JSONL + Parquet fixtures and Typst compiles the PDF

## Scope boundaries (explicit v1 non-goals)

- SAC ships experimental (built, not validated). PPO is the validated baseline.
- No recurrent policies. The 23-input observation is Markov enough; revisit only if empirics say otherwise.
- No curriculum learning.
- No animation (`animate.py` parity).
- No multi-mission or multi-planet training in a single run.
- No distributed training (single process, multi-thread Rayon for stepping, single-device PyTorch).
- No TensorBoard export.
- No NN architecture search.

## Build sequence

1. **Rust `BatchedSimulation` pyclass** - carve out the per-tick advance from `runner.rs`, wrap in pyclass, export `build_nn_input()`. Equivalence test vs `run()` gates this phase.
2. **Python `policy.py` + `export.py`** - JSON roundtrip bit-compatibility test gates this phase.
3. **Python `env.py` + `rewards.py` + `config.py`** - thin wrappers around (1), PBRS math, TOML parsing. Unit-testable independently.
4. **Python `ppo.py` + `train.py`** - CleanRL PPO recipe wired into the existing logger/TUI. Smoke test gates this phase.
5. **Python `report_rl.py` + Typst template** - swap Part 1, reuse Parts 2/3.
6. **Integration** - `train_all.sh` alias, `compare_guidance.py` wiring (mostly config).
7. **Python `sac.py`** - built experimental; SAC validation deferred.
8. **Final step** - invoke the `smart-commit` skill with "take the whole branch into account" (user's global CLAUDE.md rule).

## Success criteria (v1 ship)

- PPO-trained policy deployed via `best_model.json` matches or beats the GA-trained NN's median DV on the 1000-seed final-eval pool, with constraint violation rate no worse than GA.
- `compare_guidance.py --schemes neural_network neural_network_rl ...` runs cleanly and produces a comparison PDF.
- Full Rust + Python test suite green; no regression on existing Rust / Python tests.
- RL training produces a PDF report with Parts 1 and 2 rendering correctly (Part 3 renders correctly when the sensitivity CLI has been run separately for the RL scheme).

## Risks and open questions

- **Throughput:** a single tick involves integrator substeps + event detection + photo-record append. Target ballpark: >= 100k env-steps/sec aggregate across 64 envs on a modern laptop. If far lower, PPO's sample budget becomes expensive; investigate batching integrator substeps or shrinking the photo record during training.
- **Shaping calibration:** `alpha`, `energy_scale`, `pdyn_scale` defaults are guesses. First run will likely need a sweep. Mitigation: the PBRS theorem guarantees optimum is unchanged for any finite `alpha`, so worst case we turn shaping off and run pure terminal reward.
- **Stochastic-to-deterministic gap:** the policy trains with exploration noise but deploys deterministic. If the deterministic behavior at inference diverges from training-time captured trajectories, the validation gate will catch it but the policy will appear to regress. Mitigation: validation evaluates deterministic policy explicitly, and the training-time episodic capture rate is logged alongside.
- **Reference trajectory dependency:** shaping silently disables if `ref_trajectory.dat` missing, but RL's main value versus GA is arguably the ability to learn without a reference. Open question: do we also offer a "no-shaping" baseline config so we can measure whether shaping actually helps?

## References

- Ng, A.Y., Harada, D., Russell, S.J. (1999). "Policy invariance under reward transformations: theory and application to reward shaping."
- CleanRL: https://github.com/vwxyzjn/cleanrl
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms."
- Haarnoja et al. (2018). "Soft Actor-Critic."
- Internal: `docs/superpowers/specs/2026-04-14-explicit-seed-strategy-design.md` (seed strategy framework reused here)
- Internal: `docs/superpowers/specs/2026-04-13-nn-input-expansion-pruning-design.md` (23-input vector, `input_mask`)
