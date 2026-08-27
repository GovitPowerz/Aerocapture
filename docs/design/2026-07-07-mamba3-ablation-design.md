# Mamba-3 ablation spike -- design

Date: 2026-07-07
Branch: `feature/mamba3-ablation`
Status: approved, pre-implementation

## Motivation

The deployed recurrent headline (`Mamba_962`, the Phase 4a S6 selective-SSM core) beats
the dense baseline on the sizing tail beyond run-to-run variance. Mamba-3
(arXiv 2603.15569, March 2026) proposes three axes of improvement; two of them have a
plausible mechanism for a smooth low-bandwidth control signal like aerocapture guidance:

1. **Exponential-trapezoidal discretization** -- second-order accurate state-input
   integration (local error O(Δt²) -> O(Δt³)) via a data-dependent convex blend of the
   current and previous input.
2. **Complex-valued (rotational) state** -- diagonal complex `A` recovers state-tracking
   (parity, modular arithmetic) that real diagonal SSMs provably cannot; equivalent to
   data-dependent RoPE on `B`/`C`.

The third axis (MIMO decode efficiency) is irrelevant at ~1000 params trained by PSO and
is out of scope.

This is a **preliminary-signal** experiment, not a paper-grade result: a 2x2 ablation at
reduced budget with seed-repeats, reporting the tail metric with σ_run error bars. Its job
is to decide whether either axis is worth a full campaign. Per the project's recorded
lesson, single-run deltas are noise; every claim must clear σ_run.

## Scope

- New `Mamba3` layer type, **PSO-only** (the PPO Python path raises `NotImplementedError`,
  identical gate to the current Mamba). The proven `Mamba_962` path is left untouched so the
  headline cannot regress.
- Two independent config flags select a clean 2x2:

  | Arm | `discretization` | `state_mode` |
  |---|---|---|
  | baseline | `euler` | `real` |
  | trapz | `trapezoidal` | `real` |
  | complex | `euler` | `complex` |
  | both | `trapezoidal` | `complex` |

- `euler` + `real` is **defined to be bit-identical to the existing `MambaLayer`** -- a free
  correctness anchor in the equivalence test.
- Experiment script trains the 4 arms x 3 seed-repeats at reduced budget, evaluates on a
  shared reserved pool, and reports p50/p95/CVaR95 DV + capture rate with σ_run.

## Non-goals / deliberate simplifications (flagged for the write-up)

- **`λ` is a learned per-channel constant, not data-dependent.** The paper's `λ_t` is
  data-dependent; the minimal probe uses `λ = sigmoid(lambda_logit)` per channel. Upgrade
  path if the arm shows signal: project `λ` from `x`.
- **Complex readout reads `Re(h)` with real `B`/`C`** (half-complex, S4D-style). The paper's
  full formulation has complex `C`. Upgrade path: add a `C_imag` block to `x_proj`.
- **`θ` (rotation frequency) is a learned per-(channel, state) constant, not data-dependent.**
  Standard S4D-Lin init; the minimal probe for rotational dynamics.
- No conv1d / SiLU gating / in-out expansion (those were already deferred as "the full Mamba
  block" in Phase 4a and remain out of scope).

## Layer semantics

`Mamba3Layer` extends `MambaLayer` with two orthogonal recurrence modes. Base weights
(`x_proj_w`, `dt_proj_w`, `dt_proj_b`, `a_log`, `d_skip`) and the fused `x_proj` -> (Δ_pre, B, C)
split are unchanged. `x_proj` shape stays `(dt_rank + 2*d_state, input_size)` in all modes
(`B`, `C` remain real).

Let `α[d,n] = exp(Δ[d]·A[d,n])`, `A[d,n] = -exp(a_log[d,n])` in real mode.

### Discretization

**euler (baseline, == current Mamba):**
```
b_bar[d,n] = Δ[d] · B[n] · expm1_over_x(Δ[d]·A[d,n])
h[d,n]     = α[d,n]·h[d,n] + b_bar[d,n]·x[d]
```

**trapezoidal (strict generalization of euler):**
```
λ[d]   = sigmoid(lambda_logit[d])
h[d,n] = α[d,n]·h[d,n]
         + (1-λ[d]) · Δ[d] · α[d,n] · B_prev[n] · x_prev[d]      (cross term)
         + λ[d]     · Δ[d] · B[n] · expm1_over_x(Δ[d]·A[d,n]) · x[d]   (current term)
then: x_prev <- x, B_prev <- B
```
Note the deliberate deviation from the paper: we keep `expm1_over_x` on the current-input
term (the paper's Euler baseline uses the crude `B̄ = ΔB`). This makes our trapezoidal a
**nested** generalization -- `λ -> 1` recovers the deployed euler ZOH exactly, isolating the
cross term as the single independent variable. `lambda_logit` inits to **+4** (`λ ≈ 0.982`),
so training departs from euler rather than cold-starting.

### State mode

**real:** `h` is `(input_size, d_state)` real, `h_im` unused (zeros).

**complex:** `A[d,n] = -exp(a_log[d,n]) + i·θ[d,n]`, `θ = a_imag` (new param). State is
`(h_re, h_im)`. With `r = exp(-Δ·exp(a_log))`, `φ = Δ·θ`:
```
α          = r·(cosφ + i·sinφ)                       (complex)
b_bar      = Δ · B · expm1_over_x_complex(Δ·A)       (complex; B real)
h_new      = α·h + b_bar·x                            (complex mult, x real)
y[d]       = Σ_n Re(h_new[d,n])·C[n] + D[d]·x[d]      (real readout)
```
Complex arithmetic is implemented with **explicit (re, im) real ops on both sides** (Rust
scalar loop + Python numpy/torch real tensors), never a native complex dtype -- this keeps
the cross-language bit-identity contract. `expm1_over_x_complex(z) = (exp(z)-1)/z` with a
Taylor fallback `1 + z/2 + z²/6` for `|z| < 1e-8`, evaluated in explicit (re,im) form.

`both` composes the two: the cross term is also complex.

### Flat-weight layout (conditional on flags)

```
x_proj_w   (dt_rank + 2*d_state, input_size)   row-major
dt_proj_w  (input_size, dt_rank)               row-major
dt_proj_b  (input_size,)
a_log      (input_size, d_state)               row-major
a_imag     (input_size, d_state)   row-major   [only if state_mode == complex]
lambda_logit (input_size,)                     [only if discretization == trapezoidal]
d_skip     (input_size,)
```

```
n_params = input_size·(3·d_state + 2·dt_rank + 2)
         + [complex:      input_size·d_state]
         + [trapezoidal:  input_size]
```

### Hidden state

`LayerState::Mamba3 { h_re: DMatrix, h_im: DMatrix, x_prev: DVector, b_prev: DVector }`,
all zero-initialized. Unused slabs (h_im in real mode; x_prev/b_prev in euler mode) stay
zero and are not read -- avoids a LayerState variant explosion for the four flag combos. At
`d_state ≤ 16`, `input_size ≤ 32` the memory overhead is negligible. `reset()` zeros all four.

### Init (`initialization_v2.py`)

Mirror the Mamba branch (HiPPO `a_log = log(n+1)` centers, `inv_softplus(U(dt_min,dt_max))`
`dt_proj_b` centers with per-layer sub-RNG `_MAMBA_DT_BIAS_SEED ^ layer_idx`, Xavier x_proj,
`d_skip = 1`), plus:
- `a_imag`: S4D-Lin convention (per-(channel,state) ramp), only when complex.
- `lambda_logit`: constant `+4` + per-individual `N(0, 0.01·bound_multiplier)` jitter, only
  when trapezoidal.

## Files touched

### Rust
- `src/rust/src/data/neural/layers/mamba3.rs` (new) -- `Mamba3Layer` struct + `LayerWeights`
  impl + `forward`. Reuses `helpers::{softplus, expm1_over_x}`; adds
  `expm1_over_x_complex` (in mamba3.rs).
- `src/rust/src/data/neural/layers/mod.rs` -- export `Mamba3Layer`.
- `src/rust/src/data/neural/mod.rs` -- `Layer::Mamba3(Box<..>)` (io/to_flat/from_flat/n_params
  arms), `LayerSpec::Mamba3 { input_size, d_state, dt_rank, discretization, state_mode }`,
  `from_v2_json` / `save_json` / `from_flat_weights_v2` arms.
- `src/rust/src/data/nn_state.rs` -- `LayerState::Mamba3 { h_re, h_im, x_prev, b_prev }` +
  `for_layer` + `reset` arms.
- `src/rust/src/config.rs` -- `TomlLayerSpec::Mamba3` + `to_layer_spec` (dt_rank fallback
  `max(1, input_size/16)`; validate `discretization ∈ {euler, trapezoidal}`,
  `state_mode ∈ {real, complex}`).

### Python
- `src/python/aerocapture/training/rl/layers/mamba3.py` (new) -- torch mirror (explicit
  (re,im) arithmetic), `to_flat`/`from_flat`.
- `src/python/aerocapture/training/rl/layers/__init__.py` -- `build_layer(Mamba3Spec)` raises
  `NotImplementedError` (PSO-only gate).
- `src/python/aerocapture/training/rl/schemas.py` -- `Mamba3Spec` + discriminated-union entry.
- `src/python/aerocapture/training/encoding.py` -- `_mamba3_specs` (base Mamba specs +
  conditional a_imag / lambda_logit).
- `src/python/aerocapture/training/config.py` -- `_layer_n_params` / `_layer_output_size` /
  `describe_architecture` arms; reuse `resolve_mamba_dt_rank`.
- `src/python/aerocapture/training/initialization_v2.py` -- `_init_mamba3_layer`.
- `src/python/aerocapture/training/model_io.py` (`load_policy_from_json`) -- raise
  `NotImplementedError` on any Mamba3 layer (PSO-only).

### Experiment
- `src/python/aerocapture/training/experiments/mamba3_ablation.py` (new).
- `configs/training/mamba3/*.toml` (generated by `--generate`).
- `MAMBA3_EVAL_SEED_OFFSET = 10_000_000` added to `evaluate.py` (1M-9M taken; 8M/9M are
  headline-requote / stress; 10M is the next free slot).

## Experiment script

CLI mirroring `param_sweep.py`:

- `--generate` -- write the 4 arm configs under `configs/training/mamba3/`, each base-inheriting
  `configs/training/msr_aller_mamba_pso_train.toml` and **restating the full**
  `[[network.architecture]]` (arrays replace under deep-merge) with the Mamba layer swapped to
  `type = "mamba3"` + the arm's two flags. Also write per-seed-repeat leaves
  (`{arm}_s{r}.toml`) setting `monte_carlo.seed = BASE_SEED + r`.
- `--train` -- subprocess-train each arm x 3 repeats: `python -m aerocapture.training.train
  configs/training/mamba3/{arm}_s{r}.toml --n-gen 500 --no-tui --skip-report --output-dir
  training_output/mamba3/{arm}_s{r}`. Skip if `best_model.json` exists (`--force` to retrain).
- `--eval` -- for each arm x repeat, load `training_output/mamba3/{arm}_s{r}/best_model.json`
  (+ `best_params.json` scaffolding if present) and `run_batch` on the shared reserved pool
  (`make_reserved_seeds(0, MAMBA3_EVAL_SEED_OFFSET, n_sims)`, default n_sims=1000). Compute DV
  p50/p95/CVaR95 + capture rate per repeat.
- `--report` -- aggregate across repeats -> arm x metric table with **σ_run = std across the 3
  repeats**; explicitly flag whether any inter-arm gap on the tail metric clears σ_run
  (`|Δ| > σ_run` gate). Optional SVG.
- `--all` -- generate -> train -> eval -> report.

Reduced budget defaults (all CLI-overridable): `--n-gen 500`, `--repeats 3`, `--n-sims 1000`,
`--n-pop 64` (inherited from base config). This is a long-running job (hours); the smoke path
is not this script (see gates).

## Testing / gates

### Rust units (in `mamba3.rs` `#[cfg(test)]` or `tests.rs`)
- `mamba3_real_euler_bit_identical_to_mamba` -- same weights, same input sequence -> identical
  output (the free correctness anchor).
- `mamba3_trapezoidal_reduces_to_euler_at_high_lambda` -- `lambda_logit = +30` -> matches euler
  to < 1e-12.
- `mamba3_complex_warmup_deterministic` -- state from zero evolves deterministically; no
  step-0-vs-step-1 collapse.
- `mamba3_flat_roundtrip` -- `to_flat` -> `from_flat` -> `to_flat` bit-identical, per flag combo;
  `n_params` matches slab length.

### Cross-language equivalence (`tests/test_rust_python_mamba3_equivalence.py`)
- All 4 flag combos: 100-step sequence through `aerocapture_rs.nn_forward` vs the torch mirror
  (per-step reset), max abs diff < 1e-12 (looser than the real path's 1e-14 -- complex adds a
  multiply; expected actual ~1e-14).

### PSO integration
- `tests/test_mamba3_pso_smoke.py` (@slow) -- 2 PSO gens on a reduced `mamba3` arch, assert
  `best_model.json` is v2 with a `"mamba3"` layer and `nn_forward` returns a finite tuple.
- `tests/test_mamba3_ppo_rejection.py` (@fast) -- `build_layer` + `load_policy_from_json` both
  raise `NotImplementedError`.
- `tests/test_mamba3_encoding.py` -- `_mamba3_specs` length == `n_params` per flag combo;
  config `_layer_n_params` agrees.

### Regression
- All existing Rust guidance goldens bit-identical (new layer touches no existing path).
- Full `./check_all.sh` (Rust fmt/clippy/test + release build) and `pytest tests` green.

## Implementation order

1. Rust `Mamba3Layer` + `expm1_over_x_complex` + unit tests (real-euler anchor first).
2. Rust enum wiring (Layer/LayerSpec/LayerState/TomlLayerSpec + JSON round-trip).
3. Python mirror + schema + encoding + config + init.
4. Cross-language equivalence test (all 4 combos) -- the gate that makes the experiment
   falsifiable.
5. PSO smoke + PPO-rejection + encoding tests.
6. Rebuild PyO3 (`maturin develop --release --manifest-path ...` from repo root).
7. Experiment script + generated configs.
8. `--generate` + a tiny `--n-gen 5 --repeats 1 --n-sims 20` dry run to prove the pipeline end
   to end (NOT the real budget).
9. `check_all.sh` + `pytest tests` + goldens.
10. Invoke the `smart-commit` skill, telling it to take the whole `feature/mamba3-ablation`
    branch into account.

## Result (2026-07-08)

**Verdict: neither Mamba-3 axis improves the aerocapture sizing tail. Clean negative.**

Two experiments were run:

1. **500-gen PSO spike** (3042-param arch, 3 seed-repeats, the `mamba3_ablation.py` driver).
   Trapz/complex looked "leaning positive but within σ_run" (p95 ~215-258, σ_run 70-120). This
   directional hint turned out to be a MIRAGE of the under-trained, high-variance regime.

2. **Full-budget run** (the deployed Mamba_962 cell: Dense 17->16 -> Mamba3(16, d_state=12) ->
   Dense 16->2; GA n_pop=512, n_gen=10000, training_n_sims=2, adaptive-max seeds, scaffolding=live,
   no warm-start; configs `configs/training/mamba3_962/*.toml`). Scored on 2000 held-out sims
   (reserved 10M pool) with each model's co-trained scaffolding, via
   `aerocapture.training.experiments.mamba3_962_compare`:

   | arm | params | cap% | dvP50 | dvP95 | CVaR95 |
   |---|---|---|---|---|---|
   | baseline (euler+real) | 962 | 100.00 | 111.0 | **116.9** | **118.6** |
   | trapz | 978 | 100.00 | 110.7 | 117.0 | 119.6 |
   | complex | 1154 | 99.95 | 111.5 | 118.6 | 120.9 |
   | both | 1170 | 100.00 | 110.8 | 117.5 | 120.1 |

At full budget all four converge to a tight, excellent solution (whole tail within ~8 m/s of the
p50). Baseline (euler+real, fewest params) is best-or-tied on every tail metric; complex is
consistently the worst (+1.8 p95, +2.2 CVaR95) despite +192 `a_imag` params. **The spike's positive
hint reverses to flat-to-negative once training converges** -- an under-trained spike gives false
architecture signals; confirm at full budget.

Caveat: the full-budget comparison is SINGLE-run per arm, so the ~1-2% gaps are within run-to-run
noise -- the honest claim is "no benefit / indistinguishable," NOT "baseline significantly better."

Interpretation: Mamba-3's axes (trapezoidal accuracy, complex state-tracking, MIMO throughput) target
long-context tracking and decode throughput; aerocapture is a short, smooth, low-bandwidth signal and
the baseline 962 cell already has ample capacity. This is the data-backed answer to "why not Mamba-3?"
See memory `project-mamba3-ablation-result`.
