# CfC + xLSTM probes -- design

Date: 2026-07-07
Branch: `feature/cfc-xlstm` (stacked on `feature/mamba3-ablation`)
Status: approved, pre-implementation

## Motivation

Two recent recurrent families were identified as pertinent gaps in the architecture
comparison, each with a mechanistic hypothesis for aerocapture guidance:

1. **CfC (closed-form continuous-time, Hasani & Lechner et al. 2022)** -- input-dependent
   time constants match the fast/slow phase structure of an aerocapture pass (seconds-scale
   dynamics near periapsis, near-static coast in vacuum). A fixed-cadence GRU must learn
   that timescale switch implicitly in its gates; CfC parameterizes it directly. Hypothesis:
   matches the recurrent tail at similar params.
2. **xLSTM (Beck et al., NeurIPS 2024)** -- exponential gating lets a cell sharply *revise*
   a stored estimate when surprise arrives (the bounce, a density shock crossing the 50 km
   bank-limit switch); classic sigmoid-gated LSTM cannot. mLSTM additionally tests matrix
   memory against Mamba's diagonal state. Hypothesis: exponential gating explains part of
   the LSTM rows' underperformance.

Like the Mamba-3 spike, these are **preliminary-signal** experiments, not paper-grade
results: controlled arms at reduced budget with seed-repeats, reporting tail metrics with
sigma_run error bars. Per the project's recorded lesson, single-run deltas are noise; every
claim must clear sigma_run.

## Scope

- Three new layer types, all **PSO-only** (identical gate to Mamba/Mamba3: `build_layer`
  and `load_policy_from_json` raise `NotImplementedError`; torch mirrors exist solely for
  the cross-language equivalence tests). Cell-only, no block scaffolding.
- Two experiment scripts in `aerocapture.training.experiments`, mirroring
  `mamba3_ablation.py`'s CLI and report exactly:
  - `cfc_probe.py`: arms `{gru (baseline), cfc}`
  - `xlstm_probe.py`: arms `{lstm (baseline), slstm, mlstm}`
- Every arm (including baselines) retrained at matched param budget, same sandwich, same
  fixed seeds -- gaps attributable. Deployed champions additionally scored on the same eval
  pool as non-matched **reference rows**.
- All changes additive: existing layers, goldens, and the deployed champions untouched.

## Non-goals / deliberate simplifications (flagged for the write-up)

- **No block scaffolding.** xLSTM's up/down projections + causal conv4 and CfC's
  mixed-memory LSTM wrapper are omitted; the Dense sandwich provides the projections.
  (Same rationale as Phase 4a's "cell only, no full Mamba block".)
- **mLSTM is single-head with `d_qk = d_v = H`.** The paper's multi-head split is a
  capacity knob irrelevant at this scale.
- **Forget-gate bias init is a +2.0 center**, not the official powerlaw-blockdependent
  init. The exp-gating analogue of the LSTM forget-bias-1 precedent.
- **CfC `Δt` is fixed at one guidance tick** and absorbed into the learned time-gate heads
  `t_a`/`t_b`; the continuous-time formulation degenerates to a learned-timescale gate at
  fixed cadence. Documented, and acceptable: the hypothesis under test is input-dependent
  timescales, not variable-cadence inference.
- **CfC backbone is exactly one layer** (no `backbone_layers` knob), ncps "default" mode
  only (no `pure` / `no_gate` modes).
- **sLSTM uses full recurrent matrices, single head** (the paper's block-diagonal R is a
  multi-head artifact).

## Layer semantics

All three: cell-only, f64, explicit scalar loops on the Rust side, unbatched torch mirrors.
Gate order and flat order below are canonical -- the torch mirror defines the convention
(no PyTorch built-in exists for any of these).

### `cfc` -- spec `{ input_size: I, hidden_size: H, backbone_units: B }`

State: `h (H,)`, zero-init. `LayerState::Cfc(Vec<f64>)` (flat, like GRU).

```
cat = [x, h]                                   (I+H,)
xb  = lecun_tanh(W_bb·cat + b_bb)              (B,)    lecun_tanh(z) = 1.7159·tanh(2z/3)
g   = sigmoid(−(W_ta·xb + b_ta)·Δt + (W_tb·xb + b_tb))    Δt = 1
h'  = (1−g)⊙tanh(W_f1·xb + b_f1) + g⊙tanh(W_f2·xb + b_f2)
output = h'
```

Output bounded in (−1, 1) by construction (PSO-friendly).

Flat order: `w_bb (B×(I+H)), b_bb (B), w_ff1 (H×B), b_ff1 (H), w_ff2 (H×B), b_ff2 (H),
w_ta (H×B), b_ta (H), w_tb (H×B), b_tb (H)`, matrices row-major.
`n_params = B(I+H)+B + 4(HB+H)`.

### `slstm` -- spec `{ input_size: I, hidden_size: H }`

State: `(h, c, n, m)` each `(H,)`, all zero-init.
`LayerState::Slstm { h, c, n, m: Vec<f64> }`.
Weights: `w_ih (4H×I), w_hh (4H×H), bias (4H)`; gate order `i, f, z, o`; single bias.

```
(ĩ, f̃, z̃, õ) = W_ih·x + W_hh·h + b            per-unit slices of the 4H preactivation
m' = max(f̃ + m, ĩ)                             stabilizer
i' = exp(ĩ − m');  f' = exp(f̃ + m − m')        exponential gating
c' = f'·c + i'·tanh(z̃)
n' = f'·n + i'
h' = sigmoid(õ) ⊙ c'/n'
output = h'
```

No div-by-zero at t=0: `n₁ = i' > 0` (exp is strictly positive) and every later step adds
a positive `i'`, so `n` stays positive.
`n_params = 4HI + 4HH + 4H`.

### `mlstm` -- spec `{ input_size: I, hidden_size: H }` (single head, `d_qk = d_v = H`)

State: `(C: H×H, n: (H,), m: scalar)`, zero-init.
`LayerState::Mlstm { c: DMatrix<f64>, n: Vec<f64>, m: f64 }` (reuses the Mamba DMatrix
precedent). No recurrent weights (paper-faithful: gates read x only).

```
q = W_q·x + b_q;   k = (W_k·x + b_k)/sqrt(H);   v = W_v·x + b_v
ĩ = w_i·x + b_i (scalar);   f̃ = w_f·x + b_f (scalar)
m' = max(f̃ + m, ĩ);   i' = exp(ĩ − m');   f' = exp(f̃ + m − m')
C' = f'·C + i'·(v kᵀ)
n' = f'·n + i'·k
h' = sigmoid(W_o·x + b_o) ⊙ (C'·q) / max(|n'·q|, 1)
output = h'
```

Flat order: `w_q (H×I), b_q (H), w_k (H×I), b_k (H), w_v (H×I), b_v (H), w_o (H×I),
b_o (H), w_i (I), b_i (1), w_f (I), b_f (1)`, matrices row-major.
`n_params = 4(HI + H) + 2(I + 1)`.

### Shared helpers (`layers/helpers.rs`)

- `stabilized_exp_gates(i_pre, f_pre, m) -> (i', f', m_new)` -- used per-unit by slstm,
  per-step by mlstm. Must stay finite for large preactivations (the max-stabilizer
  guarantees both exp arguments are <= 0).
- `lecun_tanh(z) = 1.7159 * tanh(2z/3)`.

### Enum boxing

`Layer::Cfc/Slstm/Mlstm` variants are boxed if clippy `large_enum_variant` fires (expected
for mlstm at least, matching the `Mamba`/`Transformer`/`Mamba3` precedent).

## Initialization (`initialization_v2.py` arms)

- Xavier-tanh gain on tanh-feeding matrices: cfc `w_bb/w_ff1/w_ff2`, slstm `w_ih/w_hh`
  (4H-concatenated, like the LSTM arm).
- Plain Xavier: cfc `w_ta/w_tb`, mlstm `w_q/w_k/w_v/w_o` and gate vectors `w_i/w_f`.
- Biases `N(0, 0.01·bound_multiplier)` -- except forget slices: the sLSTM f-slice of
  `bias` and mLSTM `b_f` are centered at **+2.0** with widened ParamSpec bounds (the
  LSTM forget-bias `2.0·bound_multiplier` precedent applied to the exp-gating cells).

## Config / plumbing surface (extensibility contract, per-layer)

Rust: `layers/{cfc,slstm,mlstm}.rs` (struct + forward + `LayerWeights`), `neural/mod.rs`
(`LayerSpec`/`Layer` variants, `LayerSpec::io()` arms, `save_json`/`from_v2_json`/
`from_flat_weights_v2` arms), `nn_state.rs` (`LayerState` variants + `for_layer` + reset),
`config.rs` (`TomlLayerSpec` variants + validation: all dims >= 1).

Python: `rl/schemas.py` (`CfcSpec`, `SlstmSpec`, `MlstmSpec`, `Field(ge=1)`),
`rl/layers/{cfc,slstm,mlstm}.py` (torch mirrors), `rl/layers/__init__.py` (build_layer
raises `NotImplementedError`, PSO-only), `model_io.py` (load raises), `encoding.py`
(`_cfc_specs`/`_slstm_specs`/`_mlstm_specs`), `config.py` (`_layer_n_params` +
`_layer_output_size` arms; output size = H for all three), `initialization_v2.py`,
`evaluate.py` (`PROBE_EVAL_SEED_OFFSET`).

## Experiment scripts

`experiments/cfc_probe.py` and `experiments/xlstm_probe.py`, CLI and report identical to
`mamba3_ablation.py`: `--generate/--train/--eval/--report/--all`, `--repeats` (default 3),
`--n-gen` (default 500), `--training-n-sims` (default 10), `--n-sims` (default 1000),
`--sim-timeout`, `--force`, `--from-scratch`.

- Configs written to `configs/training/cfc_probe/` and `configs/training/xlstm_probe/`;
  outputs to `training_output/cfc_probe/<arm>_s<r>/` and `training_output/xlstm_probe/...`.
- Generated leaf TOMLs: `base = ["../msr_aller_nn_atan2_train.toml"]` -- the paper's
  atan2 training environment (17-input calibrated `input_mask` + `[network]`
  normalization, `full_neural` + `atan2_signed`, `scaffolding = "live"` with tuned
  nav/shaping starting points; its warm-start block is commented out upstream, which
  the PSO-only probe layers require). Sandwich `Dense(17→32, swish) → cell →
  Dense(H→2, asinh)` (the `[[network.architecture]]` array replaces the base's dense
  stack), PSO `n_pop=300`, `seed_strategy="fixed"`, `validation_n_sims=200`,
  `monte_carlo.seed = BASE_SEED + r`, `results_suffix = ".{script}_{arm}_s{r}"`.
  Because arms train with live scaffolding, `eval_arms` applies each arm's deployed
  `best_params.json` overrides at scoring (the param_sweep lesson: scoring without
  them mis-ranks architectures).
- `BASE_SEED = 20260707` (same as mamba3 -- identical training seed lists across all three
  probe scripts).
- `--generate` also writes `manifest.json` with per-arm dims and exact param counts
  (cell + total trainable, via `NetworkConfig`).

### Param matching (cell params; sandwich adds 576 + head)

| script | arm | dims | cell params | total | delta vs baseline |
|---|---|---|---|---|---|
| cfc_probe | gru (baseline) | H=32 | 6336 | 6978 | -- |
| cfc_probe | cfc | H=32, B=32 | 6304 | 6946 | -0.5% |
| xlstm_probe | lstm (baseline) | H=32 | 8448 | 9090 | -- |
| xlstm_probe | slstm | H=32 | 8320 | 8962 | -1.4% |
| xlstm_probe | mlstm | H=64 | 8514 | 9220 | +1.4% |

Head dense is `(32→2)` = 66 params for H=32 arms, `(64→2)` = 130 for mlstm; the NN
chromosome additionally carries the 3 live-scaffolding params (identical for all arms).

### Shared eval pool

New `PROBE_EVAL_SEED_OFFSET = 10_000_000` in `evaluate.py`;
`MAMBA3_EVAL_SEED_OFFSET` becomes an alias (`= PROBE_EVAL_SEED_OFFSET`) so all three probe
scripts score on the same reserved pool and reports are directly comparable. Disjoint from
all training/validation/final-eval/warm-start/report/calibration/sweep streams by
construction.

### Reference rows

At `--eval`, deployed champions are scored once on the same pool and reported under a
separate `references` key (single evaluation, no repeats, flagged not-budget-matched):

- cfc_probe: GRU champion (`training_output/neural_network_gru_pso/`,
  `configs/training/msr_aller_gru_pso_train.toml`) + Mamba champion
  (`training_output/neural_network_mamba_pso/`, `msr_aller_mamba_pso_train.toml`).
- xlstm_probe: LSTM champion (`training_output/neural_network_lstm_pso/`,
  `msr_aller_lstm_pso_train.toml`) + the same Mamba champion.

Each scored via its own training TOML + `best_model.json`, with `best_params.json`
scaffolding overrides applied when present (reuse `report.py::_load_nn_scaffolding_overrides`,
as `param_sweep._entry_overrides` does). Missing champion dirs skip with a notice.

### Report

Same table as mamba3 (cap%, rms, dvP50, dvP95 +- sigma, CVaR95 +- sigma), tail-led,
plus reference rows; sigma_run significance of each treatment arm vs the in-script
baseline arm (`cfc` vs `gru`; `slstm` and `mlstm` vs `lstm`), same
|gap| > sqrt(sigma_a^2 + sigma_b^2) rule, skipped below 2 repeats.
Results JSON: `training_output/{script}/probe_results.json`.

## Testing

Per layer, mirroring the mamba3 suite:

1. **Cross-language equivalence** (`tests/test_rust_python_{cfc,slstm,mlstm}_equivalence.py`):
   100-step `nn_forward_sequence` vs the torch mirror on reduced dims
   (e.g. Dense(4→8) → cell(8→H) → Dense(→2)), gate 1e-12 (mamba3 precedent;
   expected observed ~1e-14 -- mLSTM's matrix-state accumulation is why the
   gate is not 1e-14).
2. **Rust unit tests** (`data/neural/tests.rs`): flat round-trip, v2 JSON round-trip,
   slstm t=0 warm-up (no div-by-zero, finite h), mlstm denom-clamp path
   (|n·q| < 1 exercises the max), stabilizer under large preactivations (f̃, ĩ ~ ±50:
   finite, no inf/NaN).
3. **Encoding tests**: ParamSpec width == `n_params` per layer; forget-slice centers +2.0.
4. **Init tests**: `init_v2_population` centers/jitter per the initialization section.
5. **PPO-rejection tests**: `build_layer` + `load_policy_from_json` raise for all three.
6. **PSO plumbing smoke** (@slow): reduced arch end-to-end through
   `write_nn_json`/`flat_weights_to_json`, asserts v2 JSON layer list + finite
   `nn_forward` output.
7. **Script unit tests** (`tests/test_{cfc,xlstm}_probe.py`): config generation
   (TOML parses, arms/dims correct, param counts within +-2% of baseline), aggregation +
   significance math (mirror `test_mamba3_ablation.py`).
8. **Golden regressions**: untouched -- all changes are additive; run the suite to prove it.

## Docs

- CLAUDE.md: one paragraph describing the three experimental layer types + probe scripts
  (mamba3-paragraph style).
- README: experimental-layer note alongside mamba3's.
- This spec + the implementation plan under `docs/superpowers/`.

## File inventory (expected)

New: `src/rust/src/data/neural/layers/{cfc,slstm,mlstm}.rs`,
`src/python/aerocapture/training/rl/layers/{cfc,slstm,mlstm}.py`,
`src/python/aerocapture/training/experiments/{probe_common,cfc_probe,xlstm_probe}.py`
(`probe_common.py` holds the shared score/aggregate/report machinery; `mamba3_ablation.py`
keeps its own copies, untouched),
`tests/test_rust_python_{cfc,slstm,mlstm}_equivalence.py`, `tests/test_{cfc,xlstm}_probe.py`,
`tests/test_{cfc,xlstm}_encoding.py`, `tests/test_init_v2_cfc_xlstm.py`,
`tests/test_cfc_xlstm_ppo_rejection.py` (combined), `tests/test_{cfc,xlstm}_pso_smoke.py`,
this spec + plan.

Modified: `layers/mod.rs`, `neural/mod.rs`, `neural/tests.rs`, `nn_state.rs`, `config.rs`
(Rust); `rl/schemas.py`, `rl/layers/__init__.py`, `model_io.py`, `encoding.py`,
`config.py`, `initialization_v2.py`, `evaluate.py` (Python); `CLAUDE.md`, `README.md`.
