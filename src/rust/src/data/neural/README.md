# NN runtime (`src/rust/src/data/neural/`) and its PyTorch mirror

The neural guidance runtime: the 35-input candidate contract, the bank decoders, the model JSON
format, the ten layer types, and the differentiable PyTorch mirror
(`src/python/aerocapture/training/torch_mirror/`) that is held to it by cross-language
equivalence gates at machine epsilon. Rust is the source of truth; every Python-side view is
derived from it or gated against it.

Related: [src/rust/README.md](../../../README.md) (`gnc/guidance/neural.rs`, `nn_state.rs`,
`tick.rs`), [configs/README.md](../../../../../configs/README.md) (`[network]`,
`[guidance.neural_network]`), the training README for warm-start / calibration / ablation /
quantization, [rl/README.md](../../../../python/aerocapture/training/rl/README.md) for the
PPO-BPTT axis. Design history: `docs/design/2026-04-17-stateful-nn-runtime-infrastructure-design.md`
and the per-cell designs it links; the multi-cell program is shipped and `TODO.md` keeps its
epilogue.

## Inputs: the 35-element candidate vector

`NN_FULL_INPUT_SIZE = 35`. `build_nn_input` (`gnc/guidance/neural.rs`) extracts the raw scalars
into a `[f64; NN_FULL_INPUT_SIZE]` then applies the per-input spec in a uniform loop
(extract-then-normalize; the only non-trivial extraction is angle -> (sin, cos)). A configurable
`input_mask` selects the subset that reaches the network (absent = `[0..16]`).

| indices | content |
|---|---|
| 0-15 | orbital / aero / thermal state (incl. the bounce flag at 15, a ±1 binary) |
| 16-19 | reference trajectory interpolations |
| 20 | the always-live exit-bank teacher: the closed-loop FTC exit-phase pdyn-feedback law fed every step, degenerating to pure radial-velocity damping pre-bounce when `ref_velocity_latched = 0` |
| 21-24 | lateral-state telemetry: inclination-error rate via finite-diff `(curr - prev) / guidance_period`, previous-tick bank command normalized to `[-1, 1]`, `tanh(time_since_last_sign_flip / 30)`, `tanh(integral_deg_s / 100)` of the integrated inclination error |
| 25-30 | seam-free `(sin, cos)` bank-history pairs: exit-bank teacher (25/26), previous commanded bank (27/28), previous pilot-realized bank (29/30); cyclic encoding with no ±pi seam |
| 31 | periapsis altitude |
| 32-34 | live correction-DV components `predicted_dv1/2/3` |

The correction-DV inputs are the signed components of `maneuver::predicted_dv_for_nn` evaluated
per tick on the CURRENT osculating orbit (a causal, cost-aligned signal): dv1 = energy-closing
burn at the current periapsis (vis-viva, the Δv to shed excess energy and bring apoapsis to
target; large pre-capture, ~0 near target), dv2 = periapsis-correction (apoapsis-referenced,
`->0` for hyperbolic as the continuous limit), dv3 = inclination plane change. All three are
defined and SMOOTH across the `e=1` boundary; there is NO sentinel. `predicted_dv_for_nn` is
distinct from `compute_deltav` (the terminal-maneuver plan).

The telemetry inputs (21-24) make the supervisor's signed-bank decision Markovian from the NN's
point of view (without them, post-reversal near-duplicate states collapse the supervised MSE
target). Their backing state lives on `GuidanceState::{prev_inclination_error_for_nn,
prev_bank_for_nn, prev_realized_bank_for_nn, last_sign_flip_time_for_nn,
inclination_error_integral}` and is updated unconditionally by `tick.rs` post-guidance from the
EFFECTIVE command (the RL env's `forced_bank` when injected, else the dispatcher output), so RL
observations track the policy's own previous action; sign-flip detection compares
signum(new_bank) vs signum(prev_bank), excluding zero-crossing cases.
`NnInputContext::from_guidance_state(&GuidanceState, sim_time, target_inclination)` is the one
place that state is read.

**Normalization** is one uniform per-input transform `norm = transform((raw - center) / scale)`
with `transform in {none, asinh, tanh}` (divisor form: `scale` = characteristic magnitude,
`center` = subtracted offset; `NormSpec`, `apply_norm` in `mod.rs`). Specs resolve in this
order: the model JSON's embedded `normalization` block (self-describing, like `output_param`;
emitted by `save_json` and by the PSO `flat_weights_to_json` path) > the TOML
`[network.normalization]` override (35 entries, length-validated; also carried on
`SimData::nn_normalization_override` regardless of guidance type, so a teacher scheme's
`collect_supervised` trace normalizes on the deployed NN's scales) > the baked
`DEFAULT_NORMALIZATION` table (`mod.rs`, the single Rust source of truth). A wrong-sized
embedded block hard-errors instead of silently reverting to the default. `validate_mask` rejects
out-of-range, negative or duplicate `input_mask` indices at load. The contract is exported as
`aerocapture_rs.candidate_inputs()` (35 `{index, name, transform, scale, center}` dicts);
`training/config.py::candidate_input_names()` / `candidate_input_index()` /
`candidate_input_normalization()` derive the Python names, width and lookups from it (fallback
tuple asserted equal by `tests/test_record_index_drift.py`). Scales are data-driven via
`calibrate_inputs.py` (training README). Ablation: `ablated_input` zeros one input,
`ablated_value` freezes it to a value instead (`build_nn_input` writes it to `full_input[idx]`).
The per-tick candidate trace recorded for `collect_supervised` / `collect_nn_inputs` uses a full
mask of width `NN_FULL_INPUT_SIZE`.

## Output decoders (`output_param`)

Decoded in `nn_bank_angle` (`match` on the model's `output_param`; the knobs are TOML-overridable
and embedded in the deployed `best_model.json`):

- `atan2_signed` (default): `bank = atan2(out[0], out[1])`, 2-output head, legal in both modes
  (under `magnitude_only` it is `.abs()`'d).
- `acos_tanh`: `bank = acos(tanh(out[0]))`, 1-output `tanh` head, `magnitude_only` only; a smooth
  `[0, π]` mapping aligned with FTC's internal `cos_bank`.
- `scaled_pi`: `bank = wrap_to_pi(scaled_pi_n * pi * tanh(out[0]))`, knob `scaled_pi_n` (default
  1.0; larger pushes the ±pi seam away from the operating region). `full_neural` only.
- `delta`: `bank = wrap_to_pi(prev_realized + delta_max * tanh(out[0]))`, knob `delta_max`
  (default 0.35 rad): a bounded per-tick increment on the previous pilot-realized bank, the
  accumulator left unbounded and wrapped only at the guidance -> shaper boundary. `full_neural`
  only.

Routing (`[guidance.neural_network] mode`): `full_neural` bypasses lateral, exit and thermal
guidance; `magnitude_only` feeds `.abs()` of the decoded bank through FTC's unsigned-magnitude
pipeline. `reset_state_every_tick` (eval-only) reconstructs a zeroed `NnState` before every tick.

## Model JSON

`from_json_str` dispatches on `format_version`: v1 (dense-only `layer_sizes` + `activations`,
per-layer `w`/`b`) and v2 (`format_version: 2`, a tagged `architecture` list of `LayerSpec`
entries, `#[serde(tag = "type")]`, plus one weights entry per parameterized layer keyed by the
tensor-table names). v1 files produce bit-identical output to their v2 form. The loaders deny
unknown keys at every level (file, `LayerSpec` entry, stray tensor, stray `layer_i` entry; the
legacy `output_interpretation` key is declared and ignored). The JSON also carries `input_mask`,
`output_param` (+ `scaled_pi_n` / `delta_max`), `normalization`, `ablated_input` /
`ablated_value`. `NeuralNetModel::forward(&self, &mut NnState, &[f64])` is the stateful forward;
the model is `Arc`-shared and immutable, the per-sim state lives in `GuidanceState::nn_state: Option<NnState>`, `NnState { layer_states:
Vec<LayerState> }` (`nn_state.rs`; `NnState::for_model` eager init, `Clone` for RL rollout
snapshots, rebuilt at episode start by `GuidanceState::new(initial_bank, initial_aoa, nn_model:
Option<&NeuralNetModel>)`; `build_sim_state` asserts `nn_state.is_some() == neural_net.is_some()`).

## Layer types

Every layer declares its weights ONCE with `tensor_table!` (`layers/tensor.rs`): the listed
fields in order ARE the flat chromosome order, the JSON keys and `n_params`; flag-gated tensors
are `Option` fields. `LayerWeights` provides `tensors()` / `tensors_mut()` from the table, a
`post_load` hook, and the default `to_flat` / `from_flat` / `n_params`. `load_layers` (JSON) and
`from_flat_weights_v2` (PSO chromosome) are generic walks over the table; `save_json` walks
`tensors()` (keys in table order except `JSON_KEYS_LAST` = Mamba3 `a_imag` / `lambda_logit`, kept
last so re-saves stay byte-identical). `Layer::from_spec` is the ONE dim-validation site;
`LayerSpec::io()` gives the chain shapes. Shared kernels in `layers/helpers.rs`: `matvec`,
`dot_plus_bias`, `gelu_exact` (via `libm::erf`), `layer_norm_biased`, `build_pe_table`,
`softplus` (stable `max(x,0) + log1p(exp(-|x|))`), `expm1_over_x` (Taylor fallback at `|z| <
1e-8`), `lecun_tanh`, `stabilized_exp_gates`; sequential FIFO reductions everywhere, for
cross-language bit-identity.

| type | weights (flat order) | state | notes |
|---|---|---|---|
| `dense` | `w [O, I]`, `b [O]` | none | activations linear / tanh / swish / asinh / ... |
| `gru` | `weight_ih [3H, I]`, `weight_hh [3H, H]`, `bias_ih [3H]`, `bias_hh [3H]` | `(H,)` | PyTorch `nn.GRUCell` convention, r/z/n gates, `h_new = (1-z)*n + z*h_prev` |
| `lstm` | `weight_ih [4H, I]`, `weight_hh [4H, H]`, `bias_ih [4H]`, `bias_hh [4H]` | `(h, c)` | `nn.LSTMCell` convention, gate order i/f/g/o, no peepholes; `c_new = f*c + i*g`, `h_new = o*tanh(c_new)` |
| `window` | none (`n_params() == 0`, `to_flat() == Vec::new()`, `from_flat` consumes 0 from any slice) | `VecDeque<Vec<f64>>` pre-filled with `n_steps` zero vectors | FIFO ring buffer, output `n_steps * input_size`; spec-only JSON |
| `transformer` | `w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, w_ffn1, b_ffn1, w_ffn2, b_ffn2, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta` (16 flat keys) | `k_cache` / `v_cache` `VecDeque`s growing 0 -> `n_seq`, no zero-padding | 1-layer pre-norm block, causal window attention over the KV ring, sinusoidal PE relative to ring slot, GELU-exact FFN, biased-variance LayerNorm eps=1e-5; `d_model % n_heads == 0`; boxed (`Layer::Transformer(Box<..>)`); derived `k_pe_offsets` / `v_pe_offsets` (= `W_K @ PE_table`, `W_V @ PE_table`) rebuilt by the `post_load = rebuild_pe_offsets` hook on every load (mutating `w_k` / `w_v` by hand requires calling it yourself) |
| `mamba` | `x_proj_w [dt_rank + 2*d_state, I]` (no bias), `dt_proj_w [I, dt_rank]`, `dt_proj_b [I]`, `a_log [I, d_state]` (`A = -exp(a_log)`), `d_skip [I]` | `h: DMatrix (I, d_state)` | selective SSM core (S6): diagonal A, HiPPO-style init, ZOH discretization, input-dependent Δ/B/C; `dt_rank` defaults to `max(1, input_size // 16)` (`resolve_mamba_dt_rank` is the Python single source); boxed; `from_v2_json` rejects `dt_rank=0` |
| `mamba3` | `x_proj_w, dt_proj_w, dt_proj_b, a_log, [a_imag if complex], [lambda_logit if trapezoidal], d_skip` | `{h_re, h_im, x_prev, b_prev}` (unused slabs zero) | the Mamba-3 ablation 2x2: `discretization ∈ {euler, trapezoidal}`, `state_mode ∈ {real, complex}` (strings in TOML and JSON, bools only on the runtime `Mamba3Layer` via `mamba3_flags()`). `euler`+`real` is BIT-identical to `mamba` (`real_euler_bit_identical_to_mamba`). Trapezoidal: `h = α·h + λ·Δ·B·expm1_over_x(ΔA)·x + (1-λ)·Δ·α·B_prev·x_prev`, `λ = sigmoid(lambda_logit)` (init +4, near-euler), `λ→1` recovers euler exactly. Complex: `A = -exp(a_log) + i·θ` (`θ = a_imag`), B/C real, readout `Re(h)·C`; `expm1_over_x_complex` in explicit (re, im) arithmetic. `n_params = I·(3·d_state + 2·dt_rank + 2) + [complex: I·d_state] + [trapz: I]` |
| `cfc` | one lecun_tanh backbone + ff1 / ff2 / t_a / t_b heads | `(H,)` | closed-form continuous-time cell (ncps "default" mode): `h' = (1-g)*tanh(ff1) + g*tanh(ff2)`, `g = sigmoid(-t_a*dt + t_b)`, dt = 1 guidance tick |
| `slstm` | full recurrent matrices, single bias | `(h, c, n, m)` | xLSTM sLSTM: exponential gating with max-stabilizer, gate order i/f/z/o |
| `mlstm` | projections, no recurrent weights | `C (HxH)`, `n (H,)`, scalar `m` | xLSTM mLSTM: matrix memory, covariance update, single head `d_qk = d_v = H` |

Rust names per type: `Layer::Gru` / `Layer::Lstm` / `Layer::Window` / `Layer::Transformer(Box<TransformerLayer>)` /
`Layer::Mamba(Box<MambaLayer>)` (boxed to keep the enum's variants balanced, `large_enum_variant`), the specs
`LayerSpec::Gru { input_size, hidden_size }`, `LayerSpec::Lstm { input_size, hidden_size }`,
`LayerSpec::Window { input_size, n_steps }`, `LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq }`,
`LayerSpec::Mamba { input_size, d_state, dt_rank }` (TOML: `TomlLayerSpec::Gru` / `::Lstm` / `::Window { input_size, n_steps }`
/ `::Transformer` / `::Mamba`, `[[network.architecture]] type = "lstm"` etc.), the states `LayerState::None`,
`LayerState::Gru(Vec<f64>)`, `LayerState::Lstm { h, c }`, `LayerState::Window { buffer: VecDeque<Vec<f64>> }`
(the forward takes `&mut VecDeque` directly, `w.forward(&current, buffer)`, to avoid a double borrow across the
match), `LayerState::Transformer { k_cache: VecDeque<Vec<f64>>, v_cache: VecDeque<Vec<f64>> }`,
`LayerState::Mamba { h: DMatrix<f64> }`, `LayerState::Mamba3 { h_re, h_im, x_prev, b_prev }`; the weight impls
`LayerWeights for GruLayer` / `LstmLayer` / `WindowLayer` / `TransformerLayer` / `MambaLayer` are all the
`tensor_table!` default (`LayerWeights::from_flat` runs `post_load`). Per-cell design docs:
`docs/design/2026-04-17-phase-1-gru-mvp-design.md`, `docs/design/2026-04-18-phase-2a-lstm-mvp-design.md`,
`docs/design/2026-04-20-phase-2b-window-mlp-design.md`, `docs/design/2026-04-22-phase-3a-transformer-mvp-design.md`,
`docs/design/2026-04-24-phase-4a-mamba-ssm-mvp-design.md`, `docs/design/2026-07-07-mamba3-ablation-design.md`,
`docs/design/2026-07-07-cfc-xlstm-probes-design.md`.

`mamba3`, `cfc`, `slstm`, `mlstm` are PSO-only probe cells (no `torch_mirror/layers/` module for
BPTT; `build_layer` / `load_policy_from_json` raise `NotImplementedError`; their unbatched torch
mirrors exist only for the equivalence gates; `MAMBA3_EVAL_SEED_OFFSET` is an alias of `PROBE_EVAL_SEED_OFFSET`). Deployed training configs per cell type are in
`configs/README.md`; the Rust runtime dispatches on the JSON architecture (no per-scheme Rust).

## The PyTorch mirror (`training/torch_mirror/`)

Load-bearing for the population path (chromosome sizing, warm-start) and the base of the
shelved PPO/SAC trainer in `rl/` (which depends on it, never the reverse).

- `schemas.py` — Pydantic v2 `DenseSpec` / `GruSpec` / `LstmSpec` / `WindowSpec` /
  `TransformerSpec` / `MambaSpec` / `Mamba3Spec` / ... in `LayerSpec = Annotated[..., Discriminator("type")]`,
  `ArchitectureV2`; validators mirror Rust (`d_model % n_heads == 0`, `MambaSpec`'s
  `model_validator(mode='after')` resolving `dt_rank`).
- `layers/` — one torch module per BPTT-trainable cell with the uniform step contract
  `forward(x, state) -> (y, new_state)`: `DenseLayer`, `GruLayer` (manual r/z/n gates matching
  `nn.GRUCell` bit-for-bit; `new_state` tracks the parameter dtype so `policy.double()`
  propagates), `LstmLayer` (tuple state `forward(x, (h, c)) -> (h_new, (h_new, c_new))`),
  `WindowLayer` (class-level `_dtype_anchor: Tensor` for the non-persistent buffer),
  `TransformerLayer` (manual LN / GELU / softmax / MHA, no `nn.LayerNorm` / `F.softmax` /
  tanh-GELU), `MambaLayer` (manual softplus, NOT `F.softplus` with its `threshold=20` branch;
  `_expm1_over_x` in the additive-mix form `taylor + (exact - taylor) * gate` to sidestep the
  `torch.where` double-backward NaN pitfall). `__init__.py::build_layer(spec)` dispatches per
  `spec.type` and raises `NotImplementedError` for the PSO-only cells. Every layer has
  `to_flat()` mirroring Rust `LayerWeights::to_flat`.
- `policy.py` — `V2Policy` iterates layers with per-layer state; `log_std` is a non-exported
  `nn.Parameter` (exploration noise only). State-threaded methods: `forward_mean_logstd(obs,
  state)`, `sample(obs, state) -> (bank, raw, log_prob, new_state)`, `evaluate(obs_seq, state_0,
  dones_seq, raw_seq) -> (log_probs_seq, entropy_seq)` (the BPTT forward over a time chunk,
  zeroing per-env state on `dones_seq[t]` via `_zero_state_where_done`, whose `_zero_entry` recursion handles `None`,
  `Tensor` and `tuple`-of-the-above and raises `TypeError` on anything else), and
  `forward_seq_means` (the autograd-friendly warm-start mirror of `evaluate`).
- `export.py` — `export_v2_policy_to_json(policy, path, obs_normalizer=None)` writes format v2
  from the layer schema (Transformer's 16 keys flat at layer level, Mamba's 5, Window spec-only);
  the obs-normalizer affine is baked into layer 0 (`W_new = W/std`, `b_new = b - W @ (mean/std)`;
  `obs_normalizer` typed as the `ObsAffine` Protocol so the mirror never imports `rl/`), and
  `_check_obs_norm_bake_compatibility` rejects a non-dense layer 0 (GRU / LSTM / Window / Mamba
  cannot absorb an affine shift).
- `training/model_io.py` — `load_policy_from_json(path, device) -> V2Policy`, round-tripping
  with the exporter bit-for-bit; raises for PSO-only cells.
- `training/layer_schema.py` — tensor names / shapes / flat order per layer spec from
  `aerocapture_rs.layer_schema`, with `_fallback_layer_schema` (`tests/test_layer_schema_drift.py`).
- `training/encoding.py` — `nn_param_specs_from_v2(architecture, bound_multiplier)` via
  `_layer_param_specs` per type (`_gru_specs`: tanh-Xavier on the 3H gate matrices, `0.1 *
  bound_multiplier` on biases; `_lstm_specs`: 4H, with the forget-gate slice of `bias_ih` at `2.0
  * bound_multiplier`; `_transformer_specs`: Xavier `sqrt(6/(2*d_model))` on projections,
  `sqrt(6/(d_model+d_ffn))` on the FFN, `[1-0.01*mul, 1+0.01*mul]` on LN gamma, tight near-zero on
  biases / LN beta; `_mamba_specs`: Xavier on `x_proj_w`, Xavier * `dt_rank^{-0.5}` on
  `dt_proj_w`, `inv_softplus(U(1e-3, 1e-1))` centers on `dt_proj_b` (shared per channel via
  `_MAMBA_DT_BIAS_SEED ^ layer_idx`), HiPPO `log(n+1)` centers on `a_log`, 1.0 on `d_skip`;
  probe cells: +2.0 forget-bias centers on the sLSTM f-slice and mLSTM `b_f`, bound `3.0*mul`);
  the ordering MUST match Rust `to_flat` or PSO chromosomes scramble. All-dense architectures get
  bounds identical to the v1 `nn_param_specs_from_architecture` via `compute_layer_bound`.
- `training/initialization_v2.py` — `init_v2_population(architecture, n_pop, bound_multiplier,
  rng)`: dense uniform-in-Xavier-bound; GRU tanh-Xavier + `N(0, 0.01*mul)` biases; LSTM
  tanh-Xavier + `N(0, 0.01*mul)` i/g/o biases + `1.0 + N(0, 0.01*mul)` forget-bias on `bias_ih`
  ONLY (Jozefowicz, Zaremba & Sutskever 2015; `bias_hh` forget stays ~0 to avoid double-applying
  through the gate sum); Mamba per-individual `N(0, 0.01*bound_multiplier)` jitter around the
  shared centers (`_init_mamba_layer`; without it PSO init collapses on the non-zero-centered
  slices); Window `continue`. `train.py::build_initial_population_for_v2` normalizes the result
  to [0, 1] per ParamSpec. `compute_weight_stats` is skipped for v2.
- `training/config.py` — `NetworkConfig.architecture: list[dict] | None`, `_layer_n_params`
  (dense `I*O + O`, gru `3HI + 3HH + 6H`, window 0, ...), `_layer_output_size` (window
  `n_steps * input_size`), `describe_architecture`.

## Extensibility

**A new scalar-state layer type.** Rust: `layers/<type>.rs` (struct + forward + `zeros` +
`tensor_table!(XxxLayer { field, ... } [, post_load = hook])`), then one arm each in
`neural/mod.rs` (`LayerSpec` variant, `LayerSpec::io`, `Layer` variant + `as_weights` /
`as_weights_mut` / `from_spec`, `forward`), `nn_state.rs` (`LayerState` variant + `for_layer` +
reset arms), `config.rs` (`TomlLayerSpec` variant + `to_layer_spec` arm). No serialization code.
Python: `torch_mirror/schemas.py` (Spec class + union entry), `layer_schema.py::_fallback_layer_schema`
(one branch), ONE `ArchCase` row in `tests/nn_archs.py` (reduced arch, hand-counted `n_params`,
tolerance, seeded mirror builder; `test_archs_cover_every_layer_type` fails until it exists, and
the row drives both `tests/test_nn_equivalence.py` and `tests/test_nn_pso_smoke.py`; a
`forward_unbatched` / `new_state()` mirror needs a branch in `nn_archs._step`), `encoding.py` (a
per-tensor bound / center rule; a frozen-spec fixture guards the chromosome contract),
`_layer_output_size` in `config.py`; `torch_mirror/layers/<type>.py` + `layers/__init__.py` only
for BPTT-trainable types (`export.py` / `model_io.py` rebuild slabs from the schema and need only
a `_spec_entry` branch). No changes to `problem.py`, `dispatch.rs` or `runner.rs`. Zero-parameter
layers are an empty table (Window: `from_flat` consumes 0, `save_json` writes no entry,
`_layer_param_specs(WindowSpec)` returns `[]`, `config.py::_layer_n_params(window) == 0`, `_layer_output_size(window) == n_steps * input_size`, `init_v2_population` contributes a `continue`).

**Multi-tensor hidden states** (LSTM `(h, c)`) additionally need: (a) a branch in
`_zero_state_where_done`; (b) a `hidden_shapes` arm in `rl/train.py::_derive_hidden_shapes`
packing the state into one stacked numpy array (LSTM: `(2, H)`); (c) matching `ndim == N`
dispatch in `_np_state_to_torch` / `_torch_state_to_np`; (d) matching dispatch in
`ppo.py::ppo_update_bptt` to reconstruct the container before `policy.evaluate`. The `(B, 2, H)`
stacking is deliberate: `done`-mask zeroing via boolean row indexing zeros both `h` and `c` in
one operation, so the rollout loop needs no per-type special case; stacked LSTMs get one `(T, B,
2, H)` slab each in `RolloutBuffer.states`.

**2D single-tensor state** (Mamba `(d_inner, d_state)`) is stored as one `Tensor`, so
`_zero_state_where_done` already handles it; a Mamba PPO path would store `(T, B, d_inner,
d_state)` (`ndim == 4`) and add the matching dispatch in `ppo_update_bptt`, `hidden_shapes` and
the pack/unpack helpers.

**Derived-at-load-time fields** (the Transformer PE-offset pattern): store only trainable
parameters in the table and name a `post_load` hook; the default `from_flat` runs it at the end
of EVERY load, so both entry points rebuild without per-path code.

## Gates

- `tests/test_nn_equivalence.py` over the `tests/nn_archs.py` rows (one per layer type incl.
  the four `mamba3_*` flag combos): 100-step stateful Rust-vs-Python equivalence through
  `nn_forward_sequence` (tolerance 1e-14 mamba, 1e-12 probe types / mamba3, 1e-10 elsewhere;
  observed at machine epsilon, e.g. GRU 4.4e-16, Window 2.78e-16, Transformer 4.16e-17, Mamba
  1.11e-16; the high-`a_log` Mamba stress at 4.07e-14 is bounded by `d_state * eps *
  max|h|*max|c|` and gated at 1e-12), `test_state_evolves` / `test_deterministic` for every
  stateful row (`test_window_buffer_warmup_zero_padded`: buffer[0] lags by n_steps-1 ticks;
  `test_transformer_cache_warmup`: cache grows 0 -> n_seq), `test_archs_cover_every_layer_type`.
- `tests/test_v2_rust_python_equivalence.py`: the `export_v2_policy_to_json` seam (dense / GRU /
  LSTM; Dense + `input_mask=[0,2,4]`; `test_acos_tanh_rust_python_equivalence` < 1e-10).
- `tests/test_nn_pso_smoke.py`: PSO serialization round-trip per row with keys/shapes from
  `layer_schema` (`test_serialization_roundtrip[...]`), `test_train_two_gens[gru|lstm]`, the
  real end-to-end PSO training smoke via subprocess (n_pop=4, n_gen=1, training_n_sims=2, the full
  TOML -> NetworkConfig -> write_nn_json -> flat_weights_to_json path), `test_mamba3_encoding`,
  `test_init_v2_mamba3`. The PPO-rejection tests for the PSO-only cells (`build_layer` +
  `load_policy_from_json` both raise) are `tests/test_{transformer,mamba,mamba3,cfc_xlstm}_ppo_rejection.py` (Window's `build_layer` is enabled for warm-start; its PPO guards are `load_policy_from_json` + `rl/train.py::_derive_hidden_shapes`, tested in `tests/test_window_export_and_load.py`).
- Rust: `tests/nn_flat_order_fixtures.rs` (per-type flat vector + exact `save_json` bytes under
  `tests/fixtures/flat_order/`), `tests/nn_model_roundtrip.rs` (every committed model JSON
  load -> save -> reload; `--ignored` resave tool for byte diffs), the Mamba3 units
  (`real_euler_bit_identical_to_mamba`, `trapezoidal_reduces_to_euler_at_high_lambda`,
  `complex_warmup_deterministic`, `mamba3_json_v2_save_load_roundtrip_all_flags`), and the six
  guidance goldens (a new cell must leave them bit-identical).
- `tests/test_layer_schema_drift.py`, `tests/test_record_index_drift.py`,
  `tests/test_config_normalization_blocks.py` (the contract exports).
