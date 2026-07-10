# Quantization study appendix (Mamba-962) -- design

Date: 2026-07-10
Branch: `feature/quantization-mamba962` (ports the dense-only work from `feature/quantization`, unmerged, diverged 2026-06-05)
Status: approved, pre-implementation

## Motivation

The paper explicitly names this gap (paper.typ:949): "We have no clean campaign
study of pruning or quantizing the deployed head -- the only such cells predate
the simulator fixes in this work and are not comparable -- so deploy-size
reduction of the Mamba policy is open." This study closes the quantization half
of that sentence as a new appendix (Appendix D), on the deployed champion:

- **Target model:** `training_output/mamba_p962_long/best_model.json` --
  Dense(17->16, swish) -> Mamba(16, d_state=12, dt_rank=1) -> Dense(16->2, asinh),
  atan2_signed, 17-input calibrated mask, `scaffolding = "live"` (4 co-trained
  nav/shaping overrides in `best_params.json`), GA 512 x 20k gens.
- **Baseline quote (fresh pool, offset 8M, n=1000):** capture 100%, DV p50 109.7 /
  p95 113.8 / p99 116.0 / CVaR95 115.2 m/s (`fresh_pool_requote.json`).

Three questions, in order of scientific interest:

1. **PTQ sensitivity:** how few bits do the champion's weights need, and *which
   tensors* of a selective SSM are the bottleneck? `a_log` is the prime suspect:
   20% of params (192/962), exponentiated at runtime (`A = -exp(a_log)`), and
   one-sided (HiPPO `log(n+1)` init), so a symmetric grid wastes half its levels.
2. **QAT recoverability + trainability:** does fine-tuning the champion under a
   4-bit fitness recover the PTQ loss (recoverability), and does a from-scratch
   run under the 4-bit constraint reach the fp champion (trainability)?
3. **Deployment benefit:** measured inference cost (real int8/int4 kernels,
   microbenchmark) and exact memory footprint. NOT simulation throughput --
   fake-quant leaves sim time unchanged by construction, and the appendix says
   so explicitly rather than inviting the question.

## Prior work reused (branch `feature/quantization`)

Portable pieces: `quantize.py` (symmetric fake-quant core + sweep runner + CLI),
`charts_quant.py`, QAT hook points (`qat_bits`/`qat_granularity` on
`NetworkConfig` + `validate_qat`; population rounding in
`problem._run_batch_pyo3` pre-`run_grid`; deploy rounding in
`evaluate.write_nn_json`), tests (`test_quantize.py`, `test_qat_training.py`).
All dense-only by hard ValueError -- the extension to Mamba is the core new code.
The old dense QAT outputs (`neural_network_atan2_qat4/qat8`, 2000-gen smokes,
never fresh-pool-quoted) are scaffolding, not results; they are not cited.

Two scoring gaps in the old sweep are fixed in the port (both are the recorded
param_sweep mis-ranking lesson):

- it never applied the co-trained `best_params.json` scaffolding overrides;
- it used the config's `monte_carlo.seed` instead of a reserved pool.

## Scope / non-goals

- Weight-only fake-quant for accuracy: rounded values stored back as f64, Rust
  runtime untouched, goldens bit-identical.
- Layer types: dense + mamba only. gru/lstm/window/transformer/mamba3/cfc/slstm/mlstm
  stay rejected with the existing ValueError.
- No activation quantization study (the w8a8/w4a8 microbench kernels quantize
  activations dynamically, but that is a compute measurement, not an accuracy arm).
- No integer path wired into the simulator.
- QAT at 4 bits only (8-bit PTQ is expected lossless; if it is not, that result
  itself is reported and QAT-8 reconsidered).

## Tensor accounting (champion arch)

| tensor        | shape        | params | policy "all"        | policy "proj_only" |
|---------------|--------------|--------|---------------------|--------------------|
| dense0.w      | 16x17        | 272    | quant               | quant              |
| dense0.b      | 16           | 16     | fp                  | fp                 |
| x_proj_w      | 25x16        | 400    | quant               | quant              |
| dt_proj_w     | 16x1         | 16     | quant               | quant              |
| dt_proj_b     | 16           | 16     | fp (bias)           | fp                 |
| a_log         | 16x12        | 192    | quant               | fp                 |
| d_skip        | 16           | 16     | quant (per-tensor)  | fp                 |
| dense2.w      | 2x16         | 32     | quant               | quant              |
| dense2.b      | 2            | 2      | fp                  | fp                 |
| **total**     |              | 962    | 928 quant / 34 fp   | 720 quant / 242 fp |

Rules: biases always fp (dense convention preserved; `dt_proj_b` is a bias).
1-D tensors get per-tensor scale (per-channel on a vector is per-element =
lossless = meaningless). 2-D tensors: per-channel = one scale per output row.

## Study design

### 1. PTQ sweep

Grid: bits {8, 6, 4, 3, 2} x granularity {per_channel, per_tensor} x policy
{all, proj_only} = 20 variants. Plus leave-one-out at 4 bits: quantize exactly
one tensor group at a time from {dense0.w, x_proj_w, dt_proj_w, a_log, d_skip,
dense2.w} (granularity from the grid's best 4-bit cell) = 6 variants. Each
variant: n=1000 on the eval pool below. Total compute: minutes.

### 2. QAT arms (launched after the PTQ verdict)

The PTQ result picks the granularity + tensor policy the QAT arms train under
(pre-registered rule: the 4-bit cell with the best CVaR95; ties break toward
per_channel + all).

- **Fine-tune:** copy the champion's latest `checkpoint_g*.{json,npz}` pair into
  `training_output/quant/mamba962_qat4_finetune/`, resume with `qat_bits = 4`
  for +3000 gens (auto-resume; `--n-gen` means additional). Chromosome width is
  unchanged by qat knobs, so the resume-width guard passes; the resume path
  re-validates the checkpointed best under the now-quantized objective, which is
  exactly the wanted semantics. The champion dir is never written to. (Verified:
  the champion has no `warm_start_bounds.json` sidecar -- it trained from
  scratch under static bounds -- so the checkpoint pair is the only state to
  copy and the resumed ParamSpec bounds match by construction.)
- **From-scratch:** GA 512 x 20000, matched to the champion budget (~2.5-3 days
  wall), `training_output/quant/mamba962_qat4_scratch/`.

Both leaf configs live in `configs/training/quant/`, base-inherit the champion
pipeline (same regime as `configs/training/sweep/mamba_p962.toml`: 17-input
mask, calibrated normalization, scaffolding = live, adaptive-max curation,
training_n_sims 2), override only `[network] qat_bits/qat_granularity/
qat_tensor_policy`, the deploy path, and (scratch) `n_gen = 20000`.

Both arms are single runs: every QAT-vs-champion delta is quoted against the
probe-campaign sigma_run, same honesty rule as Appendix C.

### 3. Microbenchmark + memory table

`src/rust/benches/quant_forward.rs` (criterion, dev-dependency only). Four
kernels hard-coded to the champion arch, self-contained in the bench file:

- f64 baseline via `NeuralNetModel::forward` (the deployed path);
- f32 (dtype-width reference);
- w8a8: per-channel i8 weights, dynamic per-tick i8 activation quant,
  i32 accumulate, f64 dequant;
- w4a8: same with packed nibbles (2 weights/byte, unpack in kernel).

SSM recurrence (softplus, exp, state update) stays fp in ALL variants --
quantization accelerates the projections only, and the bench reports that
dilution honestly. Startup self-check asserts each kernel against a plain-Rust
reference implementation of itself (not against the f64 model output -- w8a8
output differs from fake-quant by activation rounding, by design).

Outputs: ns per guidance-tick forward; x measured ticks/sim (from a nominal
run's trajectory length) -> per-sim ms bound placed next to the paper's
3.68 ms/sim number. Memory table is analytic and exact per (bits, granularity,
policy): quantized bytes = ceil(n_quant * b / 8), + one f32 scale per row-group,
+ fp remainder; headline ~7.7 KB (f64) -> ~1 KB (int8) -> ~0.6 KB (int4).

### 4. Finalists

Baseline (champion), best PTQ-4bit cell, QAT-fine-tune, QAT-scratch re-scored at
n=10000 on the same pool (baseline re-run at that depth so all four rows are
same-pool, same-depth).

## Eval protocol

- Pool: `make_reserved_seeds(base_mc_seed, HEADLINE_REQUOTE_SEED_OFFSET=8_000_000, n)`
  -- the champion's fresh-requote pool, so every number sits next to the paper
  quote.
- Scaffolding: apply the champion's `best_params.json` overrides (the 4 keys in
  `fresh_pool_requote.json`: command_shaping.enabled/max_bank_acceleration,
  navigation.density_filter_gain/density_gain_max_delta). QAT arms use their OWN
  co-trained `best_params.json` (they retrain the live scaffolding genes).
- Metrics, tail-led: capture %, DV p50/p95/p99/CVaR95, constraint-violation %,
  deltas vs baseline. `mean_cost` kept for continuity with the old sweep but not
  the headline.
- Reporting: the model file evaluated is always pinned explicitly
  (`data.neural_network` override), never read from the shared TOML deploy path
  (concurrent-run clobber lesson).

## Code changes

- `src/python/aerocapture/training/quantize.py` (port + extend):
  - `quantize_model_weights(model_json, n_bits, granularity, tensor_policy,
    only_tensor=None)` replaces `quantize_dense_weights` (no external callers;
    branch was never merged). Mamba arm per the tensor table; `only_tensor`
    drives the LOO.
  - `quantize_flat_weights_batch(..., tensor_policy)` gains the mamba arm using
    the canonical flat order [x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip].
    Invariant: operates on the NN-weight slab only -- scaffolding genes travel
    through `run_grid` overrides, never through this array; the existing
    exact-width assert enforces it and a test pins it.
  - `run_quant_sweep`: reserved pool + scaffolding overrides + `--model` /
    `--params` pinning (ablation.py CLI conventions), tail metrics, LOO mode.
- `src/python/aerocapture/training/config.py`: `validate_qat` accepts
  dense+mamba, gains `qat_tensor_policy` ("all" | "proj_only", default "all");
  `NetworkConfig` + train.py TOML plumbing for the new key.
- `src/python/aerocapture/training/problem.py` + `evaluate.py`: thread
  `qat_tensor_policy` through the two existing hook points. Audit that every
  eval path (per-gen fitness, validation gate, final selection, final eval,
  report) sees the rounded policy; the deploy writer hook covers the
  temp-JSON/run_batch paths, the problem hook covers run_grid -- verified by a
  test that trains 2 gens with qat_bits=8 and asserts the deployed weights are
  idempotent under re-quantization (quantize(deployed) == deployed).
- `src/python/aerocapture/training/charts_quant.py`: sweep curve y-axes become
  capture + DV CVaR95 (tail-led), + LOO bar chart, + QAT convergence overlay
  (champion vs fine-tune vs scratch best-cost curves).
- `configs/training/quant/mamba962_qat4_finetune.toml`,
  `configs/training/quant/mamba962_qat4_scratch.toml`.
- `experiments/paper/15_quantization.sh`: PTQ sweep -> (gate: pick QAT cell) ->
  QAT arms -> finalist requotes -> bench run -> collect JSONs into
  `articles/paper/data/quant/`.
- `src/rust/benches/quant_forward.rs` + criterion dev-dependency + `[[bench]]`
  entry. No src/ changes; goldens untouched.

## Testing

Port `test_quantize.py` + `test_qat_training.py`, then extend:

- mamba grid round-trip: flat-path and JSON-path quantization of the same model
  produce identical weights (the two hook points agree);
- policy exclusions: a_log/d_skip untouched under proj_only; dt_proj_b and all
  dense biases untouched under every policy;
- LOO isolation: only_tensor=x quantizes exactly that group;
- 1-D per-tensor rule: d_skip identical under both granularities;
- scaffolding-slab safety: width assert fires when handed a 962+3 matrix;
- validate_qat accept/reject matrix incl. mamba accepted, mamba3/gru rejected;
- QAT smoke (@slow): 2 gens on a tiny dense->mamba->dense arch with qat_bits=8,
  deployed weights idempotent under re-quantization;
- PTQ sweep smoke (@slow): reduced grid end-to-end, JSON + SVG artifacts exist;
- Rust: bench startup self-check (kernel vs its reference impl); `cargo test`
  and the 6 guidance goldens must stay bit-identical (no src changes expected).

## Sequencing

1. Port + extend quantizer, tests, charts (0.5-1 day).
2. PTQ sweep + LOO on the champion (minutes). **Gate:** pick the QAT cell.
3. Launch QAT-scratch (2.5-3 days wall) and QAT-fine-tune (~0.5 day).
4. While training: microbench + memory table (~1 day), appendix skeleton.
5. Finalist requotes at n=10000 (minutes).
6. `docs/paper/quantization_appendix.md` working notes (mirroring
   `architecture_probes_appendix.md`) + Appendix D in `paper.typ` + update the
   future-work sentence at paper.typ:949 + data bundle.

## Contingency

If the LOO fingers `a_log`, add an affine (asymmetric, zero-point) quantizer
variant for one-sided tensors (~10 lines in `_quantize_matrix`) and one sweep
row; report symmetric-vs-affine for that tensor. Not built preemptively.

## Cost estimate

PTQ + LOO + finalist requotes: < 1 hour of compute. QAT-fine-tune: ~0.5 day.
QAT-scratch: ~2.5-3 days (matched 512 x 20k). Code + bench + appendix text:
~2-3 days of work. All numbers on the dev machine, single concurrent training
(per the recorded training-throughput constraints).
