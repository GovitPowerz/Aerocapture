# Quantization of the deployed Mamba-962 guidance head

Status: WORKING NOTES, complete and integration-ready (2026-07-12). NOT yet in
paper.typ -- integration deferred to a dedicated writing session (target:
Appendix D, plus the future-work sentence rewrite at paper.typ:949). This file
is the single source for that session; every number below is generated from the
committed JSONs in `articles/paper/data/quant/`, none hand-typed.

## Provenance

- Branch `feature/quantization-mamba962`. Spec
  `docs/superpowers/specs/2026-07-10-quantization-study-appendix-design.md`;
  plan `docs/superpowers/plans/2026-07-10-quantization-study-appendix.md`.
- Target: `training_output/mamba_p962_long` -- Dense(17->16, swish) ->
  Mamba(16, d_state 12, dt_rank 1) -> Dense(16->2, asinh), 962 NN params,
  atan2_signed, 17-input calibrated mask, scaffolding = live (4 co-trained
  nav/shaping overrides), GA 512 x 20k.
- PTQ: weight-only symmetric fake-quant (rounded values stored back as f64;
  runtime and goldens untouched). Grid bits {8,6,4,3,2} x granularity
  {per_channel, per_tensor} x tensor policy {all, proj_only}, n=1000/cell;
  leave-one-out at 4 bits over the 6 quantizable tensor groups. Tooling:
  `aerocapture.training.quantize` (sweep CLI), `experiments/paper/17_quantization.sh`.
- QAT: GA-in-the-loop fake-quant (`[network] qat_bits/qat_granularity/
  qat_tensor_policy`; population rounded before every `run_grid` eval, deploy
  writer rounds `best_model.json` -- both deployed models audited on-grid at
  atol 1e-12). Cell pinned by the pre-registered PTQ verdict rule (best-capture
  then best-CVaR95 at 4 bits): **4b per_channel proj_only**.
  - Fine-tune arm: champion checkpoint g20000 resumed +3000 gens (exactly
    g23000), `training_output/quant/mamba962_qat4_finetune/`.
  - From-scratch arm: GA 512 x 20000, matched champion budget,
    `training_output/quant/mamba962_qat4_scratch/`. Wall time ~6.5 h at
    1.2-4 s/gen (QAT gens are much cheaper than the champion's late-phase
    gens -- use this pace for future QAT budget estimates).
- Eval: every row scored on the reserved fresh pool
  (`HEADLINE_REQUOTE_SEED_OFFSET = 8_000_000`, base seed 42) with the model's
  co-trained `best_params.json` scaffolding applied (recorded per row in the
  JSON). Sanity gate passed: the sweep baseline reproduced
  `fresh_pool_requote.json` to the decimal (capture 1.000, p50 109.7, CVaR95
  115.2), and champion_fp at n=10k (115.8) matches the n=1k quote within noise.
- Bench: `src/rust/benches/quant_forward.rs`, criterion medians on Apple
  Silicon (dev machine, idle), plain scalar Rust, SSM recurrence fp in ALL
  variants. Runs 2026-07-12; sim throughput untouched by construction.
- Ops incidents during the campaign (reproducibility notes, not paper
  content): pymoo 0.6.2 re-entered via a lock refresh and SIGABRT'd both QAT
  arms after 2 gens (now pinned <0.6.2 in pyproject); `--sim-timeout 120`
  added to the QAT phases per the recorded NaN-hang lesson.

## One-line result

Weight-only 8-bit quantization of the deployed selective-SSM head is free; at
4 bits the SSM dynamics parameters (`a_log`, `d_skip`) are the bottleneck --
keeping those 242 scalars in fp holds 100% capture, and 3000 generations of
quantization-aware fine-tuning recover 98% of the remaining tail cost
(CVaR95 116.6 vs fp 115.8 m/s, +0.9 -- within the probe-campaign sigma_run of
1.2-2.1); the deployment benefit is memory (7696 B -> 1564 B), not compute.

## Headline: finalists (n = 10000, same pool, scaffolding applied)

| variant | capture | DV p50 | p95 | p99 | CVaR95 | d CVaR95 vs fp | viol % |
|---|---|---|---|---|---|---|---|
| champion_fp | 1.0000 | 109.7 | 114.2 | 116.8 | 115.8 | +0.0 | 0.00 |
| ptq4_verdict | 1.0000 | 126.7 | 143.4 | 153.5 | 149.6 | +33.8 | 0.00 |
| qat4_finetune | 1.0000 | 110.2 | 115.2 | 117.6 | 116.6 | +0.9 | 0.00 |
| qat4_scratch | 1.0000 | 112.4 | 121.1 | 126.6 | 124.5 | +8.7 | 0.00 |

Reading: PTQ at the best 4-bit cell costs +33.8 m/s of sizing tail.
Fine-tuning under the quantized fitness recovers it to +0.9 -- statistically
indistinguishable from fp against sigma_run(CVaR95) ~ 1.2-2.1 m/s measured in
the Appendix C probe repeats (probe budget; directional context, not a formal
test at this budget). From-scratch QAT at the matched 512x20k budget reaches
+8.7: the 4-bit constraint is trainable to a competent policy but does not
reach the champion's basin; fine-tuning is the better path. Val-RMS ordering
agrees (fp 1.331e6 < finetune 1.350e6 < scratch 1.457e6).

## Sensitivity: which parts of a selective SSM tolerate 4 bits

Leave-one-out at 4 bits (per_channel; one tensor group quantized, rest fp):

| tensor | params | capture | d CVaR95 |
|---|---|---|---|
| layer_0.w (input dense) | 272 | 1.000 | +59.0 |
| layer_1.x_proj_w | 400 | 1.000 | +15.8 |
| layer_1.dt_proj_w | 16 | 1.000 | +0.0 |
| layer_1.a_log | 192 | 1.000 | +53.0 |
| layer_1.d_skip | 16 | 1.000 | +47.4 |
| layer_2.w (output dense) | 32 | 1.000 | +22.5 |

The spec's hypothesis is confirmed: the SSM dynamics parameters are
disproportionately sensitive -- `a_log` (+53.0; exponentiated at runtime,
A = -exp(a_log), and one-sided so a symmetric grid wastes half its levels) and
`d_skip` (+47.4 from just 16 scalars, the per-channel residual gains). The
input dense layer (+59.0) is the other large contributor. `dt_proj_w` is
exactly 0 by construction: per-channel scaling of a (16,1) matrix is one scale
per element, i.e. lossless. LOO deltas do not add up to the all-tensors cell
(+119.9) -- interactions compound. This motivates the `proj_only` policy
(quantize projections, keep the 242 dynamics/bias scalars fp), and is the
appendix's SSM-specific claim: it echoes the outlier-sensitivity of SSM
dynamics reported in the Mamba-quantization literature, reproduced here on a
control task with a closed-loop, tail-led metric.

## PTQ grid (n = 1000/cell)

| bits | granularity | policy | capture | CVaR95 | d CVaR95 | viol % |
|---|---|---|---|---|---|---|
| 8 | per_channel | all | 1.000 | 115.7 | +0.4 | 0.00 |
| 8 | per_channel | proj_only | 1.000 | 115.8 | +0.6 | 0.00 |
| 8 | per_tensor | all | 1.000 | 119.6 | +4.4 | 0.00 |
| 8 | per_tensor | proj_only | 1.000 | 119.0 | +3.8 | 0.00 |
| 6 | per_channel | all | 1.000 | 137.4 | +22.2 | 0.00 |
| 6 | per_channel | proj_only | 1.000 | 134.3 | +19.1 | 0.00 |
| 6 | per_tensor | all | 1.000 | 123.7 | +8.4 | 0.00 |
| 6 | per_tensor | proj_only | 1.000 | 121.5 | +6.2 | 0.00 |
| 4 | per_channel | all | 0.772 | 235.1 | +119.9 | 8.90 |
| 4 | per_channel | proj_only | 1.000 | 147.9 | +32.7 | 0.00 |
| 4 | per_tensor | all | 0.846 | 204.0 | +88.8 | 6.50 |
| 4 | per_tensor | proj_only | 1.000 | 151.4 | +36.2 | 0.00 |
| 3 | per_channel | all | 0.863 | 209.9 | +94.7 | 1.30 |
| 3 | per_channel | proj_only | 0.932 | 204.0 | +88.7 | 1.40 |
| 3 | per_tensor | all | 1.000 | 149.7 | +34.5 | 0.00 |
| 3 | per_tensor | proj_only | 1.000 | 185.1 | +69.9 | 0.00 |
| 2 | per_channel | all | 0.519 | 291.7 | +176.5 | 5.80 |
| 2 | per_channel | proj_only | 0.188 | 347.7 | +232.5 | 60.60 |
| 2 | per_tensor | all | 0.000 | - | - | 100.00 |
| 2 | per_tensor | proj_only | 0.000 | - | - | 100.00 |

Reading: 8-bit per-channel is free (+0.4). Degradation is visible from 6 bits
and catastrophic below 4 (2-bit collapses capture entirely). Note the
non-monotone granularity interaction at 6 and 3 bits (per_tensor beats
per_channel in several cells, against the usual expectation); the cells are
single-model evaluations at n=1000 with no seed-repeats, so adjacent cells
within a few m/s should not be over-read -- but the 4-bit policy split
(proj_only holds capture 1.000 where all collapses to 0.77-0.85) is far above
any noise floor. The verdict rule (max capture, then min CVaR95, ties to
per_channel/all) selected 4b per_channel proj_only.

## Deployment benefit (the honest version)

Memory (analytic, exact; b-bit packed weights + f32 scales + f32 fp remainder;
f64 baseline 7696 B):

| bits | granularity | policy | quant | scales | fp | total | vs f64 |
|---|---|---|---|---|---|---|---|
| 8 | per_channel | proj_only | 720 | 236 | 968 | 1924 | x4.0 |
| 4 | per_channel | proj_only (deployed QAT cell) | 360 | 236 | 968 | **1564** | x4.9 |
| 4 | per_tensor | proj_only | 360 | 16 | 968 | 1344 | x5.7 |
| 4 | per_tensor | all | 464 | 24 | 136 | 624 | x12.3 |

The 624 B all-tensors cell is quoted for contrast only -- it is
accuracy-broken at 4 bits (capture 0.85). The honest headline is the deployed
cell: **7696 B -> 1564 B (x4.9)**, bounded below by the 968 B of fp dynamics
parameters the accuracy study says must stay fp.

Compute (criterion medians, ns per guidance tick, Apple Silicon, scalar Rust,
SSM recurrence fp in all variants; x 644 ticks/sim -> ms/sim next to the
paper's 3.68 ms/sim):

| kernel | ns/tick | vs f64 model | ms/sim (644 ticks) |
|---|---|---|---|
| f64 (deployed NeuralNetModel::forward) | 1888 | -- | 1.22 |
| f64 hand-rolled | 1670 | -11.6% | 1.08 |
| f32 hand-rolled | 1290 | -31.7% | 0.83 |
| w8a8 (int8 weights + dynamic int8 activations) | 1735 | -8.1% | 1.12 |
| w4a8 (packed int4 weights + int8 activations) | 1747 | -7.5% | 1.13 |

Reading: quantization is NOT a compute win at this scale. The NN forward is
~1.22 ms of the 3.68 ms/sim (33%); f32 is the compute sweet spot (-32%/tick,
~0.39 ms/sim); the integer kernels beat f64 by only ~8% because dynamic
activation quantization overhead plus the fp SSM recurrence (192 exp calls per
tick) dominate a 962-parameter workload, and w4a8's nibble unpacking cancels
its bandwidth advantage (~= w8a8). Simulation throughput in the accuracy study
is unchanged BY CONSTRUCTION (fake-quant stores rounded weights back as f64) --
any sim-time claim would be an artifact. The deployment case for 4-bit is
memory footprint and fixed-point-capable projection arithmetic, not speed.

## Caveats

- Weight-only accuracy study: activations, hidden state, and input
  normalization stay f64; the w8a8/w4a8 kernels quantize activations for the
  compute measurement only.
- Both QAT arms are single runs; deltas are quoted against
  sigma_run(CVaR95) ~ 1.2-2.1 m/s from the Appendix C probe repeats, which
  were measured at the probe budget (GA 300 x 5000), not at 512 x 20k.
- PTQ grid cells are n=1000, single model, no repeats; the finalists table is
  n=10000 on the same pool.
- Bench numbers are Apple Silicon scalar Rust; ratios do not transfer directly
  to flight processors (no wide FPU, different memory hierarchy) -- the memory
  table does.
- Fine-tune keeps the champion's scaffolding genes live during QAT (they moved
  within their bounds); the comparison is policy+scaffolding vs
  policy+scaffolding, same as every other row.

## Contingency status

The spec armed an affine (asymmetric) quantizer variant for one-sided tensors
if the LOO fingered `a_log`. It did (+53.0). The variant was NOT run --
`proj_only` sidesteps the issue at a 968 B fp cost, and the QAT result makes
the added arm unnecessary for the appendix's claims. It remains a one-line
extension (`_quantize_matrix` zero-point) plus one sweep row if a reviewer
asks whether `a_log`'s sensitivity is representational (symmetric-grid waste)
or intrinsic.

## Integration checklist (for the writing session)

1. Target: `= Appendix D: quantization of the deployed Mamba head` after
   Appendix C (paper.typ ~line 1066+), following C's table idioms.
2. Rewrite the future-work sentence at paper.typ:949. Current: "We have no
   clean campaign study of pruning or quantizing the deployed head -- the only
   such cells predate the simulator fixes in this work and are not comparable
   -- so deploy-size reduction of the Mamba policy is open." Suggested: point
   at Appendix D for the quantization half (PTQ sensitivity + two QAT arms);
   pruning remains open.
3. Candidate body pointer: the compute-cost paragraph at paper.typ ~940 (the
   3.68 ms vs 2.40 ms discussion) can cite the bench: the Mamba head is ~1.22
   ms of the 3.68, f32 would shave ~0.4 ms, quantization does not.
4. Suggested claims, in strength order: (a) 8-bit free; (b) SSM dynamics
   params are the 4-bit bottleneck (LOO); (c) QAT fine-tune = tail-equivalent
   4-bit head (+0.9 vs sigma_run 1.2-2.1); (d) scratch trainable but worse
   (+8.7); (e) memory x4.9, compute not a win. Lead tail-first per the
   sizing-metric convention (CVaR95/p95 before p50).
5. Data (committed, `articles/paper/data/quant/`): quantization_results.json
   (grid + LOO + verdict + memory + pool), finalists_results.json (headline,
   per-row scaffolding provenance), bench_forward.json (criterion medians +
   CI95), ticks_per_sim.json.
6. Figures (committed, `articles/paper/figures/`): quantization_sweep.svg
   (capture + CVaR95 vs bits, 4 series), quantization_loo.svg (tensor bars),
   quant_qat_convergence.svg (fp champion / fine-tune / scratch best-cost
   overlay). Tables may suffice for the appendix; figures are ready if wanted.
7. Numbers hygiene: everything above regenerates from the JSONs via the table
   script in the plan (Task 11 Step 1); the commit message of a83450b quotes
   1592 B for the deployed int4 cell -- a typo for 1564 B; the JSON is
   authoritative.
8. Reproduction: `./experiments/paper/17_quantization.sh {ptq|bench|
   qat_finetune|qat_scratch|finalists|collect}` from the repo root on the main
   checkout; QAT configs in `configs/training/quant/`; the two training dirs
   carry full checkpoints, JSONL logs, final_selection.json, and reports.
