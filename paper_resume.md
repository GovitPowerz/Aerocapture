# Aerocapture NN Paper — Session Resume

> **Purpose:** let a fresh session pick up the paper work without re-reading the whole history.
> **As of:** 2026-06-18. **Phase:** campaign 00-09 DONE + analysed; 5b/5c (compute + robustness) DONE; **headline NN FINAL** (515-net plateaued, see below; 972-net confirms bigger-is-worse-for-GA). 10 (arch sweep) about to run; 11 PENDING. Then figures + Typst.
> **Branch:** `feature/parameter_sweep`. Never push. The user runs the heavy training; the assistant sets up configs/runners, analyses results, keeps docs current.

## TL;DR

Comprehensive **Typst** research paper — the follow-up to **Gelly & Vernis, AIAA GNC 2009** ("Neural Networks as a Guidance Solution for Soft-Landing and Aerocapture"). Benchmarks the repo's NN aerocapture guidance vs classical + predictor-corrector schemes, and presents the training methodology. On 2026-06-12 the campaign was **reorganized for reproducibility and reset to one regime**: study-named runners in `experiments/paper/`, cell-named configs in `configs/training/paper/`, all prior outputs wiped (except footnoted legacy dirs). As the campaign ran (06-14 → 06-18) several headline claims FLIPPED — read "Live results" + "Headline" below, not the older numbers in the spec.

## Authoritative docs (read first, in order)

1. **Plan v2 (THE live build plan: aggregate → figures → Typst → smart-commit):** `docs/superpowers/plans/2026-06-12-aerocapture-nn-article-v2.md`. Its §8.x results bullets carry the up-to-date per-study readings. (The 06-08 plan is bannered DO NOT EXECUTE.)
2. **Campaign guide:** `experiments/paper/README.md` ← run order, reuse cells, reporting rules, legacy dirs.
3. **Reorg spec (layout + wipe policy + bundle):** `docs/superpowers/specs/2026-06-12-paper-experiments-reorg-design.md`.
4. **Original study spec:** `docs/superpowers/specs/2026-06-08-aerocapture-nn-article-design.md` ← study DEFINITIONS only; its §5 numbers are HISTORICAL (pre-wipe), bannered/annotated as superseded.
5. **Source extracts (prose + citations + voice):** `articles/markdown/00..05`.

## The narrative (current)

17-year arc: 2009 feed-forward NN + GA for aerocapture → 2015-17 recurrent NN for speech (CG-LSTM, QPSO, divide-and-conquer, custom losses) → now stateful NN guidance trained for robustness, vs FTC + predictor-correctors on a bit-validated simulator.

**Two-part contribution (the spine):**

1. **The moving-environment training methodology IS the contribution (Study C).** GA does NOT win because it is robust to a moving objective — it WINS BECAUSE IT NEEDS one. Under FIXED seeds GA is the WORST optimizer (160.3, overfits the repeated scenarios); non-stationary (adaptive) seeds RESCUE it: fixed→rotating→adaptive = 160.3→120.0→118.0 (−42 m/s, the campaign's biggest single effect). CMA-ES is FLAT (~127). Iso-compute clincher: GA rotating-vs-fixed is +40 m/s at 1.14× compute; CMA-ES is +0 at EXACT iso-compute → it's the seed strategy, not compute. So the **adaptive-seed methodology converts GA from worst optimizer to best** — without it you'd deploy CMA-ES; with it, GA dominates. The quartet (GA + adaptive seeds + cubed transform + max bucket) is a matched SYSTEM, not four independent wins.

2. **Worst-case-leaning objective shaping minimizes the mission cost (Studies D + C-sub).** Mission framing (Grégory): propellant (ergols) is sized for the FAR-tail design case (3σ ≈ p99.87 / CVaR99.9 / worst-case), NOT p95 — the tail IS the mission cost function, the mean is operationally near-irrelevant. `cost_transform = cubed` (tail-weighted) and `bucket_selection = max` (hardest seed per cost-CDF bin) are the SAME mechanism: both compress the design-case extreme tail by forcing the policy onto hard cases. Both are VINDICATED at the correct sizing depth (n=10000 far-tail eval), where shallow metrics (mean, CVaR95) would have favored milder choices. Legs 3+4 of the quartet are one idea.

**Plus:** the **parameter-efficiency** story (capability floor: a ~515-param net is the sweet spot, beats 3998; no collapse down to 102 params) and the **allocation** story (Study F: many generations × few sims/gen beats balanced — few-sims noise is bought out by more non-stationary diversity, deepening Study C).

## THE HEADLINE NN (supersedes ga_300 AND adaptive_2) — FINAL (plateaued 2026-06-18)

> **⚠ HEADLINE SETTLED (10c three-way, 2026-06-25): the sizing headline is Mamba_962** (`training_output/mamba_p962_long/`), dense_515 = efficiency reference. 3-seed σ_run far-tail CVaR99.9 mean: **MAMBA 124.5 < LSTM 129.2 < DENSE 139.2**; max: MAMBA 127.6 < LSTM 132.4 < DENSE 159.0. BOTH recurrent nets beat dense beyond σ_run (LSTM max [126-138] non-overlapping with dense [146-184]) — the "single lucky arch" rebuttal is dead. Mamba edges LSTM by ~4-5 m/s on every tail metric AND is the most consistent (CVaR95 std 2.65 vs lstm 4.39) → Mamba deploys, LSTM close 2nd. dense_515's s1 was a lucky draw (CVaR99.9 128.1 vs its 3-run mean 139.2). The dense-515 finalization below (bundle/aggregator/ablation) now describes the EFFICIENCY reference; re-point the SIZING headline to Mamba_962.

**`training_output/dense_p515_ga_paper_best/` — 515-param dense net, GA + adaptive + cubed + max, n_sims=2, n_pop=512, 20000 gens.** (n_pop=512 verified from the checkpoint population shape (512, 518), NOT 300 — challenger/repeat runs must match.)
Deployed (n=1000): **109.7 mean / 117.0 CVaR95 / 120.6 CVaR99**, 100% capture.
Far-tail (n=10000, the SIZING depth): **CVaR99.9 128.1 / max 146.7**. val RMS plateaued at **1.326e6**.

Why it's the headline: it **dominates the design tail** at **1/8 the parameters** of the 3998-param candidates, and beats the best classical (joint-FTC / FNPAG, CVaR99.9 ~164/165) by **~36 m/s at the sizing tail** and ~15 m/s on the mean. Candidate hierarchy (far-tail CVaR99.9): **515@n=2/20k 128.1** < adaptive_2 (3998@n=2/10k) 139.3 < ga_300 (3998@n=10/2k) 152.9. The smaller net gives the TIGHTER worst-case — the parameter-efficiency point at the metric that matters.

**Plateau + the GA-dimensionality result (2026-06-18, head-to-head 515 vs 972, SAME allocation n=2/20000 gens):** the 515 val RMS decelerated to a plateau (1.81→1.54→1.376→1.330e6, drops −0.27/−0.16/−0.046 per 5k). The 972-param net, trained identically to 20000 gens, **plateaued HIGHER (val RMS 1.433e6 vs 1.326e6) and deploys WORSE** (112.4 / CVaR95 121.6 / CVaR99.9 130.7 vs 109.7 / 117.0 / 128.1 — 515 better on mean/CVaR95/CVaR99/CVaR99.9 by 2.6-4.6 m/s; only the single-sample max is worse). So **more parameters did NOT help — they hurt**: bigger nets are harder for the GA to optimize and give no capacity benefit. This CONFIRMS the GA-dimensionality hypothesis and REFUTES the "more plasticity → learn faster/better" intuition (a gradient-descent phenomenon that does NOT transfer to a gradient-free GA). The 1000-net "plasticity" test is thereby answered (negatively) by the 972 run. (The non-stationary objective still means training is compute-bound — both nets kept improving for ~15-20k gens before plateauing — but capacity above ~515 is wasted.)

**Provenance caveat:** trained via 4× resume (5000-gen steps). Resume is a VALID continuation (see "Resume equivalence" below) but a DIFFERENT allocation (n_sims=2/20000) than the controlled studies (n_sims=10/2000) — state this in the methods. Not yet in the committed bundle / aggregator paired tables.

## Live campaign results (deployed DV mean / CVaR95 m/s, 100% capture unless noted)

- **Classical (01):** FNPAG **124.3 / 144.0** (surprise — the 2026-06 density-fix made it the best classical; converged by gen 59 so 371 gens was plenty), pred_guid 167.4 / 227.1, FTC-fixed 170.7 / 244.1, EC 176.7 / 245.8 (99.6%), eqglide 200.3 / 327.6 (99.5%), piecewise 258.3 / 421.1 (99.8%). Ordering flipped from pre-fix (pred_guid sign-fix + FNPAG density-fix).
- **Study A optimizer × budget (02, dense_p3998):** GA best @150 (118.0) & @300 (120.4) but **GA@60 COLLAPSED (166.3** — n_pop=60 starves the 4000-dim search; "GA dominates at every budget" REFUTED, GA@60 is N=1). islands budget-robust (123.7/120.1/122.2).
- **Optimizer × dimensionality (03):** at **26p (FTC) all optimizers TIE ~170** (CMA-ES-low-dim hypothesis REFUTED; GA actually worst on the 26p tail); GA separates only at 515/3998. → optimizer matters for NN weights, NOT classical-gain tuning. Confound stated: 26p is FTC (different scheme), not a pure width sweep.
- **Study B output-param (03, all GA):** atan2 117.4/128.7 > delta 119.9/141.6 > scaledpi 122.2/140.4. The pre-fix 25 m/s gap was a different-optimizer artifact; real edge is ~12 m/s on the TAIL.
- **Study C seed-strategy (04):** the methodology result — see narrative (GA needs non-stationarity; 42 m/s swing; iso-compute clincher).
- **Study D cost_transform (05) + far-tail:** the optimal transform DEPENDS on the sizing percentile. Shallow (CVaR95): sqrt edges cubed. FAR tail (n=10000): TIE at p99/CVaR99 (~143-145), but cubed WINS CVaR99.9 (153.0) & max (160.1, vs 162-181) by compressing the extreme. log worst everywhere. → cubed VINDICATED for far-tail sizing; the earlier "cubed past the optimum / switch to sqrt" was a too-shallow (CVaR95) artifact, WALKED BACK.
- **Study C-sub curation (06) + far-tail:** bucket=max VINDICATED — dominates the far tail (CVaR99.9 153.0 vs random 173, middle 194, min 226). min has BEST mean (117.8) but catastrophic extreme (max 245) — the optimize-the-average-blow-up-the-worst-case illustration. Trim refuted again.
- **Study E joint reference (07) — user hypothesis CONFIRMED:** co-optimizing the constant-bank reference recovers FTC 170.7→**126.2** (−44 m/s, paired p=3e-165, 100% win), EC →142.1 (−35), pred_guid →144.2 (−23). FTC's weakness WAS the reference. **joint-FTC (126.2 / CVaR95 142.9) now MATCHES FNPAG (124.3 / 144.0)** and is ANALYTIC/fast → the new best classical.
- **Study F training_n_sims (08):** view A (rotating, fixed gens) sweet spot n_sims=10; view B (adaptive, allocation) **n_sims=2 @ 10000 gens dominates** (109.9 mean) — but used 1.8× actual sims (21M vs 11.6M; validation/curation scale with n_gen, ~8 m/s is genuine allocation gain). User's efficiency note: n_sims=2 is best but ~2× the wall-time of n_sims=5.
- **Capability floor (09):** NO collapse — even a 102-param net guides at 100% capture (120.8 mean). **Sweet spot ~515 params (117.4/128.7) BEATS 3998 (120.4/137.6)** → 3998 is over-parameterized. Below ~300p the tail degrades gracefully while mean/capture hold. This is why the headline is a 515-net.
- **5c robustness (deployed policies on high atmo/density/nav, 9M stress pool):** joint-FTC MOST robust (94.5% capture, −5.5) > NN 93.8% > FNPAG 92.4% > PredGuid 90.8% >> **FTC-fixed 67.1% (catastrophic)**. A well-referenced FTC > FNPAG on robustness (confirms the hypothesis); FTC-fixed's fragility is the reference (ties Study E — reference drives FTC accuracy AND robustness).
- **5b compute (1 core, idle-box publication-grade):** FTC 1.29 ms/sim, **NN 3.21 (2.5× FTC), FNPAG 87.0 (68× FTC, 27× NN)**. FNPAG is DOMINATED: joint-FTC matches its accuracy + beats its robustness at 68× less compute; the NN beats its accuracy at 27× less.

**The 3-way deployability triangle (NN / joint-FTC / FNPAG):** accuracy NN > joint-FTC ≈ FNPAG · compute FTC ≈ NN ≫ FNPAG (68×) · robustness joint-FTC > NN > FNPAG. With the 515-net headline the NN's accuracy lead over the best classical is ~13-15 m/s mean / **~35 m/s at the design tail**.

## Pending decisions & next steps (priority order)

1. ~~FINALIZE the 515-net as headline~~ **DONE (commit aed0fe1).** Bundled `dense_p515` + `dense_p972` under `runs/headline/`; aggregator HEADLINE + `nn_vs_{ftc,fnpag,jointftc}` + `nn515_vs_nn972` re-pointed (NN−jointFTC **−16.5 m/s**, 99.9% win, p=3e-165; 515−972 **−2.69**, 78% win, p=1e-83; `ga300_vs_*` kept as the controlled-allocation reference). Fresh-pool (8M) bundled: **109.5 p50 / 116.9 CVaR95 / 117.9 p99**, matches the 2M pool (no selection optimism). Ablation: leans on **hdot_nominal (+4.06)** + autoregressive **predicted_dv3/1/2** + radial_velocity + pdyn_error. Input-report: at the cost threshold ALL trajectories are low-DV (no failure tail); even at a median split no single input separates outcomes (sep <0.05) → residual DV is irreducible scenario noise, not a policy weakness. **Canonical headline config: `configs/training/msr_aller_nn_atan2_best_paper.toml`** (base-inherits `sweep/dense_p515.toml`, overrides arch + deploy path; `_best_paper_1000.toml` is the 972). STILL TODO: re-run far-tail `robustness_stress.py` / `compute_benchmark.py` with the 515-net as the NN (both still point at `paper/optimizer_budget/ga_300`).
2. ~~+5000 gens~~ DONE — 515 trained to 20000 gens, plateaued (val RMS 1.330e6, deployed 109.7).
3. ~~1000-net plasticity test~~ ANSWERED by the 972-net: more params plateau worse (1.433e6 vs 1.326e6) and deploy worse — GA-dimensionality confirmed, plasticity intuition refuted.
4. ~~Run 10 (architecture sweep)~~ **DONE** (user ran it at n_sims=2 / n_gen=5000 / **n_pop=512**, all 24 cells 100% capture). Results (`configs/training/sweep/pareto_results.json`, 1000-seed dv50): all archs perform well; the sweep is now an explicit per-cell resumable runner (`10_architecture_sweep.sh`, skip-if-`final_selection.json`). KEY NUANCE for the paper: at the EQUAL 5000-gen budget dense is **mid-pack** — best cells are recurrent (lstm_p1082 **111.6**, gru_p4082 112.2, mamba ~114) vs best dense cell 115.8. So "dense is best" is a CHOICE argument (simplest arch reaching the best LONG-budget result = 109.7 headline; fastest; no degenerate regime), NOT equal-budget dominance. Transformer is worst (762: 121.4) — NOT "few params" (762 > dense 515): at that budget attention is forced to d_model=8 / 2-dims-per-head; high attention overhead + GA-hostility, never good even at 3822. Recurrent ≈ dense because the engineered autoregressive inputs already carry the temporal info (ablation confirms the headline leans on predicted_dv3/1/2 + bank-history-cos + hdot_nominal). mamba is consistently ~114 (real small edge); lstm/gru are SPIKY (one good cell each = likely single-run luck — no per-cell repeats).
5. ~~Run 10b (long recurrent challengers)~~ **DONE — and it REVERSES "dense is best" on the SIZING TAIL (my prediction was wrong).** lstm_p1082 / mamba_p962 / gru_p1014 extended to 20000 gens (n=2/512) at `training_output/<arch>_p<N>_long/`. Far-tail (n=10000), sizing order by CVaR99.9 / max: **Mamba_962 122.0 / 124.4** < **LSTM_1082 123.2 / 126.0** < GRU_1014 126.1 / 130.6 < **DENSE_515 (headline) 128.1 / 146.7** < DENSE_972 130.7 / 137.0. ALL THREE recurrent nets beat the dense headline on the tank-sizing tail; Mamba trims CVaR99.9 −6.1 and the worst case **−22 m/s**. The dense net's tight median (109.2) masked a FAT extreme tail (max 146.7). **Equal-capacity control proves architecture not size:** mamba_962 (122.0) vs dense_972 (130.7) at ~960 params → recurrent −8.7 m/s CVaR99.9 (and dense_972 worse than dense_515 — GA-dimensionality, dense can't buy out with params). Mechanism: engineered autoregressive inputs flatten the BULK (all archs median ~108–112) but genuine internal state wins the EXTREME tail. Val RMS corroborates: lstm 1.276M < dense_515 1.326M ≈ mamba 1.331M. CAVEAT: single runs — magnitude needs σ_run (→ 10c). Tradeoff: recurrent wins the tail at ~2× params + stateful runtime; dense_515 stays efficiency-optimal.
6. ~~Run 10c (tail σ_run)~~ **DONE — VERDICT: Mamba_962 beats Dense_515 on the sizing tail BEYOND σ_run. The deployed headline should become Mamba_962.** s2/s3 fresh runs (`training_output/paper/tail_repeats/{dense515,mamba962}_s{2,3}`), all 100% capture at n=10000. σ_run over {s1,s2,s3} (far-tail CVaR99.9 mean / range): **DENSE 139.2 / 21.5** (s1 128.1 was the LUCKY draw — typical ~140!) vs **MAMBA 124.5 / 9.3**. max: DENSE 159.0 / 38.3 (s3 hit **184**) vs MAMBA 127.6 / 15.2. Overlap test deepening into the tail: CVaR95 ΔMean −5.6 (overlap), CVaR99.9 −14.7 (barely overlap), **max −31.4 NON-OVERLAPPING (mamba's worst run 136.9 < dense's best 146.1)**. So (a) my 10b quote used dense's favorable s1 — the REAL gap is ~−15 CVaR99.9, not −6; (b) Mamba is also ~2× more CONSISTENT (half the σ_run) — the dense net's extreme tail is a high-variance crapshoot, mamba's state gives robust tail behavior. The advantage GROWS deeper into the tail (where tanks are sized). Headline → **Mamba_962** (sizing); dense_515 = efficiency reference (½ params, memoryless). **LSTM s2/s3 DONE — three-way σ_run settles it: MAMBA wins** (CVaR99.9 mean MAMBA 124.5 < LSTM 129.2 < DENSE 139.2; max 127.6 < 132.4 < 159.0). Both recurrent beat dense beyond σ_run; mamba edges lstm ~4-5 m/s on every tail metric + tightest σ_run (CVaR95 std 2.65 vs 4.39). **HEADLINE LOCKED = Mamba_962.**
7. **FINALIZE the Mamba_962 headline — NEXT.** Same pass done for dense_515 (commit aed0fe1): add `mamba_p962_long` to `collect_runs.py` HEADLINE + bundle; re-point aggregator HEADLINE + `nn_vs_{ftc,fnpag,jointftc}` paired tables from dense_p515 → mamba_p962_long; fresh-pool requote (8M); ablation + nn_input_report on mamba_p962_long; re-run far-tail robustness/compute with Mamba as the NN. Keep dense_515 bundled as the efficiency reference. Also bundle the 6 tail_repeats cells + lstm_p1082_long as the σ_run evidence.
8. ~~Run 11 (seed-repeats)~~ **OBSOLETE / SKIP.** Never run (0 cells, aggregator sigma_run empty). It would σ_run the OPTIMIZER studies (02/03) at n=10/2000 on the MEAN — but (a) those studies' real effects (GA@60 collapse, adaptive≫fixed, CMA-ES high-dim) are 10-40× σ_run, no error bars needed; (b) the tight ties (GA@150≈@300) are reported "indistinguishable", supportable by the σ_run 10c ALREADY measured; (c) it calibrates the MEAN, but the paper leads with the TAIL. If a stated optimizer RANKING is ever wanted, run a REDUCED version on the cells actually ranked + compute σ_run on CVaR95, not the mean.
8. Then Phase 2/3 of plan v2: aggregate → figures (Task 3) → Typst (Tasks 6-14) → smart-commit.

## Analysis tooling (`articles/paper/scripts/`)

- `aggregate_results.py` → `articles/paper/data/results.json`: per-run capture/DV stats (incl. p99, CVaR95, CVaR99, p99.9 + bootstrap CIs), within-transform best-val, actual-sims accounting, 15 named PAIRED tables (paired bootstrap + Wilcoxon, dispersion-fingerprint asserted), σ_run pooling from the seed_repeats triplets, fresh-pool headline slot. Run after each `12_collect_results.sh`.
- `far_tail_eval.py` — **the sizing eval.** Re-runs a deployed cell on the full reserved 2M pool at **n=10000** (training-disjoint) for stable p99.9 / CVaR99 / CVaR99.9 with CIs (n=1000 can't estimate the far tail). Accumulates cells across runs into `far_tail_eval.json`. Label = dir under `training_output/paper` or `training_output`; PASS THE RIGHT LABEL (a bad label silently scores an untrained default — happened once).
- `fresh_pool_requote.py` — the ABSTRACT number: re-quotes the deployed headline on a FRESH pool (`HEADLINE_REQUOTE_SEED_OFFSET = 8_000_000`, disjoint), avoiding selection-on-test (the headline was chosen by sweeps scored on the 2M pool).
- `compute_benchmark.py` — single-core ms/sim per scheme (needs an IDLE box; the whole-sim ratio under-states the pure guidance-cost gap). DONE (publication-grade): FTC 1.29 / NN 3.21 / FNPAG 87.0.
- `robustness_stress.py` (`STRESS_EVAL_SEED_OFFSET = 9_000_000`) — deployed policies on a harder MC regime (atmo/density/nav/nav_filter = high). DONE. Schemes: NN / joint-FTC / FTC-fixed / FNPAG / PredGuid.

Seed offsets (all disjoint): VALIDATION 1M, FINAL_EVAL 2M, RL 3M, WARM_START 4M, NN_INPUT_REPORT 5M, CALIBRATION 6M, SWEEP_EVAL 7M, HEADLINE_REQUOTE 8M, STRESS_EVAL 9M.

## Methodology notes

- **Sizing metric = FAR tail.** Quote p99 + CVaR95 AND p99.9 + CVaR99.9 with bootstrap CIs; CVaR99.9 is the headline sizing metric; the sample max (≈p99.99 at n=10000) is a descriptive bound. The far tail is unreliable at n=1000 (~1-10 samples) → use `far_tail_eval.py` (n=10000) for any sizing decision.
- **Resume equivalence (4×5000 ≈ 1×20000? — yes as optimization, not bit-identical).** On resume the trainer RNG state (`rng.bit_generator.state`), the full GA population (pop_X+pop_F from the .npz), and the SeedCurator state (seed_list, last_curation_gen) are all RESTORED — so the **non-stationary seed schedule + population + best-so-far continue faithfully**. Only pymoo's operator RNG (SBX/PM) is re-realized (`warm_start_algorithm` calls `setup(seed=None)`, train.py:1371) — but that stream is UNSEEDED even within a single run, so resume adds NO extra non-determinism. GA has no generation-dependent operators (fixed eta/mutation_prob) so the reset is clean (NB: QPSO would re-anneal alpha on resume; GA dodges it). Consequence: not bit-reproducible from `--seed` alone (it never was) — reproduce via the deployed `best_model.json` + checkpoints. This is also WHY study 11 (seed-repeats) exists.
- **Compute accounting:** "compute-matched" claims must report ACTUAL sims (from JSONL: training = n_pop×n_sims×n_gen; validation fires ~58-80% of gens × 1000; curation ≈1000 sims/event every ≤2 gens). The nominal n_sims×n_gen budget under-counts because validation/curation scale with n_gen.
- **Paired comparisons** on the shared 1000-seed final-eval pool (offset 2M; prefix property: n=1000 ⊂ n=2000 ⊂ n=10000). σ_run from 11 calibrates every N=1 comparison.

## The campaign (`experiments/paper/`, run from repo root, in order)

| # | Script | Study | Status |
|---|---|---|---|
| 00 | `00_prereqs.sh` | corridor + mission ref + PC classical row | DONE |
| 01 | `01_classical_baselines.sh` | classical GA @2000×300 (ftc/eqglide/ec/pred_guid/fnpag) | DONE |
| 02 | `02_optimizer_budget.sh` | Study A (6 opt × 3 budgets, dense_p3998) | DONE |
| 03 | `03_optimizer_dimensionality.sh` | opt × width (26p FTC / 515p) + Study B | DONE |
| 04 | `04_seed_strategy.sh` | Study C fixed/rotating (adaptive = 02 @150) | DONE |
| 05 | `05_cost_transform.sh` | Study D (cubed = 02 ga_300) | DONE |
| 06 | `06_curation_shaping.sh` | C-sub bucket + trim (max = 02 ga_300) | DONE |
| 07 | `07_joint_reference.sh` | Study E (baselines from 01) | DONE |
| 08 | `08_training_n_sims.sh` | Study F (rotating noise floor + adaptive allocation) | DONE |
| 09 | `09_capability_floor.sh` | sub-500 dense (p102/201/298/416) | DONE |
| 10 | `10_architecture_sweep.sh` | 6-family Pareto (GA post-fix) via param_sweep | **PENDING** (allocation decision — see next-steps #4) |
| 11 | `11_seed_repeats.sh` | σ_run repeats (s1 from 01/02/03) | **PENDING** (add 515-net/adaptive_2 repeat) |
| 12 | `12_collect_results.sh` | → `articles/paper/data/runs/` committed bundle | rerun after each study |

Plus the 5b/5c/far-tail/fresh-pool eval scripts (above) — NOT in the numbered runners. All runners: skip-if-done per cell, `--sim-timeout 5`. Never run two cells of the same config TOML concurrently; never regenerate `training_output/mars/` while a ref-tracking scheme trains.

## Defaults in `common.toml` (the ONE controlled-study regime)

`cost_transform = "cubed"` · `curation_bucket_selection = "max"` · `algorithm = "ga"` · `seed_strategy = "adaptive"` · `training_n_sims = 10` · `curation_top_k = 1` · `seed_pool_interval = 2` · `validation_n_sims = 1000`. Every campaign cell inherits these; per-cell deltas live in the cell config name. NB the deployed HEADLINE uses a different allocation (n_sims=2/15000) — that is the best *deployable* config, not the controlled-study regime.

## Legacy dirs (preserved, PRE-FIX regime — footnote when quoted)

RL (`neural_network_rl` 636/973/1185, `neural_network_gru_ppo` 513/829, `neural_network_atan2_{ppo,rl,best}`, `neural_network_rl_explore`), warm-start/joint (`paper_opt_warmstart` 132.4, `{best_,}neural_network_joint`, `neural_gru_joint`, `neural_network_warm`), quant/pruning (`neural_network_atan2` 119.0/132.0/165.2 base, `_qat8` 125.1, `_qat4` 128.7, all `*pruned*` + bases). Bundled under `articles/paper/data/runs/legacy/`. Conclusions (RL ~5× worse, warm-start below plain GA, QAT/pruning deployability) are regime-insensitive; NOT re-run. Pre-wipe historical numbers: spec §5 + git history.

## How to extract numbers (reusable)

Capture = `ifinal==3 & eccentricity<1.0`; DV = `dv_total_m_s` over captured; one `final_eval.parquet` per run dir (under `training_output/paper/<study>/<cell>/`, canonical scheme/sweep dirs, or the committed bundle).

```python
import numpy as np, pyarrow.parquet as pq
def stats(path):  # n=1000 -> mean/CVaR95 reliable; far tail needs far_tail_eval.py (n=10000)
    df = pq.read_table(f"{path}/final_eval.parquet").to_pandas()
    cap = (df["ifinal"]==3) & (df["eccentricity"]<1.0)
    dvc = df.loc[cap,"dv_total_m_s"].to_numpy()
    cv = lambda lv: float(np.sort(dvc)[-max(1,int(round(len(dvc)*(1-lv)))):].mean())
    return dict(n=len(df), cap=round(100*cap.mean(),1), mean=round(dvc.mean(),1),
                p95=round(float(np.percentile(dvc,95)),1), p99=round(float(np.percentile(dvc,99)),1),
                cvar95=round(cv(.95),1), cvar99=round(cv(.99),1))
```

(`cost_transform` rescales the TRAINING best-val, so it is NOT comparable across transforms — use deployed DV. Within one transform, training best-val from `run_*.jsonl` `validation.rms_cost` is comparable, but it's in cubed-cost space.)

## Process notes

- Flow: `brainstorming` → spec → `writing-plans` → inline execution. New knobs added this project (all in `train.py`/`OptimizerConfig`/`SeedCurator`): `--output-dir`, `--seed-strategy`, `--training-n-sims`, `--seed`, `[optimizer] curation_trim_fraction` (null result), `[optimizer] curation_bucket_selection`.
- The interrupted-run report bug was fixed (an interrupted training no longer writes a self-certifying `final_eval.parquet`). The far_tail/compute/robustness/fresh-pool scripts pin the run-local `best_model.json` + co-trained scaffolding so they score the right model.
- Memory files updated this project: `project_seed_strategy_result` (GA needs non-stationarity), `project_tail_sizing_rationale`, `project_cost_function_design` (cubed vindicated far-tail).
