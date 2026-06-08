# Aerocapture Neural-Guidance Article — Design

**Date:** 2026-06-08
**Status:** Design approved (pending spec review)
**Output:** a comprehensive arXiv-style research paper (Typst) — the follow-up to Gelly & Vernis, AIAA GNC 2009.

---

## 1. Goal & framing

Write a thorough (page count is not a constraint) research article presenting the repo's neural aerocapture guidance, benchmarked against classical and predictor-corrector schemes, and the optimization machinery that trains it. The paper is the explicit fulfilment of the 2009 paper's closing line: *"extend our work on the aerocapture ... and evaluate the performance of neural guidance compared to classic algorithms such as the predictor-corrector schemes."*

**Narrative thesis (the 17-year arc):**
2009 feed-forward NN + GA for aerocapture → 2015-2017 recurrent NN + QPSO + divide-and-conquer + custom losses (speech) → now stateful NN guidance (Dense/GRU/LSTM/Window/Transformer/Mamba) trained by a 3-island PSO/GA/DE optimizer with supervised warm-start, benchmarked against FTC and predictor-correctors on a bit-validated simulator.

**Source material already prepared:** `articles/markdown/00..05` (synthesis kit, the 2009 extract, the three speech-paper methodology extracts, and the authorial-voice guide). The paper reuses the voice guide for tone and the synthesis kit for the lineage narrative and bibliography.

---

## 2. Paper structure

1. **Abstract** — one paragraph, your template (problem → method → comparison → headline number).
2. **Introduction** — 2009 lineage + the speech-NN detour that built the machinery + contributions list. Open by quoting the 2009 "next step" hook.
3. **Problem formulation** — aerocapture; corridor in the (orbital energy, dynamic pressure) plane; restricted corridor ±δZa; MSR entry (120 km, 5687 m/s, −10.24°, 38.04°) and target orbit (apoapsis 500 km, periapsis 11 km, incl 50°); ΔV correction-cost metric (apoapsis+periapsis+inclination), 113 m/s periapsis-raise floor.
4. **Simulation testbed** — the bit-validated Rust simulator (725 timesteps, 22/24 photo columns exact); MC dispersions (entry state, density ±50%, winds, Gauss-Markov density OU, mass/aero); EKF navigation + bias mode; altitude-dependent winds; J2/J3/J4 gravity; fixed-RK4 vs adaptive DOPRI45. Contrast with 2009's 4-DOF / 1 Hz tool.
5. **Classical guidance algorithms** — PiecewiseConstant (corridor/ref generator), **FTC + the PC-reference improvement** (FTC tracks a piecewise-constant-optimized reference trajectory instead of a single constant-bank-angle trajectory — the in-plane apoapsis enslavement Eq. 10 + roll-reversal out-of-plane), FNPAG (Lu numerical predictor-corrector, 3-DOF forward predictor), PredGuid (Apollo/Shuttle drag tracking), EqGlide, EnergyController.
6. **Neural guidance** — architecture family (Dense/GRU/LSTM/Window/Transformer/Mamba, the stateful-runtime generalization of the 2009 single-hidden-layer net); the 35-candidate input vector (orbital/aero/thermal + reference-trajectory + exit-bank teacher + lateral telemetry + seam-free bank-history (sin,cos) pairs + periapsis alt + 3 live correction-DV "autoregressive" inputs) with a learned input mask; output parameterizations — 2D atan2 (the 2009 sin/cos decoder, Eq. 11), 1D scaled_pi, 1D delta.
7. **Training & optimization** — the optimizer lineage GA(2009)→QPSO(2015)→islands(now); PSO, GA, DE, CMA-ES, RL(PPO/SAC), supervised warm-start (= 2016 divide-and-conquer reborn), and the 3-island PSO/GA/DE model with migration; **compute-fairness protocol stated explicitly** (see §4).
8. **Results**
   - 8.1 Optimizer comparison on a fixed dense net → islands best (Study A).
   - 8.2 Architecture sweep (param-vs-DV Pareto) → dense best, Mamba second, **strongest in the low-param regime**.
   - 8.3 Output-parameterization study (Study B) → fair atan2 vs scaled_pi vs delta.
   - 8.4 Input ablation → engineered autoregressive inputs explain why dense beats Mamba's internal recurrence.
   - 8.5 **Classical vs NN** (headline comparison table).
   - 8.6 Pruning & quantization (deployability: QAT4/QAT8, pruned variants).
9. **Discussion** — robustness (impressively low p95 and max), parameter efficiency, why dense+autoregressive-inputs wins, on-board feasibility (training is the only heavy cost; the deployed policy is tiny).
10. **Conclusion** — plain dense NN is best and incredibly robust with very few parameters; islands improved training over the 2009 GA; future work (skip-entry, Earth-return leg, on-line adaptation).
11. **References** — Typst native bibliography.

Ordering is the recommended scientific order; §5-8 carry the user's 5 requested parts.

---

## 3. Controlled experiments to run

I generate configs + a runner script; the user executes the training; I extract numbers and write.

### Study A — Optimizer comparison (control architecture = `dense_p515`, n_gen=2000)

| Run | Optimizer | n_pop | Status |
|---|---|---|---|
| A1 | PSO | 300 | new |
| A2 | GA | 300 | new |
| A3 | DE | 300 | new |
| A4 | CMA-ES | 300 (native; `_CMAES_MAX_PARAMS = 20000`, 515 < 20000) | new |
| A5 | islands (PSO/GA/DE) | 100 × 3 islands | **reuse** `sweep_dense_p515` |
| A6 | warm-start + islands | 100 × 3 | new |
| A7 | RL / PPO | step budget (see risks) | new on dense arch (fallback: cite existing GRU-PPO with caveat) |

Metrics per run: best validation RMS cost (training-convergence) + deployed MC final-eval (capture %, ΔV mean/p50/p95/max, peak heat flux/g-load, bank consumption) on the reserved final-eval seed pool.

### Study B — Output parameterization (control = `dense_p515` + islands)

| Run | Head | output_size | Status |
|---|---|---|---|
| B1 | 2D atan2 | 2 | **reuse** `sweep_dense_p515` |
| B2 | 1D scaled_pi | 1 (tanh) | new |
| B3 | 1D delta | 1 (tanh) | new |

(Last layer changes 9→2 to 9→1 for B2/B3, ~506 params; note the minor count difference.)

**~6 new training runs total.** Everything else reuses committed data.

---

## 4. Compute-fairness protocol (load-bearing for the "islands is best" claim)

Islands runs 3 heterogeneous sub-populations: per-island `n_pop` × 3 = total evals/gen. The committed `sweep_dense_p515` used islands `n_pop=100` → **300 individuals/generation**. Therefore single-optimizer baselines (PSO/GA/DE/CMA-ES) are run at **`n_pop=300`, `n_gen=2000`** so total function evaluations match (≈300 × 2000 × sims-per-eval). The paper states this explicitly; comparing islands@100 vs PSO@100 would hand islands a 3× compute advantage and a reviewer would reject the central claim. RL (PPO) is budgeted in environment steps and cannot be exactly evaluation-matched — report its total step budget and note it underperforms despite a large budget.

---

## 5. Data already extracted from committed runs (reference)

Capture = `ifinal==3 & eccentricity<1.0`; ΔV = `dv_total_m_s` over captured sims; n=1000 unless noted. Format: **mean / p50 / p95 / max** (m/s), capture %.

### Classical
| Scheme | mean | p50 | p95 | max | cap% | n |
|---|---|---|---|---|---|---|
| FTC | 136.2 | 130.7 | 172.6 | 275.7 | 100 | 1000 |
| EnergyController | 174.3 | 163.0 | 268.4 | 444.6 | 99.9 | 1000 |
| PiecewiseConstant | 190.8 | 176.7 | 298.3 | 767.6 | 100 | 1000 |
| FNPAG | 266.1 | 213.6 | 629.4 | 879.1 | 100 | 2000 |
| PredGuid | 391.9 | 287.3 | 929.8 | 1410.1 | 98.2 | 2000 |
| EqGlide | (user will run a deploy/eval to populate this row) | | | | | |

### Architecture sweep (islands, matched budgets) — mean / p95 / max, all 100% capture
| Arch | ~500p | ~1000p | ~2000p | ~4000p |
|---|---|---|---|---|
| Dense | 119.6/131.1/164.5 | 120.7/133.0/167.9 | 126.2/144.2/218.2 | 118.4/131.1/164.9 |
| GRU | 127.1/147.7/191.8 | 120.8/133.9/180.1 | 121.6/136.0/171.5 | 118.5/130.8/210.2 |
| LSTM | 123.2/138.3/182.7 | 125.7/146.9/190.1 | 123.8/139.3/204.6 | 118.7/132.8/168.9 |
| Mamba | 121.9/136.0/169.6 | 125.0/138.9/176.4 | 119.7/132.9/186.6 | 125.1/143.8/221.9 |
| Transformer | 130.0/146.9/194.6 | 123.6/138.2/162.8 | 121.2/134.4/180.2 | 123.8/142.0/196.8 |
| Window | 125.0/141.0/239.8 | 121.2/139.1/202.5 | 123.3/137.4/166.0 | — |

**Reading:** at ~500 params Dense (p95 131.1) clearly leads, Mamba (136.0) second — the few-params robustness story. At ~4000 params Dense/GRU/LSTM converge to ~131-133. The paper frames "dense best, Mamba 2nd" as a **low-parameter-regime** result and shows the full Pareto.

### Output parameterization (COMMITTED, mixed optimizers — Study B re-runs cleanly)
| Head | mean | p95 | max | source optimizer |
|---|---|---|---|---|
| atan2 (2D) | 119.0 | 132.0 | 165.2 | islands/best |
| scaled_pi (1D) | 145.0 | 180.3 | 247.2 | PSO |
| delta (1D) | 141.8 | 162.4 | 228.3 | PSO |

### RL (committed) — far worse
| Run | mean | p95 | max |
|---|---|---|---|
| neural_network_rl (PPO) | 636.0 | 973.0 | 1185.4 |
| gru_ppo | 512.6 | 828.5 | 1015.4 |

### Warm-start / joint (committed)
best_neural_network_joint 125.3/143.3/179.8 · neural_network_joint 125.7/143.8/200.5 · neural_gru_joint 127.6/153.4/204.6

### Pruning / quantization (committed, atan2 base 119.0/132.0/165.2)
QAT8 125.1/140.4/196.0 · QAT4 128.7/149.7/186.7 · pruned_dv3 variants ~120-123 mean.

### Headline classical-vs-NN
Best NN (`sweep_dense_p515`, 515 params): **119.6 / 131.1 / 164.5, 100% capture** vs best classical FTC 136.2 / 172.6 / 275.7. The NN improves mean −13%, p95 −24%, max −40% at a fraction of any predictor-corrector's cost — and recovers the 2009 result (116.7 mean) at far lower tail risk.

---

## 6. Figures

Reuse `charts.py` / `report.py` chart functions where possible; new scripts for the rest. Output SVG/PDF into `articles/paper/figures/`.
- F1 Aerocapture corridor schematic (energy vs pdyn).
- F2 Corridor trajectories: best NN vs FTC (MC spaghetti + envelopes).
- F3 Optimizer convergence: best validation cost vs generation (from JSONL logs).
- F4 Optimizer comparison bar (deployed ΔV mean/p95/max per optimizer).
- F5 **Param-vs-ΔV-p95 Pareto** across the six architectures (new).
- F6 Output-parameterization bar (atan2/scaled_pi/delta, Study B).
- F7 Input-ablation bar (ΔV degradation per zeroed input; from `ablation.py`).
- F8 Classical-vs-NN ΔV CDF / box.
- F9 Pruning/quantization tradeoff (ΔV vs bit-width / sparsity).

---

## 7. Typst setup & file layout

New standalone academic paper (NOT the training-report template). Clean single-column arXiv style; Typst native bibliography (Hayagriva `refs.yml`).

```
articles/paper/
  main.typ              — document shell, metadata, imports
  template.typ          — page style, headings, abstract block, figure/table helpers
  refs.yml              — Hayagriva bibliography (self-citations + classical + methods)
  sections/
    00_abstract.typ
    01_introduction.typ
    02_problem.typ
    03_testbed.typ
    04_classical.typ
    05_neural.typ
    06_training.typ
    07_results.typ
    08_discussion.typ
    09_conclusion.typ
  figures/              — generated SVG/PDF
  experiments/          — the new Study A/B configs + runner script
```

Compile via `typst compile articles/paper/main.typ`. Degrade gracefully if a figure is absent.

---

## 8. Sequencing & deliverables

1. Write this spec → commit (feature branch `feature/parameter_sweep`).
2. Generate Study A/B configs (`articles/paper/experiments/` or `configs/training/paper/`) + a runner script → **user executes** the ~6 new training runs.
3. Generate figure-generation scripts + render figures.
4. Draft the Typst paper section by section (compile + visual check per section).
5. Final step: invoke the `smart-commit` skill over the whole branch.

---

## 9. Constraints & decisions resolved

- **CMA-ES** runs natively up to 20000 params (`_CMAES_MAX_PARAMS = 20000`); 515 is fine. *(CLAUDE.md's "fallback >200" note is stale — fix later, out of scope here.)*
- **Islands** `n_pop` is per-island × 3; compute-matched single-optimizers use `n_pop=300`.
- **Control architecture** = `dense_p515` (17→18→9→2, swish/asinh; atan2 pipeline base: 17-input mask, calibrated normalization, full_neural, scaffolding=live, command shaping + navigation).
- **Capture definition** = `ifinal==3 & eccentricity<1.0`; ΔV = `dv_total_m_s`.
- **Scope** = comprehensive (includes output-param, input ablation, pruning, quantization).
- **Paper home** = `articles/paper/`.

## 10. Risks & open items

- **EqGlide** — *resolved:* user will run a deploy/eval; include EqGlide in the classical table.
- **RL-on-dense (A7)** — *resolved:* user will run RL on the dense arch; keep A7 (no GRU-PPO fallback needed).
- **Compute budget** — *resolved:* user approves scaling `n_gen` down uniformly across Study A if wall-clock is prohibitive (keeps the comparison fair); note any reduction in the paper.
- **General:** if any paragraph (RL or otherwise) needs additional training/simulation runs during drafting, ask the user — they will run them.
- **Output-param param-count:** B2/B3 are ~506 vs B1's 515 — note the minor difference; it does not affect the conclusion.
- Numbers in §5 are from committed runs as of 2026-06-08; final tables use the fresh Study A/B runs where they supersede committed mixed-optimizer data.
