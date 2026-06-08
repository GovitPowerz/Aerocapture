# Synthesis Writing Kit — New Aerocapture Neural-Guidance Article

> **Start here.** This maps the four prior papers + the current repo onto a section-by-section scaffold for the new (arXiv) article. Companion files:
> [01 — 2009 AIAA full extract](01_2009_AIAA_neural_guidance.md) ·
> [02 — 2015 VAD methodology](02_2015_MWE_RNN_VAD.md) ·
> [03 — 2016 LID D&C methodology](03_2016_LID_divide_and_conquer.md) ·
> [04 — 2017 LID Angular Proximity methodology](04_2017_LID_angular_proximity.md) ·
> [05 — Authorial voice & style](05_authorial_voice_and_style.md)

---

## 0. The one-sentence thesis (and why it's a strong paper)

> *The 2009 paper's own closing line asked for exactly this work — neural aerocapture guidance benchmarked against predictor-corrector schemes — and the intervening decade of recurrent-NN + swarm-optimization work (yours, in speech) is precisely the machinery that now makes it deliverable.*

The narrative arc is your competitive advantage. No one else can write this paper, because it is **your** 17-year through-line:

```
2009  Feed-forward NN + Genetic Algorithm  →  aerocapture & soft-landing (Mars, MSR)
                                                  [single hidden layer; sin/cos bank output; beats FTC by 19% ΔV]
   |   "next step: compare to predictor-corrector schemes"   <-- explicit hook, unfulfilled until now
   v
2015  Augmented LSTM (BLSTM+) + custom loss + QPSO + hybrid backprop  →  speech (VAD)   [cites the 2009 paper]
2016  LSTM+ + Divide-and-Conquer smart-init                          →  speech (LID)
2017  CG-LSTM + language-vector + angular-proximity loss             →  speech (LID)
   |   recurrent cells, swarm training, smart initialization, custom differentiable losses
   v
NOW   Stateful NN guidance (GRU/LSTM/Window/Transformer/Mamba) + PSO/PPO/SAC + warm-start + scaffolding
      benchmarked on identical MC scenarios against FTC AND predictor-correctors (FNPAG, PredGuid)
      bit-validated Rust simulator; 7 guidance schemes
```

---

## 1. Section-by-section scaffold

### Abstract
Follow your template (file 05 §1). One paragraph: aerocapture problem → "we train stateful neural-network guidance policies" → "trained by particle-swarm optimization with supervised warm-start" → "compared on identical Monte-Carlo scenarios against FTC and predictor-corrector schemes" → headline ΔV / capture-rate number.

### 1. Introduction
- **Lineage paragraph** (institutional + your own): the 2009 result, then the speech detour, then "this paper brings that machinery back." Use the ready-made opener in [05 §7].
- **Why NN for atmospheric guidance** — reuse the "no analytic solution" framing ([01 §I], [02 reusable one-liners]).
- **Contributions list** ("Here we introduce..."): (i) stateful guidance policies spanning recurrent/attention/SSM cells; (ii) a swarm-with-warm-start training pipeline; (iii) a bit-validated high-fidelity simulator; (iv) the first head-to-head of neural vs predictor-corrector aerocapture guidance under identical dispersions.
- Section roadmap sentence.

### 2. Problem formulation — Aerocapture
Reuse directly from [01 §IV]:
- Definition (propulsion-free hyperbolic→elliptic insertion; only control = bank-angle modulation; authority ∝ dynamic pressure).
- MSR entry conditions (120 km, 5687 m/s, −10.24°, 38.04°) and target orbit (apoapsis 500 km, periapsis 11 km, incl 50°).
- **Corridor in the (orbital energy, dynamic pressure) plane** + restricted corridor with $\pm\delta Z_a$ (Eq. set in [01 §IV.B]). EI energy 4.91 → exit −5.87 MJ/kg.
- Performance metric = $\Delta V$ correction cost (sum of apoapsis/periapsis/inclination $\Delta V_i$); periapsis-raise floor = 113 m/s.
- **Update vs 2009:** the repo now models far more (EKF navigation, altitude-dependent winds, Gauss-Markov density perturbations, J2/J3/J4 gravity, thermal limits, adaptive integration). State that the fidelity jump is part of the contribution.

### 3. Classic baselines
- **FTC / Cerimele-Gamble** (in-plane apoapsis enslavement Eq. 10 + roll-reversal out-of-plane) — from [01 §IV.F]. This is the 2009 baseline; keep it for continuity.
- **Predictor-correctors** — the *new* baselines the 2009 paper asked for: **FNPAG** (Lu's numerical predictor-corrector, 3-DOF forward predictor) and **PredGuid** (Apollo/Shuttle drag tracking). Plus EqGlide, EnergyController, PiecewiseConstant for breadth (7 schemes total).

### 4. Neural guidance policies
- **Output parameterization** — open with the 2009 `sin/cos → atan2` decoder (Eq. 11, [01 §IV.G]); then present the new decoders that attack the ±π wrap seam: `atan2_signed`, `acos_tanh` (magnitude-only), `scaled_pi`, `delta`. This is a clean "what we kept / what we improved" story.
- **Stateful cells** — cite your own CG-LSTM lineage ([04 §1]) for the recurrent guidance cell; present the architecture family (Dense / GRU / LSTM / Window / Transformer / Mamba) as the modern generalization of the 2009 single-hidden-layer net.
- **Inputs** — contrast 2009's 5 hand-picked inputs (orbital energy, eccentricity, inclination, velocity, non-gravitational acceleration) with the modern 35-candidate input vector + learned input mask. Frame it as the same idea (orbital + aero state) grown up.

### 5. Training
- **Optimizer lineage** — GA (2009) → QPSO (2015) → PSO/DE/CMA-ES/islands + PPO/SAC (now). Reuse the [02 §3] lesson: *global search + local gradient, used alternately, beats either alone* — that's exactly warm-start (BPTT pretrain) + PSO.
- **Warm-start = Divide-and-Conquer reborn** — [03 §2]: decompose, train easy sub-problems (supervisor schemes), recombine into a smart initialization, then optimize. Cite your 2016 D&C as the conceptual ancestor.
- **Custom cost design** — [02 §2]: a smooth differentiable surrogate of a discontinuous mission metric. Map your L3-with-τ-smoothing precedent onto `dv_cost` (C-∞ softplus-quadratic), energy-proportional virtual-DV, and `cost_transform` tail compression.
- **Dispersed training / seed rotation** — the 2009 "we regularly change the set of dispersed conditions during training" ([01 §II.B.2]) is the direct ancestor of the rotating/adaptive seed strategy. Plus hard-example mining ([03/04]) ≈ adaptive seed curation.

### 6. Experimental setup
- Simulator fidelity + **bit-level validation against the legacy reference** (725 timesteps, 22/24 photo columns exact) — a credibility anchor 2009 didn't have.
- MC dispersions table (extend [01 Table 6] with winds, density-perturbation OU process, navigation errors).
- Metrics: ΔV correction cost, capture rate, peak heat flux / g-load / heat load, bank-angle consumption.
- **Fair-comparison guardrail** (your signature): identical MC scenarios across all schemes (`compare_guidance.py`), same seed pools, disjoint train/validation/final-eval seeds.

### 7. Results
- Lead with the **scheme-vs-scheme comparison matrix** in your (−x%) delta style ([01 Table 9] is the template).
- Per-architecture ablation (which cell type wins; PSO vs PPO/SAC — and note your standing finding that **PSO empirically beats PPO/SAC here**).
- Corridor plots, ΔV CDFs, constraint-margin tables.

### 8. Conclusion
Restate gains in numbers, the honest drawback (training cost / on-board feasibility — but now mitigated by deploying a fixed trained policy), and a forward hook (skip-entry, Earth-return leg, on-line adaptation).

---

## 2. Reusable equation bank (ready to paste)

**NN forward (feed-forward, 2009 — the base case):**
$$z = \sigma_2\!\left(W_2\,\sigma_1\!\left(W_1 p + b_1\right) + b_2\right)$$

**Bank-angle decoders:**
$$\text{2009 (atan2): } \sin\mu = \tfrac{o_1}{\lVert o\rVert},\ \cos\mu = \tfrac{o_2}{\lVert o\rVert} \;\Rightarrow\; \mu = \operatorname{atan2}(o_1,o_2)$$
$$\text{acos\_tanh: } \mu = \arccos(\tanh(o_1)) \qquad \text{scaled\_pi: } \mu = \mathrm{wrap}_\pi\!\big(n\,\pi\,\tanh(o_1)\big) \qquad \text{delta: } \mu = \mathrm{wrap}_\pi\!\big(\mu_{prev} + \Delta_{\max}\tanh(o_1)\big)$$

**FTC in-plane law (baseline):**
$$\cos\mu_{com} = \cos\mu_{ref} + G_{\dot h}\frac{\dot h - \dot h_{ref}}{q} + G_q\frac{q - q_{ref}}{q}$$

**CG-LSTM cell (your custom recurrent cell):** see [03 Eqs. 3–13] / [04 §1].

**Custom-loss design principle:** smooth surrogate of a discontinuous metric (your L3 + τ smoothing [02 §2]) ⇒ guidance `dv_cost` softplus-quadratic knee.

---

## 3. Headline numbers worth quoting from 2009 (for the "since then" framing)

- Neural aerocapture: mean ΔV **116.7 m/s vs FTC 144.8 m/s (−19%)**, floor 113 m/s.
- Neural soft-landing: final-position worst case **2.1 m vs Apollo 9.68 m (−78%)**, fuel −35%.
- The mechanism: NN handles in-plane + out-of-plane **jointly**, not decoupled like FTC.

---

## 4. Drop-in related-work paragraph (your voice)

> *Neural networks have long been attractive for guidance problems that admit no closed-form optimal solution. An early demonstration trained a single-hidden-layer feed-forward network with a genetic algorithm to perform both the soft-landing and the aerocapture of a Mars Sample Return vehicle, outperforming an Apollo-E descent law and a Cerimele-Gamble-derived feedback trajectory controller [Gelly & Vernis 2009]. The same evolutionary-training philosophy was later carried to recurrent architectures for speech, where a coordinated-gate LSTM cell trained by quantum-behaved particle-swarm optimization, with a divide-and-conquer initialization and task-aligned differentiable losses, set the approach used here [Gelly & Gauvain 2015, 2016, 2017]. Classic aerocapture guidance, by contrast, relies on analytic or numerical predictor-corrector laws — from the Cerimele-Gamble corridor logic to Lu's FNPAG and Apollo/Shuttle drag-tracking (PredGuid) — against which neural policies have not, to our knowledge, been systematically benchmarked under identical dispersions. This paper provides that comparison.*

---

## 5. Consolidated bibliography (self-citations + key external)

**Your through-line (cite all four):**
- Gelly G., Vernis P., *Neural Networks as a Guidance Solution for Soft-Landing and Aerocapture*, AIAA GNC Conference, Chicago, 2009.
- Gelly G., Gauvain J.-L., *Minimum Word Error Training of RNN-based Voice Activity Detection*, Interspeech, 2015.
- Gelly G., Gauvain J.-L., Le V., Messaoudi A., *A Divide-and-Conquer Approach for Language Identification based on Recurrent Neural Networks*, Interspeech, 2016, pp. 3231–3235.
- Gelly G., Gauvain J.-L., *Spoken Language Identification using LSTM-based Angular Proximity*, Interspeech, 2017.
- (earlier) Gelly G., Ferreira E., *Guidance Algorithm Using Neural Networks for a Soft-Landing Application*, EUCASS, 2007.
- (earlier) Vernis P., Gelly G., et al., *Guidance Trade-Off for Aerocapture Missions*, ESA GNC, 2004.

**Classic baselines / methods:**
- Cherry G. W., *A general explicit, optimizing guidance law for rocket-propelled spacecraft*, Apollo Guidance and Navigation, 1964.
- Cerimele C. J., Gamble J. D., *A Simplified Guidance Algorithm for Lifting Aeroassist Orbital Transfer Vehicles*, AIAA-85-0348.
- Hochreiter S., Schmidhuber J., *Long Short-Term Memory*, Neural Computation 9(8):1735–1780, 1997.
- Graves A., *Supervised Sequence Labelling with Recurrent Neural Networks*, Springer, 2012.
- Werbos P. J., *Backpropagation through time: what it does and how to do it*, Proc. IEEE 78(10):1550–1560, 1990.
- Kennedy J., Eberhart R., *A new optimizer using particle swarm theory*, 1995; Clerc & Kennedy, IEEE Trans. Evol. Comp. 6(1), 2002; Sun et al. (QPSO), 2004/2012.
- *(add for the new paper)* Lu P. et al., FNPAG / numerical predictor-corrector aerocapture; Apollo/Shuttle PredGuid drag-tracking references; Mamba/Transformer/SMORMS3 as used.

---

## 6. Loose ends to fill from the repo (not in the old papers)
- Exact new MC dispersion ranges (winds, OU density τ/σ, navigation errors) — pull from the actual training/nominal TOMLs.
- Final scheme-vs-scheme comparison numbers — from a `compare_guidance.py` run.
- Which architecture + optimizer combination wins (the paper's punchline) — from `training_output/` results.
- Cite the validation claim precisely (725 timesteps, 22/24 columns) — already in repo CLAUDE.md.
