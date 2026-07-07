# Minimum Word Error Training of RNN-based Voice Activity Detection

> **Methodology extract** (not a full transcription). Mining the parts that transfer to the aerocapture-NN article: the augmented LSTM cell, custom task-aligned loss design, and QPSO + hybrid backprop — the optimization lineage the repo now uses.

**Authors:** Grégory Gelly, Jean-Luc Gauvain — LIMSI-CNRS, Spoken Language Processing Group, Orsay / Paris-Sud University
**Venue:** Interspeech 2015
**Note:** This paper **cites the 2009 aerocapture paper** as ref [15] — your own through-line from aerospace GA to speech QPSO.

---

## Why this paper matters to the new article

It is the bridge between the 2009 feed-forward+GA aerocapture work and the modern repo. Three transferable ideas, each with a repo counterpart:

| 2015 speech idea | Repo counterpart today |
|---|---|
| Augmented LSTM cell (gate-to-gate links), "BLSTM+" | `LstmLayer` / `GruLayer` stateful runtime; the "coordinated-gate" idea motivates richer recurrent cells |
| Custom task-aligned loss (minimize the *real* end metric, not a proxy) | `compute_cost` / `dv_cost` softplus-quadratic + `cost_transform` |
| QPSO global search + RPROP local refinement, **alternated** | PSO population eval + warm-start/BPTT gradient pretrain (hybrid global+local) |
| Joint optimization of *feature extraction + weights + post-processing* in one optimizer | `scaffolding = "full"/"live"` co-optimizing nav/shaping/exit params **with** NN weights |

---

## 1. Architecture

### 1.1 MLP baseline (same form as the 2009 net)

$$z = \sigma_z\!\left(W_z \cdot \sigma_h\!\left(W_h \cdot p + b_h\right) + b_z\right) \tag{1}$$

### 1.2 Standard RNN / LSTM

Standard RNN over input sequence $p=(p_1,\dots,p_T)$, with the previous hidden state stacked into the input $\tilde p_t = [\,p_t\;;\;h_{t-1}\,]$:

$$h_t = \sigma_1\!\left(W_1 \cdot \tilde p_t + b_1\right) \tag{2}, \qquad z_t = \sigma_z\!\left(W_z \cdot h_t + b_z\right) \tag{3}$$

LSTM cell (peephole form — $W_i^c, W_f^c, W_o^c$ **diagonal**, "so that each heart of a cell is only visible to the gates of the same cell"):

$$i_t = \sigma\!\left(W_i\,\tilde p_t + W_i^c\, c_{t-1} + b_i\right) \tag{4}$$
$$f_t = \sigma\!\left(W_f\,\tilde p_t + W_f^c\, c_{t-1} + b_f\right) \tag{5}$$
$$c_t = \mathrm{diag}(f_t)\, c_{t-1} + \mathrm{diag}(i_t)\, \sigma_c\!\left(W_c\,\tilde p_t + b_c\right) \tag{6}$$
$$o_t = \sigma\!\left(W_o\,\tilde p_t + W_o^c\, c_t + b_o\right) \tag{7}$$
$$h_t = \mathrm{diag}(o_t)\, \sigma_h\!\left(c_t\right) \tag{8}$$

**BLSTM**: two LSTMs run forward and backward, outputs combined into the output layer — "BLSTM networks always perform better than unidirectional ones."

### 1.3 Augmented BLSTM (**BLSTM+**) — the signature cell

Add **direct links between the three gates** of a cell. The intent (verbatim engineering honesty worth emulating): *"It was meant to avoid some of the LSTM cells to get stuck in a saturated state when training on long sequences. It did not do much for our saturation problem but it improved the performance of the VAD so we kept it."*

Gate-coupling pre-activations injected into the gate equations:

$$\tilde\imath_t = W_i^{\,i} i_{t-1} + W_i^{\,f} f_{t-1} + W_i^{\,o} o_{t-1} \tag{9}$$
$$i_t = \sigma\!\left(W_i\,\tilde p_t + W_i^c\, c_{t-1} + \tilde\imath_t + b_i\right) \tag{10}$$
$$\tilde f_t = W_f^{\,i} i_{t-1} + W_f^{\,f} f_{t-1} + W_f^{\,o} o_{t-1} \tag{11}$$
$$f_t = \sigma\!\left(W_f\,\tilde p_t + W_f^c\, c_{t-1} + \tilde f_t + b_f\right) \tag{12}$$
$$\tilde o_t = W_o^{\,i} i_{t} + W_o^{\,f} f_{t} + W_o^{\,o} o_{t-1} \tag{13}$$
$$o_t = \sigma\!\left(W_o\,\tilde p_t + W_o^c\, c_t + \tilde o_t + b_o\right) \tag{14}$$

The nine matrices $W_{\{i,f,o\}}^{\{i,f,o\}}$ are **diagonal** → a gate only sees the gates of the same cell. "With these new links the three gates of a cell can interact more efficiently and improve the behavior of the cell."

---

## 2. Minimum Word Error (MWE) training — custom loss design

> Design goal stated plainly: an optimization framework that copes with "stringent requirements such as starting VAD training **from scratch**, on noisy data, with **small training datasets**." This is the same spirit as training a guidance policy with no demonstration trajectories.

The "right" objective (true WER) is too expensive to evaluate per candidate, so **three surrogate losses** were designed, increasing in alignment with the end metric:

- **$L_1$ — frame error vs human reference.** Weighted frame error; usable **before the ASR exists** (no system output needed). $\alpha$ trades speech-frame vs noise-frame errors; FER-optimal $\alpha=0.5$, but best WER at $\alpha=0.6$ (missing speech is costlier).
  $$L_1 = \alpha \sum_{s\in S}\delta_s(z) + (1-\alpha)\sum_{n\in N}\delta_n(z) \tag{15}$$
- **$L_2$ — frame error vs ASR output.** Same form, but speech/non-speech sets are redefined from ASR correctness (correct + substitutions = speech; silences/deletions/insertions = non-speech). Best $\alpha=0.85$ (higher confidence in the tagging).
- **$L_3$ — WER-like metric.** Reflects that *every word weighs the same* regardless of length:
  $$L_3 = \frac{pS + pD + pI + \tau_i + \tau_d}{N_{words}} \tag{16}, \qquad \tau_i = \!\!\sum_{w\in pW_I}\!\!\tau_i^w,\quad \tau_d = \!\!\sum_{w\in W_C\cup W_S}\!\!\tau_d^w \tag{17}$$
  The two $\tau$ terms "were introduced to **smoothen the discontinuities** of the WER metric. As a direct result, the optimization algorithm is less prone to being trapped on a plateau of the loss function." A differentiable surrogate $L_{3b}$ (a weighted cross-entropy, Eq. 18) is provided for the backprop path.

> **Transfer to aerocapture:** this is exactly the philosophy behind the repo's cost design — a primary objective ($\Delta V$) plus smoothing terms that remove plateaus/cliffs so the population optimizer keeps a usable gradient (`dv_cost` C-∞ softplus-quadratic; energy-proportional virtual DV for crashes; `cost_transform` tail compression). Cite this paper as the methodological precedent for *designing a smooth surrogate of a discontinuous mission metric*.

---

## 3. Optimization — QPSO + hybrid backprop (the PSO lineage)

- "We saw firsthand in [15] (the 2009 aerocapture paper) the interest of optimization techniques such as **Genetic Algorithms** for minimizing complex loss functions with an important number of parameters (> 10). Since then, similar but more efficient methods such as **Quantum-behaved Particle Swarm Optimization (QPSO)** were developed."
- QPSO (Sun et al.; PSO variant of Kennedy-Eberhart / Clerc): "comparable in performance with the Genetic Algorithms approach, [but] QPSO proved to be a more powerful tool than both of them when performing **difficult optimization tasks**."
- **Joint optimization in one optimizer:** "For the NN-based VADs, QPSO is used to optimize **simultaneously MFCCs extraction parameters, neural networks weights and final smoothing parameters.**" → direct precedent for joint NN-weights + scaffolding co-optimization.
- **Hybrid global+local:** RPROP backprop (and BPTT / its LSTM version for the recurrent nets) "is used to **locally improve the best solution found by QPSO**."

**Ablation of the optimization process (Table 3, Vietnamese eval WER):**

| Optimization process (using $L_3$) | WER |
|---|---|
| Baseline (GMM multilingual VAD) | 57.2 |
| BackProp only | 58.0 |
| BackProp + QPSO(smoothing) | 57.5 |
| QPSO only | 56.9 |
| Hybrid QPSO–BackProp | 55.5 |
| QPSO + BackProp | 55.0 |
| QPSO + BackProp + QPSO(smoothing) | **54.6** |

> Lesson the new article can reuse verbatim in spirit: *"both algorithms are needed to achieve the best tuning and they are best used alternatively."* Global search escapes plateaus; local gradient polishes. Pure backprop alone is *worse than baseline*.

---

## 4. Results worth citing (headline)

- BLSTM+ gives an **absolute 2.5-point WER gain (4.4% relative)** over the multilingual baseline on IARPA Babel Vietnamese; the gain comes mainly from **fewer insertions** (6.4% → 3.6%) — the BLSTM+ learns to discard signal (even speech) that makes the ASR insert words.
- Across all VADs, **$L_3$ yields the best results** — the loss most aligned with the end metric wins.
- All nets fixed at **6000 weights** for a fair comparison; "using more weights did not improve their performance."
- RNN-based VAD runs in **1/1000 real time** on a desktop CPU — the heavy cost is training, not inference (same trade-off flagged in the 2009 aerocapture conclusion).

---

## Reusable one-liners (your phrasing, for the new paper's related-work / method)

- "Neural Networks are especially valuable to handle problems with no analytic solution."
- "We wanted to develop an optimization framework that would be able to cope with stringent requirements such as starting training from scratch... with small training datasets."
- "Both algorithms are needed to achieve the best tuning and they are best used alternatively."
- "It did not do much for our [original] problem but it improved the performance so we kept it." *(model of honest reporting)*
