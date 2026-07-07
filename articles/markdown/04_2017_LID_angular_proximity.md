# Spoken Language Identification using LSTM-based Angular Proximity

> **Methodology extract.** Transferable ideas: the **Coordinated-Gate LSTM (CG-LSTM)** naming, the **architecture-for-convergence** principle (fewer layers → less gradient dilution → faster training), **learnable per-layer magnitude weighting**, and a **custom differentiable geometry-aware loss** (angular proximity) as an alternative to softmax + cross-entropy.

**Authors:** G. Gelly, J.-L. Gauvain — LIMSI, CNRS, Univ. Paris-Sud
**Venue:** Interspeech 2017
**Datasets:** NIST LRE07 (14 languages), NIST LRE15 (20 closely-related languages/dialects)

---

## Why this paper matters to the new article

Two ideas with direct guidance-NN analogues:

| 2017 LID idea | Use in the aerocapture-NN article |
|---|---|
| **CG-LSTM** (Coordinated-Gate LSTM = the augmented cell, final name) | Canonical citation for your custom recurrent cell; the lineage 2015 BLSTM+ → 2016 LSTM+ → 2017 CG-LSTM |
| Reduce stacked-layer count to avoid **gradient dilution** → faster convergence | Justifies shallow-but-stateful guidance policies; explains warm-start convergence behavior |
| Learnable per-layer magnitude weights $\alpha_1,\alpha_2$ (soft attention over depth) | A cheap mechanism to let the optimizer pick the most informative layer without constraining the rest |
| Custom **differentiable** loss with analytic gradients, **boundary-focused** | Same design pattern as the repo's bespoke `compute_cost`; the logistic margin idea ≈ softplus knee |

---

## 1. Coordinated-Gate LSTM (CG-LSTM)

Same augmented cell as 2015/2016: direct links between the three gates of a cell. "With these new links the three gates of a cell can interact more efficiently and improve the cell behavior. We call this new model **CG-LSTM for Coordinated-Gate LSTM**." As shown in the results, "this added flexibility allows CG-LSTM cells to **outperform standard LSTM cells**." (Equations identical to [03] Eqs. 3–13.)

---

## 2. Language Vector (LV) extractor — architecture for faster convergence

**Diagnosis (verbatim intuition — model of how you motivate a design):**
> "Our intuition was that the **number of stacked layers (RNN layers + MLP) diluted the gradients during backpropagation** and thus slowed the convergence especially during the first training steps. The D&C training process tackles this part of the problem by separating the multi-class problem into $n$ smaller problems... Here we introduce an RNN architecture that deals **directly** with this problem."

**Design moves:**
1. **Remove the MLP / decision network** — "to reduce the overall number of layers... and at the same time to obtain more homogeneous gradients since the link between the errors and the LSTM cells is more direct."
2. **Concatenate the outputs of both RNN layers**, each scaled by a learnable **magnitude coefficient $\alpha_1, \alpha_2$**. "The magnitude coefficients... allow the optimization process to easily **focus on the most discriminative layer(s)** for the classification task at hand, without directly constraining the outputs of the recurrent layers." (A soft attention over depth.)
3. **Average over time** → single vector → **$L_2$-normalize onto the unit hypersphere**: "the information **only resides in the direction** of the vector." Final unit-vector dimension = sum of all layers' dimensions (here $2\times124=248$).

---

## 3. Angular Proximity (AP) loss — custom differentiable objective

A reference unit direction $c_l$ is learned per language **in the same vector space** as the output $z$. For an output $z$:

$$\theta_l(z) = \arccos\!\left(c_l \cdot z\right) \tag{1}, \qquad l^*(z) = \arg\min_{l\in[1,N]} \theta_l(z) \tag{2}$$

The loss for a vector $z$ of true language $l$ (sum over the other languages; $\sigma$ = logistic):

$$L(z,l) = \sum_{l'\neq l} \sigma\!\left(\theta_l(z) - \theta_{l'}(z)\right) \tag{3}$$

> Rationale: the logistic "brings faster and better convergence by **focusing the training effort on the cases that are close to the boundaries between languages**." (A margin/boundary-focused loss — same spirit as a softplus knee that concentrates gradient where it matters.)

**Analytic gradients** (the loss trains the RNN weights *and* the reference directions jointly). With $\delta_{l'}(z) = c_{l'}\cdot z$ and $\Delta_{ll'}(z) = \sigma(\theta_l-\theta_{l'})\,(1-\sigma(\theta_l-\theta_{l'}))$ [Eq. 5]:

$$\frac{\partial L}{\partial z} = \sum_{l'\neq l}\Delta_{ll'}(z)\left(\frac{c_{l'}}{\sqrt{1-\delta_{l'}(z)^2}} - \frac{c_{l}}{\sqrt{1-\delta_{l}(z)^2}}\right) \tag{4}$$
$$\frac{\partial L}{\partial c_{l'}} = \frac{\Delta_{ll'}(z)}{\sqrt{1-\delta_{l'}(z)^2}}\,z \quad (l'\neq l)\ \ (6), \qquad \frac{\partial L}{\partial c_{l}} = -\sum_{l'\neq l}\frac{\Delta_{ll'}(z)}{\sqrt{1-\delta_{l}(z)^2}}\,z \ \ (7)$$

(Related to FaceNet triplet loss / Bredin's TristouNet speaker-turn embedding — cite as the embedding-loss family.)

---

## 4. Training settings (reusable practical defaults)

- Input: 8 PLP + Δ + ΔΔ = 24-dim, 10 ms frames, VTLN then cepstral mean/variance normalization.
- **Chunking + overlap sweep:** chop into 320-frame (3.2 s) overlapping segments. Best chunk = **3.2 s**, best **overlap = 75%** (Table 2). "All setups give better results than using the original full sequence."
- Optimizer: BPTT + **SMORMS3**.
- Hard-example mining: ~1000 balanced random segments + 200 worst-error segments per iteration (same as 2016).

---

## 5. Results worth citing

**Binary task (Arabic-or-not), 2-class LER — cell × loss ablation (Table 1):**

| Model | loss | 3 s | 10 s | 30 s |
|---|---|---|---|---|
| MLP | CE | 33.7 | 25.0 | 20.0 |
| RNN | CE | 33.9 | 21.4 | 12.6 |
| LSTM | CE | 27.5 | 14.4 | 8.4 |
| CG-LSTM | CE | 26.2 | 13.4 | 7.5 |
| LSTM | AP | 25.9 | 12.3 | 7.2 |
| **CG-LSTM** | **AP** | **24.1** | **11.2** | **5.0** |

→ CG-LSTM always beats standard LSTM; the LV+AP architecture gives up to **33% relative** gain on 30 s segments.

**Full LID (CG-LSTM, ~400k weights), LER / $C_{avg}$:**

| System | LRE07 LER | LRE07 $C_{avg}$ | LRE15 LER | LRE15 $C_{avg}$ |
|---|---|---|---|---|
| PHO | 16.2 | 8.7 | 23.5 | 15.1 |
| IV | 23.6 | 12.7 | 26.6 | 17.4 |
| CG-LSTM Std | 25.6 | 13.8 | 30.9 | 20.8 |
| CG-LSTM D&C | 21.9 | 11.8 | 22.8 | 14.6 |
| **CG-LSTM LV** | **20.2** | **10.9** | **20.7** | **13.3** |
| PH + IV | 14.1 | 7.6 | 18.6 | 11.6 |
| **PH + LV** | **12.3** | **6.6** | **15.9** | **9.9** |

**Convergence (the key selling point):** "the language error rate drops much faster when using a language vector extractor architecture trained with the angular proximity loss." LV training time is **half** the D&C method and **a third** of standard training → **66% reduction** in training time vs standard, with *better* accuracy.

---

## Reusable one-liners

- "With its innate ability to exploit long range dependencies, LSTM neural networks were natural candidates as purely acoustic classifiers."
- "Our intuition was that the number of stacked layers diluted the gradients during backpropagation and thus slowed the convergence."
- "...a vector space where angular proximity corresponds to a measure of similarity."
- "This error reduction comes with an important reduction in the training duration."
