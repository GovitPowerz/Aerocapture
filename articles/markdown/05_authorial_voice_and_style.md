# Authorial Voice & Style Guide

> Distilled from your four papers (2009 AIAA, 2015 VAD, 2016 LID-D&C, 2017 LID-AP) so the new aerocapture-NN article can be drafted **in your voice**. Concrete patterns + reusable sentence templates, not abstract advice.

---

## 1. Macro-structure you consistently use

Every paper follows the same skeleton — keep it:

1. **Abstract** — one paragraph: problem → "this paper presents/describes" → method in one clause → "experiments compare ... to [baselines]" → headline result. No hedging.
2. **Introduction** — institutional/historical lineage first ("since the early 1990s...", "Astrium has demonstrated..."), then *why this method is a natural candidate*, then "Here, we introduce [N] innovations", then a one-paragraph section roadmap ("The next section describes... Section 3 presents...").
3. **Method** — architecture with full equations, numbered; then training/optimization.
4. **Experimental setup** — mission/corpus, testbed, dispersions/datasets, metrics (defined explicitly).
5. **Results** — tables first, prose interpreting them second; always against baselines.
6. **Conclusion** — restate gains in numbers, state the one honest drawback, then "**The next step would be to...**".

## 2. Tone — "quirky but precise senior engineer"

- **First-person plural, active voice.** "We designed...", "we successfully trained...", "we came to the conclusion that...". Never "it was decided that".
- **Confident, never breathless.** Claims are quantified, not adjectival. You earn "very efficient" / "significant" with a number next to it.
- **Intellectually honest.** You report what didn't work and why you kept it anyway. This is a signature — preserve it:
  - *"It did not do much for our saturation problem but it improved the performance ... so we kept it."*
  - The one aerocapture worst-case $Z_a$ regression (+110%) is reported, not buried.
- **Motivate every design choice with an intuition before the math.** *"Our intuition was that the number of stacked layers diluted the gradients..."* then the architecture that fixes it.
- **Engineering pragmatism foregrounded:** computational burden, on-board feasibility, design margins, fair comparison (same parameter count), real-time factor. You always close the loop to *deployability*.

## 3. Rhetorical micro-patterns (lift these directly)

**"Natural candidate" framing** — justify the model class before introducing it:
> "With its innate ability to exploit long range dependencies, [recurrent/attention policies] were natural candidates as [a guidance law for the ascending leg]."

**"No analytic solution" framing** — your standard motivation for NNs:
> "Neural networks are especially valuable to handle problems with no analytic solution. This is typically the case when confronted to the guidance of an atmospheric flight. Usually... one makes strong simplifications... which usually lead to a non-optimal resolution of the guidance problem."

**"First attempt failed → diagnosis → fix" arc** — your favorite narrative engine:
> "Our first attempt to train [a single monolithic policy] gave not-competitive results... which suggested that the problem was not with the [network] itself or its size but with the **training process**. To address this..."

**"One shortcoming... there is no reason not to..."** — introducing an extension:
> "One shortcoming of [the feed-forward policy] is that it [ignores temporal context]. For [aerocapture guidance] purposes there is no reason not to exploit [the trajectory history] as well."

**"Fair comparison" guardrail** — pre-empt the reviewer:
> "To have a fair comparison between the [N] [policies], all of them were designed with the same number of weights."

**Dual absolute+relative gains:**
> "...an absolute gain of 2.5 points on the WER (4.4% relative)." → for aerocapture: "...a mean correction cost of 116.7 m/s (−19% vs FTC)."

**"Note that..."** for caveats and scope limits. **"As could be expected..."** when a result confirms intuition. **"It has to be noted that..."** for a technical aside (2009 usage).

**The closing hook** — you always end forward-looking:
> "The next step would be to extend our work on the aerocapture to skip-entry missions and evaluate the performance of neural guidance compared to classic algorithms such as the predictor-corrector schemes." *(literally your 2009 closer — the new paper can open by quoting it and announcing it is the answer.)*

## 4. Vocabulary fingerprint

Words/phrases that recur and read as *you*: "enslavement" (FTC tracking), "innate ability", "natural candidates", "smart initialization point", "fair comparison", "important margins", "important reduction", "yields", "outperforms", "competitive / not-competitive", "stringent requirements", "from scratch", "robust to dispersions and uncertainties", "compromise between", "with respect to".

Aerospace register (2009, co-authored with Vernis) is a touch more formal ("This attractive concept thereby reduces...", "Thanks to genetic algorithms, we successfully trained..."); the LIMSI speech papers are crisper and more first-person. The new single-author arXiv paper should sit between the two: **crisp and personal like 2015–2017, but with the aerospace rigor of 2009.**

## 5. Equation & table conventions

- Number every non-trivial equation; refer to them as "Eq. 11" in prose.
- Define every symbol in a **Nomenclature** block (2009 style) or inline at first use (speech style). For a long aerospace paper, bring back the explicit Nomenclature.
- Tables: mean / std / max / min columns for MC results; a final **comparison table with (−x%) deltas in parentheses** is your signature summary device — reuse it for the new scheme-vs-scheme matrix.
- Always state the metric definitions explicitly (you define LER, $C_{avg}$, $\Delta V$ correction cost before using them).

## 6. What to consciously modernize (don't ape 2009 where the field moved)

- 2009 used GA; 2015+ moved to QPSO; the repo now uses **PSO / PPO / SAC + warm-start**. Tell that evolution as a *story you lived*, not a literature survey.
- 2009 nets were 1-hidden-layer feed-forward; the new work is **stateful (GRU/LSTM/Window/Transformer/Mamba)**. The 2015–2017 papers are your evidence that recurrent cells + custom losses + smart init pay off — cite *yourself*.
- Keep the honesty about the on-board-training drawback, but update it: warm-start + a deployable trained policy is the answer 2009 lacked.

## 7. A ready-to-adapt opening paragraph (your voice, new topic)

> *In 2009 we showed that a feed-forward neural network, trained by a genetic algorithm, could perform the aerocapture of a Mars Sample Return vehicle more efficiently than a Cerimele-Gamble-derived feedback scheme. We closed that work by noting that the natural next step would be to compare neural guidance against predictor-corrector algorithms and to extend it beyond the single capture maneuver. In the years since, working on recurrent neural networks for speech, we developed the coordinated-gate LSTM cell, particle-swarm training, and divide-and-conquer initialization. This paper brings that machinery back to where it started: we train stateful neural guidance policies for aerocapture and benchmark them, on identical Monte-Carlo scenarios, against FTC and against numerical predictor-corrector schemes (FNPAG, PredGuid).*

*(Trim/retune to the actual contributions — but this is the arc and the register.)*
