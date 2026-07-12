# Quantization Appendix Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, serial -
> Gregory's subagent-budget rule). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold the completed quantization campaign (working notes
`docs/paper/quantization_appendix.md`, data `articles/paper/data/quant/`) into the revised paper as
new **Appendix C** (mission cards shift to D), with the finalists requoted on the frozen
confirmatory pool per the revision's pool discipline.

**Architecture:** One data task (confirmatory quotes for the three quant finalists, incl.
materializing the PTQ-verdict model), one writing task (the appendix section, following the notes'
claim order: 8-bit free -> SSM-dynamics bottleneck -> QAT-finetune tail-equivalence -> scratch
worse -> memory-not-compute), one body-integration task (Section 9 future-work recount, Section 7.2
compute sentence, appendix relabeling), then proofs/docs and smart-commit.

**Tech Stack:** Typst paper, Python eval scripts (`confirmatory_eval.py`), the quantize tooling
from the rebased branch (`aerocapture.training.quantize`).

## Global Constraints

- Branch: work on `feature/quantization-mamba962` (already rebased on main). Never push.
- Compile from repo ROOT: `typst compile articles/paper/paper.typ articles/paper/paper.pdf`;
  page-proof changed pages as PNGs and view them; tables verified structurally (cells per row vs
  `columns:`) AND on final renders.
- Committed-data rule: every quoted number regenerates from `articles/paper/data/quant/*.json` or
  `articles/paper/data/confirmatory_eval.json`. The notes flag one known typo: a83450b's commit
  message says 1592 B for the deployed int4 cell - the JSON's **1564 B** is authoritative.
- De-tooling rule: no CLI/paths/module names in paper prose (language/library names OK; the
  reproduction pointer stays generic - the released tooling covers it).
- Decisions locked: quantization = **Appendix C**; cards -> **Appendix D**; figures = the sweep
  figure only (`quantization_sweep.svg`), tables for everything else.
- Typst gotchas in force: `#box[..];` / `#super[..];` swallows the semicolon (use `\;`);
  `$..$` math digits; captions above tables.
- Pool discipline: the PTQ grid + LOO stay quoted on their fresh-pool cells (exploratory,
  labeled as such); the FINALISTS table quotes the confirmatory pool (selection happened on the
  fresh pool; quote-once discipline stated in the appendix text).
- Number provenance for compute: the appendix's kernel arithmetic uses the campaign's measured
  **644 forward-pass invocations per simulation** (`ticks_per_sim.json`); Section 7.2's ~5 us/update
  uses mean flight seconds at the 1 s cadence (734) - different denominators, both stated where
  used, never mixed. Whole-sim costs quote the CURRENT benchmark (3.59/2.35 ms), not the notes'
  3.68/2.40.

---

### Task 1: Confirmatory-pool quotes for the quant finalists

**Files:**
- Create: `training_output/quant/ptq4_verdict/best_model.json` (+ `best_params.json` copied from
  the champion) - materialized PTQ model for pinning
- Modify: `articles/paper/data/confirmatory_eval.json` (3 new cells, via the existing script)

**Interfaces:**
- Consumes: `articles/paper/scripts/confirmatory_eval.py` (`--cells label:toml`, `--scaffolding-from`),
  `aerocapture.training.quantize` (the mamba-aware `quantize_model_weights` with tensor policy -
  read its signature in `src/python/aerocapture/training/quantize.py` first; the sweep CLI in the
  same module shows the exact call used for the verdict cell: bits=4, granularity="per_channel",
  policy="proj_only").
- Produces: confirmatory cells `quant/ptq4_verdict`, `quant/mamba962_qat4_finetune`,
  `quant/mamba962_qat4_scratch` with pooled CVaR95/CVaR99.9 + replicate SEs. Task 2 quotes them.
  (champion_fp's confirmatory row already exists: `mamba_p962_long`, CVaR99.9 123.3 +/- 0.1.)

- [ ] **Step 1: Materialize the PTQ-verdict model.** Read
  `src/python/aerocapture/training/quantize.py` for the exact quantize-and-save entry point (the
  sweep produced the verdict cell with bits=4 / per_channel / proj_only), then:

```bash
mkdir -p training_output/quant/ptq4_verdict
uv run python -c "
from aerocapture.training.quantize import <verified entry point>
# quantize training_output/mamba_p962_long/best_model.json at 4b/per_channel/proj_only
# -> training_output/quant/ptq4_verdict/best_model.json
"
cp training_output/mamba_p962_long/best_params.json training_output/quant/ptq4_verdict/
```
  Sanity: score the materialized model on the fresh pool at n=1000 and check it reproduces the
  grid's 4b/per_channel/proj_only cell (capture 1.000, CVaR95 ~147.9) - proving the file equals
  the sweep's in-memory transform - before spending confirmatory compute.

- [ ] **Step 2: Run the three confirmatory cells** (box idle; ~25 min each):

```bash
uv run python articles/paper/scripts/confirmatory_eval.py --replicates 10 --n 100000 --cells \
  quant/ptq4_verdict:configs/training/sweep/mamba_p962.toml \
&& uv run python articles/paper/scripts/confirmatory_eval.py --replicates 10 --n 100000 --cells \
  quant/mamba962_qat4_finetune:configs/training/quant/<finetune toml, from configs/training/quant/> \
&& uv run python articles/paper/scripts/confirmatory_eval.py --replicates 10 --n 100000 --cells \
  quant/mamba962_qat4_scratch:configs/training/quant/<scratch toml>
```
  (Read `configs/training/quant/` and `experiments/paper/17_quantization.sh` for the exact TOML
  names before launching. Labels resolve to `training_output/quant/<cell>`; run-local
  `best_model.json` + co-trained `best_params.json` pin each row - the finetune arm's scaffolding
  moved during QAT, so its own best_params matters.)

- [ ] **Step 3: Sanity + the headline question.** Compare qat4_finetune's confirmatory CVaR95
  against the fresh-pool 116.6 (expect within ~1 m/s) and record its CVaR99.9 against the
  deployed champion's 123.3 +/- 0.1 - this is the number the appendix's tail-equivalence claim
  now rests on at full sizing depth.
- [ ] **Step 4: Commit** `articles/paper/data/confirmatory_eval.json` (+ the materialized model
  dir is training_output - NOT committed; gitignored as usual):

```bash
git add articles/paper/data/confirmatory_eval.json
git commit -m "data(quant): confirmatory-pool quotes for the three quantization finalists"
```

---

### Task 2: Appendix C - the quantization section

**Files:**
- Modify: `articles/paper/paper.typ` (insert `= Appendix C: quantizing the deployed Mamba head`
  between Appendix B (probes) and the mission cards; relabel cards heading to Appendix D)
- Uses figure: `articles/paper/figures/quantization_sweep.svg` (committed, ready)

**Interfaces:**
- Consumes: `docs/paper/quantization_appendix.md` (the single content source),
  `articles/paper/data/quant/{quantization_results,finalists_results,bench_forward,ticks_per_sim}.json`,
  Task 1's confirmatory cells.
- Produces: `<tbl-quant-finalists>`, `<tbl-quant-loo>`, `<tbl-quant-memory>`, `<tbl-quant-bench>`,
  `<fig-quant-sweep>` labels Task 3 may reference.

- [ ] **Step 1: Write the section** (~3 pages), following the notes' claim order and Appendix B's
  prose idiom. Structure:
  1. Motivation paragraph: deploy-size reduction of the 962-parameter head; weight-only
     fake-quant methodology (rounded values stored back as 64-bit floats, so the accuracy study
     changes nothing else); selection on the fresh re-quote pool, finalists quoted once on the
     frozen confirmatory pool (pool discipline sentence).
  2. Finalists table (confirmatory-primary: capture / p50 / CVaR95 / CVaR99.9 +/- replicate SE per
     variant; fresh-pool n=10k values may ride as a parenthetical column if space allows). The
     tail-equivalence claim quoted against BOTH sigma_run(CVaR95) ~ 1.2-2.1 (probe budget,
     directional) AND the confirmatory replicate SEs.
  3. LOO table + the SSM-specific claim (a_log +53.0 / d_skip +47.4 from 242 scalars; a_log
     one-sided so a symmetric grid wastes half its levels; dt_proj_w exactly 0 by construction;
     LOO deltas do not add up - interactions compound - hence proj_only).
  4. PTQ grid: the sweep figure `<fig-quant-sweep>` + a compact reading paragraph (8-bit free at
     +0.4; collapse below 4 bits; the 4-bit policy split far above the n=1000 noise floor; the
     non-monotone granularity cells explicitly not over-read). The full 20-row grid table is NOT
     reproduced - the figure + the committed data carry it.
  5. Memory table (7696 B -> 1564 B x4.9 deployed cell; the 624 B cell quoted for contrast only,
     accuracy-broken) + compute table (criterion medians; f32 the sweet spot at -32%/tick; integer
     kernels ~-8% because dynamic activation quantization + the fp SSM recurrence dominate a
     962-parameter workload; w4a8 nibble unpacking cancels its bandwidth edge). State the 644
     measured forward invocations per simulation as the ticks basis, and quote the head share
     against the CURRENT 3.59 ms/sim (1888 ns x 644 = 1.22 ms, ~34%).
  6. Caveats block from the notes (weight-only scope; single-run QAT arms; n=1000 grid cells;
     Apple-silicon ratios do not transfer, the memory table does; scaffolding live in both arms).
  7. One-line closing: the deployment benefit is memory and fixed-point projection arithmetic,
     not speed; the affine-quantizer question for a_log stays open (one line, mirrors the notes'
     contingency).
- [ ] **Step 2: Relabel the cards appendix**: `= Appendix C: per-scheme mission reports` -> `=
  Appendix D: per-scheme mission reports`; sweep the ~4 literal "Appendix B/C" body mentions
  (`rg -n "Appendix [BCD]" articles/paper/paper.typ`) so probes stay B, quantization = C, cards =
  D; check `articles/paper/appendix.typ` for its own self-references.
- [ ] **Step 3: Number verification script pass.** Cross-check every typed table value against the
  JSONs (print-and-diff, the campaign plan's Task-11 pattern):

```bash
uv run python -c "
import json
q = json.load(open('articles/paper/data/quant/quantization_results.json'))
f = json.load(open('articles/paper/data/quant/finalists_results.json'))
b = json.load(open('articles/paper/data/quant/bench_forward.json'))
c = {x['label']: x for x in json.load(open('articles/paper/data/confirmatory_eval.json'))['cells']}
# print the exact values the appendix tables must show, in table order
..."
```
  Every printed value must match the typed Typst cell; fix the Typst side on any mismatch.
- [ ] **Step 4: Compile + proof** the new appendix pages (render PNGs, check both tables
  structurally and visually; the sweep figure legible at full width). Commit:

```bash
git add articles/paper/paper.typ
git commit -m "paper: Appendix C -- quantizing the deployed Mamba head (cards shift to Appendix D)"
```

---

### Task 3: Body integrations

**Files:**
- Modify: `articles/paper/paper.typ` - Section 9 future-work paragraph (`rg -n "pruning or
  quantizing"`), Section 7.2 compute paragraph (`rg -n "price of the tail it buys"`).

- [ ] **Step 1: Section 9 rewrite.** Current thread: "We deliberately leave three threads as
  future work. We have no clean campaign study of pruning or quantizing the deployed head -- the
  only such cells predate the simulator fixes in this work and are not comparable -- so
  deploy-size reduction of the Mamba policy is open. ..." Replace the count and the first thread:

```typst
Two of the threads earlier drafts left open are now closed in the appendices: the state-ablation
controls of Section 6.3, and the deploy-size question -- Appendix C quantizes the deployed head
(weight-only), finding 8-bit free, the SSM dynamics parameters the 4-bit bottleneck, and a
quantization-aware fine-tune that holds the sizing tail at a $4.9 times$ memory reduction; pruning
remains open.
```
  Adjust the surrounding sentence flow so the paragraph still reads as one narrative (the
  variance-calibration sentence follows unchanged).
- [ ] **Step 2: Section 7.2 pointer.** After "the price of the tail it buys." insert:

```typst
The head itself accounts for $approx 1.2$ ms of that cost; single-precision arithmetic would
roughly halve it, while integer quantization buys memory rather than speed at this scale
(Appendix C).
```
- [ ] **Step 3: Compile, proof the two touched pages, commit**:

```bash
git add articles/paper/paper.typ
git commit -m "paper: sec 9 deploy-size thread closed by Appendix C; sec 7.2 head-cost pointer"
```

---

### Task 4: Proofs, docs, memory

- [ ] **Step 1: Full compile; page count** (expect ~58-59 pp); render + view the abstract page
  (unchanged - verify no accidental drift), the new Appendix C pages, the relabeled Appendix D
  heading page, Section 9.
- [ ] **Step 2:** Update `docs/paper/quantization_appendix.md` header: "Status: INTEGRATED into
  paper.typ Appendix C (2026-07-12); this file remains the working-notes source." Update
  `paper_resume.md` (one paragraph: appendix integrated, confirmatory quant cells, lettering
  C/D). Update memory `project_paper_state.md` (appendix in; qat4_finetune confirmatory tail
  number) and `project_quantization_result.md` (integrated + confirmatory-grade numbers).
- [ ] **Step 3:** Remind Gregory: Desktop PDF copy (TCC), and the repo-URL placeholder still open
  from the revision.
- [ ] **Step 4: Commit** docs:

```bash
git add docs/paper/quantization_appendix.md paper_resume.md
git commit -m "docs: quantization appendix integrated -- notes/resume updated"
```

---

### Task 5: smart-commit close-out

- [ ] **Step 1:** Invoke the `smart-commit` skill, telling it to take the whole
  `feature/quantization-mamba962` branch into account (campaign machinery + the integration), so
  CLAUDE.md documents the quantize tooling/QAT knobs if the campaign commits did not already, then
  final commit. Never push - Gregory pushes and merges.

---

## Self-review notes

- The two `<... toml>` markers in Task 1 Step 2 are read-then-fill (exact filenames live in
  `configs/training/quant/`, verified against `17_quantization.sh` before launching) - a defined
  lookup, not an open TBD; same for the quantize entry point in Step 1 (name verified from the
  module before use, with the sanity gate proving equivalence to the sweep's transform).
- Coverage vs the notes' integration checklist: item 1 (Task 2), item 2 (Task 3.1), item 3
  (Task 3.2), item 4 claim order (Task 2.1), item 5 data (Tasks 1-2), item 6 figures (sweep only,
  per decision), item 7 numbers hygiene (Task 2 Step 3 + the 1564-vs-1592 note in Global
  Constraints), item 8 reproduction (covered by the released tooling; de-tooled in prose).
- New vs the notes: confirmatory-pool quotes (revision discipline), the 3.68->3.59 requote, the
  644-vs-734 denominator scoping, and the appendix lettering decision (C, cards->D).
