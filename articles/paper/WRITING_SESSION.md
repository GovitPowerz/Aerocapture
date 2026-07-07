# Writing-session instructions (paper prose)

> **STATUS — DONE (commit c7b4907).** The prose is written: `articles/paper/paper.typ` is filled
> (all sections, tables T1-T4, `refs.bib` added) and compiles to a 19-page PDF. The three decisions
> below were resolved as: (1) methodology-first section order; (2) abstract leads with the
> architecture result; (3) dense_515 as full efficiency-reference rows. Single author (G. Gelly,
> no affiliation). The notes below are kept as the provenance of how the paper was written.
>
> Hand-off for the session that wrote the paper. ALL prep was done — data, 14 figures,
> outline, and a compilable Typst skeleton. The job was the PROSE, in the author's voice.
> Nothing needs training or re-evaluation; do not re-run experiments.

## Read first, in this order
1. `paper_resume.md` (repo root) — the campaign state + every result/number, with the headline.
2. `articles/paper/OUTLINE.md` — section-by-section structure, the one claim per section, the
   figure/table each carries, and the locked numbers. THIS is the writing map.
3. `articles/paper/paper.typ` — the skeleton to fill (see "How to write" below).
4. `articles/markdown/05_authorial_voice_and_style.md` — the author's voice. Match it.
   (`00_synthesis_writing_kit.md` + `01_2009_AIAA_neural_guidance.md` = the 2009 predecessor + kit.)

## The deliverable
A single paper: `articles/paper/paper.typ` → `articles/paper/paper.pdf`.
Follow-up to **Gelly & Vernis, AIAA GNC 2009**. Benchmarks NN aerocapture guidance vs classical /
predictor-corrector schemes + the moving-environment training methodology.

## How to write (fill the skeleton)
`paper.typ` has three kinds of marker — turn the document from scaffolding into prose:
- `#todo[...]` — yellow box = PROSE YOU MUST WRITE. Replace each with real prose, then the marker is gone.
- `#guide[...]` — blue box = the section's claim + the locked numbers (your writing brief). DELETE each
  as you finish the section (it is scaffolding, not paper content).
- `#fig(...)` — figures are already wired with DRAFT captions. Keep the include; refine the caption.

Compile (from repo ROOT, so figure paths resolve):
```
typst compile articles/paper/paper.typ articles/paper/paper.pdf
```
(`pdftoppm`/poppler not installed here — open the PDF in a viewer to proofread.)

## Three decisions to make FIRST (flagged at the top of paper.typ + in OUTLINE.md)
1. **Section order** — the skeleton leads with methodology (§4). Alternative: results-first (move the
   architecture/classical sections up). Pick one.
2. **Abstract lead** — open with the ARCHITECTURE result (Mamba beats classical at the sizing tail) or
   the METHODOLOGY (the moving environment). Both are contributions; the spine slightly favors architecture.
3. **dense_515** — full row everywhere (efficiency reference) or a footnote to the parameter-efficiency
   + GA-dimensionality story.

## Data sources (for ANY number — never invent)
All committed under `articles/paper/data/`:
- `results.json` — `runs` (per-cell capture/mean/p95/CVaR95/p99/CVaR99/heat/g) + `paired` (the
  cross-cell comparisons, now WITH `delta_p95` / `delta_cvar95` tail deltas) + `headline_fresh_pool`.
- `far_tail_eval.json` — n=10000 tail (CVaR99 / CVaR99.9 / max) for the headline + classical cells.
- `robustness_stress.json` (5c, off-nominal) · `compute_benchmark.json` (5b, ms/sim).
- `plateau.json` (val-RMS-vs-gen) · `corridor.npz` (the MC ensemble) · `runs/` (the committed bundle).
- Regenerate the aggregate after any data change: `uv run python articles/paper/scripts/aggregate_results.py`.

## The 14 figures (all built, in `articles/paper/figures/`, wired into paper.typ)
Headline: `fig_arch_tail` (the 10c σ_run result), `fig_classical_vs_nn` (deployability scatter), `fig_pareto`.
Methodology: `fig_seed_strategy`, `fig_cost_transform`, `fig_curation`, `fig_training_n_sims`, `fig_plateau`, `fig_optimizer`.
Classical: `fig_joint_reference`, `fig_robustness`. Other: `fig_ablation`, `fig_corridor`, `fig_output_param`.
Each `fig_*.py` reads only committed data; rebuild one with `cd articles/paper/scripts && uv run python fig_<name>.py`.

## Tables to fill (skeletons marked in paper.typ)
- T1 (§3) MC dispersion domains (26) — from `configs/missions/*.toml` + `dispersions.rs`.
- T2 (§4) scheme summary (signed/unsigned bank, compute class, reference-dependence).
- T3 (§8) final MC performance, all schemes — AUTO-FILL from `results.json` `runs` (capture, mean, p95,
  CVaR95, CVaR99.9, max, violation %). Lead with the tail columns.
- T4 (§8) paired comparisons — from `results.json` `paired` (`nn_vs_*`, `headline_vs_*`): dMean, dP95,
  dCVaR95, win-rate, p. (Consider a tiny script to emit T3/T4 as Typst tables from results.json.)

## Bibliography
Create `articles/paper/refs.bib`, then uncomment the `#bibliography("refs.bib")` line in paper.typ.
Minimum: Gelly & Vernis 2009 (the predecessor); Lu (FNPAG / numerical predictor-corrector); the FTC /
apoapsis-enslavement heritage; Ng-Harada-Russell 1999 (PBRS, the RL appendix if kept); the Mamba/SSM
and LSTM-init references for the architecture section.

## LOCKED headline numbers (do not re-derive)
- Deployed sizing headline = **Mamba_962** (`training_output/mamba_p962_long/`). dense_515 = efficiency reference.
- Mamba fresh-pool (8M, the abstract number): p50 **109.7** / CVaR95 **115.2** / p99 116.0; 100% capture.
- Far-tail σ_run (3 seeds, n=10000) CVaR99.9: Mamba **124.5** < LSTM **129.2** < Dense **139.2**;
  max 127.6 < 132.4 < 159.0.
- vs best classical (joint-FTC): `nn_vs_jointftc` dMean **−16.4** / dCVaR95 **−27.6** (99.9% win, p=3e-165).
- Compute (5b): NN-mamba 3.68 / NN-dense 2.40 / FTC 1.25 / FNPAG 86.1 ms/sim (NN 23× < FNPAG).
- Robustness (5c, off-nominal, HONEST CAVEAT): joint-FTC most robust (capture drop 5.5% / CVaR95 +197)
  vs NN-mamba (9.9% / +402). Frame: NN wins NOMINAL sizing, joint-FTC generalizes better.
- Methodology: GA fixed→rotating→adaptive seeds = 160.3 → 120.0 → 118.0 mean (−42 m/s, biggest effect).

## Typst gotchas (these bit during skeleton authoring — avoid in prose)
- `_word_` is EMPHASIS — code identifiers (`predicted_dv2`, `hdot_nominal`) italicize; wrap in
  `` #raw("predicted_dv2") `` or backticks if you want literal monospace.
- `@name` is a REFERENCE — write "at population 150", not "@150".
- `<name>` is a LABEL — keep `<` out of markup (use math mode `$E < 0$` for inequalities).
