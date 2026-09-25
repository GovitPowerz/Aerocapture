# Changes since arxiv-v3

The committed `paper.pdf` is recompiled from the Typst source below (`make -C articles/paper pdf`,
last on 2026-09-25 with #156 and #166); the arxiv-v3 build is the `arxiv-v3` tag.

- 2026-09-16 (#108, ADR-0006): the abstract and the conclusion lead with the per-scenario-noise
  result (fine-tuned Mamba, three-seed means: CVaR99.9 163.2 +- 1.3 m/s, 99.996% capture of 10^6,
  73 m/s below FNPAG and the best dense network) and state the shared-path result (123.3 +- 0.1 at
  100% capture) as the historical headline that Appendix E corrects. Sections 5-7 and their tables
  still quote the shared-path regime and say so; per-scenario re-quotes of those sections are not
  part of this change.
- 2026-09-22 (#101): Section 5's reinforcement-learning sentence is re-quoted from
  protocol-matched cells. The old sentence ("636 m/s mean for the dense PPO policy and 513 for
  the recurrent one", footnoted as predating simulator fixes) mislabelled its artifacts: both
  bundled legacy RL cells (`legacy/neural_network_rl`, `legacy/neural_network_gru_ppo`) are one
  Dense -> GRU(16) -> Dense architecture on a 23-input mask at 15M / 60M env steps, matched to no
  population cell. The new baseline (`experiments/paper/18_rl_baseline.sh`, configs under
  `configs/training/paper/rl/`) trains PPO from scratch and PPO warm-started from the per-scenario
  champions `ou_marginal/ft_dense_p515` and `ou_marginal/ft_gru_p1014` under their exact
  observation contract, scaffolding, per_draw regime and reserved pools; the quote reads the new
  `rl/*` and `ou_marginal/*` keys of `data/results.json` (2M pool, n = 1000, four `ppo_*` paired
  tables). The footnote is gone. The README results card gains the two dense PPO rows on the
  10^6 confirmatory pools (`experiments/ou_marginal/confirmatory_marginal.json`) and a
  contributions block; the review workflow artifacts moved from `articles/paper/` to
  `docs/paper/reviews/`.
- 2026-09-22 (#120): the headline tables read the bundle at compile time. `results.typ`
  wraps `data/results.json`, `data/confirmatory_eval.json` and `data/quant/finalists_results.json`;
  every cell of the performance table, the paired-comparison table and the quantization
  finalists table is an accessor call (the performance table's Viol. column stays transcribed:
  the bundle carries no violation field). A colophon reads `data/provenance.json` (paper-inputs
  digest, Release tag, toolchain versions, noise regime) and the git head `make pdf` passes in.
  One cell moved: the QAT fine-tune CVaR99.9 is 122.85 pooled, which rounds to 122.9 (the
  transcription had rounded it down); the two prose quotes follow. Every other converted cell
  reproduces its literal.
- 2026-09-23 (#134): every prose quote of the 99.996% capture rate (abstract, Section 9,
  conclusion) is labelled as a three-fine-tune-seed mean, the label the CVaR99.9 beside it
  already carried; Section 9 and the conclusion also give the deployed seed's 99.9995%, the
  value in Appendix E's far-tail confirmatory table. The conclusion's second quote of the
  CVaR99.9 (163.2) gains the same label; the table's 163.0 is the deployed seed. Per-seed
  captures on the 10^6 confirmatory pool: 999995 / 999970 / 999920 (s1 deployed / s2 / s3).
  Review follow-up (PR #136): the abstract also gives the deployed seed's 99.9995%, and every
  three-seed +- (abstract, Section 9, conclusion, Appendix E, README, TODO) is labelled one seed
  standard deviation; the far-tail table's +- stays the replicate standard error. No number
  changes.
- 2026-09-23 (#137): the per-scenario headline reads the bundle. `data/confirmatory_marginal.json`
  holds the cells and fields the paper quotes from `experiments/ou_marginal/confirmatory_marginal.json`
  (per_draw, 10 x 100,000): it is written by `scripts/extract_confirmatory_marginal.py`
  (`make confirmatory-marginal`, also a step of `make paper`), checked against its source by
  `make check`, and covered by `data/SHA256SUMS` and the provenance digest. `results.typ` gains
  `marg()` (pooled capture from n_captured / n, scenarios lost, CVaR95, CVaR99.9 with its
  replicate s.e., worst case, violation %; the per_draw regime is asserted at load) and `mean_sd()`
  (mean and sd, over the three fine-tune seeds); every cell must cover the full pool, and
  `paper.typ` asserts the 10 x 100,000 shape and the three seeds' zero violation share that its
  prose quotes. Now accessor calls: every cell of the
  Appendix E far-tail table (its bold marks the lowest CVaR95 / CVaR99.9, derived rather than
  flagged) and its caption's FNPAG non-capture share, the table's prose and the seed-robustness
  paragraph with its loss rates, and the per-scenario quotes of the abstract, Section 9 and the
  conclusion (99.996%, 99.9995%, 163.2 +- 1.3, 163.0 +- 0.3, the 73 m/s margin, "past 236"). The
  colophon names the new file and its regime. No value moved: every page except the colophon
  renders pixel-identical to the #134 build. Left transcribed at the time: "near 237" (abstract,
  conclusion) and the n = 1000 per-scenario quotes, whose source
  `experiments/ou_marginal/quote_results.json` was outside the bundle (brought in by #157 below).
- 2026-09-25 (#154): the four Section 5 PPO cells are retrained on the parity environment.
  The 2026-09-22 cells had trained against a one-tick observation lag and an action injected
  past the command shaper (fixed 2026-09-24, gate `src/rust/tests/rl_env_parity.rs`), a shaping
  potential kept at terminations, and three episode-boundary leaks (#150, #151); their bundled
  `rl/*` artifacts, `results.json` rows and 10^6 confirmatory-marginal rows are replaced. The
  Section 5 sentence now reads: scratch 180 / 284 (dense, 99.9% capture, 0.6% heat-flux) and
  382 / 412 (GRU, 100%, 0.8%), paired +67 / +257 m/s, both still improving at the budget; the
  dense warm start deploys the champion (+1.1 paired) then walks off it (validation capture 83%
  by 17M steps); the GRU warm start is off the champion at its first gate and plateaus at +11
  paired. Superseded: 237 / 316 (4.8% heat-flux), 284 / 435 (47% heat-flux + 31% g-load), warm
  +0.3 / +3.4 then capture 2% / 91%. The change is not attributable to any one of the fixes.
  The two champions' 2M-pool parquets were regenerated on the same build (only the #141 time
  columns and one non-capture virtual DV moved; every quoted statistic is unchanged).
- 2026-09-25 (#157): the last transcribed numbers read the bundle. `data/quote_marginal.json`
  holds the cells and fields Appendix E quotes from `experiments/ou_marginal/quote_results.json`
  (the paired n = 1000 pool scored under both regimes: "frozen" pins the shared noise path,
  "marginal" re-seeds `simulation.random_seed` per scenario, both under legacy seeding): written
  by `scripts/extract_quote_marginal.py` (`make quote-marginal`, also a step of `make paper`),
  checked against its source by `make check`, covered by `data/SHA256SUMS` and the provenance
  digest. `results.typ` gains `ou()` (capture %, CVaR95, heat-load violation % of one cell under
  one regime; the regime pair is asserted at load) and `paper.typ` asserts the n = 1000 pool, the
  counts its prose states (fourteen of fifteen clean scratch repeats, three of five clean
  fine-tunes, the one unclean repeat a dense-515 seed losing one scenario, clean constraints for
  the deployed FNPAG, the GRU fine-tune regressing) and that the dense fine-tune and FNPAG both sit
  within 1 m/s of the "near 237" the abstract and conclusion quote (now the rounded larger of the
  two CVaR99.9). Now accessor calls: every cell of the shared-path-versus-per-scenario table and of
  the retraining table (the scratch mean +- sd computed from the three repeats; the PredGuid / FTC
  row's ranges and every "a--b" range in prose derived as min--max), the tail losses of the laws
  and of the networks (11--31, 54--102: table prose and conclusion), FNPAG's 154.3, the two
  fine-tune CVaR95 (129.8, 138.6) and their 16--25 margin, the scratch retrains' 17--69 margin, the
  3 m/s compression of the three close scratch means, the +-1.6 against +-5.7--12.7 consistency
  quote, the pilot's 138.3, the LSTM fine-tune's 10.8% violation. The colophon lists the two
  tables, Appendix E and the new file among those read at compile time. No value moved: every page
  but the colophon's renders pixel-identical to the #154 build.
  No number the paper quotes from `experiments/ou_marginal/` is transcribed any more; the "2--4 times" ratio of
  network to classical tail loss (abstract, Section 1, the table caption) stays a rounded prose
  characterization of those two ranges, not a bundle value.
- 2026-09-25 (#166): Appendix E's three prose claims are narrowed to what `data/quote_marginal.json`
  shows, and the regime check tests the source. The shared-path-versus-per-scenario table notes,
  beside every value under either regime, a capture rate below 100% and any heat-load violation, the
  one rule for every row and the one the prose introducing the table now states (two rows carried a
  note before, under an intro whose rule the notes did not follow): the per-scenario GRU (98.0%
  capture, 1.5% violation), Dense 972 (99.3%, 0.5%), Dense 515 (98.4%, 0.8%), FNPAG (99.4%) and
  PredGuid (99.9%) join the Mamba (now with its 1.2% violation) and the LSTM (17.0%, and 14.6%
  under the shared path); an assert holds that every network sheds capture or feasibility. The
  first conclusion no longer says the scratch retrains "edge FNPAG's 154.3 by roughly one
  sigma_run": four of the five scratch means sit below it, by 0.6--6.2 m/s, and the dense-515 mean
  sits 4.3 above, both asserted. The consistency quote's range covers all four other cells
  (+-3.8--12.7, not +-5.7--12.7) and asserts the Mamba spread the smallest of the five.
  `experiments/ou_marginal/quote_marginal.py` writes its protocol into `quote_results.json`
  (`regimes`: the noise seeding and per-seed override of each regime; `seed_pool`: the rng, seed
  and range of the shared pool); `scripts/extract_quote_marginal.py` copies that record and exits
  on a source without it, so the load-time regime assert of `results.typ` tests the scoring
  script rather than the extractor's own constant. Re-run on 2026-09-25: all 64 previously scored
  cells reproduce bit-identically (four `ft_mamba_p962_s2` / `_s3` cells trained since are new to
  the source and unquoted); no quoted value moved.
- 2026-09-25 (#156): Section 7.3's centered high-regime cells are quoted at sizing depth, and the
  claim changes. `data/centered_depth.json` (`scripts/centered_depth_eval.py`, `make
  mc-centered-depth`) scores the three centered-Mamba trainer seeds and the two joint-FTC references
  on the 9M stress pool at n = 10,000 (its first 1000 seeds are the n = 1000 pool), paired on
  scenario, bootstrap 95% CIs, under both noise regimes, each labelled; `results.typ` gains
  `centered()` and `centered_paired()` and asserts the regime pair at load. Two new tables (shared
  path; per-scenario noise) replace the transcribed three-seed sentence; the figure's Mamba bar
  becomes the three n = 10,000 seeds with CI whiskers, the joint-FTC line the n = 10,000 reference
  with its interval, every bar under the shared path. Under the shared path the n = 1000 claim
  survives: every seed beats both references on the tail with paired CIs excluding zero
  (104--178 m/s below the retrained joint-FTC, 22--97 below the medium-deployed one) at capture
  within half a point; the tails read 314--388 m/s at 95.9--96.0% capture, not the 231--273 at
  94.8--95.0% the n = 1000 quote gave. Under per-scenario noise the reversal does not survive: the
  retrained joint-FTC captures 96.3% against the seeds' 94.3--95.8% and out-tails two of three,
  the medium-deployed one holds the best conditional tail (457 against 535--610). Section 7.3, its
  figure caption and the Discussion now say so (the centered cells trained on the shared path;
  the open follow-up is a centered retrain under per-scenario seeding), asserted in `paper.typ`;
  the "not intrinsic to neural guidance" conclusion is scoped to the shared path. The Discussion's
  echo of Appendix E's FNPAG margin follows #166 (four of five scratch means below FNPAG, by at
  most 6.2 m/s). The colophon lists the two tables and the new file.
