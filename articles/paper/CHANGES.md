# Changes since arxiv-v3

The committed `paper.pdf` is recompiled from the Typst source below (`make -C articles/paper pdf`,
last on 2026-10-01 with #170 and the Pareto sweep cells); the arxiv-v3 build is the `arxiv-v3` tag.

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
  under the shared path). The asymmetry sentence under the table names the feasibility split the
  notes show (every network adds heat-load violations, no law incurs any; capture slips in both
  families), each half asserted; it had contrasted the networks as the ones that "shed capture or
  feasibility" beside FNPAG's 99.4% and PredGuid's 99.9% capture. The abstract and the conclusion
  no longer say the retraining "restores 100% capture ... for every cell": the scratch retrains are
  feasible in every cell and capture all but one of 15,000 scenarios, asserted. The
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
- 2026-09-26 (#156): Section 7.3's centered high-regime cells are quoted at sizing depth, and the
  claim is scoped to the shared noise path. `data/centered_depth.json` (`scripts/centered_depth_eval.py`,
  `make mc-centered-depth`) scores the three centered-Mamba trainer seeds and the two joint-FTC
  baselines on the 9M stress pool at n = 10,000 (its first 1000 seeds are the n = 1000 pool), paired
  on scenario, with bootstrap 95% CIs on capture, mean and CVaR95 and on the paired capture and
  CVaR95 deltas, under both noise regimes, each labelled. `results.typ` gains `centered()` and
  `centered_paired()` (regime pair asserted at load) and `span()` (Appendix E's `ou_span`, moved;
  Appendix E renders unchanged). Two tables (shared path; per-scenario noise) replace the
  transcribed three-seed sentence; the figure's Mamba bar becomes the three n = 10,000 seeds with CI
  whiskers and the joint-FTC line the n = 10,000 baseline with its interval, every bar under the
  shared path. Under the shared path the claim survives: every seed beats both baselines on the
  conditional tail with paired CIs excluding zero (147--178 m/s below the retrained joint-FTC,
  65--97 below the medium-deployed one) at capture within half a point (314--345 m/s at
  95.9--96.0%, not the 231--273 at 94.8--95.0% of the n = 1000 quote). Under per-scenario noise it
  does not: the retrained joint-FTC out-captures every seed and out-tails two, and the
  medium-deployed one holds the best conditional tail (457 against 535--604) but is out-captured by
  seed 3. Section 7.3's title and prose, the figure caption, the Discussion and the abstract's
  off-nominal clause now say so, asserted in `paper.typ`. Seed 1's run-local model had been
  overwritten on 2026-07-11 by the seed 3 repeat through the shared config's deploy path; it was
  rebuilt from seed 1's final checkpoint and reproduces the committed n = 1000 quote exactly
  (94.9%, CVaR95 272.8), and the eval script exits when two seeds share a model file. The same
  deploy-path overwrite reached three other run directories; their repair is a separate change.
  The Discussion's echo of Appendix E's FNPAG margin follows #166 (four of five scratch means below
  FNPAG, by at most 6.2 m/s). The colophon lists the two tables and the new file. Section 7.2's
  lexicographic rule states the half-point capture-parity band Section 7.3 reads with (seed 3's
  resolved 0.15-point deficit to the retrained joint-FTC sits inside it; the per-scenario ranking
  deltas sit outside it except seed 2's parity, asserted), and the pool table gives the stress
  pool's n = 10,000 depth and one query per policy and noise regime.
- 2026-09-28 (#170): Appendix E's Dense 515 regime row is re-quoted from the deployed champion.
  `training_output/dense_p515_ga_paper_best/best_model.json` had been overwritten by the
  `paper/tail_repeats/dense515_s3` repeat through the shared config's deploy path, so the n = 1000
  quote flew that repeat's weights (CVaR95 126.2 shared path / 228.0 per-scenario, 98.4% capture and
  0.8% violation per-scenario). The champion, byte-identical to the bundle's `headline/dense_p515`,
  gives 117.4 / 201.8 at 100% capture and 0.3% violation; `quote_marginal.py --only dense_p515`
  rewrote those two cells of `experiments/ou_marginal/quote_results.json` and
  `data/quote_marginal.json` follows. The paper compiles with every assertion passing; the
  committed `paper.pdf` is not recompiled here. The trainer now writes `[data] neural_network` only
  when it lies inside the run's output dir, and `experiments/paper/audit_deployed_models.py`
  rebuilds every run's winner from its final checkpoint: every rebuildable run under
  `training_output/` deploys its own winner, including the three other overwritten directories.
- 2026-10-01 (Pareto sweep cells): the Section 6 Pareto panel plots all 28 sweep cells.
  `sweep_gru_p1014`, `sweep_lstm_p1082` and `sweep_mamba_p962` had never been bundled:
  `collect_runs.py` skips a run dir whose `best_model.json` is newer than its
  `final_eval.parquet`. The GRU and LSTM dirs held another run's model from 2026-06-25 and
  2026-06-27 (the trainer copied every checkpoint's network to the config's `[data]
  neural_network`, so `gru_p1014_long` and the LSTM repeat s3 overwrote them) until their
  restoration on 2026-09-26 (#170); the Mamba dir's final selection was re-run on 2026-08-27 (same
  champion, not promoted). `fig_pareto.py` had excused the three as extended in place by
  `10b_arch_long_challengers.sh`, which seeds separate `*_long` dirs. Re-flown on the
  1000-scenario pool (shared-path noise), each current model reproduces its June
  `final_eval.parquet` bit for bit except the five #141 time-label columns, which move the same
  way for a bundled control cell per family; the June parquets are bundled and `fig_pareto.py`
  errors on any missing cell. The panel gains LSTM 1082 (dv99 124.3), Mamba 962 (127.3) and GRU
  1014 (132.8). Section 6's best mean per family becomes LSTM 112.0, GRU 112.8, Mamba 114.9
  against the best dense 116.8 (was GRU 112.8, Mamba 114.9, LSTM 116.0): LSTM 1082 has the best
  mean of the sweep, so the tail-reversal aside no longer credits the GRU with it and names the
  1014-parameter GRU cell taken to convergence.
  Every cell still captures 100% and Transformer 762 stays the worst (121.9). `results.json`
  gains the three runs and no other entry moves. `runs/headline/dense_p515/ablation_results.json`
  (2026-06-19, skipped by the same guard while that dir held a clobbered model; read by no
  figure) is bundled with them, so a full `12_collect_results.sh` leaves the tracked bundle as is.
- 2026-10-01 (#170 follow-up): two more consumers of the overwritten models are re-flown.
  - Appendix B: the CfC and xLSTM probes' GRU and LSTM reference rows (scored 2026-07-10 from
    `training_output/sweep_*/best_model.json`) had flown `gru_p1014_long` and the LSTM repeat s3,
    not the 5000-generation sweep cells the budget caveat describes. Re-scored on the restored
    cells with the nine arms reproducing bit-identically: the LSTM sweep cell stays at 120.2 p95
    and the GRU sweep cell moves from 117.3 to 125.4, 1.6 m/s worse than its in-regime baseline
    (123.7 +- 1.5). The caveat now states that the Mamba and LSTM references sit 4--5 m/s better
    and the GRU one 1.6 worse, reads `data/probes/` and asserts that ordering.
  - Compute: the NN-dense row of `compute_benchmark.json` (2026-07-12) flew the overwritten model,
    whose flights are 7% shorter than the champion's (721 s against 778 s mean on the benchmark's
    200 scenarios). Re-run on one idle core (rustc 1.98.1, no simulator change since): dense 1.98
    ms (was 1.88), Mamba 3.07 (3.14), FTC 0.91 (0.90), FNPAG 85.5 (87.1). The Mamba's cost over
    the dense network is 1.5x (was 1.7x); the 3.1 ms, the 28x, the three and a half times FTC,
    the 4 us per update, the 0.27 ms per replan and the factor of thirty between deadline margins
    hold. Every compute quote (abstract, Section 1, the scheme table caption, Section 7.2, the
    Discussion, the conclusion, Appendix A's timing paragraph, Appendix C) now reads the file;
    Appendix A names its spread as the standard deviation over the five repeats, 0.6--2.0% of the
    median (0.1--2.2% on the old run). `fig_classical_vs_nn` moves with the four timings.
  - The "2--4 times" of the abstract, Section 2 and the regime-table caption is now asserted as
    each network's tail loss over the classical laws' mean loss (2.5--4.0; the overwritten dense
    row had put it at 4.8). The colophon lists `compute_benchmark.json` and `data/probes/`. The
    committed `paper.pdf` is not recompiled here.
- 2026-10-06 (#179): the bundle carries the per-scenario regime end to end. The 24
  `runs/ou_marginal/` cells gain their `run.jsonl.gz` (stripped with `strip_run_logs.py`, 114 MB
  for the tree; `SHA256SUMS.runlogs` lists 119 logs), so their `results.json` rows carry
  `actual_sims` and the best validation RMS like every other study's; the #173 cells count their
  training evals at 512 x 2 (`aggregate_results._infer_training_n_sims`). The Release asset the
  logs live in is named once, `write_provenance.RELEASE_TAG` (`arxiv-v4`, uploaded by the v4
  release, #183), and `fetch_run_logs.sh` reads the URL from the committed `provenance.json`.
  `data/confirmatory_marginal.json` extracts every cell of the scorer's manifests
  (`confirmatory_cells.txt` + `confirmatory_cells_v4.txt`: 56 rows, was 6) plus the per-10^6
  heat-load violation count and each row's `eval_commit` (49 rows were scored after the file's
  `freeze_commit`); `data/quote_marginal.json` extracts every cell of
  `quote_results.json` (118 rows, was 37); new `data/heat_load_slope.json` copies the #192
  ceiling study verbatim. `results.typ` gains `per_draw_regime(key)` (the v4 performance-table
  rows read `results.json` through it only, as the shared-path rows read `legacy_regime`),
  `slope()` / `slope_summary()` over the ceiling study, whose protocol record is asserted equal to
  `quote_marginal.json`'s marginal regime; `check_results_schema.py` refuses a `rl/` or
  `ou_marginal/` run flagged legacy. `make paper` runs the third extract, `make check` its
  `--check`, and the v4 scorers have opt-in targets (`mc-confirmatory-v4`, `mc-quote-marginal`,
  `mc-heat-load-slope`). `provenance.json`'s `noise_regime` names per_draw as the main-body regime
  and the shared-path files as the development regime. No shared-path bundle file changes bytes,
  every arxiv-v3 row of the two grown extracts keeps its values, no figure changes; the text is
  untouched (the single-regime rewrite is #181).
  `collect_runs.py` gzips every `run_*.jsonl` fragment of a resumed run (later fragments supersede
  the generations they re-log); it took the newest only, so `ou_marginal/hl_gru_p1014` (resumed at
  generation 16606) counted 3395 of its 20001 generations in `actual_sims`. The five `headline/`
  cells (5001 or 15001 of 20001) and `classical_baselines/fnpag` (6 of 372) are re-collected too:
  only their `actual_sims` moves, the best validation RMS and every figure stay. `make paper` runs
  `check-logs` after `fetch-logs`, which is a no-op once any log is present.
- 2026-10-08 (#176): `compute_benchmark.json` is re-measured under rustc 1.99.0 (the #170 follow-up
  run used 1.98.1) on one idle core: Mamba 2.84 ms (was 3.07), dense 1.86 (1.98), FTC 0.85
  (0.91), FNPAG 82.4 (85.5). The network's factor below FNPAG moves from 28x to 29x wherever it is
  quoted, the scheme table caption's slow class from 86 to 82 ms/sim, FNPAG's per-replan share
  from 0.27 to 0.26 ms (27 to 26 ms at the 100x scaling), and Appendix A's compiler to 1.99 with a
  repeat spread of 0.2--0.8% of the median. The 1.5x cost of state, the 4 us per update and the
  factor of thirty between deadline margins hold. The Compute paragraph's "roughly three and a
  half times FTC" (3.36 on this run) now reads the file. `fig_classical_vs_nn` moves with the four
  timings. The committed `paper.pdf` is not recompiled here.
- 2026-10-08 (#180, first batch): six figures regenerate from the per-scenario bundle, the
  development-regime figures keep their bytes. `fig_arch_tail` plots the #173 headline-allocation
  cells per seed (CVaR95 and CVaR99.9 on the 10^6 per-scenario pool, replicate-SE whiskers, the
  retuned joint-FTC and FNPAG as reference lines, seed 1 of Mamba-962 starred); `fig_classical_vs_nn`
  reads the same pool (three-seed means for the two network points) with axis limits derived from the
  data; `fig_survival` draws the survival curves of the five finalists from a `survival_sample` the
  extract now carries for those cells only (`extract_confirmatory_marginal.SURVIVAL_CELLS`);
  `fig_loss_vs_tail` plots the nine headline-allocation runs (filled) and the four bundled fine-tunes
  (hollow), the Spearman over the nine quoted as descriptive; `fig_ablation` reads the v4 champion's
  `runs/ou_marginal/hl_mamba_p962/ablation_results.json` (bundled with its `fresh_pool_requote.json`,
  #176); `fig_robustness` reads `stress_depth.json` (n = 10 000, bootstrap CI whiskers) against each
  scheme's 10^6 nominal. `figlib` gains `marginal()` and `stress_depth()`, both asserting per_draw.
  `fig_classical_vs_nn` reads the rustc 1.99.0 benchmark above. Appendix D keeps the three
  schemes v4 keeps (the deployed Mamba champion, the retuned joint-FTC and FNPAG), each card
  re-flown by `collect_appendix.py` on the final-evaluation pool under per-scenario noise (stated in
  its `stats.json`, asserted by `appendix.typ`); the seven other cards are gone with their
  `figures/appendix/` directories, and the appendix intro names the three. Left for the rest of
  #180: `fig_objective_centering` (#177) and `quantization_sweep` (#178). The text is otherwise
  untouched (#181).
