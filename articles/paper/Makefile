# Rebuild the paper's figures, results.json and PDF from the committed bundle.
#
#   make -C articles/paper                 figures (default): the 18 fig_*.svg from data/ + the bundle
#   make -C articles/paper paper           fetch-logs (if absent) -> check-logs -> results -> confirmatory-marginal -> quote-marginal -> heat-load-slope -> figures -> provenance -> pdf
#   make -C articles/paper check           SHA256SUMS + results.json schema + the three per-scenario extracts + figures/results.json unchanged in git
#   make -C articles/paper -B figures      force every figure (CI does this, then `check`)
#   make -C articles/paper -n paper        print what would run
#
# Bundle-only by default: no Rust build, no simulator, no training_output/. The
# `mc-*` targets at the bottom re-fly cells and are opt-in (hours of compute; never
# CI). Every recipe runs from the repo root; the scripts resolve REPO from their own
# path and typst resolves figure paths relative to paper.typ. GNU make 3.81 (macOS).

HERE := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ifneq ($(CURDIR),$(HERE))
$(error run as: make -C articles/paper <target>)
endif
ROOT := $(abspath $(HERE)/../..)
UVPY := cd $(ROOT) && uv run python
SHASUM := shasum -a 256

SCRIPTS := scripts
DATA := data
# The SHA256SUMS domain, run from $(DATA): sums writes it, check-sums requires it verbatim.
TRACKED_DATA_SUMS = git ls-files -z . | tr '\0' '\n' | grep -v -e '^SHA256SUMS' -e '^provenance.json$$' | LC_ALL=C sort | xargs $(SHASUM)
RUNS := $(DATA)/runs
FIGS := figures
PDF ?= articles/paper/paper.pdf

.DEFAULT_GOAL := figures
.PHONY: figures results confirmatory-marginal quote-marginal heat-load-slope fetch-logs provenance pdf paper check check-sums check-logs check-schema \
        check-provenance check-marginal check-quote check-slope sums help \
        need-sim mc-data mc-far-tail mc-confirmatory mc-corridor mc-appendix mc-robustness mc-robustness-retrain \
        mc-objective-centering mc-centered-depth mc-compute-benchmark mc-requote mc-quant probes \
        mc-confirmatory-v4 mc-quote-marginal mc-heat-load-slope mc-stress-depth mc-centered-depth-v4 mc-objective-centering-v4

# ---------------------------------------------------------------------------
# Figures: figures/fig_<name>.svg <- scripts/fig_<name>.py + figlib + fonts + data
# ---------------------------------------------------------------------------
FIG_NAMES := ablation arch_tail classical_vs_nn corridor cost_transform curation islands joint_reference \
             loss_vs_tail objective_centering optimizer output_param pareto plateau robustness seed_strategy \
             survival training_n_sims
FIGURES := $(FIG_NAMES:%=$(FIGS)/fig_%.svg)
FONTS := fonts/STIXTwoText.ttf fonts/STIXTwoText-Italic.ttf
FIGLIB := $(SCRIPTS)/figlib.py $(FONTS)

figures: $(FIGURES)

$(FIGS)/fig_%.svg: $(SCRIPTS)/fig_%.py $(FIGLIB)
	$(UVPY) articles/paper/$<

# Per-figure data inputs (the recipe comes from the pattern rule above). A missing
# input is a make error, never a thinner figure -- except the $(wildcard) sweep
# parquets, which make cannot miss; fig_pareto.py errors on an absent cell itself.
SWEEP_MANIFESTS := $(ROOT)/configs/training/sweep/manifest.json $(ROOT)/configs/training/sweep/manifest_floor.json
SWEEP_PARQUETS := $(wildcard $(RUNS)/architecture_sweep/*/final_eval.parquet)

$(FIGS)/fig_ablation.svg: $(RUNS)/ou_marginal/hl_mamba_p962/ablation_results.json
$(FIGS)/fig_arch_tail.svg: $(DATA)/confirmatory_marginal.json
$(FIGS)/fig_classical_vs_nn.svg: $(DATA)/confirmatory_marginal.json $(DATA)/compute_benchmark.json
$(FIGS)/fig_corridor.svg: $(DATA)/corridor.npz
$(FIGS)/fig_cost_transform.svg: $(DATA)/far_tail_eval.json
$(FIGS)/fig_curation.svg: $(DATA)/results.json $(DATA)/far_tail_eval.json
$(FIGS)/fig_islands.svg:
$(FIGS)/fig_joint_reference.svg: $(DATA)/results.json
$(FIGS)/fig_loss_vs_tail.svg: $(DATA)/results.json $(DATA)/confirmatory_marginal.json
$(FIGS)/fig_objective_centering.svg: $(DATA)/objective_centering.json $(DATA)/centered_depth.json
$(FIGS)/fig_optimizer.svg: $(DATA)/results.json
$(FIGS)/fig_output_param.svg: $(DATA)/results.json
$(FIGS)/fig_pareto.svg: $(SWEEP_MANIFESTS) $(SWEEP_PARQUETS)
$(FIGS)/fig_plateau.svg: $(DATA)/plateau.json
$(FIGS)/fig_robustness.svg: $(DATA)/stress_depth.json $(DATA)/confirmatory_marginal.json
$(FIGS)/fig_seed_strategy.svg: $(DATA)/results.json $(DATA)/sigma_extras.json
$(FIGS)/fig_survival.svg: $(DATA)/confirmatory_marginal.json
$(FIGS)/fig_training_n_sims.svg: $(DATA)/results.json

# results.json is a committed bundle product consumed by the figures as a plain file
# (deliberately no rule: a fresh clone's mtimes must not trigger a rebuild that needs
# the ~310 MB run logs; a missing file is "No rule to make target"). `make results`
# regenerates it explicitly; aggregate_results.py exits non-zero without the logs.

# ---------------------------------------------------------------------------
# results.json <- aggregate_results.py over the bundle parquets + the run logs
# ---------------------------------------------------------------------------
RUN_LOGS := $(shell find $(RUNS) -name run.jsonl.gz 2>/dev/null)

results:
	$(UVPY) articles/paper/scripts/aggregate_results.py

# ---------------------------------------------------------------------------
# confirmatory_marginal.json <- the 10^6 per-scenario confirmatory (issues #137, #174)
# ---------------------------------------------------------------------------
# Every cell of the scorer's manifests (experiments/ou_marginal/confirmatory_cells*.txt) and the fields
# the paper quotes, extracted from experiments/ou_marginal/confirmatory_marginal.json (written by
# confirmatory_marginal.py there: hours per cell, never a make recipe; `mc-confirmatory-v4` below). Pure stdlib.
confirmatory-marginal:
	$(UVPY) articles/paper/scripts/extract_confirmatory_marginal.py

# ---------------------------------------------------------------------------
# quote_marginal.json <- the n = 1000 paired per-scenario quotes (issues #157, #175)
# ---------------------------------------------------------------------------
# Every cell of experiments/ou_marginal/quote_results.json (written by quote_marginal.py there: every
# cell flown under both regimes; never a make recipe; `mc-quote-marginal` below), the quoted fields only. Pure stdlib.
quote-marginal:
	$(UVPY) articles/paper/scripts/extract_quote_marginal.py

# ---------------------------------------------------------------------------
# heat_load_slope.json <- the heat-load ceiling sensitivity (issue #192)
# ---------------------------------------------------------------------------
# Verbatim copy of experiments/ou_marginal/heat_load_slope.json (written by heat_load_slope.py there;
# `mc-heat-load-slope` below). Pure stdlib.
heat-load-slope:
	$(UVPY) articles/paper/scripts/extract_heat_load_slope.py

# The run.jsonl.gz of data/SHA256SUMS.runlogs are a Release asset (URL: data/provenance.json's
# run_logs_asset, named by write_provenance.py), gitignored. Network.
fetch-logs:
	@if [ -z "$(RUN_LOGS)" ]; then bash $(SCRIPTS)/fetch_run_logs.sh; else echo "run logs present: $(words $(RUN_LOGS)) files"; fi

# ---------------------------------------------------------------------------
# Provenance, PDF, the one top-level target
# ---------------------------------------------------------------------------
provenance:
	$(UVPY) articles/paper/scripts/write_provenance.py

# SOURCE_DATE_EPOCH = the HEAD commit time, so the PDF's creation date is reproducible; git_head
# feeds the colophon (`-dirty` when a tracked paper input differs from HEAD).
# `make pdf PDF=/tmp/paper.pdf` compiles without touching the committed PDF (CI).
GIT_HEAD = $$(git rev-parse --short HEAD)$$(test -z "$$(git status --porcelain -- articles/paper ':(exclude)articles/paper/paper.pdf')" || echo -dirty)
pdf:
	cd $(ROOT) && SOURCE_DATE_EPOCH=$$(git log -1 --format=%ct) typst compile --input git_head=$(GIT_HEAD) articles/paper/paper.typ $(PDF)

# Recursive so the log list is re-evaluated after fetch-logs. check-logs first: fetch-logs is a no-op
# once any log is present, and an older asset's subset would drop actual_sims from the rows it lacks.
paper: fetch-logs
	$(MAKE) check-logs
	$(MAKE) results
	$(MAKE) confirmatory-marginal
	$(MAKE) quote-marginal
	$(MAKE) heat-load-slope
	$(MAKE) figures
	$(MAKE) provenance
	$(MAKE) pdf

# ---------------------------------------------------------------------------
# FROZEN inputs: committed data files with NO producer in the tree. They were
# hand-extracted once (the producing commit and the method are recorded here);
# a missing one is an error, never a silent re-derivation (the recipe only checks
# existence so `make -B` still works).
#   plateau.json                     b85db68  running-min validation RMS per generation for headline/dense_p515 and
#                                             headline/dense_p972, read off the 4-segment training JSONLs (fig_plateau)
#   sigma_extras.json                7234f40  n=1000 final-eval stats of the 16_sigma_extras.sh seed repeats
#                                             (ga/cmaes x fixed/rotating, s1-s3); the eval step lived in a plan doc
#   selection_gate.json              20482aa  validation-gate query count for the pool-roles table
#   nominal_floor.json               fd396b4  undispersed-nominal correction-DV floor (section on the objective)
#   fnpag_failure_classification.json f73a454 FNPAG confirmatory failed seeds: timeout vs physical
#   s2_failure_classification.json   2b683e0  Mamba s2 confirmatory failed seeds: timeout vs physical (61/1e6 physical)
#   quant/ticks_per_sim.json         d9a413e  guidance ticks per sim behind the Appendix C compute table
# ---------------------------------------------------------------------------
FROZEN := $(DATA)/plateau.json $(DATA)/sigma_extras.json $(DATA)/selection_gate.json $(DATA)/nominal_floor.json \
          $(DATA)/fnpag_failure_classification.json $(DATA)/s2_failure_classification.json $(DATA)/quant/ticks_per_sim.json
$(FROZEN):
	@test -f $@ || { echo "error: $@ is a FROZEN input with no producer in the tree (see the FROZEN block in articles/paper/Makefile)"; exit 1; }

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
# $(FROZEN) as prerequisites: a missing frozen input fails here with its FROZEN message
# (the block above is defined first: make expands prerequisite lists when it reads the rule).
# `git diff HEAD` so a regenerated-and-staged file still counts as changed.
check: check-sums check-schema check-marginal check-quote check-slope check-provenance $(FROZEN)
	cd $(ROOT) && git diff HEAD --exit-code --stat -- articles/paper/figures articles/paper/data/results.json articles/paper/data/provenance.json
	cd $(ROOT) && test -z "$$(git status --porcelain -- articles/paper/figures)" || { git status --porcelain -- articles/paper/figures; echo "error: untracked/modified figures"; exit 1; }

# provenance.json must be what this tree yields (only per-input facts, so it changes only with an input).
check-provenance:
	$(UVPY) articles/paper/scripts/write_provenance.py --check

# data/confirmatory_marginal.json must be what its experiments/ source yields.
check-marginal:
	$(UVPY) articles/paper/scripts/extract_confirmatory_marginal.py --check

# data/quote_marginal.json must be what its experiments/ source yields.
check-quote:
	$(UVPY) articles/paper/scripts/extract_quote_marginal.py --check

# data/heat_load_slope.json must be what its experiments/ source yields.
check-slope:
	$(UVPY) articles/paper/scripts/extract_heat_load_slope.py --check

# data/SHA256SUMS = what `sums` writes: every tracked file under data/ except itself and
# provenance.json. Recomputed and diffed (not `shasum --check`, which only verifies the
# files LISTED and would pass a bundle file added without `make sums`).
check-sums:
	cd $(DATA) && $(TRACKED_DATA_SUMS) | diff SHA256SUMS - || { echo "error: data/SHA256SUMS differs from the tree (< committed, > tree): inspect, then make -C articles/paper sums"; exit 1; }

# The run logs (after fetch-logs): the extracted Release asset, file by file.
check-logs:
	cd $(DATA) && $(SHASUM) --check --quiet SHA256SUMS.runlogs

check-schema:
	$(UVPY) articles/paper/scripts/check_results_schema.py

# Regenerate both checksum files after a deliberate bundle change (12_collect_results.sh, mc-* targets).
# SHA256SUMS.runlogs only when at least as many run logs are present as it lists: with none or a
# subset (a fresh clone, or one re-collected cell) the find would shrink the manifest `check-logs`
# verifies the Release asset against. To drop a cell's log on purpose, delete the manifest first.
# A subset is not fixed by `fetch-logs` (a no-op once any log is present), and fetch_run_logs.sh
# untars over the re-collected logs: set them aside, fetch, put them back, then sums.
sums:
	cd $(DATA) && $(TRACKED_DATA_SUMS) > SHA256SUMS
	@cd $(DATA) && have=$$(find runs -name run.jsonl.gz | wc -l | tr -d ' ') && listed=$$(cat SHA256SUMS.runlogs 2>/dev/null | wc -l | tr -d ' ') && \
	if [ $$have -eq 0 ] || [ $$have -lt $$listed ]; then \
	  echo "run logs: $$have present, $$listed listed: SHA256SUMS.runlogs left as is. None present: make fetch-logs, then sums." \
	    "Some present: fetch_run_logs.sh overwrites them, so set re-collected logs aside, run it, put them back, then sums."; \
	else find runs -name run.jsonl.gz | LC_ALL=C sort | xargs $(SHASUM) > SHA256SUMS.runlogs; fi

help:
	@sed -n '2,8p' $(lastword $(MAKEFILE_LIST))

# ---------------------------------------------------------------------------
# mc-data: OPT-IN producers that re-fly deployed cells. Never default, never CI.
# Need training_output/ (the cells are not committed), the PyO3 build (./build.sh),
# and every committed number was produced under the legacy noise regime (the
# scripts pin it; ADR-0003/0006) except the labelled per_draw cells of
# centered_depth.json. Wall times are single-machine estimates.
# ---------------------------------------------------------------------------
need-sim:
	@test -d $(ROOT)/training_output || { echo "error: training_output/ missing -- mc-* targets re-fly cells (experiments/paper/README.md)"; exit 1; }
	@$(UVPY) -c "import aerocapture_rs" 2>/dev/null || { echo "error: aerocapture_rs not built (./build.sh)"; exit 1; }

mc-data: mc-far-tail mc-confirmatory mc-corridor mc-appendix mc-robustness mc-robustness-retrain \
         mc-objective-centering mc-centered-depth mc-compute-benchmark mc-requote mc-quant probes

# far_tail_eval.json -- n=10000 reserved 2M pool per cell (~minutes/cell, FNPAG ~50x slower).
# The cell list is label:toml[:bundle_key]; the committed file records only labels, so pass CELLS=...
# (the 26 committed labels: `python -c "import json;print([c['label'] for c in json.load(open('data/far_tail_eval.json'))['cells']])"`).
mc-far-tail: need-sim
	@test -n "$(CELLS)" || { echo "error: CELLS='label:toml ...' required (see the comment above mc-far-tail)"; exit 1; }
	$(UVPY) articles/paper/scripts/far_tail_eval.py --n-sims 10000 --cells $(CELLS)

# confirmatory_eval.json -- 10 x 100000 scenarios per cell, [2^31, 2^32) pool: HOURS per cell.
# The committed file is self-describing (label, toml, bundle_key per cell), so the cell list comes from it;
# --force re-flies every cell it holds. Rerun after an interruption, it reuses the interrupted cell's
# finished replicates (<out>.partial/) but re-flies the cells it had completed: to resume by hand, run
# confirmatory_eval.py --force --cells with the cells not yet re-flown.
mc-confirmatory: need-sim
	$(UVPY) articles/paper/scripts/confirmatory_eval.py --force --replicates 10 --n 100000 --cells \
	    $$($(UVPY) -c "import json; print(' '.join(':'.join(filter(None, (c['label'], c['toml'], c.get('bundle_key')))) for c in json.load(open('articles/paper/data/confirmatory_eval.json'))['cells']))")

# corridor.npz -- 300000 randomized piecewise-constant sims (~1 h) + boundary traces.
mc-corridor: need-sim
	$(UVPY) articles/paper/scripts/collect_corridor.py --n-sims 300000

# figures/appendix/<scheme>/{7 SVGs, stats.json} -- 1000 sims x 3 schemes with trajectories, per-scenario noise (~2 min).
mc-appendix: need-sim
	$(UVPY) articles/paper/scripts/collect_appendix.py --n-sims 1000

# robustness_stress.json / robustness_retrain.json / objective_centering.json -- 1000 sims per cell on the 9M stress pool.
mc-robustness: need-sim
	$(UVPY) articles/paper/scripts/robustness_stress.py --n-sims 1000
mc-robustness-retrain: need-sim
	$(UVPY) articles/paper/scripts/robustness_retrain_eval.py --n-sims 1000
mc-objective-centering: need-sim
	$(UVPY) articles/paper/scripts/objective_centering_eval.py --n-sims 1000

# centered_depth.json -- the three centered-Mamba seeds + the two joint-FTC baselines at n = 10000 on the
# 9M stress pool, both regimes, bootstrap CIs and paired deltas (issue #156; ~minutes).
mc-centered-depth: need-sim
	$(UVPY) articles/paper/scripts/centered_depth_eval.py --n-sims 10000

# stress_depth.json -- the v4 champion + the four #172 classical retunes at n = 10000 on the 9M stress pool,
# per-scenario noise only, paired with bootstrap CIs (issue #176; ~30 min, FNPAG-dominated).
mc-stress-depth: need-sim
	$(UVPY) articles/paper/scripts/stress_depth_eval.py --n-sims 10000

# centered_depth_v4.json -- the three centered-Mamba seeds retrained under per-scenario noise + the two joint-FTC
# baselines retuned the same way (classical/ftc_joint, classical/ftc_joint_high) at n = 10000 on the 9M stress
# pool, per-scenario noise only, bootstrap CIs and paired deltas (issue #177; ~minutes once
# experiments/paper/14_objective_centering.sh v4 has trained the cells, which it ends with this scorer).
mc-centered-depth-v4: need-sim
	$(UVPY) articles/paper/scripts/centered_depth_eval.py --v4 --n-sims 10000

# objective_centering_v4.json -- the five dense lever cells retrained under per-scenario noise at n = 1000 on the
# 9M stress pool, per_draw (issue #177, the figure's optional cells; 14_objective_centering.sh v4-dense).
mc-objective-centering-v4: need-sim
	$(UVPY) articles/paper/scripts/objective_centering_eval.py --v4 --n-sims 1000

# compute_benchmark.json -- 200 single-threaded sims per scheme; wall-clock, never diffable.
mc-compute-benchmark: need-sim
	$(UVPY) articles/paper/scripts/compute_benchmark.py --n-sims 200

# runs/headline/mamba_p962/fresh_pool_requote.json -- the headline cell on the untouched 8M pool (1000 sims).
mc-requote: need-sim
	$(UVPY) articles/paper/scripts/fresh_pool_requote.py training_output/mamba_p962_long --toml configs/training/sweep/mamba_p962.toml --n-sims 1000

# quant/{quantization_results,finalists_results,bench_forward}.json + figures/quantization_sweep.svg:
# copied from the quantization campaign's training_output by its runner.
mc-quant: need-sim
	cd $(ROOT) && ./experiments/paper/17_quantization.sh collect

# ---------------------------------------------------------------------------
# v4 per-scenario scorers (issue #179): every one flies under per_draw density noise (ADR-0006,
# one realization per scenario) and writes under experiments/ou_marginal/, from where the
# extracts above copy into the bundle. Resumable: each persists per replicate / per cell.
# ---------------------------------------------------------------------------
# experiments/ou_marginal/confirmatory_marginal.json -- 10 x 100000 per-scenario scenarios per cell of
# confirmatory_cells_v4.txt (issue #174): HOURS per cell, FNPAG ~2 h; scored cells are skipped.
mc-confirmatory-v4: need-sim
	$(UVPY) -u experiments/ou_marginal/confirmatory_marginal.py --manifest experiments/ou_marginal/confirmatory_cells_v4.txt

# experiments/ou_marginal/quote_results.json -- every OU-marginal cell on the paired n = 1000 pool under
# both regimes (frozen = the shared path, marginal = per-scenario; issues #157, #175; ~minutes per cell).
mc-quote-marginal: need-sim
	$(UVPY) -u experiments/ou_marginal/quote_marginal.py

# experiments/ou_marginal/heat_load_slope.json -- the six #192 ceiling legs on the paired n = 1000 pool,
# each at its own ceiling and at the v4 limit (marginal regime; ~minutes).
mc-heat-load-slope: need-sim
	$(UVPY) -u experiments/ou_marginal/heat_load_slope.py

# stress_depth.json (mc-stress-depth above) and the champion's runs/ou_marginal/hl_mamba_p962/
# {ablation_results,fresh_pool_requote}.json (collect_runs.py copies them) are #176's; fig_robustness and
# fig_ablation read them since #180, their results.typ accessors come with #181, as do those of the two
# #177 files (mc-centered-depth-v4, mc-objective-centering-v4 above: committed with the issue's outcome,
# then `make sums`). Not yet a producer: quant_v4/ (#178, 17_quantization.sh collect) gets its mc-* target,
# accessor and check rule with the issue that writes the file.

# probes/<x>_probe_results.json <- the probe drivers' training_output/<x>_probe/probe_results.json.
probes: $(DATA)/probes/cfc_probe_results.json $(DATA)/probes/mamba3_probe_results.json $(DATA)/probes/xlstm_probe_results.json
$(DATA)/probes/%_probe_results.json: $(ROOT)/training_output/%_probe/probe_results.json
	cp $< $@
